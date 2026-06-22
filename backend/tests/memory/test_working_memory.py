"""
tests/memory/test_working_memory.py — Unit tests for WorkingMemoryManager.

WHY fakeredis
-------------
Running tests against a real Redis instance creates an external dependency
that breaks CI environments without Redis installed.  fakeredis implements
the full redis-py API in-memory, so our tests exercise the exact same code
paths as production — including TTL tracking, SCAN, RPUSH, and GET —
without requiring a live Redis process.

WHY unittest.IsolatedAsyncioTestCase
--------------------------------------
WorkingMemoryManager is fully async.  IsolatedAsyncioTestCase provides an
event loop for each test method and handles proper setup/teardown without
requiring pytest-asyncio or asyncio.run() boilerplate.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest

# Ensure backend package is importable regardless of how tests are invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import fakeredis.aioredis as fakeredis_async

from app.memory.working_memory import WorkingMemoryManager
from app.memory.exceptions import MemoryStoreError, MemoryRetrievalError


def _make_manager() -> WorkingMemoryManager:
    """
    Return a WorkingMemoryManager whose Redis client is replaced with a
    fakeredis instance.  We patch _client directly (rather than mocking
    from_url) so all real code paths inside the manager are exercised.
    """
    mgr = WorkingMemoryManager(redis_url="redis://localhost:6379/0")
    # Bypass connect() — inject fakeredis directly.
    fake = fakeredis_async.FakeRedis(decode_responses=True)
    mgr._client = fake
    return mgr


class TestWorkingMemoryStoreRetrieve(unittest.IsolatedAsyncioTestCase):
    """Core store/retrieve cycle."""

    async def asyncSetUp(self) -> None:
        self.mgr = _make_manager()
        self.task_id = "test-task-001"

    async def asyncTearDown(self) -> None:
        await self.mgr.clear(self.task_id)

    async def test_store_and_retrieve_string(self) -> None:
        await self.mgr.store(self.task_id, "plan", "my plan", category="planning")
        result = await self.mgr.retrieve(self.task_id, "plan")
        self.assertEqual(result, "my plan")

    async def test_store_and_retrieve_dict(self) -> None:
        payload = {"subtasks": ["a", "b"], "count": 2}
        await self.mgr.store(self.task_id, "execution_plan", payload, category="planning")
        result = await self.mgr.retrieve(self.task_id, "execution_plan")
        self.assertEqual(result, payload)

    async def test_retrieve_returns_none_for_missing_key(self) -> None:
        result = await self.mgr.retrieve(self.task_id, "nonexistent_key_xyz")
        self.assertIsNone(result)

    async def test_store_overwrites_existing_value(self) -> None:
        await self.mgr.store(self.task_id, "counter", 1)
        await self.mgr.store(self.task_id, "counter", 99)
        result = await self.mgr.retrieve(self.task_id, "counter")
        self.assertEqual(result, 99)

    async def test_retrieve_searches_across_categories(self) -> None:
        """
        If a key is stored with a non-default category, retrieve() should
        still find it via the SCAN fallback path.
        """
        await self.mgr.store(
            self.task_id, "output", "researcher result", category="subtask_outputs"
        )
        # Retrieve without specifying category — should still find it.
        result = await self.mgr.retrieve(self.task_id, "output")
        self.assertEqual(result, "researcher result")


class TestWorkingMemoryAppend(unittest.IsolatedAsyncioTestCase):
    """Append-to-list operations."""

    async def asyncSetUp(self) -> None:
        self.mgr = _make_manager()
        self.task_id = "test-task-002"

    async def asyncTearDown(self) -> None:
        await self.mgr.clear(self.task_id)

    async def test_append_builds_list(self) -> None:
        await self.mgr.append(self.task_id, "error_log", "error 1")
        await self.mgr.append(self.task_id, "error_log", "error 2")
        await self.mgr.append(self.task_id, "error_log", {"msg": "error 3"})

        # Verify the list contents directly via the raw Redis client.
        # append() stores list values as JSON strings under the "list" category.
        # Redis LIST keys must be read with LRANGE, not GET — retrieve_all()
        # uses GET and correctly skips list-type keys.
        raw_list = await self.mgr._client.lrange(
            f"agentis:task:{self.task_id}:list:error_log", 0, -1
        )
        import json
        parsed = [json.loads(item) for item in raw_list]
        self.assertEqual(parsed, ["error 1", "error 2", {"msg": "error 3"}])

    async def test_append_different_tasks_dont_interfere(self) -> None:
        await self.mgr.append(self.task_id, "messages", "task1 msg")
        await self.mgr.append("other-task-999", "messages", "task2 msg")

        raw_task1 = await self.mgr._client.lrange(
            f"agentis:task:{self.task_id}:list:messages", 0, -1
        )
        self.assertEqual(len(raw_task1), 1)


class TestWorkingMemoryClear(unittest.IsolatedAsyncioTestCase):
    """clear() removes all keys for a task."""

    async def asyncSetUp(self) -> None:
        self.mgr = _make_manager()
        self.task_id = "test-task-003"

    async def test_clear_removes_all_task_keys(self) -> None:
        await self.mgr.store(self.task_id, "key1", "v1", category="planning")
        await self.mgr.store(self.task_id, "key2", "v2", category="subtask_outputs")
        await self.mgr.append(self.task_id, "errors", "e1")

        await self.mgr.clear(self.task_id)

        result1 = await self.mgr.retrieve(self.task_id, "key1")
        result2 = await self.mgr.retrieve(self.task_id, "key2")
        self.assertIsNone(result1)
        self.assertIsNone(result2)

    async def test_clear_does_not_affect_other_tasks(self) -> None:
        other_task = "other-task-000"
        await self.mgr.store(self.task_id, "mykey", "todelete")
        await self.mgr.store(other_task, "safekey", "keepme")

        await self.mgr.clear(self.task_id)

        # The other task's key must survive.
        safe = await self.mgr.retrieve(other_task, "safekey")
        self.assertEqual(safe, "keepme")

        # Clean up the other task manually.
        await self.mgr.clear(other_task)


class TestWorkingMemoryTTL(unittest.IsolatedAsyncioTestCase):
    """TTL is set on stored keys."""

    async def asyncSetUp(self) -> None:
        self.mgr = _make_manager()
        self.task_id = "test-task-004"

    async def asyncTearDown(self) -> None:
        await self.mgr.clear(self.task_id)

    async def test_ttl_is_set_on_store(self) -> None:
        await self.mgr.store(self.task_id, "mykey", "myvalue")
        redis_key = self.mgr._key(self.task_id, "mykey", "general")
        ttl = await self.mgr._client.ttl(redis_key)
        # TTL should be positive (key has an expiry set).
        self.assertGreater(ttl, 0)

    async def test_ttl_is_set_on_append(self) -> None:
        await self.mgr.append(self.task_id, "log", "entry")
        list_key = f"agentis:task:{self.task_id}:list:log"
        ttl = await self.mgr._client.ttl(list_key)
        self.assertGreater(ttl, 0)


class TestWorkingMemoryRetrieveAll(unittest.IsolatedAsyncioTestCase):
    """retrieve_all() returns all keys for a category."""

    async def asyncSetUp(self) -> None:
        self.mgr = _make_manager()
        self.task_id = "test-task-005"

    async def asyncTearDown(self) -> None:
        await self.mgr.clear(self.task_id)

    async def test_retrieve_all_returns_category_data(self) -> None:
        await self.mgr.store(self.task_id, "subtask_1", "output A", category="outputs")
        await self.mgr.store(self.task_id, "subtask_2", "output B", category="outputs")

        result = await self.mgr.retrieve_all(self.task_id, "outputs")

        self.assertIn("subtask_1", result)
        self.assertIn("subtask_2", result)
        self.assertEqual(result["subtask_1"], "output A")
        self.assertEqual(result["subtask_2"], "output B")

    async def test_retrieve_all_empty_category(self) -> None:
        result = await self.mgr.retrieve_all(self.task_id, "nonexistent_category")
        self.assertEqual(result, {})


class TestWorkingMemoryGetTaskContext(unittest.IsolatedAsyncioTestCase):
    """get_task_context() returns a full snapshot organised by category."""

    async def asyncSetUp(self) -> None:
        self.mgr = _make_manager()
        self.task_id = "test-task-006"

    async def asyncTearDown(self) -> None:
        await self.mgr.clear(self.task_id)

    async def test_get_task_context_includes_all_categories(self) -> None:
        await self.mgr.store(self.task_id, "plan", "supervisor plan", category="planning")
        await self.mgr.store(self.task_id, "output", "writer output", category="results")

        ctx = await self.mgr.get_task_context(self.task_id)

        self.assertIn("planning", ctx)
        self.assertIn("results", ctx)
        self.assertEqual(ctx["planning"]["plan"], "supervisor plan")
        self.assertEqual(ctx["results"]["output"], "writer output")

    async def test_get_task_context_empty_task(self) -> None:
        ctx = await self.mgr.get_task_context("empty-task-xyz")
        self.assertEqual(ctx, {})


if __name__ == "__main__":
    unittest.main()
