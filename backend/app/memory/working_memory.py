"""
memory/working_memory.py — Short-term per-task working memory backed by Redis.

WHAT THIS MODULE PROVIDES
--------------------------
A per-task key-value scratchpad that lives in Redis for the duration of a
task run (up to 24 hours, then auto-expired).  It allows LangGraph nodes
to stash and retrieve intermediate results without bloating AgentState with
ephemeral data that doesn't belong in the PostgreSQL task record.

WHY REDIS FOR WORKING MEMORY
------------------------------
Redis stores data in RAM, making it orders of magnitude faster than
PostgreSQL for frequent small reads and writes within a single task run.
It also provides native TTL support: we set a 24-hour expiry on every key
so stale task data is never left behind, even if the cleanup path (clear())
is never reached due to a crash.

WHY PER-TASK NAMESPACED KEYS
------------------------------
All keys follow the pattern `agentis:task:{task_id}:{category}:{key}`.
The namespace has three levels:
  - "agentis"  — prevents collisions with any other Redis clients sharing
                  the same Redis instance.
  - task_id    — isolates one task's data from another.  Concurrent tasks
                  can write to Redis without interfering.
  - category   — groups related keys (e.g. "subtask_outputs", "errors"),
                  making retrieve_all() and debugging easier.

WHY JSON SERIALISATION (not pickle)
-------------------------------------
JSON produces human-readable values that can be inspected with redis-cli.
Pickle produces opaque binary blobs, introduces security risks (arbitrary
code execution on deserialise), and breaks across Python versions.
The cost is that only JSON-serialisable values can be stored — dicts, lists,
strings, numbers, and booleans.  This is intentional: if you need to store
a complex object, serialise it to a dict first.

WHY SCAN INSTEAD OF KEYS
--------------------------
`KEYS pattern` is a blocking O(N) command that freezes Redis for all other
clients while it runs.  `SCAN` is a cursor-based command that iterates in
small batches and yields control between batches.  For a production Redis
shared with other services, SCAN is the safe choice.
"""
from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from app.memory.config import memory_settings
from app.memory.exceptions import MemoryRetrievalError, MemoryStoreError

log = structlog.get_logger(__name__)


class WorkingMemoryManager:
    """
    Async Redis-backed working memory for a single Agentis task run.

    Lifecycle:
        wm = WorkingMemoryManager()
        await wm.connect()
        try:
            await wm.store(task_id, "plan", plan_dict, category="planning")
            data = await wm.retrieve(task_id, "plan")
        finally:
            await wm.clear(task_id)
            await wm.close()
    """

    # Key prefix for all Agentis working-memory keys.
    # Having a distinct prefix means we can inspect or flush only Agentis
    # keys in redis-cli without touching other tenants.
    _PREFIX = "agentis:task"

    def __init__(self, redis_url: str | None = None) -> None:
        """
        Initialise the manager with an optional Redis URL override.

        WHY URL OVERRIDE: Tests pass a fakeredis URL; production uses
        the value from MemorySettings.  Accepting None means production
        code doesn't need to know about MemorySettings at all — it just
        instantiates WorkingMemoryManager() with no arguments.
        """
        self._redis_url = redis_url or memory_settings.redis_url
        self._client: aioredis.Redis | None = None
        self._ttl = memory_settings.working_memory_ttl_seconds

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open the Redis connection pool.

        WHY NOT CONNECT IN __init__
        ----------------------------
        __init__ is synchronous; aioredis.from_url() is also synchronous in
        terms of object creation (it doesn't open a socket yet), but the
        idiomatic pattern is to defer network I/O to an async method so
        tests and callers control when the connection is established.
        The connection pool is created lazily on the first command anyway,
        but calling connect() makes the lifecycle explicit and testable.
        """
        try:
            self._client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Ping to verify the connection is reachable.
            await self._client.ping()
            log.info("working_memory_connected", redis_url=self._redis_url)
        except Exception as exc:
            log.warning("working_memory_connect_failed", error=str(exc))
            raise MemoryStoreError(f"Redis connect failed: {exc}") from exc

    async def close(self) -> None:
        """
        Close the Redis connection pool and release resources.

        WHY ALWAYS CALL THIS: aioredis maintains a connection pool.  If
        close() is not called, the pool's sockets linger until the process
        exits.  In a long-running FastAPI server, repeated task runs without
        close() would leak connections.
        """
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:
                log.warning("working_memory_close_failed", error=str(exc))
            finally:
                self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, task_id: str, key: str, category: str = "general") -> str:
        """Build the full Redis key for a task/category/key triple."""
        return f"{self._PREFIX}:{task_id}:{category}:{key}"

    def _assert_connected(self) -> aioredis.Redis:
        """Return the Redis client or raise if connect() was never called."""
        if self._client is None:
            raise MemoryStoreError(
                "WorkingMemoryManager.connect() must be called before use."
            )
        return self._client

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def store(
        self,
        task_id: str,
        key: str,
        value: Any,
        category: str = "general",
    ) -> None:
        """
        Serialise `value` to JSON and store it in Redis with a TTL.

        WHY TTL ON EVERY WRITE
        -----------------------
        Even if clear() is called on task completion, the TTL acts as a
        safety net for tasks that are abandoned or crash mid-run.  Without
        TTL, orphaned keys would accumulate indefinitely.

        Parameters
        ----------
        task_id  : Unique task identifier from AgentState.
        key      : Logical name for this piece of data (e.g. "execution_plan").
        value    : Any JSON-serialisable Python object.
        category : Grouping label (e.g. "planning", "subtask_outputs", "errors").
        """
        client = self._assert_connected()
        redis_key = self._key(task_id, key, category)
        try:
            serialised = json.dumps(value)
            await client.set(redis_key, serialised, ex=self._ttl)
            log.debug("working_memory_store", task_id=task_id, key=redis_key)
        except (TypeError, ValueError) as exc:
            raise MemoryStoreError(
                f"Failed to serialise value for key {redis_key!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise MemoryStoreError(
                f"Redis write failed for key {redis_key!r}: {exc}"
            ) from exc

    async def retrieve(self, task_id: str, key: str) -> Any | None:
        """
        Retrieve and deserialise a value from Redis.

        If the key is not found in the "general" category, SCAN all categories
        for this task to find it.  This makes retrieve() forgiving of
        category mismatches without requiring callers to know the category.

        Returns None if the key does not exist in any category.
        """
        client = self._assert_connected()
        # Try the "general" category first — fast path.
        redis_key = self._key(task_id, key, "general")
        try:
            raw = await client.get(redis_key)
            if raw is not None:
                return json.loads(raw)
        except Exception as exc:
            raise MemoryRetrievalError(
                f"Redis read failed for key {redis_key!r}: {exc}"
            ) from exc

        # Slow path: scan all categories for this task.
        pattern = f"{self._PREFIX}:{task_id}:*:{key}"
        try:
            async for found_key in client.scan_iter(pattern):
                raw = await client.get(found_key)
                if raw is not None:
                    log.debug(
                        "working_memory_retrieve_found_in_category",
                        task_id=task_id,
                        found_key=found_key,
                    )
                    return json.loads(raw)
        except Exception as exc:
            raise MemoryRetrievalError(
                f"Redis scan failed for pattern {pattern!r}: {exc}"
            ) from exc

        return None

    async def retrieve_all(self, task_id: str, category: str) -> dict[str, Any]:
        """
        Return all key-value pairs stored under a given task/category prefix.

        Uses SCAN (not KEYS) to avoid blocking Redis on large keyspaces.

        Returns an empty dict if no keys exist for the given category.
        """
        client = self._assert_connected()
        pattern = f"{self._PREFIX}:{task_id}:{category}:*"
        prefix_len = len(f"{self._PREFIX}:{task_id}:{category}:")
        result: dict[str, Any] = {}

        try:
            async for full_key in client.scan_iter(pattern):
                # Use TYPE to skip non-string keys (e.g. Redis LIST keys created
                # by append()).  Calling GET on a LIST key raises WRONGTYPE.
                key_type = await client.type(full_key)
                if key_type != "string":
                    continue
                raw = await client.get(full_key)
                if raw is not None:
                    # Strip the namespace prefix to return bare key names.
                    short_key = full_key[prefix_len:]
                    result[short_key] = json.loads(raw)
        except Exception as exc:
            raise MemoryRetrievalError(
                f"Redis retrieve_all failed for pattern {pattern!r}: {exc}"
            ) from exc

        log.debug(
            "working_memory_retrieve_all",
            task_id=task_id,
            category=category,
            count=len(result),
        )
        return result

    async def append(self, task_id: str, key: str, value: Any) -> None:
        """
        Append `value` to a Redis list identified by task_id + key.

        Used for accumulating ordered sequences such as error_log entries or
        agent_messages.  Each element is stored as a JSON string inside the
        Redis list so that the list can hold heterogeneous values.

        The TTL is refreshed on every append to ensure the list lives at
        least as long as the most recent write.
        """
        client = self._assert_connected()
        redis_key = self._key(task_id, key, "list")
        try:
            serialised = json.dumps(value)
            await client.rpush(redis_key, serialised)
            await client.expire(redis_key, self._ttl)
            log.debug(
                "working_memory_append", task_id=task_id, list_key=redis_key
            )
        except (TypeError, ValueError) as exc:
            raise MemoryStoreError(
                f"Failed to serialise value for list {redis_key!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise MemoryStoreError(
                f"Redis RPUSH failed for key {redis_key!r}: {exc}"
            ) from exc

    async def clear(self, task_id: str) -> None:
        """
        Delete ALL Redis keys belonging to `task_id`.

        WHY SCAN + DELETE (not FLUSHDB or a pattern-based DEL)
        -------------------------------------------------------
        FLUSHDB would delete every key in Redis, including data from other
        services or tasks.  A pattern-based DEL (`DEL key1 key2 ...`) with
        KEYS blocks Redis.  SCAN + DELETE is safe: it iterates in small
        batches and only deletes the exact keys it finds, leaving unrelated
        data untouched.
        """
        client = self._assert_connected()
        pattern = f"{self._PREFIX}:{task_id}:*"
        deleted = 0

        try:
            async for key in client.scan_iter(pattern):
                await client.delete(key)
                deleted += 1
        except Exception as exc:
            raise MemoryStoreError(
                f"Redis clear failed for task {task_id!r}: {exc}"
            ) from exc

        log.info("working_memory_cleared", task_id=task_id, keys_deleted=deleted)

    async def get_task_context(self, task_id: str) -> dict[str, Any]:
        """
        Return a full snapshot of all working memory for `task_id`, organised
        by category.

        Example return value:
            {
                "general":         {"some_key": "some_value"},
                "planning":        {"execution_plan": {...}},
                "subtask_outputs": {"subtask_1": "researcher output ..."},
                "list":            {"error_log": ["err1", "err2"]},
            }

        WHY THIS EXISTS
        ---------------
        Debugging a crashed task is much easier when you can dump the entire
        working-memory snapshot in one call rather than guessing category names.
        The LangSmith trace shows agent outputs, but working memory captures
        intermediate scratchpad data that never enters the graph state.
        """
        client = self._assert_connected()
        pattern = f"{self._PREFIX}:{task_id}:*"
        context: dict[str, dict[str, Any]] = {}

        try:
            async for full_key in client.scan_iter(pattern):
                # Skip non-string keys (e.g. LIST keys from append()) to avoid
                # WRONGTYPE errors when calling GET on them.
                key_type = await client.type(full_key)
                if key_type != "string":
                    continue
                raw = await client.get(full_key)
                if raw is None:
                    continue
                # full_key = "agentis:task:{task_id}:{category}:{key}"
                parts = full_key.split(":", maxsplit=4)
                # parts[3] = category, parts[4] = key
                if len(parts) < 5:
                    continue
                category = parts[3]
                short_key = parts[4]
                context.setdefault(category, {})[short_key] = json.loads(raw)
        except Exception as exc:
            raise MemoryRetrievalError(
                f"Redis get_task_context failed for task {task_id!r}: {exc}"
            ) from exc

        log.debug(
            "working_memory_context_snapshot",
            task_id=task_id,
            categories=list(context.keys()),
        )
        return context
