"""
tests/memory/test_management.py — Unit tests for MemoryManager.

WHY WE MOCK SemanticMemoryManager AND ChromaDB
-----------------------------------------------
MemoryManager depends on SemanticMemoryManager for ChromaDB access.  We
inject a mock so that tests are:
- Fast (no network I/O, no Docker required)
- Reproducible (controlled test data, not real DB state)
- Safe to run in any environment (CI, local, Docker-free)

WHY WE MOCK call_llm FOR CONSOLIDATION TESTS
---------------------------------------------
consolidate_memories() calls call_llm(role="specialist") to generate merged
summaries.  In tests we don't want real OpenAI API calls (cost, latency,
non-determinism).  Mocking call_llm allows us to verify that:
1. consolidate_memories() calls call_llm when appropriate.
2. The returned summary is stored correctly.

WHY unittest.IsolatedAsyncioTestCase
--------------------------------------
All MemoryManager methods are async.  IsolatedAsyncioTestCase provides a
fresh event loop per test without requiring pytest-asyncio.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.memory.models import MemoryStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_semantic_manager() -> MagicMock:
    """Return a MagicMock SemanticMemoryManager."""
    mgr = MagicMock()
    mgr.query_similar_tasks = AsyncMock(return_value=[])
    mgr.query_approaches = AsyncMock(return_value=[])
    mgr.query_user_preferences = AsyncMock(return_value=[])
    mgr.query = AsyncMock(return_value=[])
    mgr.delete_user_memories = AsyncMock(return_value=0)
    return mgr


def _make_mock_collection(
    ids: list[str] | None = None,
    docs: list[str] | None = None,
    metas: list[dict] | None = None,
    count: int = 0,
) -> MagicMock:
    """Return a MagicMock ChromaDB collection."""
    col = MagicMock()
    col.count.return_value = count or len(ids or [])
    col.get.return_value = {
        "ids": ids or [],
        "documents": docs or [],
        "metadatas": metas or [],
    }
    col.query.return_value = {
        "ids": [ids or []],
        "documents": [docs or []],
        "metadatas": [metas or []],
        "distances": [[0.0] * len(ids or [])],
    }
    col.add = MagicMock()
    col.update = MagicMock()
    col.delete = MagicMock()
    return col


def _days_ago_iso(days: int) -> str:
    """Return an ISO timestamp for a datetime `days` ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Tests: calculate_importance
# ---------------------------------------------------------------------------

class TestCalculateImportance(unittest.TestCase):
    """calculate_importance() uses the composite formula described in the docstring."""

    def _make_manager(self) -> "object":
        from app.memory.management import MemoryManager
        return MemoryManager(semantic_memory=_make_mock_semantic_manager())

    def test_fresh_high_access_success_scores_high(self) -> None:
        """
        A memory created today, accessed 10+ times, from a successful task
        should score close to 1.0.
        """
        manager = self._make_manager()
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance_score": 0.9,
            "access_count": 15,
            "success": True,
        }
        score = manager.calculate_importance(metadata)
        # Fresh + high access + success + high base → should be > 0.85
        self.assertGreater(score, 0.85)
        self.assertLessEqual(score, 1.0)

    def test_old_zero_access_failure_scores_low(self) -> None:
        """
        A memory from 90 days ago, never accessed, from a failed task
        should score close to the minimum.
        """
        manager = self._make_manager()
        metadata = {
            "timestamp": _days_ago_iso(90),
            "importance_score": 0.3,
            "access_count": 0,
            "success": False,
        }
        score = manager.calculate_importance(metadata)
        # Old + no access + failure + low base → should be < 0.4
        self.assertLess(score, 0.4)
        self.assertGreaterEqual(score, 0.0)

    def test_score_is_clamped_to_unit_interval(self) -> None:
        """calculate_importance always returns a value in [0.0, 1.0]."""
        manager = self._make_manager()
        # Extreme inputs
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance_score": 999.0,
            "access_count": 10000,
            "success": True,
        }
        score = manager.calculate_importance(metadata)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_missing_timestamp_does_not_raise(self) -> None:
        """calculate_importance handles missing or empty timestamp gracefully."""
        manager = self._make_manager()
        metadata = {
            "importance_score": 0.5,
            "access_count": 3,
            "success": True,
        }
        score = manager.calculate_importance(metadata)
        # Should return a valid float in [0.0, 1.0]
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_medium_memory_scores_in_middle(self) -> None:
        """A 30-day-old memory accessed 5 times should score in the middle range."""
        manager = self._make_manager()
        metadata = {
            "timestamp": _days_ago_iso(30),
            "importance_score": 0.5,
            "access_count": 5,
            "success": True,
        }
        score = manager.calculate_importance(metadata)
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.85)


# ---------------------------------------------------------------------------
# Tests: decay_memories
# ---------------------------------------------------------------------------

class TestDecayMemories(unittest.IsolatedAsyncioTestCase):
    """decay_memories() reduces importance for stale memories and expires them."""

    def _make_manager_with_collection(self, mock_collection: MagicMock) -> "object":
        from app.memory.management import MemoryManager
        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(return_value=mock_collection)
        return MemoryManager(semantic_memory=mock_semantic)

    async def test_stale_memory_importance_is_reduced(self) -> None:
        """
        A memory older than 30 days should have its importance decayed
        by the decay_rate factor.
        """
        from app.memory.management import MemoryManager

        old_timestamp = _days_ago_iso(60)
        mock_col = _make_mock_collection(
            ids=["mem-1"],
            docs=["Old task memory"],
            metas=[{
                "user_id": "default",
                "timestamp": old_timestamp,
                "importance_score": 0.8,
                "access_count": 0,
                "expired": False,
                "consolidated": False,
            }],
        )
        manager = self._make_manager_with_collection(mock_col)

        expired = await manager.decay_memories(decay_rate=0.95, min_importance=0.1)

        # 0.8 * 0.95 = 0.76, above min_importance → should not be expired.
        self.assertEqual(expired, 0)
        # update() should have been called to reduce importance.
        mock_col.update.assert_called()

    async def test_very_stale_memory_is_expired(self) -> None:
        """
        A memory whose importance after decay falls below min_importance
        should be marked as expired.
        """
        from app.memory.management import MemoryManager

        old_timestamp = _days_ago_iso(365)
        stale_col = _make_mock_collection(
            ids=["mem-old"],
            docs=["Very old memory"],
            metas=[{
                "user_id": "default",
                "timestamp": old_timestamp,
                "importance_score": 0.05,  # already very low
                "access_count": 0,
                "expired": False,
                "consolidated": False,
            }],
        )
        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])

        mock_semantic = _make_mock_semantic_manager()
        # Only task_memories has the stale document; other collections are empty.
        mock_semantic._get_collection = MagicMock(
            side_effect=lambda name: stale_col if name == "task_memories" else empty_col
        )
        manager = MemoryManager(semantic_memory=mock_semantic)

        expired = await manager.decay_memories(decay_rate=0.95, min_importance=0.1)

        # 0.05 * 0.95 = 0.0475 < 0.1 → should be expired exactly once.
        self.assertEqual(expired, 1)
        # Verify that update was called with expired=True.
        call_kwargs = stale_col.update.call_args
        updated_metas = call_kwargs.kwargs.get("metadatas", call_kwargs.args[1] if len(call_kwargs.args) > 1 else [])
        if updated_metas:
            self.assertTrue(updated_metas[0].get("expired", False))

    async def test_recent_memory_not_decayed(self) -> None:
        """Memories created recently (< 30 days) are not decayed."""
        from app.memory.management import MemoryManager

        recent_timestamp = _days_ago_iso(5)
        mock_col = _make_mock_collection(
            ids=["mem-recent"],
            docs=["Recent task memory"],
            metas=[{
                "user_id": "default",
                "timestamp": recent_timestamp,
                "importance_score": 0.8,
                "access_count": 2,
                "expired": False,
                "consolidated": False,
            }],
        )
        manager = self._make_manager_with_collection(mock_col)

        expired = await manager.decay_memories(decay_rate=0.95, min_importance=0.1)

        self.assertEqual(expired, 0)
        # update() should NOT have been called (recent memory is skipped).
        mock_col.update.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: consolidate_memories
# ---------------------------------------------------------------------------

class TestConsolidateMemories(unittest.IsolatedAsyncioTestCase):
    """consolidate_memories() merges near-duplicate memories using the LLM."""

    async def test_two_similar_memories_are_consolidated(self) -> None:
        """
        Two near-identical memories in the same collection should be merged
        into a single consolidated summary.
        """
        from app.memory.management import MemoryManager

        doc_a = "Task: Research AI market\nPlan: Use researcher → writer"
        doc_b = "Task: Research AI industry\nPlan: Use researcher → writer"

        # The query result simulates high similarity between the two docs.
        task_col = _make_mock_collection(
            ids=["mem-a", "mem-b"],
            docs=[doc_a, doc_b],
            metas=[
                {
                    "user_id": "default",
                    "timestamp": _days_ago_iso(1),
                    "importance_score": 0.7,
                    "access_count": 2,
                    "consolidated": False,
                    "task_type": "research",
                    "success": True,
                },
                {
                    "user_id": "default",
                    "timestamp": _days_ago_iso(2),
                    "importance_score": 0.6,
                    "access_count": 1,
                    "consolidated": False,
                    "task_type": "research",
                    "success": True,
                },
            ],
        )
        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])

        # Make query() return the second doc as a high-similarity result for the first.
        task_col.query.return_value = {
            "ids": [["mem-a", "mem-b"]],
            "documents": [[doc_a, doc_b]],
            "metadatas": [[{}, {}]],
            "distances": [[0.0, 0.1]],  # 0.1 → similarity=0.9, above threshold 0.85
        }

        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(
            side_effect=lambda name: task_col if name == "task_memories" else empty_col
        )

        manager = MemoryManager(semantic_memory=mock_semantic)

        # call_llm is imported inside the method body with a local import, so we
        # patch it at its source (app.llm.client) which is what the local import resolves to.
        with patch("app.llm.client.call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Consolidated: Research AI market/industry using researcher → writer"
            count = await manager.consolidate_memories(user_id="default", similarity_threshold=0.85)

        # Both source memories should be counted.
        self.assertGreaterEqual(count, 2)
        # A new consolidated memory should have been added.
        task_col.add.assert_called_once()
        # Both originals should be marked as consolidated.
        self.assertGreaterEqual(task_col.update.call_count, 2)

    async def test_single_memory_not_consolidated(self) -> None:
        """A collection with only one memory cannot be consolidated."""
        from app.memory.management import MemoryManager

        single_col = _make_mock_collection(
            ids=["mem-1"],
            docs=["Only one memory"],
            metas=[{
                "user_id": "default",
                "timestamp": _days_ago_iso(5),
                "importance_score": 0.5,
                "access_count": 0,
                "consolidated": False,
            }],
        )
        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])
        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(
            side_effect=lambda name: single_col if name == "task_memories" else empty_col
        )

        manager = MemoryManager(semantic_memory=mock_semantic)

        # Patch at the source; call_llm should NOT be called for a single-doc collection.
        with patch("app.llm.client.call_llm", new_callable=AsyncMock) as mock_llm:
            count = await manager.consolidate_memories(user_id="default")

        # No consolidation possible with a single memory.
        self.assertEqual(count, 0)
        mock_llm.assert_not_called()
        single_col.add.assert_not_called()

    async def test_already_consolidated_memories_skipped(self) -> None:
        """Memories with consolidated=True are not included in new clusters."""
        from app.memory.management import MemoryManager

        done_col = _make_mock_collection(
            ids=["mem-c"],
            docs=["Already merged memory"],
            metas=[{
                "user_id": "default",
                "timestamp": _days_ago_iso(1),
                "importance_score": 0.7,
                "access_count": 0,
                "consolidated": True,  # already done
            }],
        )
        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])
        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(
            side_effect=lambda name: done_col if name == "task_memories" else empty_col
        )

        manager = MemoryManager(semantic_memory=mock_semantic)

        with patch("app.llm.client.call_llm", new_callable=AsyncMock) as mock_llm:
            count = await manager.consolidate_memories(user_id="default")

        self.assertEqual(count, 0)
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: get_dashboard / get_stats
# ---------------------------------------------------------------------------

class TestGetDashboard(unittest.IsolatedAsyncioTestCase):
    """get_dashboard() aggregates per-collection stats into a summary dict."""

    async def test_dashboard_with_memories_returns_totals(self) -> None:
        """Dashboard total is the sum of all per-collection counts."""
        from app.memory.management import MemoryManager

        task_col = _make_mock_collection(
            ids=["t1", "t2"],
            docs=["Task A", "Task B"],
            metas=[
                {"user_id": "default", "timestamp": _days_ago_iso(1), "importance_score": 0.8, "access_count": 5},
                {"user_id": "default", "timestamp": _days_ago_iso(2), "importance_score": 0.6, "access_count": 2},
            ],
        )
        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])
        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(
            side_effect=lambda name: task_col if name == "task_memories" else empty_col
        )

        manager = MemoryManager(semantic_memory=mock_semantic)
        dashboard = await manager.get_dashboard(user_id="default")

        self.assertEqual(dashboard["total_memories"], 2)
        self.assertIn("by_collection", dashboard)
        self.assertEqual(dashboard["by_collection"]["task_memories"], 2)
        self.assertIn("top_accessed", dashboard)
        self.assertIn("recent", dashboard)
        self.assertIn("avg_importance", dashboard)

    async def test_dashboard_empty_store_returns_zeros(self) -> None:
        """An empty memory store returns a dashboard with all zeros."""
        from app.memory.management import MemoryManager

        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])
        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(return_value=empty_col)

        manager = MemoryManager(semantic_memory=mock_semantic)
        dashboard = await manager.get_dashboard(user_id="default")

        self.assertEqual(dashboard["total_memories"], 0)
        self.assertEqual(dashboard["avg_importance"], 0.0)
        self.assertEqual(dashboard["top_accessed"], [])
        self.assertEqual(dashboard["recent"], [])


class TestGetStats(unittest.IsolatedAsyncioTestCase):
    """get_stats() returns a valid MemoryStats model."""

    async def test_stats_is_memory_stats_instance(self) -> None:
        """get_stats() returns a MemoryStats Pydantic model."""
        from app.memory.management import MemoryManager

        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])
        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(return_value=empty_col)

        manager = MemoryManager(semantic_memory=mock_semantic)
        stats = await manager.get_stats(user_id="default")

        self.assertIsInstance(stats, MemoryStats)
        self.assertIsInstance(stats.total_memories, int)
        self.assertIsInstance(stats.by_collection, dict)
        self.assertIsInstance(stats.avg_importance, float)
        self.assertIsInstance(stats.most_accessed, list)

    async def test_stats_total_matches_dashboard_total(self) -> None:
        """stats.total_memories should equal dashboard['total_memories']."""
        from app.memory.management import MemoryManager

        task_col = _make_mock_collection(
            ids=["t1"],
            docs=["Task A"],
            metas=[{"user_id": "default", "timestamp": _days_ago_iso(1),
                    "importance_score": 0.7, "access_count": 3}],
        )
        empty_col = _make_mock_collection(ids=[], docs=[], metas=[])
        mock_semantic = _make_mock_semantic_manager()
        mock_semantic._get_collection = MagicMock(
            side_effect=lambda name: task_col if name == "task_memories" else empty_col
        )

        manager = MemoryManager(semantic_memory=mock_semantic)
        stats = await manager.get_stats(user_id="default")

        self.assertEqual(stats.total_memories, 1)


if __name__ == "__main__":
    unittest.main()
