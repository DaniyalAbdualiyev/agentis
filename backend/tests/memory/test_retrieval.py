"""
tests/memory/test_retrieval.py — Unit tests for PlanningMemoryRetriever.

WHY WE MOCK SemanticMemoryManager (not use the real ChromaDB client)
---------------------------------------------------------------------
PlanningMemoryRetriever depends on SemanticMemoryManager for all ChromaDB
access.  By injecting a mock SemanticMemoryManager, tests are:
- Fast (no network I/O or embedding API calls)
- Reproducible (controlled fixtures, not real DB state)
- Safe to run anywhere (no Docker required)

The mock verifies that PlanningMemoryRetriever calls SemanticMemoryManager
with the correct arguments and assembles MemoryContext correctly — the right
level of testing for an orchestration class.

WHY unittest.IsolatedAsyncioTestCase
--------------------------------------
PlanningMemoryRetriever is fully async.  IsolatedAsyncioTestCase provides
a fresh event loop for each test method without requiring pytest-asyncio.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.memory.models import MemoryContext, MemoryDocument


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------

def _make_memory_doc(
    doc_id: str = "doc-1",
    content: str = "Task: Research AI trends\nPlan reasoning: ...",
    collection: str = "task_memories",
    similarity: float = 0.9,
    metadata: dict | None = None,
) -> MemoryDocument:
    """Return a realistic MemoryDocument fixture."""
    return MemoryDocument(
        id=doc_id,
        content=content,
        metadata=metadata or {"user_id": "default", "access_count": 2, "success": True},
        collection=collection,
        similarity_score=similarity,
    )


def _make_approach_doc(
    outcome: str = "success",
    problem_type: str = "research",
    confidence: float = 0.85,
) -> MemoryDocument:
    """Return a MemoryDocument fixture representing an approach memory."""
    return MemoryDocument(
        id="approach-1",
        content=(
            f"Problem type: {problem_type}\n"
            f"Approach: Use researcher → analyst → writer\n"
            f"Task: Research something"
        ),
        metadata={
            "user_id": "default",
            "outcome": outcome,
            "problem_type": problem_type,
            "confidence_score": confidence,
            "access_count": 1,
        },
        collection="approach_memories",
        similarity_score=0.88,
    )


def _make_mock_semantic_manager() -> MagicMock:
    """
    Return a MagicMock SemanticMemoryManager with async methods set up.

    All query methods default to returning empty lists so individual tests
    can override only the methods they care about.
    """
    mgr = MagicMock()
    mgr.query_similar_tasks = AsyncMock(return_value=[])
    mgr.query_approaches = AsyncMock(return_value=[])
    mgr.query_user_preferences = AsyncMock(return_value=[])
    mgr.query = AsyncMock(return_value=[])
    return mgr


# ---------------------------------------------------------------------------
# Tests: retrieve_context
# ---------------------------------------------------------------------------

class TestRetrieveContextWithMemories(unittest.IsolatedAsyncioTestCase):
    """retrieve_context() returns a populated MemoryContext when memories exist."""

    async def test_returns_similar_tasks(self) -> None:
        """Similar tasks from SemanticMemoryManager appear in MemoryContext.similar_tasks."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        doc = _make_memory_doc()
        mock_mgr.query_similar_tasks.return_value = [doc]

        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        context = await retriever.retrieve_context("Research AI market", user_id="default")

        self.assertIsInstance(context, MemoryContext)
        self.assertEqual(len(context.similar_tasks), 1)
        self.assertEqual(context.similar_tasks[0].id, "doc-1")
        self.assertTrue(context.has_relevant_memories)

    async def test_returns_effective_approaches(self) -> None:
        """Successful approaches populate MemoryContext.effective_approaches."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        approach = _make_approach_doc(outcome="success")
        mock_mgr.query_approaches.return_value = [approach]

        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        context = await retriever.retrieve_context("Analyse competitors", user_id="default")

        self.assertEqual(len(context.effective_approaches), 1)
        self.assertTrue(context.has_relevant_memories)

    async def test_failed_approaches_separated_from_effective(self) -> None:
        """Failed approaches go to failed_approaches, not effective_approaches."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        success_approach = _make_approach_doc(outcome="success")
        failed_approach = _make_approach_doc(outcome="failed")
        mock_mgr.query_approaches.return_value = [success_approach, failed_approach]

        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        context = await retriever.retrieve_context("Analyse market", user_id="default")

        self.assertEqual(len(context.effective_approaches), 1)
        self.assertEqual(context.effective_approaches[0].metadata["outcome"], "success")
        self.assertEqual(len(context.failed_approaches), 1)
        self.assertEqual(context.failed_approaches[0].metadata["outcome"], "failed")

    async def test_user_preferences_returned(self) -> None:
        """User preferences from the memory manager appear in the context."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        pref_doc = MemoryDocument(
            id="pref-1",
            content="User prefers concise bullet-point summaries",
            metadata={"user_id": "default"},
            collection="user_preferences",
            similarity_score=None,
        )
        mock_mgr.query_user_preferences.return_value = [pref_doc]

        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        context = await retriever.retrieve_context("Write a report", user_id="default")

        self.assertEqual(len(context.user_preferences), 1)
        self.assertIn("bullet-point", context.user_preferences[0].content)
        self.assertTrue(context.has_relevant_memories)


class TestRetrieveContextNoMemories(unittest.IsolatedAsyncioTestCase):
    """retrieve_context() returns has_relevant_memories=False when store is empty."""

    async def test_empty_store_returns_false_flag(self) -> None:
        """All lists empty → has_relevant_memories is False."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        # All methods return empty lists (default).

        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        context = await retriever.retrieve_context("New topic no one has asked", user_id="default")

        self.assertFalse(context.has_relevant_memories)
        self.assertEqual(context.similar_tasks, [])
        self.assertEqual(context.effective_approaches, [])
        self.assertEqual(context.failed_approaches, [])
        self.assertEqual(context.user_preferences, [])
        self.assertEqual(context.domain_context, [])

    async def test_empty_context_is_valid_memory_context(self) -> None:
        """An empty MemoryContext is a valid model instance."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        context = await retriever.retrieve_context("X", user_id="default")

        self.assertIsInstance(context, MemoryContext)
        # Ensure it can be serialised (used by memory_retrieval_node).
        dumped = context.model_dump()
        self.assertIn("has_relevant_memories", dumped)
        self.assertFalse(dumped["has_relevant_memories"])


class TestRetrieveContextGracefulDegradation(unittest.IsolatedAsyncioTestCase):
    """retrieve_context() degrades gracefully when SemanticMemoryManager raises."""

    async def test_similar_tasks_failure_returns_empty_list(self) -> None:
        """If query_similar_tasks raises, similar_tasks is [] and no exception propagates."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        mock_mgr.query_similar_tasks.side_effect = Exception("ChromaDB unavailable")

        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        # Should not raise.
        context = await retriever.retrieve_context("Research something", user_id="default")

        self.assertEqual(context.similar_tasks, [])
        # has_relevant_memories may still be True if other queries succeeded;
        # in this case all are empty so it should be False.
        self.assertFalse(context.has_relevant_memories)

    async def test_all_failures_still_returns_memory_context(self) -> None:
        """If every query method raises, retrieve_context still returns a valid MemoryContext."""
        from app.memory.retrieval import PlanningMemoryRetriever

        mock_mgr = _make_mock_semantic_manager()
        mock_mgr.query_similar_tasks.side_effect = RuntimeError("fail")
        mock_mgr.query_approaches.side_effect = RuntimeError("fail")
        mock_mgr.query_user_preferences.side_effect = RuntimeError("fail")
        mock_mgr.query.side_effect = RuntimeError("fail")

        retriever = PlanningMemoryRetriever(semantic_memory=mock_mgr)
        context = await retriever.retrieve_context("Research something", user_id="default")

        self.assertIsInstance(context, MemoryContext)
        self.assertFalse(context.has_relevant_memories)


# ---------------------------------------------------------------------------
# Tests: format_for_prompt
# ---------------------------------------------------------------------------

class TestFormatForPrompt(unittest.IsolatedAsyncioTestCase):
    """format_for_prompt() produces readable prompt text from MemoryContext."""

    def _make_retriever(self) -> "object":
        from app.memory.retrieval import PlanningMemoryRetriever
        return PlanningMemoryRetriever(semantic_memory=_make_mock_semantic_manager())

    def test_empty_context_returns_empty_string(self) -> None:
        """An empty MemoryContext produces no prompt text."""
        retriever = self._make_retriever()
        context = MemoryContext()
        result = retriever.format_for_prompt(context)
        self.assertEqual(result, "")

    def test_similar_tasks_section_included(self) -> None:
        """Similar tasks appear in the formatted prompt."""
        retriever = self._make_retriever()
        doc = _make_memory_doc(
            content="Task: Research AI trends\nPlan reasoning: ...",
            metadata={"success": True, "agents_used": "researcher,analyst,writer"},
        )
        context = MemoryContext(
            similar_tasks=[doc],
            has_relevant_memories=True,
        )
        result = retriever.format_for_prompt(context)
        self.assertIn("Similar Past Tasks", result)
        self.assertIn("Research AI trends", result)

    def test_effective_approaches_section_included(self) -> None:
        """Effective approaches appear in the formatted prompt."""
        retriever = self._make_retriever()
        approach = _make_approach_doc(outcome="success", problem_type="research")
        context = MemoryContext(
            effective_approaches=[approach],
            has_relevant_memories=True,
        )
        result = retriever.format_for_prompt(context)
        self.assertIn("Effective Approaches", result)

    def test_failed_approaches_section_included(self) -> None:
        """Failed approaches appear under 'Approaches to Avoid'."""
        retriever = self._make_retriever()
        failed = _make_approach_doc(outcome="failed", problem_type="research")
        context = MemoryContext(
            failed_approaches=[failed],
            has_relevant_memories=True,
        )
        result = retriever.format_for_prompt(context)
        self.assertIn("Approaches to Avoid", result)

    def test_user_preferences_section_included(self) -> None:
        """User preferences appear in the formatted prompt."""
        retriever = self._make_retriever()
        pref = MemoryDocument(
            id="pref-1",
            content="Prefers concise bullet points",
            metadata={"user_id": "default"},
            collection="user_preferences",
            similarity_score=None,
        )
        context = MemoryContext(
            user_preferences=[pref],
            has_relevant_memories=True,
        )
        result = retriever.format_for_prompt(context)
        self.assertIn("User Preferences", result)
        self.assertIn("concise bullet points", result)

    def test_empty_sections_not_included(self) -> None:
        """Sections with no data are not included in the prompt output."""
        retriever = self._make_retriever()
        doc = _make_memory_doc()
        context = MemoryContext(
            similar_tasks=[doc],
            # All other sections empty
            has_relevant_memories=True,
        )
        result = retriever.format_for_prompt(context)
        self.assertIn("Similar Past Tasks", result)
        self.assertNotIn("Effective Approaches", result)
        self.assertNotIn("Approaches to Avoid", result)
        self.assertNotIn("User Preferences", result)

    def test_format_includes_header(self) -> None:
        """The formatted output starts with the 'Past Context (from memory)' header."""
        retriever = self._make_retriever()
        doc = _make_memory_doc()
        context = MemoryContext(
            similar_tasks=[doc],
            has_relevant_memories=True,
        )
        result = retriever.format_for_prompt(context)
        self.assertIn("Past Context (from memory)", result)


if __name__ == "__main__":
    unittest.main()
