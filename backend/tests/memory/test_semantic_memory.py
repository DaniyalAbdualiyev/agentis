"""
tests/memory/test_semantic_memory.py — Unit tests for SemanticMemoryManager.

WHY WE MOCK ChromaDB (not use the real HTTP client)
----------------------------------------------------
SemanticMemoryManager connects to a ChromaDB Docker container at runtime.
In CI and local dev there is no guarantee that the container is running.
We mock the chromadb.HttpClient and the collection interface so tests are:
  - Fast (no network I/O)
  - Reproducible (no external state)
  - Safe to run anywhere (no Docker required)

The mock verifies that SemanticMemoryManager calls ChromaDB with the correct
arguments — which is the right level of testing for an I/O adapter class.

WHY WE MOCK AsyncOpenAI AND OpenAIEmbeddingFunction
-----------------------------------------------------
SemanticMemoryManager's __init__ creates an OpenAIEmbeddingFunction which
in turn instantiates AsyncOpenAI.  If OPENAI_API_KEY is not set (typical in
CI), AsyncOpenAI raises immediately.  We patch AsyncOpenAI at the module level
to prevent the error, and then replace _embedding_fn on each manager instance
with a deterministic lambda that returns a fixed vector — avoiding real OpenAI
API calls in tests entirely.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call
from typing import Any

# Ensure backend package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers to build realistic ChromaDB mock responses
# ---------------------------------------------------------------------------

def _make_query_result(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
) -> dict[str, Any]:
    """Build a ChromaDB query result dict matching the real API shape."""
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _make_get_result(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> dict[str, Any]:
    """Build a ChromaDB get result dict matching the real API shape."""
    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }


def _make_mock_collection(
    count: int = 5,
    query_result: dict | None = None,
    get_result: dict | None = None,
) -> MagicMock:
    """Return a mock ChromaDB Collection object."""
    col = MagicMock()
    col.count.return_value = count
    if query_result is not None:
        col.query.return_value = query_result
    else:
        col.query.return_value = _make_query_result([], [], [], [])
    if get_result is not None:
        col.get.return_value = get_result
    else:
        col.get.return_value = _make_get_result([], [], [])
    return col


# ---------------------------------------------------------------------------
# Helper: build a SemanticMemoryManager with mocked dependencies
# ---------------------------------------------------------------------------

def _make_semantic_manager(mock_http_client_cls: MagicMock, collection_factory=None) -> "Any":
    """
    Construct a SemanticMemoryManager with:
    - ChromaDB HttpClient replaced by mock_http_client_cls
    - OpenAI embedding function replaced with a deterministic lambda

    WHY A FACTORY FUNCTION (not setUp/setUpClass)
    ---------------------------------------------
    IsolatedAsyncioTestCase runs each test in its own event loop, so a single
    setUp that patches would need careful teardown.  A simple factory called
    inside each test is simpler and less error-prone.
    """
    from app.memory.semantic_memory import SemanticMemoryManager

    mock_client_instance = MagicMock()
    mock_http_client_cls.return_value = mock_client_instance

    if collection_factory is not None:
        mock_client_instance.get_or_create_collection.side_effect = collection_factory

    mgr = SemanticMemoryManager()
    # Replace the embedding function so no OpenAI API calls are made.
    mgr._embedding_fn = lambda texts: [[0.1] * 10 for _ in texts]
    return mgr, mock_client_instance


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestStoreTaskCompletion(unittest.IsolatedAsyncioTestCase):
    """store_task_completion() should write to task_memories and approach_memories."""

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_store_creates_document(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        # Set up mock collections
        task_col = _make_mock_collection()
        approach_col = _make_mock_collection()

        def get_or_create(name: str, **kwargs: Any) -> MagicMock:
            if name == "task_memories":
                return task_col
            return approach_col

        mgr, _ = _make_semantic_manager(mock_http_client, get_or_create)

        memory_id = await mgr.store_task_completion(
            task_id="abc-123",
            user_id="default",
            task_summary={
                "original_task": "Analyse the AI market",
                "plan_reasoning": "Use researcher → analyst → writer",
                "final_output_preview": "AI market grew by 35% in 2025.",
            },
            execution_data={
                "subtask_count": 3,
                "retry_count": 0,
                "error_count": 0,
                "agents_used": ["researcher", "analyst", "writer"],
            },
        )

        # A non-empty memory_id (UUID) should be returned.
        self.assertIsInstance(memory_id, str)
        self.assertTrue(len(memory_id) > 0)

        # task_memories.add() must have been called once.
        task_col.add.assert_called_once()
        call_kwargs = task_col.add.call_args
        # The ID passed to add() must match the returned memory_id.
        self.assertIn(memory_id, call_kwargs.kwargs.get("ids", call_kwargs.args[0] if call_kwargs.args else []))

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_store_also_writes_approach_memory(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        task_col = _make_mock_collection()
        approach_col = _make_mock_collection()

        def get_or_create(name: str, **kwargs: Any) -> MagicMock:
            if name == "task_memories":
                return task_col
            return approach_col

        mgr, _ = _make_semantic_manager(mock_http_client, get_or_create)

        await mgr.store_task_completion(
            task_id="abc-456",
            user_id="default",
            task_summary={
                "original_task": "Write a competitive analysis",
                "plan_reasoning": "Research first then write",
                "final_output_preview": "Competitive landscape summary.",
            },
            execution_data={
                "subtask_count": 3,
                "retry_count": 1,
                "error_count": 0,
                "agents_used": ["researcher", "writer"],
            },
        )

        # Both collections should have had add() called.
        task_col.add.assert_called_once()
        approach_col.add.assert_called_once()


class TestQuery(unittest.IsolatedAsyncioTestCase):
    """query() should return MemoryDocument list filtered by relevance threshold."""

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_query_returns_relevant_results(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        from app.memory.models import MemoryDocument

        # Similarity = 1 - distance.  Set distance=0.1 → similarity=0.9 (above threshold).
        query_result = _make_query_result(
            ids=["doc-1"],
            documents=["Task: Research AI trends"],
            metadatas=[{"user_id": "default", "task_type": "research", "access_count": 0}],
            distances=[0.1],
        )
        task_col = _make_mock_collection(count=1, query_result=query_result)
        empty_col = _make_mock_collection(count=0)

        mgr, _ = _make_semantic_manager(
            mock_http_client,
            lambda name, **kw: task_col if name == "task_memories" else empty_col,
        )

        results = await mgr.query("AI trends", user_id="default")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], MemoryDocument)
        self.assertEqual(results[0].id, "doc-1")
        self.assertAlmostEqual(results[0].similarity_score, 0.9, places=2)

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_query_filters_low_similarity(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        """Results below the relevance threshold (0.7) should be excluded."""
        # distance=0.4 → similarity=0.6, below threshold of 0.7.
        query_result = _make_query_result(
            ids=["doc-low"],
            documents=["Weak match"],
            metadatas=[{"user_id": "default", "access_count": 0}],
            distances=[0.4],
        )
        task_col = _make_mock_collection(count=1, query_result=query_result)
        empty_col = _make_mock_collection(count=0)

        mgr, _ = _make_semantic_manager(
            mock_http_client,
            lambda name, **kw: task_col if name == "task_memories" else empty_col,
        )

        results = await mgr.query("something unrelated", user_id="default")

        # Below-threshold result should be filtered out.
        self.assertEqual(results, [])

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_query_empty_collections_returns_empty_list(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        empty_col = _make_mock_collection(count=0)
        mock_http_client.return_value = MagicMock()
        mock_http_client.return_value.get_or_create_collection.return_value = empty_col

        mgr, _ = _make_semantic_manager(mock_http_client)

        results = await mgr.query("any query", user_id="default")
        self.assertEqual(results, [])


class TestDeleteUserMemories(unittest.IsolatedAsyncioTestCase):
    """delete_user_memories() should remove all documents for a user."""

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_delete_removes_documents_across_collections(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        task_col = MagicMock()
        task_col.get.return_value = _make_get_result(
            ids=["doc-1", "doc-2"],
            documents=["a", "b"],
            metadatas=[{}, {}],
        )
        empty_col = MagicMock()
        empty_col.get.return_value = _make_get_result([], [], [])

        mgr, _ = _make_semantic_manager(
            mock_http_client,
            lambda name, **kw: task_col if name == "task_memories" else empty_col,
        )

        total = await mgr.delete_user_memories(user_id="test-user")

        # 2 docs deleted from task_memories, 0 from the rest.
        self.assertEqual(total, 2)
        task_col.delete.assert_called_once_with(ids=["doc-1", "doc-2"])

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_delete_user_with_no_memories_returns_zero(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        empty_col = MagicMock()
        empty_col.get.return_value = _make_get_result([], [], [])

        mgr, _ = _make_semantic_manager(mock_http_client)
        mgr._client.get_or_create_collection.return_value = empty_col

        total = await mgr.delete_user_memories(user_id="ghost-user")
        self.assertEqual(total, 0)


class TestGetMemoryById(unittest.IsolatedAsyncioTestCase):
    """get_memory_by_id() fetches a document by exact ID."""

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_returns_document_when_found(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        from app.memory.models import MemoryDocument

        task_col = MagicMock()
        task_col.get.return_value = _make_get_result(
            ids=["known-id"],
            documents=["content here"],
            metadatas=[{"task_type": "research"}],
        )

        mgr, _ = _make_semantic_manager(mock_http_client)
        mgr._client.get_or_create_collection.return_value = task_col

        doc = await mgr.get_memory_by_id("known-id")

        self.assertIsNotNone(doc)
        self.assertIsInstance(doc, MemoryDocument)
        self.assertEqual(doc.id, "known-id")
        self.assertEqual(doc.content, "content here")
        self.assertIsNone(doc.similarity_score)

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_returns_none_when_not_found(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        empty_col = MagicMock()
        empty_col.get.return_value = _make_get_result([], [], [])

        mgr, _ = _make_semantic_manager(mock_http_client)
        mgr._client.get_or_create_collection.return_value = empty_col

        result = await mgr.get_memory_by_id("does-not-exist")
        self.assertIsNone(result)


class TestQuerySimilarTasks(unittest.IsolatedAsyncioTestCase):
    """query_similar_tasks() targets only the task_memories collection."""

    @patch("app.memory.semantic_memory.AsyncOpenAI")
    @patch("app.memory.semantic_memory.chromadb.HttpClient")
    async def test_returns_list_of_memory_documents(
        self, mock_http_client: MagicMock, _mock_openai: MagicMock
    ) -> None:
        from app.memory.models import MemoryDocument

        query_result = _make_query_result(
            ids=["t1", "t2"],
            documents=["Task A", "Task B"],
            metadatas=[
                {"user_id": "default", "access_count": 0},
                {"user_id": "default", "access_count": 1},
            ],
            distances=[0.05, 0.15],
        )
        task_col = _make_mock_collection(count=2, query_result=query_result)

        mgr, _ = _make_semantic_manager(mock_http_client)
        mgr._client.get_or_create_collection.return_value = task_col

        results = await mgr.query_similar_tasks("research AI landscape")
        self.assertEqual(len(results), 2)
        for doc in results:
            self.assertIsInstance(doc, MemoryDocument)


if __name__ == "__main__":
    unittest.main()
