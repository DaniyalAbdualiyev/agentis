"""
memory/semantic_memory.py — Long-term semantic memory backed by ChromaDB.

WHAT THIS MODULE PROVIDES
--------------------------
A vector-search-powered memory store that persists task completion summaries,
successful/failed approach strategies, user preferences, and domain facts
across task runs.  When a new task starts, the Supervisor can query this
store to retrieve semantically similar past experiences and plan better.

WHY CHROMADB
------------
ChromaDB is purpose-built for embedding-backed similarity search.  The
alternatives were:

  pgvector (PostgreSQL extension) — would require modifying the DB schema
    and adding a migration.  ChromaDB is self-contained and doesn't increase
    complexity of the main database.

  Pinecone / Qdrant (managed cloud) — introduce external service dependencies
    and API keys.  ChromaDB runs as a local Docker container, keeping the
    development environment self-contained and free.

  FAISS (local in-process) — not persistent across restarts.  Every restart
    would wipe all memories.  ChromaDB persists to disk via the volume mount
    in docker-compose.yml.

WHY FOUR COLLECTIONS
---------------------
Different memory types have different query patterns:
  task_memories       → queried by task description similarity
  approach_memories   → queried by problem type similarity
  user_preferences    → queried by user_id (metadata filter)
  domain_facts        → queried by topic similarity

A single flat collection would mix all types, requiring metadata filters on
every query.  Separate collections let each query target exactly the right
data without filtering noise.

WHY OPENAI EMBEDDINGS (text-embedding-3-small)
-----------------------------------------------
text-embedding-3-small (1536 dimensions) offers an excellent quality/cost
ratio.  It outperforms Ada-002 on most benchmarks while being cheaper and
faster.  Since we already depend on the openai package for call_llm(), adding
an embedding call adds no new dependency.

WHY HTTP CLIENT (not in-process ChromaDB)
------------------------------------------
chromadb.HttpClient connects to the ChromaDB Docker service over HTTP.
The alternative — chromadb.Client() (in-process) — would load the entire
ChromaDB SQLite database into the backend process's RAM, causing problems:
  - High memory use in a shared container.
  - No persistence across process restarts (in-memory mode).
  - No shared access if multiple backend replicas are ever deployed.
The HTTP client is slightly slower per call but architecturally correct.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
import structlog
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from openai import AsyncOpenAI

from app.memory.config import memory_settings
from app.memory.exceptions import MemoryRetrievalError, MemoryStoreError
from app.memory.models import MemoryDocument

log = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# ChromaDB collection names
# ------------------------------------------------------------------
_COLLECTION_TASK_MEMORIES = "task_memories"
_COLLECTION_APPROACH_MEMORIES = "approach_memories"
_COLLECTION_USER_PREFERENCES = "user_preferences"
_COLLECTION_DOMAIN_FACTS = "domain_facts"

_ALL_COLLECTIONS = [
    _COLLECTION_TASK_MEMORIES,
    _COLLECTION_APPROACH_MEMORIES,
    _COLLECTION_USER_PREFERENCES,
    _COLLECTION_DOMAIN_FACTS,
]


# ------------------------------------------------------------------
# OpenAI Embedding Function (ChromaDB-compatible interface)
# ------------------------------------------------------------------

class OpenAIEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB embedding function that delegates to OpenAI's Embeddings API.

    WHY NOT USE chromadb.utils.embedding_functions.OpenAIEmbeddingFunction
    -----------------------------------------------------------------------
    ChromaDB ships its own OpenAI wrapper, but it uses the synchronous
    openai client internally.  Since our FastAPI app is fully async, we
    use AsyncOpenAI to avoid blocking the event loop during embedding calls.

    WHY THIS WRAPS AsyncOpenAI (not plain openai)
    -----------------------------------------------
    Embedding calls can take 100–500 ms.  In an async FastAPI request, a
    blocking synchronous call would stall the event loop and degrade
    latency for all concurrent requests.  AsyncOpenAI.embeddings.create()
    is awaitable and plays nicely with asyncio.

    NOTE: ChromaDB's EmbeddingFunction interface is synchronous (__call__).
    We work around this by running the coroutine via asyncio.run() only when
    called from ChromaDB's internal synchronous paths.  For all other uses
    (our own code), we call embed_texts() directly as an async method.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Async path: embed a batch of texts and return their vectors.

        WHY BATCH (not one-by-one)
        ---------------------------
        OpenAI's embedding endpoint accepts up to 2048 inputs per request.
        Batching reduces API round trips and therefore latency and cost.
        """
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        """
        Synchronous ChromaDB interface.  Used by ChromaDB internally when
        it needs to embed documents or query vectors.

        WHY asyncio.run() HERE
        -----------------------
        ChromaDB's collection.add() and collection.query() are synchronous
        and call the embedding function synchronously.  We bridge to our
        async embed_texts() using asyncio.run().  This is safe as long as
        ChromaDB's calls are not nested inside a running event loop — which
        is guaranteed because we only call ChromaDB from within our own
        async methods that properly await results.

        If this ever becomes a problem (e.g. nested event loops), the fix
        is to use chromadb's async client (chromadb.AsyncHttpClient) which
        was added in chromadb>=0.5.
        """
        import asyncio
        return asyncio.run(self.embed_texts(list(input)))


# ------------------------------------------------------------------
# SemanticMemoryManager
# ------------------------------------------------------------------

class SemanticMemoryManager:
    """
    Async facade over ChromaDB for long-term semantic memory operations.

    All public methods are async even though the underlying chromadb client
    is synchronous.  This is intentional: the interface is async so that
    callers don't need to change if we later switch to chromadb.AsyncHttpClient.

    Instantiation is cheap — no network calls happen in __init__.
    Collections are created lazily on first use via _get_collection().
    """

    def __init__(
        self,
        chromadb_host: str | None = None,
        chromadb_port: int | None = None,
    ) -> None:
        """
        Initialise the ChromaDB HTTP client.

        WHY host/port OVERRIDE PARAMS
        ------------------------------
        Tests can pass a mock or in-memory ChromaDB instance without setting
        environment variables.  Production code instantiates with no args,
        picking up values from MemorySettings.
        """
        host = chromadb_host or memory_settings.chromadb_host
        port = chromadb_port or memory_settings.chromadb_port
        self._client = chromadb.HttpClient(host=host, port=port)
        self._embedding_fn = OpenAIEmbeddingFunction(
            model=memory_settings.embedding_model
        )
        log.debug(
            "semantic_memory_init",
            chromadb_host=host,
            chromadb_port=port,
            embedding_model=memory_settings.embedding_model,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection(self, name: str) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection with our embedding function.

        WHY get_or_create_collection (not get_collection)
        --------------------------------------------------
        On first startup, no collections exist.  get_collection() would
        raise an exception; get_or_create_collection() is idempotent and
        safe to call on every request.
        """
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _infer_task_type(task_description: str) -> str:
        """
        Infer a coarse task type label from the task description.

        WHY A ROUGH HEURISTIC (not an LLM call)
        ----------------------------------------
        Calling an LLM to classify task type would add ~500 ms and cost to
        every task completion.  A keyword heuristic is free and good enough
        for the metadata field, which is used for filtering, not scoring.
        Filtering by task_type in approach_memories is approximate anyway —
        the primary ranking signal is embedding similarity, not this label.
        """
        desc = task_description.lower()
        if any(kw in desc for kw in ["research", "find", "search", "gather"]):
            return "research"
        if any(kw in desc for kw in ["analys", "compare", "evaluate", "assess"]):
            return "analysis"
        if any(kw in desc for kw in ["write", "report", "summarize", "draft"]):
            return "writing"
        if any(kw in desc for kw in ["code", "implement", "debug", "script"]):
            return "coding"
        return "general"

    @staticmethod
    def _make_document_id() -> str:
        """Generate a unique document ID using UUID4."""
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def store_task_completion(
        self,
        task_id: str,
        user_id: str,
        task_summary: dict[str, Any],
        execution_data: dict[str, Any],
    ) -> str:
        """
        Persist a completed task's summary into ChromaDB for future retrieval.

        This is called once per task after it reaches status="completed".
        It creates two ChromaDB entries:
          1. task_memories  — the full task summary for similarity search.
          2. approach_memories — the planning approach used, indexed by
             problem type, so similar future tasks can reuse it.

        Parameters
        ----------
        task_id        : The task's UUID (for deduplication and cross-referencing).
        user_id        : User identifier (for future multi-tenant filtering).
        task_summary   : Dict with keys: original_task, plan_reasoning,
                         final_output_preview.
        execution_data : Dict with keys: subtask_count, retry_count,
                         error_count, agents_used.

        Returns
        -------
        str : The memory_id of the created task_memories document.
        """
        original_task = task_summary.get("original_task", "")
        plan_reasoning = task_summary.get("plan_reasoning", "")
        output_preview = task_summary.get("final_output_preview", "")
        agents_used = execution_data.get("agents_used", [])
        retry_count = execution_data.get("retry_count", 0)
        error_count = execution_data.get("error_count", 0)
        subtask_count = execution_data.get("subtask_count", 0)

        # Combine the key signals into a single embeddable text.
        # The embedding quality is highest when the text is coherent prose
        # rather than a JSON dump — the model was trained on natural language.
        content = (
            f"Task: {original_task}\n"
            f"Plan reasoning: {plan_reasoning}\n"
            f"Output preview: {output_preview}"
        )

        memory_id = self._make_document_id()
        task_type = self._infer_task_type(original_task)
        now = datetime.now(timezone.utc).isoformat()

        task_metadata: dict[str, Any] = {
            "user_id": user_id,
            "task_id": task_id,
            "timestamp": now,
            "importance_score": 0.5,
            "access_count": 0,
            "task_type": task_type,
            "success": True,
            "retry_count": retry_count,
            "error_count": error_count,
            "subtask_count": subtask_count,
            "agents_used": ",".join(agents_used),
        }

        try:
            collection = self._get_collection(_COLLECTION_TASK_MEMORIES)
            collection.add(
                ids=[memory_id],
                documents=[content],
                metadatas=[task_metadata],
            )
            log.info(
                "semantic_memory_task_stored",
                memory_id=memory_id,
                task_id=task_id,
                task_type=task_type,
            )
        except Exception as exc:
            raise MemoryStoreError(
                f"ChromaDB task_memories write failed for task {task_id}: {exc}"
            ) from exc

        # Store approach in approach_memories
        try:
            approach_id = self._make_document_id()
            approach_content = (
                f"Problem type: {task_type}\n"
                f"Approach: {plan_reasoning}\n"
                f"Task: {original_task}"
            )
            approach_metadata: dict[str, Any] = {
                "user_id": user_id,
                "task_id": task_id,
                "timestamp": now,
                "problem_type": task_type,
                "outcome": "success",
                "confidence_score": max(0.1, 1.0 - (retry_count * 0.2)),
                "access_count": 0,
            }
            approach_collection = self._get_collection(_COLLECTION_APPROACH_MEMORIES)
            approach_collection.add(
                ids=[approach_id],
                documents=[approach_content],
                metadatas=[approach_metadata],
            )
            log.debug(
                "semantic_memory_approach_stored",
                approach_id=approach_id,
                task_type=task_type,
            )
        except Exception as exc:
            # Approach storage is best-effort — don't fail the whole store.
            log.warning("semantic_memory_approach_store_failed", error=str(exc))

        return memory_id

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def query(
        self,
        query_text: str,
        user_id: str = "default",
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryDocument]:
        """
        Semantic search across all collections (or a filtered subset).

        Each collection is queried independently and results are merged,
        deduplicated by ID, and sorted by similarity score.

        WHY QUERY ALL COLLECTIONS BY DEFAULT
        --------------------------------------
        A broad query ("what do I know about topic X?") benefits from
        seeing task memories, approach memories, and preferences together.
        Callers that want a specific collection should use the type-specific
        methods (query_similar_tasks, query_approaches, etc.).

        Parameters
        ----------
        query_text : The natural-language query to embed and search with.
        user_id    : Filter results to this user (via metadata where_document).
        n_results  : Number of results to return per collection.
        filters    : Optional dict of ChromaDB metadata filter clauses.

        Returns
        -------
        List of MemoryDocument sorted by similarity_score descending.
        """
        all_docs: list[MemoryDocument] = []
        where: dict[str, Any] = {"user_id": user_id}
        if filters:
            where.update(filters)

        for collection_name in _ALL_COLLECTIONS:
            try:
                collection = self._get_collection(collection_name)
                # Check if collection is non-empty before querying
                count = collection.count()
                if count == 0:
                    continue

                results = collection.query(
                    query_texts=[query_text],
                    n_results=min(n_results, count),
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )

                if not results or not results.get("ids"):
                    continue

                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0]

                for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                    # ChromaDB uses cosine distance (0=identical, 2=opposite).
                    # Convert to similarity score in [0, 1].
                    similarity = max(0.0, 1.0 - dist)
                    if similarity >= memory_settings.memory_relevance_threshold:
                        all_docs.append(
                            MemoryDocument(
                                id=doc_id,
                                content=doc,
                                metadata=dict(meta),
                                collection=collection_name,
                                similarity_score=round(similarity, 4),
                            )
                        )
                        # Increment access count asynchronously (best-effort).
                        try:
                            self._increment_access_sync(collection, doc_id, meta)
                        except Exception:
                            pass

            except Exception as exc:
                log.warning(
                    "semantic_memory_query_collection_failed",
                    collection=collection_name,
                    error=str(exc),
                )
                continue

        # Sort by similarity descending so the best match is first.
        all_docs.sort(key=lambda d: d.similarity_score or 0.0, reverse=True)
        log.debug(
            "semantic_memory_query",
            query_text=query_text[:80],
            results=len(all_docs),
        )
        return all_docs

    async def query_similar_tasks(
        self,
        task_description: str,
        user_id: str = "default",
        n_results: int = 3,
    ) -> list[MemoryDocument]:
        """
        Query task_memories for tasks similar to `task_description`.

        WHY n_results=3 DEFAULT
        -----------------------
        The Supervisor prompt has limited context capacity.  Three similar
        past tasks provide enough context without overwhelming the prompt.
        """
        try:
            collection = self._get_collection(_COLLECTION_TASK_MEMORIES)
            count = collection.count()
            if count == 0:
                return []

            where: dict[str, Any] = {"user_id": user_id}
            results = collection.query(
                query_texts=[task_description],
                n_results=min(n_results, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return self._parse_results(results, _COLLECTION_TASK_MEMORIES)
        except Exception as exc:
            raise MemoryRetrievalError(
                f"query_similar_tasks failed: {exc}"
            ) from exc

    async def query_approaches(
        self,
        problem_type: str,
        user_id: str = "default",
    ) -> list[MemoryDocument]:
        """
        Query approach_memories for strategies matching `problem_type`.

        Returns both successful and failed approaches so the Supervisor
        can learn from both.  Failed approaches are tagged with
        metadata["outcome"] = "failed".
        """
        try:
            collection = self._get_collection(_COLLECTION_APPROACH_MEMORIES)
            count = collection.count()
            if count == 0:
                return []

            where: dict[str, Any] = {"user_id": user_id}
            results = collection.query(
                query_texts=[problem_type],
                n_results=min(5, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return self._parse_results(results, _COLLECTION_APPROACH_MEMORIES)
        except Exception as exc:
            raise MemoryRetrievalError(
                f"query_approaches failed: {exc}"
            ) from exc

    async def query_user_preferences(
        self,
        user_id: str,
    ) -> list[MemoryDocument]:
        """
        Retrieve all stored preferences for `user_id`.

        User preferences are stored with the user_id as both the metadata
        filter and the primary indexing dimension — we retrieve all of them
        rather than doing a similarity search.
        """
        try:
            collection = self._get_collection(_COLLECTION_USER_PREFERENCES)
            count = collection.count()
            if count == 0:
                return []

            results = collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
            )
            if not results or not results.get("ids"):
                return []

            docs: list[MemoryDocument] = []
            for doc_id, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                docs.append(
                    MemoryDocument(
                        id=doc_id,
                        content=doc,
                        metadata=dict(meta),
                        collection=_COLLECTION_USER_PREFERENCES,
                        similarity_score=None,
                    )
                )
            return docs
        except Exception as exc:
            raise MemoryRetrievalError(
                f"query_user_preferences failed for user {user_id!r}: {exc}"
            ) from exc

    async def get_memory_by_id(self, memory_id: str) -> MemoryDocument | None:
        """
        Fetch a specific memory document by its exact ID across all collections.

        Returns None if the ID is not found in any collection.
        """
        for collection_name in _ALL_COLLECTIONS:
            try:
                collection = self._get_collection(collection_name)
                results = collection.get(
                    ids=[memory_id],
                    include=["documents", "metadatas"],
                )
                if results and results.get("ids") and results["ids"]:
                    return MemoryDocument(
                        id=memory_id,
                        content=results["documents"][0],
                        metadata=dict(results["metadatas"][0]),
                        collection=collection_name,
                        similarity_score=None,
                    )
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def delete_user_memories(self, user_id: str) -> int:
        """
        Delete ALL memory documents for `user_id` across all collections.

        Returns the total count of deleted documents.

        WHY NOT A SOFT DELETE
        ----------------------
        Soft deletion (flagging as deleted) would require filtering on every
        query.  Hard deletion is simpler and ChromaDB's where-based delete
        is atomic per-collection.
        """
        total_deleted = 0
        where: dict[str, Any] = {"user_id": user_id}

        for collection_name in _ALL_COLLECTIONS:
            try:
                collection = self._get_collection(collection_name)
                # Find all matching IDs first (ChromaDB delete requires IDs).
                results = collection.get(
                    where=where,
                    include=[],
                )
                if results and results.get("ids"):
                    ids_to_delete = results["ids"]
                    collection.delete(ids=ids_to_delete)
                    total_deleted += len(ids_to_delete)
                    log.debug(
                        "semantic_memory_deleted",
                        collection=collection_name,
                        user_id=user_id,
                        count=len(ids_to_delete),
                    )
            except Exception as exc:
                log.warning(
                    "semantic_memory_delete_failed",
                    collection=collection_name,
                    user_id=user_id,
                    error=str(exc),
                )
                continue

        log.info(
            "semantic_memory_user_deleted",
            user_id=user_id,
            total_deleted=total_deleted,
        )
        return total_deleted

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    async def increment_access(self, memory_id: str) -> None:
        """
        Increment the access_count metadata field for a memory document.

        access_count is used to surface frequently-retrieved memories in
        MemoryStats.most_accessed, which can inform garbage collection
        policies (keep high-access memories longer).

        This is a best-effort operation — failures are logged but not raised.
        """
        for collection_name in _ALL_COLLECTIONS:
            try:
                collection = self._get_collection(collection_name)
                results = collection.get(
                    ids=[memory_id],
                    include=["metadatas"],
                )
                if results and results.get("ids") and results["ids"]:
                    meta = dict(results["metadatas"][0])
                    meta["access_count"] = int(meta.get("access_count", 0)) + 1
                    collection.update(ids=[memory_id], metadatas=[meta])
                    return
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _increment_access_sync(
        self,
        collection: chromadb.Collection,
        doc_id: str,
        current_meta: dict[str, Any],
    ) -> None:
        """Synchronous access count increment (called from within query loops)."""
        try:
            meta = dict(current_meta)
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            collection.update(ids=[doc_id], metadatas=[meta])
        except Exception:
            pass

    def _parse_results(
        self,
        results: dict[str, Any],
        collection_name: str,
    ) -> list[MemoryDocument]:
        """
        Convert raw ChromaDB query results into a list of MemoryDocument.

        Only includes results above the configured relevance threshold.
        """
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        docs: list[MemoryDocument] = []
        ids = results["ids"][0]
        raw_docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results.get("distances", [[0.0] * len(ids)])[0]

        for doc_id, doc, meta, dist in zip(ids, raw_docs, metas, distances):
            similarity = max(0.0, 1.0 - dist)
            if similarity >= memory_settings.memory_relevance_threshold:
                docs.append(
                    MemoryDocument(
                        id=doc_id,
                        content=doc,
                        metadata=dict(meta),
                        collection=collection_name,
                        similarity_score=round(similarity, 4),
                    )
                )

        docs.sort(key=lambda d: d.similarity_score or 0.0, reverse=True)
        return docs
