"""
memory/management.py — Memory lifecycle management for the semantic memory store.

WHY THIS MODULE EXISTS
-----------------------
Without active management, a vector store grows indefinitely:
- Redundant memories accumulate, degrading retrieval quality (noise drowns signal).
- Old, unused memories consume storage and slow query times.
- Frequently-used memories are treated identically to rarely-used ones, so
  retrieval ranking cannot distinguish quality from noise.

This module provides three operations to keep the store healthy:

1. IMPORTANCE SCORING — a multi-factor formula that ranks how valuable a
   memory is, so future retrieval and cleanup decisions are data-driven.

2. CONSOLIDATION — clusters of near-duplicate memories are merged into a
   single high-quality summary, reducing noise while preserving information.

3. DECAY & EXPIRATION — importance scores of stale memories are reduced over
   time; memories that fall below a minimum threshold are marked as expired
   and excluded from future queries.

WHY THREE SEPARATE OPERATIONS (not one "clean" function)
---------------------------------------------------------
Each operation has a distinct trigger cadence:
- Importance scoring: called after every retrieval (cheap, per-doc).
- Consolidation: called periodically (e.g. nightly), or on demand via API.
- Decay: called periodically (e.g. weekly), separate from consolidation.

Combining them into one function would make it impossible to run them at
different intervals without splitting it again later.

WHY GRACEFUL DEGRADATION IS THE DEFAULT
-----------------------------------------
Memory management is a background concern — it must NEVER crash the main
task pipeline.  All public methods log errors and return safe default values
(counts, empty dicts) rather than raising.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from app.memory.models import MemoryDocument, MemoryStats
from app.memory.semantic_memory import (
    SemanticMemoryManager,
    _ALL_COLLECTIONS,
    _COLLECTION_TASK_MEMORIES,
    _COLLECTION_APPROACH_MEMORIES,
    _COLLECTION_USER_PREFERENCES,
    _COLLECTION_DOMAIN_FACTS,
)

log = structlog.get_logger(__name__)


class MemoryManager:
    """
    Handles memory lifecycle: importance scoring, consolidation, and expiration.

    WHY THESE THREE OPERATIONS
    --------------------------
    Without management, the memory store grows indefinitely and becomes noisy:
    - Importance scoring ensures frequently-useful memories rank higher in retrieval
    - Consolidation merges redundant memories into summaries, reducing noise
    - Expiration removes stale memories that haven't been useful, keeping retrieval fast

    INSTANTIATION
    -------------
    Cheap — no network calls in __init__.  The SemanticMemoryManager is created
    lazily or injected (useful for testing via mock injection).
    """

    def __init__(
        self, semantic_memory: SemanticMemoryManager | None = None
    ) -> None:
        """
        Initialise the manager.

        WHY DEPENDENCY INJECTION FOR semantic_memory
        ---------------------------------------------
        Tests can inject a mock SemanticMemoryManager so that unit tests for
        MemoryManager do not require a live ChromaDB service.  Production code
        calls MemoryManager() with no args.
        """
        self._semantic_memory = semantic_memory or SemanticMemoryManager()

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------

    def calculate_importance(self, memory_metadata: dict) -> float:
        """
        Calculate an importance score (0.0–1.0) for a memory document based on
        recency, access frequency, task success, and its existing importance.

        WHY A COMPOSITE FORMULA (not just access_count)
        ------------------------------------------------
        Access count alone would favour old, frequently-retrieved memories
        even if they are no longer relevant (stale but popular).  The recency
        factor decays old memories, the success factor boosts memories from
        tasks that completed successfully, and the base_importance carries
        forward any human-set priority.  Together they produce a balanced score.

        FORMULA BREAKDOWN
        -----------------
        recency_factor  = max(0.1, 1.0 - days_old * 0.01)
            Newer memories score closer to 1.0; memories 90 days old score ~0.1.

        access_factor   = min(1.0, access_count * 0.1)
            Each retrieval adds 0.1 to the factor, capped at 1.0 (10+ accesses).

        success_factor  = 1.0 if success else 0.7
            Memories from successful tasks are weighted higher because they
            demonstrate strategies that actually worked.

        importance = (base_importance * 0.4
                      + recency_factor  * 0.3
                      + access_factor   * 0.2
                      + success_factor  * 0.1)

        Weights reflect their relative information value:
        - base_importance carries previous scoring (40% — highest weight)
        - recency reflects current relevance (30%)
        - access reflects proven usefulness (20%)
        - success is binary but meaningful (10%)

        Parameters
        ----------
        memory_metadata : dict
            The metadata dict from a ChromaDB document.

        Returns
        -------
        float clamped to [0.0, 1.0].
        """
        # Parse stored timestamp to calculate age in days.
        timestamp_str = memory_metadata.get("timestamp", "")
        days_old = 0.0
        if timestamp_str:
            try:
                stored_at = datetime.fromisoformat(timestamp_str)
                # Ensure timezone-aware comparison.
                if stored_at.tzinfo is None:
                    stored_at = stored_at.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                days_old = max(0.0, (now - stored_at).total_seconds() / 86400)
            except (ValueError, TypeError):
                days_old = 0.0

        base_importance = float(memory_metadata.get("importance_score", 0.5))
        access_count = int(memory_metadata.get("access_count", 0))
        success = bool(memory_metadata.get("success", True))

        recency_factor = max(0.1, 1.0 - (days_old * 0.01))
        access_factor = min(1.0, access_count * 0.1)
        success_factor = 1.0 if success else 0.7

        importance = (
            base_importance * 0.4
            + recency_factor * 0.3
            + access_factor * 0.2
            + success_factor * 0.1
        )

        # Clamp to [0.0, 1.0] to guard against edge cases in the inputs.
        return max(0.0, min(1.0, importance))

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def consolidate_memories(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
    ) -> int:
        """
        Find clusters of near-duplicate memories and merge them into summaries.

        WHY CONSOLIDATION IS NEEDED
        ----------------------------
        Over many task runs, the same problem type (e.g. "research AI market")
        will produce many nearly-identical approach memories.  These duplicates:
        - Dilute retrieval quality (the top results are all near-identical).
        - Waste ChromaDB storage and embedding budget.
        Consolidation merges a cluster of similar memories into one concise
        summary, preserving all key information in fewer documents.

        ALGORITHM
        ---------
        1. Fetch all memories for the user from each collection.
        2. For each collection, compute pairwise similarity between memories
           using their stored content (re-embedding is done by ChromaDB query).
        3. Group memories into clusters using greedy connected-components
           (if A is similar to B, and B is similar to C, all three are one cluster).
        4. For each cluster with 2+ members:
           a. Use call_llm(role="specialist") to generate a consolidated summary.
           b. Store the summary as a new memory with averaged metadata.
           c. Mark the originals with metadata["consolidated"] = True so they
              can be filtered out of future queries or deleted in a cleanup pass.
        5. Return the count of source memories that were consolidated.

        WHY GREEDY CONNECTED-COMPONENTS (not k-means)
        -----------------------------------------------
        k-means requires choosing k upfront, which is unknown.  Connected-
        components clustering is parameter-free (the threshold is the only knob)
        and handles clusters of any size.  For the memory sizes typical in this
        system (< 1000 documents), the O(n²) pairwise comparison is fast enough.

        Parameters
        ----------
        user_id : str
            Only memories for this user are considered.
        similarity_threshold : float
            Two memories are considered near-duplicates if their cosine similarity
            exceeds this value.  Default 0.85 is intentionally strict to avoid
            merging genuinely different memories.

        Returns
        -------
        int : Count of source memories that were consolidated into summaries.
        """
        from app.llm.client import call_llm

        total_consolidated = 0

        for collection_name in _ALL_COLLECTIONS:
            try:
                collection = self._semantic_memory._get_collection(collection_name)
                # Fetch all user memories from this collection.
                results = collection.get(
                    where={"user_id": user_id},
                    include=["documents", "metadatas"],
                )
                if not results or not results.get("ids") or len(results["ids"]) < 2:
                    continue

                ids: list[str] = results["ids"]
                docs: list[str] = results["documents"]
                metas: list[dict] = [dict(m) for m in results["metadatas"]]

                # Skip already-consolidated memories.
                active_indices = [
                    i for i, m in enumerate(metas)
                    if not m.get("consolidated", False)
                ]
                if len(active_indices) < 2:
                    continue

                # Build clusters via greedy connected-components.
                # For each memory, find others with similarity > threshold.
                clusters: list[set[int]] = []

                # We use ChromaDB's query to find similar documents.
                # For each document, query the collection with its own content
                # to find near-duplicates.
                grouped: set[int] = set()

                for i in active_indices:
                    if i in grouped:
                        continue
                    cluster = {i}
                    # Query for similar documents using this doc's content.
                    try:
                        query_result = collection.query(
                            query_texts=[docs[i]],
                            n_results=min(10, len(active_indices)),
                            where={"user_id": user_id},
                            include=["documents", "metadatas", "distances"],
                        )
                        if query_result and query_result.get("ids"):
                            result_ids = query_result["ids"][0]
                            result_distances = query_result.get("distances", [[]])[0]
                            for result_id, dist in zip(result_ids, result_distances):
                                similarity = max(0.0, 1.0 - dist)
                                if similarity >= similarity_threshold and result_id != ids[i]:
                                    # Find the index of this result_id in our ids list.
                                    try:
                                        j = ids.index(result_id)
                                        if j in active_indices:
                                            cluster.add(j)
                                    except ValueError:
                                        continue
                    except Exception as query_exc:
                        log.warning(
                            "consolidation_query_failed",
                            collection=collection_name,
                            error=str(query_exc),
                        )

                    if len(cluster) >= 2:
                        clusters.append(cluster)
                        grouped.update(cluster)
                    else:
                        grouped.add(i)

                # Process each cluster.
                for cluster in clusters:
                    cluster_indices = sorted(cluster)
                    cluster_docs = [docs[i] for i in cluster_indices]
                    cluster_metas = [metas[i] for i in cluster_indices]

                    # Ask the LLM to produce a consolidated summary.
                    combined = "\n\n---\n\n".join(
                        f"Memory {k+1}:\n{doc}" for k, doc in enumerate(cluster_docs)
                    )
                    consolidation_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "You are a memory consolidation assistant. "
                                "Your job is to merge related memory entries into "
                                "one concise summary that preserves all key information."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Combine these related memories into a single concise "
                                f"summary that preserves all key information:\n\n{combined}"
                            ),
                        },
                    ]
                    try:
                        summary: str = await call_llm(
                            role="specialist",
                            messages=consolidation_prompt,
                        )
                    except Exception as llm_exc:
                        log.warning(
                            "consolidation_llm_failed",
                            collection=collection_name,
                            error=str(llm_exc),
                        )
                        continue

                    # Build averaged metadata for the consolidated document.
                    avg_importance = sum(
                        float(m.get("importance_score", 0.5)) for m in cluster_metas
                    ) / len(cluster_metas)
                    avg_access = sum(
                        int(m.get("access_count", 0)) for m in cluster_metas
                    ) // len(cluster_metas)
                    consolidated_meta: dict[str, Any] = {
                        "user_id": user_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "importance_score": avg_importance,
                        "access_count": avg_access,
                        "consolidated": True,
                        "source_count": len(cluster),
                    }
                    # Carry forward relevant type-specific fields from the first member.
                    for field in ("task_type", "problem_type", "outcome", "success"):
                        if field in cluster_metas[0]:
                            consolidated_meta[field] = cluster_metas[0][field]

                    import uuid
                    new_id = str(uuid.uuid4())
                    try:
                        collection.add(
                            ids=[new_id],
                            documents=[summary],
                            metadatas=[consolidated_meta],
                        )
                    except Exception as add_exc:
                        log.warning(
                            "consolidation_add_failed",
                            collection=collection_name,
                            error=str(add_exc),
                        )
                        continue

                    # Mark originals as consolidated so they are excluded from
                    # future queries via metadata filter.
                    for idx in cluster_indices:
                        try:
                            updated_meta = dict(metas[idx])
                            updated_meta["consolidated"] = True
                            collection.update(
                                ids=[ids[idx]], metadatas=[updated_meta]
                            )
                        except Exception:
                            pass

                    total_consolidated += len(cluster)
                    log.info(
                        "memories_consolidated",
                        collection=collection_name,
                        cluster_size=len(cluster),
                        new_memory_id=new_id,
                    )

            except Exception as exc:
                log.warning(
                    "consolidation_collection_failed",
                    collection=collection_name,
                    user_id=user_id,
                    error=str(exc),
                )
                continue

        log.info(
            "consolidation_complete",
            user_id=user_id,
            total_consolidated=total_consolidated,
        )
        return total_consolidated

    # ------------------------------------------------------------------
    # Decay & expiration
    # ------------------------------------------------------------------

    async def decay_memories(
        self,
        decay_rate: float = 0.95,
        min_importance: float = 0.1,
    ) -> int:
        """
        Apply importance decay to memories that haven't been accessed recently.

        WHY DECAY IS NEEDED
        --------------------
        Without decay, memories accumulate indefinitely.  A memory from two
        years ago about "how to research AI trends" is less relevant today than
        one from last week — but without decay, both would have the same
        importance score if they had the same access count.  Decay ensures
        that relevance tracks recency as well as usage.

        ALGORITHM
        ---------
        For each memory not accessed in the last 30 days:
          new_importance = current_importance * decay_rate
          if new_importance < min_importance:
              mark as expired (metadata["expired"] = True)
          else:
              update importance_score in metadata

        WHY 30 DAYS (not 7 or 90)
        --------------------------
        7 days would decay memories that are still fresh from recent tasks.
        90 days would allow too much stale data to accumulate.  30 days is a
        reasonable "cold" threshold — if a memory hasn't been retrieved in a
        month, it's worth starting to reduce its priority.

        Parameters
        ----------
        decay_rate : float
            Multiplicative factor applied to each stale memory's importance
            (0 < decay_rate < 1).  Default 0.95 applies gentle 5% decay.
        min_importance : float
            Importance scores that fall below this threshold after decay are
            marked as expired.  Default 0.1 prevents memories from lingering
            indefinitely with near-zero importance.

        Returns
        -------
        int : Count of memories that were marked as expired.
        """
        expired_count = 0
        staleness_threshold_days = 30
        now = datetime.now(timezone.utc)

        for collection_name in _ALL_COLLECTIONS:
            try:
                collection = self._semantic_memory._get_collection(collection_name)
                results = collection.get(include=["metadatas"])
                if not results or not results.get("ids"):
                    continue

                ids: list[str] = results["ids"]
                metas: list[dict] = [dict(m) for m in results["metadatas"]]

                for doc_id, meta in zip(ids, metas):
                    # Skip already expired or consolidated memories.
                    if meta.get("expired") or meta.get("consolidated"):
                        continue

                    # Determine when the memory was last accessed.
                    # We use the stored timestamp as a proxy for "last accessed"
                    # because ChromaDB does not track access time natively.
                    # The access_count field is a rough signal of recency.
                    timestamp_str = meta.get("timestamp", "")
                    if not timestamp_str:
                        continue

                    try:
                        stored_at = datetime.fromisoformat(timestamp_str)
                        if stored_at.tzinfo is None:
                            stored_at = stored_at.replace(tzinfo=timezone.utc)
                        days_since = (now - stored_at).total_seconds() / 86400
                    except (ValueError, TypeError):
                        continue

                    if days_since < staleness_threshold_days:
                        continue  # Recently created — skip.

                    current_importance = float(meta.get("importance_score", 0.5))
                    new_importance = current_importance * decay_rate

                    updated_meta = dict(meta)
                    if new_importance < min_importance:
                        updated_meta["expired"] = True
                        updated_meta["importance_score"] = 0.0
                        expired_count += 1
                        log.debug(
                            "memory_expired",
                            collection=collection_name,
                            memory_id=doc_id,
                            old_importance=current_importance,
                        )
                    else:
                        updated_meta["importance_score"] = round(new_importance, 4)

                    try:
                        collection.update(ids=[doc_id], metadatas=[updated_meta])
                    except Exception as update_exc:
                        log.warning(
                            "decay_update_failed",
                            collection=collection_name,
                            memory_id=doc_id,
                            error=str(update_exc),
                        )

            except Exception as exc:
                log.warning(
                    "decay_collection_failed",
                    collection=collection_name,
                    error=str(exc),
                )
                continue

        log.info("decay_complete", expired_count=expired_count)
        return expired_count

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    async def get_dashboard(self, user_id: str) -> dict:
        """
        Generate a dashboard overview of a user's memory store.

        Returns a dict with:
        - total_memories     : count across all collections
        - by_collection      : {collection_name: count}
        - top_accessed       : top 5 most-accessed memories (id, preview, count)
        - recent             : last 5 memories added (id, preview, timestamp)
        - avg_importance     : mean importance_score across all memories

        WHY A DICT (not a Pydantic model)
        ----------------------------------
        The dashboard is used by the API router, which serialises it directly
        to JSON.  A plain dict avoids an extra model import in the router and
        keeps the dashboard flexible — fields can be added without a model
        migration.

        GRACEFUL DEGRADATION
        --------------------
        Each collection is fetched independently.  A failure in one collection
        contributes 0 documents and does not prevent the others from appearing.
        """
        by_collection: dict[str, int] = {}
        all_memories: list[dict] = []

        for collection_name in _ALL_COLLECTIONS:
            try:
                collection = self._semantic_memory._get_collection(collection_name)
                results = collection.get(
                    where={"user_id": user_id},
                    include=["documents", "metadatas"],
                )
                if not results or not results.get("ids"):
                    by_collection[collection_name] = 0
                    continue

                ids = results["ids"]
                docs = results["documents"]
                metas = [dict(m) for m in results["metadatas"]]

                by_collection[collection_name] = len(ids)
                for doc_id, doc, meta in zip(ids, docs, metas):
                    all_memories.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                        "collection": collection_name,
                    })
            except Exception as exc:
                log.warning(
                    "dashboard_collection_failed",
                    collection=collection_name,
                    user_id=user_id,
                    error=str(exc),
                )
                by_collection[collection_name] = 0

        total = sum(by_collection.values())

        # Top 5 most accessed.
        sorted_by_access = sorted(
            all_memories,
            key=lambda m: int(m["metadata"].get("access_count", 0)),
            reverse=True,
        )
        top_accessed = [
            {
                "id": m["id"],
                "content_preview": m["content"][:120],
                "access_count": int(m["metadata"].get("access_count", 0)),
            }
            for m in sorted_by_access[:5]
        ]

        # Last 5 added (sort by timestamp descending).
        sorted_by_time = sorted(
            all_memories,
            key=lambda m: m["metadata"].get("timestamp", ""),
            reverse=True,
        )
        recent = [
            {
                "id": m["id"],
                "content_preview": m["content"][:120],
                "timestamp": m["metadata"].get("timestamp", ""),
            }
            for m in sorted_by_time[:5]
        ]

        # Average importance.
        importances = [
            float(m["metadata"].get("importance_score", 0.0))
            for m in all_memories
        ]
        avg_importance = (
            sum(importances) / len(importances) if importances else 0.0
        )

        return {
            "total_memories": total,
            "by_collection": by_collection,
            "top_accessed": top_accessed,
            "recent": recent,
            "avg_importance": round(avg_importance, 4),
        }

    async def get_stats(self, user_id: str) -> MemoryStats:
        """
        Return a structured MemoryStats for a user.

        WHY get_stats IS SEPARATE FROM get_dashboard
        ---------------------------------------------
        get_dashboard returns a rich dict for human-readable display.
        get_stats returns a Pydantic model — it is typed and machine-readable,
        useful for health checks, metrics export, and structured API responses.
        Keeping both means neither becomes bloated trying to serve both purposes.
        """
        dashboard = await self.get_dashboard(user_id)

        most_accessed_ids = [
            entry["id"] for entry in dashboard.get("top_accessed", [])
        ]

        return MemoryStats(
            total_memories=dashboard["total_memories"],
            by_collection=dashboard["by_collection"],
            avg_importance=dashboard["avg_importance"],
            most_accessed=most_accessed_ids,
        )
