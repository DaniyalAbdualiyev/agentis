"""
routers/memory.py — FastAPI router for the Phase 2B memory management API.

WHY A DEDICATED ROUTER (not adding endpoints to routers/tasks.py)
------------------------------------------------------------------
Memory management is a distinct concern from task submission and status
polling.  Mixing them in tasks.py would violate the single-responsibility
principle and make the router harder to read.  A dedicated router also
makes it easy to apply different authentication, rate-limiting, or versioning
policies to memory endpoints in the future without touching task endpoints.

ENDPOINTS PROVIDED
------------------
GET  /api/memory/dashboard/{user_id}     — rich usage overview
GET  /api/memory/stats/{user_id}         — typed MemoryStats model
GET  /api/memory/search                  — semantic search across all collections
DELETE /api/memory/user/{user_id}        — GDPR: delete all memories for a user
DELETE /api/memory/{memory_id}           — delete a specific memory by ID
POST /api/memory/consolidate/{user_id}   — trigger memory consolidation on demand

WHY PREFIX /api/memory (not /memory)
-------------------------------------
The /api/ prefix groups all programmatic endpoints together, making it easy
to apply API-level middleware (auth, rate limiting, versioning) in one place.
The /health endpoint deliberately does NOT have this prefix because it must
be reachable without authentication by load balancers.

GRACEFUL DEGRADATION
--------------------
All endpoints catch exceptions from the memory layer and return HTTP 500
with a descriptive message, rather than letting FastAPI return a generic 500.
This means the frontend always gets a JSON error body rather than an HTML
error page.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query

from app.memory.management import MemoryManager
from app.memory.models import MemoryDocument, MemoryStats
from app.memory.semantic_memory import SemanticMemoryManager

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _get_memory_manager() -> MemoryManager:
    """
    Instantiate a MemoryManager with a fresh SemanticMemoryManager.

    WHY A FACTORY FUNCTION (not a module-level singleton)
    -----------------------------------------------------
    A module-level singleton would share state across requests in an async
    FastAPI server.  SemanticMemoryManager's ChromaDB client is synchronous
    under the hood, so keeping one instance per request avoids potential
    concurrency issues.  The cost of constructing SemanticMemoryManager is
    trivial (just an HttpClient init — no network call in __init__).
    """
    return MemoryManager()


def _get_semantic_manager() -> SemanticMemoryManager:
    """Instantiate a fresh SemanticMemoryManager per request."""
    return SemanticMemoryManager()


# ---------------------------------------------------------------------------
# GET /api/memory/dashboard/{user_id}
# ---------------------------------------------------------------------------

@router.get("/dashboard/{user_id}")
async def get_dashboard(user_id: str) -> dict:
    """
    Return a rich usage overview for the given user.

    Response fields:
    - total_memories  : int — total document count across all collections
    - by_collection   : dict[str, int] — per-collection counts
    - top_accessed    : list of {id, content_preview, access_count}
    - recent          : list of {id, content_preview, timestamp}
    - avg_importance  : float — mean importance score

    WHY NO PAGINATION HERE
    ----------------------
    The dashboard aggregates data for display, not for iteration.  It always
    returns the top 5 in each category — suitable for a UI widget.  If a
    caller needs all memories, they should use the /search endpoint with a
    broad query.
    """
    manager = _get_memory_manager()
    try:
        dashboard = await manager.get_dashboard(user_id=user_id)
        log.info("dashboard_fetched", user_id=user_id, total=dashboard.get("total_memories"))
        return dashboard
    except Exception as exc:
        log.error("dashboard_failed", user_id=user_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch memory dashboard for user {user_id!r}: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /api/memory/stats/{user_id}
# ---------------------------------------------------------------------------

@router.get("/stats/{user_id}", response_model=MemoryStats)
async def get_stats(user_id: str) -> MemoryStats:
    """
    Return a typed MemoryStats model for the given user.

    WHY BOTH /dashboard AND /stats
    --------------------------------
    /dashboard returns a rich dict for human-readable display (flexible schema).
    /stats returns a validated Pydantic model — useful for typed consumers,
    health checks, and metrics pipelines that need a stable schema.
    """
    manager = _get_memory_manager()
    try:
        stats = await manager.get_stats(user_id=user_id)
        log.info("stats_fetched", user_id=user_id, total=stats.total_memories)
        return stats
    except Exception as exc:
        log.error("stats_failed", user_id=user_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch memory stats for user {user_id!r}: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /api/memory/search
# ---------------------------------------------------------------------------

@router.get("/search", response_model=list[MemoryDocument])
async def search_memories(
    q: str = Query(..., description="Semantic search query text"),
    user_id: str = Query(default="default_user", description="User ID to filter by"),
    n: int = Query(default=10, ge=1, le=50, description="Maximum number of results"),
) -> list[MemoryDocument]:
    """
    Perform a semantic search across all memory collections.

    WHY SEMANTIC SEARCH (not keyword search)
    ----------------------------------------
    Memories are stored as embeddings.  A keyword search for "AI market" would
    miss memories stored as "artificial intelligence industry analysis".  Semantic
    search finds conceptually similar memories regardless of exact wording.

    Parameters
    ----------
    q       : The natural-language query to embed and search with.
    user_id : Filter results to this user's memories.
    n       : Maximum number of results to return (capped at 50 to prevent
              huge responses — for bulk retrieval, use the dashboard instead).
    """
    semantic_memory = _get_semantic_manager()
    try:
        results = await semantic_memory.query(
            query_text=q,
            user_id=user_id,
            n_results=n,
        )
        log.info(
            "memory_search",
            query=q[:80],
            user_id=user_id,
            results=len(results),
        )
        return results
    except Exception as exc:
        log.error("memory_search_failed", query=q[:80], error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Memory search failed: {exc}",
        )


# ---------------------------------------------------------------------------
# DELETE /api/memory/user/{user_id}
# ---------------------------------------------------------------------------

@router.delete("/user/{user_id}")
async def delete_user_memories(user_id: str) -> dict:
    """
    Delete ALL memories for a user across all collections.

    WHY THIS ENDPOINT EXISTS (GDPR compliance)
    ------------------------------------------
    Under GDPR (and similar regulations), users have the right to erasure —
    the "right to be forgotten".  This endpoint provides a single call to
    remove all of a user's stored memories from ChromaDB.

    IMPORTANT: This is a hard delete.  There is no soft delete or undo.
    Callers should confirm the intent before hitting this endpoint.

    Returns
    -------
    {"deleted_count": int, "message": str}
    """
    semantic_memory = _get_semantic_manager()
    try:
        deleted_count = await semantic_memory.delete_user_memories(user_id=user_id)
        log.info(
            "user_memories_deleted",
            user_id=user_id,
            deleted_count=deleted_count,
        )
        return {
            "deleted_count": deleted_count,
            "message": f"All memories for user {user_id!r} deleted",
        }
    except Exception as exc:
        log.error("delete_user_memories_failed", user_id=user_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete memories for user {user_id!r}: {exc}",
        )


# ---------------------------------------------------------------------------
# DELETE /api/memory/{memory_id}
# ---------------------------------------------------------------------------

@router.delete("/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """
    Delete a specific memory document by its ID.

    WHY INDIVIDUAL DELETION (in addition to bulk delete)
    ----------------------------------------------------
    Bulk delete is for user account closure.  Individual deletion is for
    fine-grained memory curation — e.g. a user notices that a specific
    memory contains incorrect information and wants to remove only that one.

    This endpoint searches all collections for the memory_id.  ChromaDB does
    not provide a cross-collection delete-by-ID, so we iterate collections.

    Returns
    -------
    {"deleted": True, "memory_id": memory_id}

    Raises HTTP 404 if the memory_id is not found in any collection.
    """
    from app.memory.semantic_memory import _ALL_COLLECTIONS

    semantic_memory = _get_semantic_manager()

    # First, find which collection contains this memory.
    found_collection: str | None = None
    try:
        doc = await semantic_memory.get_memory_by_id(memory_id)
        if doc is not None:
            found_collection = doc.collection
    except Exception as exc:
        log.warning("delete_memory_lookup_failed", memory_id=memory_id, error=str(exc))

    if found_collection is None:
        raise HTTPException(
            status_code=404,
            detail=f"Memory {memory_id!r} not found in any collection.",
        )

    try:
        collection = semantic_memory._get_collection(found_collection)
        collection.delete(ids=[memory_id])
        log.info(
            "memory_deleted",
            memory_id=memory_id,
            collection=found_collection,
        )
        return {"deleted": True, "memory_id": memory_id}
    except Exception as exc:
        log.error("delete_memory_failed", memory_id=memory_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete memory {memory_id!r}: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /api/memory/consolidate/{user_id}
# ---------------------------------------------------------------------------

@router.post("/consolidate/{user_id}")
async def consolidate_memories(user_id: str) -> dict:
    """
    Trigger manual memory consolidation for a user.

    WHY AN ON-DEMAND ENDPOINT (in addition to scheduled consolidation)
    ------------------------------------------------------------------
    Automatic consolidation runs on a schedule, but users or operators may
    want to trigger it immediately after a large batch of tasks — e.g. after
    running 50 tasks about the same topic, they want to merge the redundant
    memories right away.

    This is an async operation — for large memory stores it may take several
    seconds due to the ChromaDB queries and LLM calls inside consolidation.
    Consider adding a background task queue in future phases if latency
    becomes an issue.

    Returns
    -------
    {"consolidated_count": int, "message": str}
    """
    manager = _get_memory_manager()
    try:
        consolidated_count = await manager.consolidate_memories(user_id=user_id)
        message = (
            f"Consolidated {consolidated_count} memories for user {user_id!r}."
            if consolidated_count > 0
            else f"No memories eligible for consolidation for user {user_id!r}."
        )
        log.info(
            "consolidation_triggered",
            user_id=user_id,
            consolidated_count=consolidated_count,
        )
        return {"consolidated_count": consolidated_count, "message": message}
    except Exception as exc:
        log.error("consolidation_failed", user_id=user_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Memory consolidation failed for user {user_id!r}: {exc}",
        )
