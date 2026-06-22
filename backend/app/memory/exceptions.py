"""
memory/exceptions.py — Custom exceptions for the Phase 2A memory subsystem.

WHY CUSTOM EXCEPTIONS INSTEAD OF LETTING redis/chromadb ERRORS PROPAGATE
--------------------------------------------------------------------------
redis-py and chromadb each have their own exception hierarchies.  If we let
those propagate up to the router layer, callers would need to import from
both libraries just to handle memory errors.  That creates a coupling between
the router and the specific memory backends — if we swap Redis for Valkey or
ChromaDB for Qdrant tomorrow, the router's except clauses would break.

Custom exceptions provide a stable API surface: the router catches
MemoryStoreError and MemoryRetrievalError regardless of what's underneath.

WHY ONLY TWO EXCEPTIONS (not a richer hierarchy)
-------------------------------------------------
The calling code in routers/tasks.py needs only two decisions:
  1. Did a write fail? → log warning, continue (don't crash the task).
  2. Did a read fail? → log warning, return empty context (degrade gracefully).
A richer hierarchy (ConnectionError, SerializationError, QuotaError, …) would
require the caller to handle more cases without meaningfully changing behaviour.
Both exceptions are caught at the same level and handled the same way.
"""


class MemoryStoreError(Exception):
    """
    Raised when a write to Redis or ChromaDB fails.

    Examples:
    - Redis connection refused during working_memory.store()
    - ChromaDB HTTP 500 during semantic_memory.store_task_completion()
    - JSON serialisation failure before the Redis write

    CALLERS MUST: log a warning and continue — memory write failures must
    NOT crash the task.  The task's result is more important than its memory.
    """


class MemoryRetrievalError(Exception):
    """
    Raised when a read or query from Redis or ChromaDB fails.

    Examples:
    - Redis connection refused during working_memory.retrieve()
    - ChromaDB query timeout during semantic_memory.query()
    - Malformed JSON returned from Redis that fails to deserialise

    CALLERS MUST: log a warning and return an empty/default value — the task
    must proceed even if historical context cannot be retrieved.
    """
