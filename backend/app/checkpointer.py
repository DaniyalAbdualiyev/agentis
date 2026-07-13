"""
app/checkpointer.py — AsyncPostgresSaver singleton for Phase 3 HITL.

WHY A SEPARATE MODULE
----------------------
The checkpointer must be:
  1. Initialised ONCE at app startup (async, needs a live DB connection).
  2. Accessed from tasks.py (to invoke the graph with checkpointing).
  3. Accessed from routers/reviews.py (to resume a paused graph run).

A module-level mutable container gives all three callers access to the
same compiled-graph object via a plain `from app.checkpointer import get_graph`
import, without circular imports or god-object patterns.

WHY AsyncPostgresSaver + psycopg3 (not asyncpg)
-------------------------------------------------
langgraph-checkpoint-postgres requires psycopg3 (the `psycopg` package),
NOT the `asyncpg` driver that SQLAlchemy uses.  Both drivers speak the
PostgreSQL wire protocol, but psycopg3 is what the LangGraph checkpoint
implementation is built against.

The DATABASE_URL in .env uses `postgresql+asyncpg://...` for SQLAlchemy.
We strip the `+asyncpg` driver hint before passing to psycopg3.

WHY autocommit=True in the pool kwargs
----------------------------------------
LangGraph's checkpoint implementation issues SAVEPOINT / RELEASE SAVEPOINT
statements.  These are incompatible with an implicit transaction wrapper.
psycopg3's autocommit mode is the correct setting when LangGraph manages
its own transaction boundaries.

WHY prepare_threshold=0
------------------------
Prepared statements are cached by connection, but with a connection pool
multiple connections share the same query patterns.  Setting
prepare_threshold=0 disables server-side prepared statements so that
pgBouncer (or other poolers) can be introduced later without invalidating
the plan cache.
"""
from __future__ import annotations

import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Mutable singletons — populated by setup() during app startup.
# ---------------------------------------------------------------------------

_pool: Any = None          # psycopg_pool.AsyncConnectionPool instance
_checkpointer: Any = None  # langgraph AsyncPostgresSaver instance
_graph: Any = None         # compiled graph WITH checkpointer (used by FastAPI)


def get_checkpointer() -> Any:
    """Return the AsyncPostgresSaver instance, or None if not yet initialised."""
    return _checkpointer


def get_graph() -> Any:
    """
    Return the compiled graph that should be used for task invocation and
    graph resumption (the checkpointer-enabled version).

    Falls back to the module-level `graph` alias in graph.py (no checkpointer,
    used by LangGraph Studio) if setup() was never called.  This keeps
    unit-tests and Studio usage working without a live PostgreSQL connection.
    """
    if _graph is not None:
        return _graph
    # Fallback: compile without checkpointer (Studio / tests)
    from app.graph.graph import build_graph
    return build_graph().compile()


async def setup(db_url: str) -> None:
    """
    Initialise the psycopg3 connection pool, create the AsyncPostgresSaver,
    run checkpointer.setup() to create the checkpoint tables, and compile
    the graph with the checkpointer.

    Called once from main.py's startup handler.  All errors are caught and
    logged — if the checkpointer cannot be set up, the app continues with
    graceful degradation (graph runs without persistence, HITL is disabled).
    """
    global _pool, _checkpointer, _graph

    # Convert SQLAlchemy driver URL to a plain psycopg3 conninfo string.
    # e.g. "postgresql+asyncpg://user:pass@host:5432/db"
    #   -> "postgresql://user:pass@host:5432/db"
    conninfo = db_url.replace("postgresql+asyncpg://", "postgresql://")

    try:
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from app.graph.graph import build_graph

        _pool = AsyncConnectionPool(
            conninfo=conninfo,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await _pool.open()
        log.info("checkpointer_pool_opened")

        _checkpointer = AsyncPostgresSaver(_pool)
        await _checkpointer.setup()
        log.info("checkpointer_tables_ready")

        # Compile with interrupt_before=[] — we use interrupt() inside nodes
        # (not interrupt_before on edges), so no pre-node interrupts needed here.
        _graph = build_graph().compile(checkpointer=_checkpointer)
        log.info("graph_compiled_with_checkpointer")

    except Exception as exc:
        log.warning(
            "checkpointer_setup_failed",
            error=str(exc),
            note="HITL is disabled. Graph will run without checkpointing.",
        )
        _pool = None
        _checkpointer = None
        _graph = None


async def teardown() -> None:
    """Close the psycopg3 pool on app shutdown."""
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
            log.info("checkpointer_pool_closed")
        except Exception as exc:
            log.warning("checkpointer_pool_close_failed", error=str(exc))
        _pool = None
