"""
Agentis — Multi-Agent Research Orchestration System
FastAPI application entry point.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

# Configure structured logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Starlette lifespan context manager (replaces deprecated @app.on_event).
    Everything before `yield` runs on startup; everything after on shutdown.
    """
    # ── Startup ────────────────────────────────────────────────────────────
    log.info("agentis_startup")

    # Initialize DB (creates all SQLAlchemy-mapped tables idempotently)
    from app.db.engine import init_db
    await init_db()
    log.info("db_initialized")

    # Register tools (side-effects of importing)
    import app.tools  # noqa: F401
    log.info("tools_registered")

    # Log LangSmith status
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project = os.getenv("LANGCHAIN_PROJECT", "agentis")
    api_key_set = bool(os.getenv("LANGCHAIN_API_KEY"))
    log.info(
        "langsmith_config",
        tracing_enabled=tracing == "true",
        project=project,
        api_key_configured=api_key_set,
    )

    # ------------------------------------------------------------------
    # Phase 2A: check Redis (working memory) connectivity.
    # ------------------------------------------------------------------
    try:
        import redis.asyncio as aioredis
        from app.memory.config import memory_settings
        redis_client = aioredis.from_url(
            memory_settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis_client.ping()
        await redis_client.aclose()
        log.info("redis_connected", redis_url=memory_settings.redis_url)
    except Exception as redis_exc:
        log.warning(
            "redis_unavailable",
            error=str(redis_exc),
            note="Working memory will be disabled for task runs.",
        )

    # ------------------------------------------------------------------
    # Phase 2A: check ChromaDB (semantic memory) connectivity.
    # ------------------------------------------------------------------
    try:
        import chromadb
        from app.memory.config import memory_settings as ms
        chroma_client = chromadb.HttpClient(
            host=ms.chromadb_host,
            port=ms.chromadb_port,
        )
        chroma_client.heartbeat()
        log.info(
            "chromadb_connected",
            host=ms.chromadb_host,
            port=ms.chromadb_port,
        )
    except Exception as chroma_exc:
        log.warning(
            "chromadb_unavailable",
            error=str(chroma_exc),
            note="Semantic memory will be disabled for task runs.",
        )

    # ------------------------------------------------------------------
    # Phase 3: Initialise AsyncPostgresSaver checkpointer for HITL.
    # ------------------------------------------------------------------
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://agentis:agentis123@localhost:5432/agentisdb",
    )
    from app import checkpointer as ckpt_module
    await ckpt_module.setup(db_url)

    # ── Yield — app is running ─────────────────────────────────────────────
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    from app import checkpointer as ckpt_module  # re-import for clarity
    await ckpt_module.teardown()
    log.info("agentis_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentis",
        description="Multi-Agent Research Orchestration System",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.routers.tasks import router as tasks_router
    app.include_router(tasks_router)

    # Phase 2B: memory management + dashboard API
    from app.routers.memory import router as memory_router
    app.include_router(memory_router)

    # Phase 3: human-in-the-loop review API
    from app.routers.reviews import router as reviews_router
    app.include_router(reviews_router)

    # Phase 4: observability — execution traces, replay, and aggregated stats
    from app.routers.traces import router as traces_router
    from app.routers.stats import router as stats_router
    app.include_router(traces_router)
    app.include_router(stats_router, prefix="/stats")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
