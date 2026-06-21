"""
Agentis — Multi-Agent Research Orchestration System
FastAPI application entry point.
"""
from __future__ import annotations

import logging
import os

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


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentis",
        description="Multi-Agent Research Orchestration System",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup() -> None:
        log.info("agentis_startup")

        # Initialize DB
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

    from app.routers.tasks import router as tasks_router
    app.include_router(tasks_router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
