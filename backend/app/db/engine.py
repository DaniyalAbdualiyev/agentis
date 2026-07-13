"""
SQLAlchemy async engine and session factory.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://agentis:agentis123@postgres:5432/agentisdb",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables (idempotent).

    create_all only CREATES missing tables — it never ALTERs an existing one.
    Phase 4 added four aggregate columns to the pre-existing `tasks` table, so
    we additively backfill them with `ADD COLUMN IF NOT EXISTS` (a Postgres
    feature).  This keeps the dev workflow migration-free while remaining safe
    to run repeatedly.  For production, a proper Alembic migration would replace
    this block.
    """
    from sqlalchemy import text

    from app.db import models  # noqa: F401 — import to register models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Phase 4: additive columns on the existing tasks table.
        for ddl in (
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS total_cost_usd DOUBLE PRECISION DEFAULT 0.0",
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS total_latency_ms INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS total_input_tokens INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS total_output_tokens INTEGER DEFAULT 0",
        ):
            await conn.execute(text(ddl))
