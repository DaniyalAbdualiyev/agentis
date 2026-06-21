"""
Async database query helpers.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SubtaskLog, Task, ToolCallLog


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

async def create_task(
    session: AsyncSession,
    original_task: str,
    task_id: str | None = None,
) -> Task:
    task = Task(
        id=task_id or str(uuid.uuid4()),
        original_task=original_task,
        status="pending",
    )
    session.add(task)
    await session.flush()
    return task


async def get_task(session: AsyncSession, task_id: str) -> Task | None:
    result = await session.execute(select(Task).where(Task.id == task_id))
    return result.scalar_one_or_none()


async def update_task(
    session: AsyncSession,
    task_id: str,
    **kwargs: Any,
) -> None:
    task = await get_task(session, task_id)
    if task is None:
        return
    for k, v in kwargs.items():
        setattr(task, k, v)
    await session.flush()


# ---------------------------------------------------------------------------
# SubtaskLog helpers
# ---------------------------------------------------------------------------

async def log_subtask(
    session: AsyncSession,
    task_id: str,
    subtask_id: str,
    subtask_description: str,
    assigned_agent: str,
    status: str,
    output: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> SubtaskLog:
    log = SubtaskLog(
        id=str(uuid.uuid4()),
        task_id=task_id,
        subtask_id=subtask_id,
        subtask_description=subtask_description,
        assigned_agent=assigned_agent,
        status=status,
        output=output,
        started_at=started_at,
        completed_at=completed_at,
    )
    session.add(log)
    await session.flush()
    return log


# ---------------------------------------------------------------------------
# ToolCallLog helper
# ---------------------------------------------------------------------------

async def log_tool_call(
    session: AsyncSession,
    task_id: str | None,
    subtask_id: str | None,
    tool_name: str,
    inputs: dict | None,
    outputs: Any,
    latency_ms: int,
    success: bool,
    error_message: str | None,
) -> ToolCallLog:
    # Ensure outputs is JSON-serialisable
    if outputs is not None and not isinstance(outputs, (dict, list, str, int, float, bool)):
        outputs = str(outputs)

    log_entry = ToolCallLog(
        id=str(uuid.uuid4()),
        task_id=task_id,
        subtask_id=subtask_id,
        tool_name=tool_name,
        inputs=inputs,
        outputs=outputs if isinstance(outputs, (dict, list)) else {"result": outputs},
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
        called_at=_now(),
    )
    session.add(log_entry)
    await session.flush()
    return log_entry
