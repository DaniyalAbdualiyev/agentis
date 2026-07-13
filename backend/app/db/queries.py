"""
Async database query helpers.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ExecutionTrace, HumanReview, SubtaskLog, Task, ToolCallLog


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


# ---------------------------------------------------------------------------
# HumanReview helpers (Phase 3)
# ---------------------------------------------------------------------------

async def create_human_review(
    session: AsyncSession,
    task_id: str,
    escalation_reason: str,
    output_shown_to_human: str | None = None,
) -> HumanReview:
    record = HumanReview(
        id=str(uuid.uuid4()),
        task_id=task_id,
        escalation_reason=escalation_reason,
        output_shown_to_human=output_shown_to_human,
    )
    session.add(record)
    await session.flush()
    return record


async def get_human_review_by_task(
    session: AsyncSession, task_id: str
) -> HumanReview | None:
    result = await session.execute(
        select(HumanReview).where(HumanReview.task_id == task_id)
    )
    return result.scalar_one_or_none()


async def update_human_review(
    session: AsyncSession,
    task_id: str,
    **kwargs: Any,
) -> HumanReview | None:
    record = await get_human_review_by_task(session, task_id)
    if record is None:
        return None
    for k, v in kwargs.items():
        setattr(record, k, v)
    await session.flush()
    return record


async def list_pending_reviews(session: AsyncSession) -> list[Task]:
    """Return all tasks with status='awaiting_human_review'."""
    result = await session.execute(
        select(Task).where(Task.status == "awaiting_human_review").order_by(Task.created_at)
    )
    return list(result.scalars().all())


async def list_tasks(session: AsyncSession, limit: int = 200) -> list[Task]:
    """Return tasks newest-first — used by the Task List page and dashboards."""
    result = await session.execute(
        select(Task).order_by(Task.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# ExecutionTrace helpers (Phase 4)
# ---------------------------------------------------------------------------

async def create_execution_trace(
    session: AsyncSession,
    task_id: str,
    node_name: str,
    started_at: datetime,
    completed_at: datetime | None,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    tool_calls_count: int,
    status: str,
    error_message: str | None = None,
    llm_prompt: str | None = None,
    llm_response: str | None = None,
    trace_metadata: dict | None = None,
) -> ExecutionTrace:
    trace = ExecutionTrace(
        id=str(uuid.uuid4()),
        task_id=task_id,
        node_name=node_name,
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        tool_calls_count=tool_calls_count,
        status=status,
        error_message=error_message,
        llm_prompt=llm_prompt,
        llm_response=llm_response,
        trace_metadata=trace_metadata,
    )
    session.add(trace)
    await session.flush()
    return trace


async def list_traces_for_task(
    session: AsyncSession, task_id: str
) -> list[ExecutionTrace]:
    """Return all node traces for a task in execution order (oldest first)."""
    result = await session.execute(
        select(ExecutionTrace)
        .where(ExecutionTrace.task_id == task_id)
        .order_by(ExecutionTrace.started_at)
    )
    return list(result.scalars().all())


async def finalize_task_totals(session: AsyncSession, task_id: str) -> None:
    """
    Roll the per-node ExecutionTrace rows up into the Task's aggregate columns.

    Called after a task completes.  total_latency_ms is wall-clock
    (created_at → completed_at) rather than the sum of node latencies, because
    the spec defines it that way and because wall-clock is what a user perceives
    as "how long did my task take" (it also naturally includes queue/idle gaps).
    """
    agg = await session.execute(
        select(
            func.coalesce(func.sum(ExecutionTrace.cost_usd), 0.0),
            func.coalesce(func.sum(ExecutionTrace.input_tokens), 0),
            func.coalesce(func.sum(ExecutionTrace.output_tokens), 0),
        ).where(ExecutionTrace.task_id == task_id)
    )
    total_cost, total_in, total_out = agg.one()

    task = await get_task(session, task_id)
    if task is None:
        return

    wall_ms = 0
    if task.completed_at and task.created_at:
        wall_ms = int((task.completed_at - task.created_at).total_seconds() * 1000)

    task.total_cost_usd = round(float(total_cost), 8)
    task.total_input_tokens = int(total_in)
    task.total_output_tokens = int(total_out)
    task.total_latency_ms = wall_ms
    await session.flush()
