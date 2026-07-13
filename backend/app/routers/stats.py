"""
routers/stats.py — Phase 4 aggregated performance statistics.

ENDPOINTS (mounted under prefix="/stats" in main.py)
-----------------------------------------------------
GET /stats/performance   — system-wide averages, escalation rate, hot nodes
GET /stats/cost-by-task  — every task with its cost, most-expensive first

WHY AGGREGATE IN SQL (not in Python)
------------------------------------
The dashboard's numbers are pure aggregates (sum / avg / count / group-by) over
two tables.  Pushing them into Postgres means we transfer only the summarised
rows, not every ExecutionTrace, and the DB does the arithmetic far faster than
looping in Python — this keeps the dashboard snappy as trace volume grows.
"""
from __future__ import annotations

from typing import Optional

import structlog
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import func, select

from app.db.engine import get_db_session
from app.db.models import ExecutionTrace, Task

log = structlog.get_logger(__name__)

router = APIRouter(tags=["stats"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PerformanceStats(BaseModel):
    total_tasks: int
    avg_cost_usd: float
    avg_latency_ms: float
    total_cost_usd: float
    escalation_rate: float
    avg_tokens_per_task: float
    slowest_node: Optional[str]
    most_expensive_node: Optional[str]
    tasks_by_status: dict[str, int]


class CostByTaskItem(BaseModel):
    task_id: str
    original_task: str
    total_cost_usd: float
    total_latency_ms: int
    status: str
    created_at: str


# ---------------------------------------------------------------------------
# GET /stats/performance
# ---------------------------------------------------------------------------

@router.get("/performance", response_model=PerformanceStats)
async def performance_stats() -> PerformanceStats:
    async with get_db_session() as session:
        # ── Task-level aggregates ────────────────────────────────────────────
        task_agg = (
            await session.execute(
                select(
                    func.count(Task.id),
                    func.coalesce(func.avg(Task.total_cost_usd), 0.0),
                    func.coalesce(func.avg(Task.total_latency_ms), 0.0),
                    func.coalesce(func.sum(Task.total_cost_usd), 0.0),
                    func.coalesce(
                        func.avg(Task.total_input_tokens + Task.total_output_tokens), 0.0
                    ),
                )
            )
        ).one()
        total_tasks, avg_cost, avg_latency, total_cost, avg_tokens = task_agg

        # ── Status breakdown ─────────────────────────────────────────────────
        status_rows = (
            await session.execute(
                select(Task.status, func.count(Task.id)).group_by(Task.status)
            )
        ).all()
        tasks_by_status = {status: count for status, count in status_rows}

        # ── Escalation rate ──────────────────────────────────────────────────
        # A task counts as escalated if it currently awaits review OR any of its
        # nodes recorded status='escalated'.
        escalated_task_ids = (
            await session.execute(
                select(func.count(func.distinct(ExecutionTrace.task_id))).where(
                    ExecutionTrace.status == "escalated"
                )
            )
        ).scalar_one()
        awaiting = tasks_by_status.get("awaiting_human_review", 0)
        escalated_total = max(escalated_task_ids, awaiting)
        escalation_rate = (escalated_total / total_tasks) if total_tasks else 0.0

        # ── Per-node hot spots ───────────────────────────────────────────────
        slowest_node = (
            await session.execute(
                select(ExecutionTrace.node_name)
                .group_by(ExecutionTrace.node_name)
                .order_by(func.avg(ExecutionTrace.latency_ms).desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        most_expensive_node = (
            await session.execute(
                select(ExecutionTrace.node_name)
                .group_by(ExecutionTrace.node_name)
                .order_by(func.sum(ExecutionTrace.cost_usd).desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    return PerformanceStats(
        total_tasks=int(total_tasks),
        avg_cost_usd=round(float(avg_cost), 8),
        avg_latency_ms=round(float(avg_latency), 2),
        total_cost_usd=round(float(total_cost), 8),
        escalation_rate=round(escalation_rate, 4),
        avg_tokens_per_task=round(float(avg_tokens), 2),
        slowest_node=slowest_node,
        most_expensive_node=most_expensive_node,
        tasks_by_status=tasks_by_status,
    )


# ---------------------------------------------------------------------------
# GET /stats/cost-by-task
# ---------------------------------------------------------------------------

@router.get("/cost-by-task", response_model=list[CostByTaskItem])
async def cost_by_task() -> list[CostByTaskItem]:
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(Task).order_by(Task.total_cost_usd.desc())
            )
        ).scalars().all()

    return [
        CostByTaskItem(
            task_id=t.id,
            original_task=t.original_task,
            total_cost_usd=t.total_cost_usd or 0.0,
            total_latency_ms=t.total_latency_ms or 0,
            status=t.status,
            created_at=t.created_at.isoformat(),
        )
        for t in rows
    ]


# ---------------------------------------------------------------------------
# Per-node aggregates — powers the dashboard's cost-by-node / latency-by-node
# bar charts.  Not in the original spec's endpoint list but required by the
# dashboard UI, so exposed here alongside the other stats.
# ---------------------------------------------------------------------------

class NodeAggItem(BaseModel):
    node_name: str
    total_cost_usd: float
    avg_latency_ms: float
    executions: int


@router.get("/by-node", response_model=list[NodeAggItem])
async def stats_by_node() -> list[NodeAggItem]:
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(
                    ExecutionTrace.node_name,
                    func.coalesce(func.sum(ExecutionTrace.cost_usd), 0.0),
                    func.coalesce(func.avg(ExecutionTrace.latency_ms), 0.0),
                    func.count(ExecutionTrace.id),
                )
                .group_by(ExecutionTrace.node_name)
                .order_by(func.sum(ExecutionTrace.cost_usd).desc())
            )
        ).all()

    return [
        NodeAggItem(
            node_name=name,
            total_cost_usd=round(float(cost), 8),
            avg_latency_ms=round(float(latency), 2),
            executions=int(count),
        )
        for name, cost, latency, count in rows
    ]
