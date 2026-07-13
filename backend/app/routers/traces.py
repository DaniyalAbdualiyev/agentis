"""
routers/traces.py — Phase 4 execution-trace + replay API.

ENDPOINTS
---------
GET  /tasks/{task_id}/trace   — full per-node execution breakdown for a task
POST /tasks/{task_id}/replay  — re-run a task from scratch, return new task_id

WHY THESE LIVE IN THEIR OWN ROUTER (not tasks.py)
--------------------------------------------------
tasks.py owns task submission/status.  Tracing and replay are a distinct
observability concern (Phase 4).  Keeping them separate mirrors the existing
split between tasks.py, reviews.py and memory.py and keeps each router small.

WHY NO prefix ON THIS ROUTER
----------------------------
The routes are nested under /tasks/{id}/... to read naturally as
"the trace of this task".  The existing tasks_router uses prefix="/tasks" with
a single-segment `/{task_id}` route, which does NOT collide with the
two-segment `/{task_id}/trace` and `/{task_id}/replay` routes declared here.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.db.engine import get_db_session
from app.db.queries import create_task, get_task, list_traces_for_task

log = structlog.get_logger(__name__)

router = APIRouter(tags=["traces"])


# ---------------------------------------------------------------------------
# Response / request schemas
# ---------------------------------------------------------------------------

class TraceNode(BaseModel):
    node_name: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tool_calls_count: int
    status: str
    error_message: Optional[str] = None
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class TraceResponse(BaseModel):
    task_id: str
    original_task: str
    status: str
    total_cost_usd: float
    total_latency_ms: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    nodes: list[TraceNode]


class ReplayRequest(BaseModel):
    # Both optional — a bare {} (or no body) triggers a plain full replay.
    modify_node: Optional[str] = None
    modified_prompt_suffix: Optional[str] = None


class ReplayResponse(BaseModel):
    original_task_id: str
    new_task_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/trace
# ---------------------------------------------------------------------------

@router.get("/tasks/{task_id}/trace", response_model=TraceResponse)
async def get_task_trace(task_id: str) -> TraceResponse:
    """Return the full node-by-node execution trace for a task."""
    async with get_db_session() as session:
        task = await get_task(session, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        traces = await list_traces_for_task(session, task_id)

        nodes = [
            TraceNode(
                node_name=t.node_name,
                latency_ms=t.latency_ms,
                input_tokens=t.input_tokens,
                output_tokens=t.output_tokens,
                cost_usd=t.cost_usd,
                tool_calls_count=t.tool_calls_count,
                status=t.status,
                error_message=t.error_message,
                llm_prompt=t.llm_prompt,
                llm_response=t.llm_response,
                started_at=t.started_at.isoformat() if t.started_at else None,
                completed_at=t.completed_at.isoformat() if t.completed_at else None,
                metadata=t.trace_metadata,
            )
            for t in traces
        ]

        return TraceResponse(
            task_id=task.id,
            original_task=task.original_task,
            status=task.status,
            total_cost_usd=task.total_cost_usd or 0.0,
            total_latency_ms=task.total_latency_ms or 0,
            total_tokens=(task.total_input_tokens or 0) + (task.total_output_tokens or 0),
            total_input_tokens=task.total_input_tokens or 0,
            total_output_tokens=task.total_output_tokens or 0,
            nodes=nodes,
        )


# ---------------------------------------------------------------------------
# POST /tasks/{task_id}/replay
# ---------------------------------------------------------------------------

@router.post("/tasks/{task_id}/replay", response_model=ReplayResponse, status_code=202)
async def replay_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    body: Optional[ReplayRequest] = None,
) -> ReplayResponse:
    """
    Re-run a task from scratch as a brand-new task and return its id immediately.

    FULL REPLAY (this phase)
    ------------------------
    We load the original task's text and submit it as a new task through the
    exact same background runner used by POST /tasks.  The new run produces its
    own ExecutionTrace rows, so `GET /tasks/{new_id}/trace` can be compared
    side-by-side with the original.

    OPTIONAL MODIFICATION (lightweight divergence)
    ----------------------------------------------
    If `modified_prompt_suffix` is provided, it is appended to the task text for
    the replay so the effect of a tweak can be observed.  Full node-level
    divergence replay (re-running only from a modified node using the original's
    checkpoint) is intentionally out of scope for this phase.
    """
    body = body or ReplayRequest()

    async with get_db_session() as session:
        original = await get_task(session, task_id)
        if original is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        original_text = original.original_task

    new_text = original_text
    if body.modified_prompt_suffix:
        new_text = f"{original_text}\n\n[Replay modification] {body.modified_prompt_suffix}"

    new_task_id = str(uuid.uuid4())
    async with get_db_session() as session:
        await create_task(session, original_task=new_text, task_id=new_task_id)

    # Import here to avoid a circular import (tasks.py imports from many modules).
    from app.routers.tasks import run_task_background
    background_tasks.add_task(run_task_background, new_task_id, new_text)

    log.info(
        "task_replay_started",
        original_task_id=task_id,
        new_task_id=new_task_id,
        modified=bool(body.modified_prompt_suffix),
        modify_node=body.modify_node,
    )

    return ReplayResponse(
        original_task_id=task_id,
        new_task_id=new_task_id,
        status="pending",
        message=(
            f"Replay started. Poll GET /tasks/{new_task_id} for status and "
            f"GET /tasks/{new_task_id}/trace for the new trace."
        ),
    )
