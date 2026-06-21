"""
FastAPI routers for task management.

POST /tasks   — submit a task, get task_id back immediately
GET  /tasks/{task_id} — poll status and get final output + LangSmith trace URL
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.db.engine import get_db_session
from app.db.queries import create_task, get_task, update_task

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    task: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    original_task: str
    execution_plan: Any | None
    subtask_outputs: dict[str, str] | None
    final_output: str | None
    langsmith_trace_url: str | None
    created_at: str
    completed_at: str | None
    errors: list[str] | None


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

async def run_task_background(task_id: str, original_task: str) -> None:
    """
    Execute the full agent graph and persist results to PostgreSQL.
    LangSmith tracing is automatic via LANGCHAIN_TRACING_V2=true env var.
    """
    from app.graph.graph import compiled_graph
    from app.db.engine import get_db_session as _get_session

    log.info("task_runner_start", task_id=task_id)

    initial_state = {
        "task_id":               task_id,
        "original_task":         original_task,
        "execution_plan":        None,
        "subtask_results":       {},
        "current_subtask_index": 0,
        "review_feedback":       None,
        "retry_count":           0,
        "final_output":          None,
        "errors":                [],
    }

    # LangSmith config: tag each run with the task_id for easy trace lookup
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "agentis")
    langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    tracer = None
    if langsmith_enabled:
        try:
            from langchain_core.tracers.langchain import LangChainTracer
            tracer = LangChainTracer(project_name=langsmith_project)
        except Exception:
            tracer = None

    run_config: dict = {
        "run_name": f"agentis-task-{task_id}",
        "metadata": {
            "task_id": task_id,
            "project": langsmith_project,
        },
        "tags": ["agentis", task_id],
    }
    if tracer is not None:
        run_config["callbacks"] = [tracer]

    try:
        final_state = await compiled_graph.ainvoke(initial_state, config=run_config)
        log.info("task_runner_done", task_id=task_id)
    except Exception as exc:
        log.error("task_runner_crashed", task_id=task_id, error=str(exc))
        final_state = {**initial_state, "errors": [str(exc)]}

    # Capture LangSmith trace URL
    trace_url: str | None = None
    if tracer is not None:
        try:
            trace_url = tracer.get_run_url()
            log.info("langsmith_trace_url", url=trace_url)
        except Exception as url_exc:
            log.warning("langsmith_trace_url_failed", error=str(url_exc))
            trace_url = None

    # Fallback: construct a search URL if trace_url not available
    run_name = f"agentis-task-{task_id}"
    if trace_url is None:
        trace_url = (
            f"https://smith.langchain.com/o/~/projects/p/{langsmith_project}"
            f"?peek={run_name}"
        )

    # Persist result
    async with _get_session() as session:
        plan = final_state.get("execution_plan")
        plan_dict = plan.model_dump() if plan is not None else None

        subtask_outputs = final_state.get("subtask_results", {})
        final_output = final_state.get("final_output")
        errors = final_state.get("errors", [])

        status = "completed" if final_output else "failed"

        trace_id = trace_url or run_name

        await update_task(
            session,
            task_id,
            status=status,
            execution_plan=plan_dict,
            final_output=final_output,
            langsmith_trace_id=trace_id,
            completed_at=datetime.now(timezone.utc),
        )

        # Log subtasks
        if plan is not None:
            from app.db.queries import log_subtask
            for subtask in plan.subtasks:
                output = subtask_outputs.get(subtask.id)
                await log_subtask(
                    session=session,
                    task_id=task_id,
                    subtask_id=subtask.id,
                    subtask_description=subtask.description,
                    assigned_agent=subtask.assigned_agent,
                    status=subtask.status,
                    output=output,
                    completed_at=datetime.now(timezone.utc) if output else None,
                )

        log.info("task_persisted", task_id=task_id, status=status)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=TaskResponse, status_code=202)
async def submit_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
) -> TaskResponse:
    task_id = str(uuid.uuid4())

    async with get_db_session() as session:
        await create_task(session, original_task=request.task, task_id=task_id)

    background_tasks.add_task(run_task_background, task_id, request.task)

    log.info("task_submitted", task_id=task_id, task=request.task[:80])
    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Task submitted. Poll GET /tasks/{task_id} for status.",
    )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    async with get_db_session() as session:
        task = await get_task(session, task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # The langsmith_trace_id column stores the full URL when tracing is enabled,
    # or a run_name fallback otherwise.
    langsmith_url: str | None = task.langsmith_trace_id if task.langsmith_trace_id else None

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        original_task=task.original_task,
        execution_plan=task.execution_plan,
        subtask_outputs=None,   # fetched from subtask_logs if needed
        final_output=task.final_output,
        langsmith_trace_url=langsmith_url,
        created_at=task.created_at.isoformat(),
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
        errors=None,
    )
