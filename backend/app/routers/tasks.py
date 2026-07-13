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

    PHASE 2A ADDITIONS
    ------------------
    Integrates two memory layers:
    1. WorkingMemoryManager (Redis) — per-task scratchpad, cleared after run.
    2. SemanticMemoryManager (ChromaDB) — stores completed-task summaries.

    PHASE 3 ADDITIONS
    -----------------
    Uses the checkpointer-compiled graph (from app.checkpointer.get_graph())
    so state is persisted between nodes.  The run config includes
    thread_id=task_id so LangGraph can identify this thread for pause/resume.

    When the graph hits the human_review interrupt, GraphInterrupt is raised.
    We catch it, log it, and return early.  The task status is already set to
    "awaiting_human_review" by check_escalation_node (inside the graph) so
    no further DB updates are needed here in the interrupted case.
    """
    from app.checkpointer import get_graph
    from app.db.engine import get_db_session as _get_session
    from langgraph.errors import GraphInterrupt

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
        "memory_context":        None,  # populated by memory_retrieval_node (Phase 2B)
        # Phase 3 HITL fields — seeded with defaults so nodes never see KeyError.
        "review_score":          None,
        "requires_human_review": False,
        "escalation_reason":     "",
        "human_decision":        None,
        "human_feedback":        None,
        "human_reviewed_at":     None,
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
        # Phase 3: thread_id = task_id so LangGraph checkpoints under this key.
        # When the graph is paused and then resumed via POST /reviews/{id}/decision,
        # LangGraph loads the checkpoint using the same thread_id to find where
        # execution left off.
        "configurable": {"thread_id": task_id},
        "run_name": f"agentis-task-{task_id}",
        "metadata": {
            "task_id": task_id,
            "project": langsmith_project,
        },
        "tags": ["agentis", task_id],
    }
    if tracer is not None:
        run_config["callbacks"] = [tracer]

    # ------------------------------------------------------------------
    # Phase 2A: initialise working memory (Redis scratchpad)
    # ------------------------------------------------------------------
    working_mem = None
    try:
        from app.memory.working_memory import WorkingMemoryManager
        working_mem = WorkingMemoryManager()
        await working_mem.connect()
        log.info("working_memory_connected", task_id=task_id)
    except Exception as wm_exc:
        log.warning("working_memory_init_failed", task_id=task_id, error=str(wm_exc))
        working_mem = None  # degrade gracefully — continue without working memory

    # ------------------------------------------------------------------
    # Run the graph.  Three possible outcomes:
    #
    # 1. Normal completion: final_state returned, persist to DB as before.
    # 2. GraphInterrupt: graph paused at human_review interrupt.  Task
    #    status is already "awaiting_human_review" (set inside the graph).
    #    Clean up working memory and return — no further DB update needed.
    # 3. Unexpected exception: log and persist as "failed".
    # ------------------------------------------------------------------
    graph = get_graph()
    interrupted = False
    final_state: dict = {}

    try:
        final_state = await graph.ainvoke(initial_state, config=run_config)
        log.info("task_runner_done", task_id=task_id)

        # LangGraph 1.x: interrupt() causes ainvoke() to RETURN (no exception),
        # with the state as it was when the interrupt was hit.  Detect this by
        # checking requires_human_review in the returned state.
        # (In older LangGraph versions, GraphInterrupt was raised instead.)
        if final_state.get("requires_human_review"):
            interrupted = True
            log.info("task_awaiting_human_review", task_id=task_id)

    except GraphInterrupt:
        # Older LangGraph behaviour: interrupt() raises GraphInterrupt.
        interrupted = True
        log.info("task_awaiting_human_review", task_id=task_id)
    except Exception as exc:
        log.error("task_runner_crashed", task_id=task_id, error=str(exc))
        final_state = {**initial_state, "errors": [str(exc)]}

    # ------------------------------------------------------------------
    # Phase 2A: clean up working memory regardless of outcome.
    # ------------------------------------------------------------------
    if working_mem is not None:
        try:
            await working_mem.clear(task_id)
            await working_mem.close()
        except Exception as cleanup_exc:
            log.warning(
                "working_memory_cleanup_failed",
                task_id=task_id,
                error=str(cleanup_exc),
            )

    # If the graph paused for human review, nothing else to persist here.
    # check_escalation_node already set the task status to "awaiting_human_review".
    if interrupted:
        return

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

    # Persist result — only reached if graph completed normally (no interrupt).
    async with _get_session() as session:
        plan = final_state.get("execution_plan")
        plan_dict = plan.model_dump() if plan is not None else None

        subtask_outputs = final_state.get("subtask_results", {})
        final_output = final_state.get("final_output")
        errors = final_state.get("errors", [])

        # For human-approved/edited tasks the status may already be set by
        # human_review_node.  For normal (non-HITL) completions, determine it here.
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

        # Phase 4: roll node-level ExecutionTrace rows into the Task's
        # aggregate cost/latency/token columns.  Done in the same session/commit
        # as the status update so a completed task always has totals populated.
        try:
            from app.db.queries import finalize_task_totals
            await finalize_task_totals(session, task_id)
        except Exception as agg_exc:
            log.warning("finalize_task_totals_failed", task_id=task_id, error=str(agg_exc))

        log.info("task_persisted", task_id=task_id, status=status)

    # ------------------------------------------------------------------
    # Phase 2A: store task completion in long-term semantic memory
    # We do this AFTER the DB write so that a memory failure cannot
    # interfere with the task's recorded status in PostgreSQL.
    # ------------------------------------------------------------------
    if status == "completed":
        try:
            from app.memory.semantic_memory import SemanticMemoryManager
            memory_mgr = SemanticMemoryManager()
            await memory_mgr.store_task_completion(
                task_id=task_id,
                user_id="default_user",  # multi-user support in a future phase
                task_summary={
                    "original_task": original_task,
                    "plan_reasoning": plan.reasoning if plan else "",
                    "final_output_preview": (final_output or "")[:1500],
                },
                execution_data={
                    "subtask_count": len(plan.subtasks) if plan else 0,
                    "retry_count": final_state.get("retry_count", 0),
                    "error_count": len(errors),
                    "agents_used": [s.assigned_agent for s in plan.subtasks] if plan else [],
                },
            )
            log.info("semantic_memory_stored", task_id=task_id)
        except Exception as mem_exc:
            # Memory failures must NOT crash task execution or affect task status.
            log.warning("memory_store_failed", task_id=task_id, error=str(mem_exc))


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


class TaskListItem(BaseModel):
    task_id: str
    original_task: str
    status: str
    total_cost_usd: float
    total_latency_ms: int
    created_at: str


@router.get("", response_model=list[TaskListItem])
async def list_all_tasks() -> list[TaskListItem]:
    """
    Return all tasks newest-first for the frontend Task List page.

    Declared BEFORE GET /{task_id} so the empty-path route is unambiguous
    (FastAPI matches routes in declaration order within a router).
    """
    from app.db.queries import list_tasks

    async with get_db_session() as session:
        tasks = await list_tasks(session)

    return [
        TaskListItem(
            task_id=t.id,
            original_task=t.original_task,
            status=t.status,
            total_cost_usd=t.total_cost_usd or 0.0,
            total_latency_ms=t.total_latency_ms or 0,
            created_at=t.created_at.isoformat(),
        )
        for t in tasks
    ]


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
