"""
LangGraph state machine definition.

Flow:
  task_intake
       │
       ▼
  supervisor_planning
       │
       ▼
  specialist_execution  (researcher → analyst → writer, sequential)
       │
       ▼
  reviewer_validation
       │
       ├─ approved ──────────────────────────► END
       │
       └─ rejected & retries < 2 ──► writer_retry ──► reviewer_validation
                                              │
                              retries >= 2 ──► END
"""
from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from app.agents.reviewer import run_reviewer
from app.agents.supervisor import run_supervisor
from app.agents.specialists.researcher import run_researcher
from app.agents.specialists.analyst import run_analyst
from app.agents.specialists.writer import run_writer
from app.graph.state import AgentState, SubtaskStatus

log = structlog.get_logger(__name__)

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def task_intake_node(state: AgentState) -> dict:
    """Validate that we have an original task; nothing else to do here."""
    task = state.get("original_task", "").strip()
    if not task:
        return {"errors": ["No task provided"]}
    log.info("task_intake", task_id=state.get("task_id"), task=task[:80])
    return {}


async def supervisor_planning_node(state: AgentState) -> dict:
    return await run_supervisor(state)


async def specialist_execution_node(state: AgentState) -> dict:
    """
    Run all three specialists in sequence: researcher → analyst → writer.
    Each specialist's output feeds the next.
    """
    plan = state.get("execution_plan")
    if plan is None:
        return {"errors": ["specialist_execution: no execution plan"]}

    results: dict = dict(state.get("subtask_results", {}))
    review_feedback = state.get("review_feedback")
    task_id = state.get("task_id")

    # Identify subtasks by agent role
    subtasks_by_role = {s.assigned_agent: s for s in plan.subtasks}

    # ── Researcher ─────────────────────────────────────────────────────────
    researcher_subtask = subtasks_by_role.get("researcher")
    if researcher_subtask and researcher_subtask.id not in results:
        log.info("running_researcher", subtask_id=researcher_subtask.id)
        try:
            researcher_subtask.status = SubtaskStatus.RUNNING
            raw_findings = await run_researcher(
                subtask_description=researcher_subtask.description,
                task_id=task_id,
                subtask_id=researcher_subtask.id,
            )
            results[researcher_subtask.id] = raw_findings
            researcher_subtask.status = SubtaskStatus.COMPLETED
            log.info("researcher_completed", subtask_id=researcher_subtask.id)
        except Exception as exc:
            researcher_subtask.status = SubtaskStatus.FAILED
            log.error("researcher_node_failed", error=str(exc))
            return {"errors": [f"Researcher failed: {exc}"], "subtask_results": results}

    # ── Analyst ────────────────────────────────────────────────────────────
    analyst_subtask = subtasks_by_role.get("analyst")
    raw_findings = results.get(researcher_subtask.id, "") if researcher_subtask else ""

    if analyst_subtask and analyst_subtask.id not in results:
        log.info("running_analyst", subtask_id=analyst_subtask.id)
        try:
            analyst_subtask.status = SubtaskStatus.RUNNING
            analysis = await run_analyst(
                subtask_description=analyst_subtask.description,
                raw_findings=raw_findings,
                task_id=task_id,
                subtask_id=analyst_subtask.id,
            )
            results[analyst_subtask.id] = analysis
            analyst_subtask.status = SubtaskStatus.COMPLETED
            log.info("analyst_completed", subtask_id=analyst_subtask.id)
        except Exception as exc:
            analyst_subtask.status = SubtaskStatus.FAILED
            log.error("analyst_node_failed", error=str(exc))
            return {"errors": [f"Analyst failed: {exc}"], "subtask_results": results}

    # ── Writer ─────────────────────────────────────────────────────────────
    writer_subtask = subtasks_by_role.get("writer")
    analysis = results.get(analyst_subtask.id, "") if analyst_subtask else ""

    if writer_subtask:
        log.info(
            "running_writer",
            subtask_id=writer_subtask.id,
            has_feedback=review_feedback is not None,
        )
        try:
            writer_subtask.status = SubtaskStatus.RUNNING
            report = await run_writer(
                subtask_description=writer_subtask.description,
                structured_analysis=analysis,
                reviewer_feedback=review_feedback,
                task_id=task_id,
                subtask_id=writer_subtask.id,
            )
            results[writer_subtask.id] = report
            writer_subtask.status = SubtaskStatus.COMPLETED
            log.info("writer_completed", subtask_id=writer_subtask.id)
        except Exception as exc:
            writer_subtask.status = SubtaskStatus.FAILED
            log.error("writer_node_failed", error=str(exc))
            return {"errors": [f"Writer failed: {exc}"], "subtask_results": results}

    return {
        "subtask_results": results,
        "execution_plan": plan,   # updated statuses
    }


async def reviewer_validation_node(state: AgentState) -> dict:
    return await run_reviewer(state)


async def writer_retry_node(state: AgentState) -> dict:
    """
    Re-run the writer only (researcher + analyst outputs are cached).
    Clears the old writer output so specialist_execution will re-run just writer.
    """
    plan = state.get("execution_plan")
    if plan is None:
        return {"errors": ["writer_retry: no execution plan"]}

    writer_subtask = next(
        (s for s in plan.subtasks if s.assigned_agent == "writer"), None
    )
    if writer_subtask is None:
        return {"errors": ["writer_retry: no writer subtask"]}

    # Remove old writer result so specialist_execution re-runs the writer
    results = dict(state.get("subtask_results", {}))
    results.pop(writer_subtask.id, None)
    writer_subtask.status = SubtaskStatus.PENDING

    log.info("writer_retry_prepared", retry_count=state.get("retry_count"))
    return {"subtask_results": results, "execution_plan": plan}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_review(state: AgentState) -> str:
    """Decide what to do after reviewer runs."""
    final_output = state.get("final_output")
    retry_count = state.get("retry_count", 0)

    if final_output is not None:
        return "end"

    if retry_count < MAX_RETRIES:
        return "retry_writer"

    return "end"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("task_intake",           task_intake_node)
    graph.add_node("supervisor_planning",   supervisor_planning_node)
    graph.add_node("specialist_execution",  specialist_execution_node)
    graph.add_node("reviewer_validation",   reviewer_validation_node)
    graph.add_node("writer_retry",          writer_retry_node)

    # Entry point
    graph.set_entry_point("task_intake")

    # Linear edges
    graph.add_edge("task_intake",          "supervisor_planning")
    graph.add_edge("supervisor_planning",  "specialist_execution")
    graph.add_edge("specialist_execution", "reviewer_validation")

    # Conditional after review
    graph.add_conditional_edges(
        "reviewer_validation",
        route_after_review,
        {
            "end":          END,
            "retry_writer": "writer_retry",
        },
    )

    # After retry prep, re-run specialists (will skip researcher+analyst due to cache)
    graph.add_edge("writer_retry", "specialist_execution")

    return graph


# Compiled graph (used by the task runner)
compiled_graph = build_graph().compile()
