"""
Reviewer Agent

Scores the writer's report 1-5 and either approves it or sends it back
with actionable feedback.  Max 2 retry loops total.
"""
from __future__ import annotations

import structlog

from app.graph.state import AgentState, ReviewResult
from app.llm.client import call_llm

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a Reviewer Agent in a multi-agent research pipeline.
Your job is to critically evaluate the final report against the original task.

Scoring rubric:
5 — Excellent: comprehensive, well-structured, accurate, actionable
4 — Good: addresses the task, minor improvements possible
3 — Adequate: covers the basics but missing depth or clarity
2 — Poor: significant gaps, inaccurate, or poorly organised
1 — Unacceptable: does not address the task

Approval rule: approve (approved=true) if score >= 4.

If not approved, provide SPECIFIC, ACTIONABLE feedback:
- List exact sections that need improvement
- Describe what information is missing
- Point out any inaccuracies
- Suggest concrete changes the writer should make
"""


async def run_reviewer(state: AgentState) -> dict:
    """
    LangGraph node: reviewer_validation

    Reads the writer's output from subtask_results and produces a ReviewResult.
    """
    plan = state.get("execution_plan")
    if plan is None:
        return {"errors": ["Reviewer: no execution plan found"]}

    # Find the writer subtask output
    writer_subtask = next(
        (s for s in plan.subtasks if s.assigned_agent == "writer"),
        None,
    )
    if writer_subtask is None:
        return {"errors": ["Reviewer: no writer subtask found in plan"]}

    report = state.get("subtask_results", {}).get(writer_subtask.id, "")
    if not report:
        return {"errors": ["Reviewer: writer produced no output"]}

    retry_count = state.get("retry_count", 0)
    original_task = state.get("original_task", "")

    log.info("reviewer_start", retry_count=retry_count, report_chars=len(report))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original task: {original_task}\n\n"
                f"Report to review:\n\n{report}"
            ),
        },
    ]

    try:
        review: ReviewResult = await call_llm(
            role="reviewer",
            messages=messages,
            response_model=ReviewResult,
        )
        log.info(
            "reviewer_done",
            score=review.score,
            approved=review.approved,
            feedback_len=len(review.feedback),
        )

        updates: dict = {}

        if review.approved or retry_count >= 2:
            # Accept the report
            updates["final_output"] = report
            updates["review_feedback"] = review.feedback or ""
            log.info(
                "reviewer_accepted",
                forced=not review.approved,
                score=review.score,
            )
        else:
            # Send back to writer
            updates["review_feedback"] = review.feedback
            updates["retry_count"] = retry_count + 1
            log.info(
                "reviewer_rejected",
                score=review.score,
                retry_count=retry_count + 1,
            )

        return updates

    except Exception as exc:
        log.error("reviewer_failed", error=str(exc))
        # Fail-safe: accept whatever we have
        return {
            "final_output": report,
            "errors": [f"Reviewer failed: {exc}"],
        }
