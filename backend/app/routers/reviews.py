"""
routers/reviews.py — FastAPI router for Phase 3 Human-in-the-Loop reviews.

ENDPOINTS
---------
GET  /reviews/pending            — list tasks awaiting human review
POST /reviews/{task_id}/decision — submit a human decision, resume graph
GET  /reviews/{task_id}          — fetch a specific HumanReview record

RESUME MECHANISM
----------------
When the graph is paused at human_review_node (via interrupt()), the graph
state is checkpointed in PostgreSQL under thread_id = task_id.  To resume:

  graph.ainvoke(
      Command(resume={"decision": ..., "feedback": ...}),
      config={"configurable": {"thread_id": task_id}},
  )

LangGraph re-runs human_review_node; this time interrupt() returns the
resume payload instead of raising GraphInterrupt.  The node processes the
decision, updates the DB, and the graph continues to END.

WHY THE ROUTER DOES NOT DIRECTLY UPDATE final_output
-----------------------------------------------------
The graph node (human_review_node) handles the DB update AFTER the decision
is processed, because it is the node that knows the full state (draft output,
decision, feedback).  The router's job is only to trigger the resume.
The router does update the HumanReview record before resuming so that the
audit trail is populated even if the graph resume fails.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.engine import get_db_session
from app.db.queries import (
    get_human_review_by_task,
    get_task,
    list_pending_reviews,
    update_human_review,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/reviews", tags=["reviews"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    decision: str   # "approved" | "edited" | "rejected"
    feedback: Optional[str] = None


class PendingReviewItem(BaseModel):
    task_id: str
    original_task: str
    escalation_reason: Optional[str]
    draft_output: Optional[str]
    created_at: str


class ReviewResponse(BaseModel):
    task_id: str
    escalation_reason: str
    output_shown_to_human: Optional[str]
    human_decision: Optional[str]
    human_feedback: Optional[str]
    reviewed_by: str
    created_at: str
    reviewed_at: Optional[str]


# ---------------------------------------------------------------------------
# GET /reviews/pending
# ---------------------------------------------------------------------------

@router.get("/pending", response_model=list[PendingReviewItem])
async def list_pending() -> list[PendingReviewItem]:
    """
    Return all tasks currently paused and waiting for human review.

    Each item includes:
    - task_id, original_task     — to identify what needs reviewing
    - escalation_reason          — why the system flagged this task
    - draft_output               — the AI output that needs human review
    - created_at                 — when the task was submitted
    """
    async with get_db_session() as session:
        tasks = await list_pending_reviews(session)
        results = []
        for task in tasks:
            # Try to load the associated HumanReview for draft output.
            review = await get_human_review_by_task(session, task.id)
            results.append(PendingReviewItem(
                task_id=task.id,
                original_task=task.original_task,
                escalation_reason=task.escalation_reason,
                draft_output=review.output_shown_to_human if review else None,
                created_at=task.created_at.isoformat(),
            ))
    return results


# ---------------------------------------------------------------------------
# POST /reviews/{task_id}/decision
# ---------------------------------------------------------------------------

@router.post("/{task_id}/decision")
async def submit_decision(task_id: str, body: DecisionRequest) -> dict:
    """
    Submit a human review decision and resume the paused LangGraph run.

    decision must be one of:
      "approved" — keep AI output as-is
      "edited"   — replace AI output with feedback text
      "rejected" — discard AI output, mark task rejected

    The graph is resumed synchronously in this endpoint (the graph run
    is fast after the interrupt — it just processes the decision and ends).
    If resume fails for any reason, an HTTP 500 is returned but the
    HumanReview record has already been updated, providing an audit trail.
    """
    if body.decision not in ("approved", "edited", "rejected"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision '{body.decision}'. Must be: approved, edited, rejected",
        )

    # Verify task exists and is in the correct state.
    async with get_db_session() as session:
        task = await get_task(session, task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status != "awaiting_human_review":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Task {task_id} is not awaiting review "
                f"(current status: {task.status})"
            ),
        )

    # Persist the decision to the audit record before resuming the graph,
    # so there is always a record even if the graph resume fails.
    now = datetime.now(timezone.utc)
    async with get_db_session() as session:
        updated = await update_human_review(
            session,
            task_id,
            human_decision=body.decision,
            human_feedback=body.feedback or None,
            reviewed_at=now,
        )
        if updated is None:
            log.warning("human_review_record_missing", task_id=task_id)

    # Resume the paused graph.
    try:
        from langgraph.types import Command
        from app.checkpointer import get_graph

        graph = get_graph()
        config = {"configurable": {"thread_id": task_id}}
        resume_payload = {
            "decision": body.decision,
            "feedback": body.feedback or "",
        }

        log.info(
            "resuming_graph",
            task_id=task_id,
            decision=body.decision,
            has_feedback=bool(body.feedback),
        )

        await graph.ainvoke(Command(resume=resume_payload), config=config)
        log.info("graph_resumed_and_completed", task_id=task_id)

        # Phase 4: the human_review node just ran and set completed_at, so
        # recompute the task's aggregate cost/latency/token totals.
        try:
            from app.db.queries import finalize_task_totals
            async with get_db_session() as agg_session:
                await finalize_task_totals(agg_session, task_id)
        except Exception as agg_exc:
            log.warning("finalize_task_totals_failed", task_id=task_id, error=str(agg_exc))

    except Exception as exc:
        log.error("graph_resume_failed", task_id=task_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Graph resume failed: {exc}",
        )

    # Return the final task status.
    async with get_db_session() as session:
        task = await get_task(session, task_id)

    final_status = task.status if task else "unknown"
    log.info("decision_processed", task_id=task_id, status=final_status)

    return {
        "task_id": task_id,
        "decision": body.decision,
        "status": final_status,
        "message": f"Decision '{body.decision}' processed. Task is now '{final_status}'.",
    }


# ---------------------------------------------------------------------------
# GET /reviews/{task_id}
# ---------------------------------------------------------------------------

@router.get("/{task_id}", response_model=ReviewResponse)
async def get_review(task_id: str) -> ReviewResponse:
    """Return the full HumanReview record for a task."""
    async with get_db_session() as session:
        review = await get_human_review_by_task(session, task_id)

    if review is None:
        raise HTTPException(
            status_code=404,
            detail=f"No human review record found for task {task_id}",
        )

    return ReviewResponse(
        task_id=review.task_id,
        escalation_reason=review.escalation_reason,
        output_shown_to_human=review.output_shown_to_human,
        human_decision=review.human_decision,
        human_feedback=review.human_feedback,
        reviewed_by=review.reviewed_by,
        created_at=review.created_at.isoformat(),
        reviewed_at=review.reviewed_at.isoformat() if review.reviewed_at else None,
    )
