"""
agents/reviewer.py — The Reviewer Agent: quality gating and feedback.

SINGLE RESPONSIBILITY
---------------------
The Reviewer has exactly one job: read the Writer's report, evaluate it
against the original task, and either accept it or send it back with
specific feedback.  It does not generate new content, does not search the
web, and does not change the execution plan.

WHY A SEPARATE REVIEWER AGENT EXISTS
--------------------------------------
A common shortcut is to ask the Writer to self-evaluate.  We don't do this
for two reasons:

1. CONFLICT OF INTEREST — a model asked to write and then immediately judge
   its own writing tends to be overly generous.  A separate Reviewer prompt,
   given only the original task and the finished report (no knowledge of the
   writing process), produces more honest and useful evaluations.

2. SEPARATION OF CONCERNS — the Writer's prompt is optimised for producing
   polished prose.  The Reviewer's prompt is optimised for critical analysis
   and structured feedback.  Mixing both responsibilities into one prompt
   leads to a model that does neither well.

WHY THE REVIEWER USES STRUCTURED OUTPUT
-----------------------------------------
Same reason as the Supervisor: we need a machine-readable decision
(`approved: bool`, `score: int`) that the graph router can act on without
string-parsing.  `ReviewResult` is the contract between the Reviewer and
the graph's conditional edge.  If the LLM returned free text saying
"I think this is pretty good, maybe 4 out of 5", the router would have no
reliable way to extract the approval decision.

WHY THE APPROVAL THRESHOLD IS score >= 4
-----------------------------------------
- Score 5 (excellent) is rare — reserving approval only for perfect reports
  would cause almost everything to be retried, wasting budget.
- Score 4 (good) means the report addresses the task with only minor issues.
  That is an acceptable quality bar for a research assistant.
- Score 3 and below means substantive gaps remain, which warrants a retry.

WHY THE FAIL-SAFE ACCEPTS THE REPORT ON EXCEPTION
---------------------------------------------------
If the Reviewer's LLM call itself fails (network error, timeout, schema
validation error), we cannot leave `final_output` unset forever.  The
fail-safe sets `final_output = report` and records the error, so:
- The pipeline completes and the caller gets something.
- The error is logged and visible in LangSmith.
- The DB record shows `status = completed` with an error annotation,
  which is better than a hung task with `status = running`.
"""
from __future__ import annotations

import structlog

from app.graph.state import AgentState, ReviewResult
from app.llm.client import call_llm

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
#
# WHY THE RUBRIC IS EXPLICIT IN THE PROMPT
# -----------------------------------------
# Without a rubric, LLMs tend to score inconsistently — the same report
# might score 3 in one run and 5 in another depending on sampling temperature
# or subtle phrasing differences.  An explicit rubric anchors the scale:
# the model can compare the report against concrete criteria (comprehensive,
# well-structured, accurate, actionable) rather than making a holistic
# judgement that is difficult to reproduce.
#
# WHY "SPECIFIC, ACTIONABLE" FEEDBACK IS DEMANDED
# -------------------------------------------------
# Vague feedback like "improve the writing quality" is useless to the Writer
# on a retry.  By instructing the Reviewer to list exact sections, name
# missing information, and suggest concrete changes, we ensure the Writer
# receives feedback it can actually act on in a single pass.
# ---------------------------------------------------------------------------

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
    LangGraph node: reviewer_validation.

    Locates the Writer's output in `subtask_results`, evaluates it against
    the original task, and returns a state update that either:
      - sets `final_output` (accepted) → graph router ends the run, OR
      - sets `review_feedback` + increments `retry_count` (rejected) →
        graph router sends the run back to the writer.

    HOW THE REVIEWER FINDS THE WRITER'S OUTPUT
    -------------------------------------------
    Rather than assuming the writer's output is at a fixed index in
    `subtask_results`, we look up the writer subtask by its `assigned_agent`
    field.  This is robust against plan variations: if the Supervisor assigns
    a different subtask_id to the writer role, the lookup still works.

    WHY retry_count IS READ HERE (not in the graph router)
    -------------------------------------------------------
    The Reviewer checks `retry_count >= 2` before deciding whether to
    accept or reject.  This means the Reviewer's response dict never sets
    `review_feedback` when retries are exhausted — it always sets
    `final_output` instead.  The graph router (`route_after_review`) then
    simply checks whether `final_output` is set to decide the next step.
    This avoids a race condition where the router and the reviewer disagree
    about whether to retry.

    WHY THE ORIGINAL TASK IS INCLUDED IN THE REVIEW PROMPT
    -------------------------------------------------------
    The Reviewer must evaluate the report relative to what was asked, not as
    a standalone document.  A report on "PM tools for small teams" that only
    covers enterprise tools would score 5 as a piece of writing but 1 as a
    response to the task.  Including the original task in the prompt forces
    the model to evaluate fit-to-purpose, not just prose quality.
    """
    plan = state.get("execution_plan")
    if plan is None:
        return {"errors": ["Reviewer: no execution plan found"]}

    # Locate the writer subtask by role — robust against variable subtask IDs.
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
                # Both the original task and the report are provided so the
                # Reviewer can assess fit-to-purpose, not just writing quality.
                f"Original task: {original_task}\n\n"
                f"Report to review:\n\n{report}"
            ),
        },
    ]

    try:
        # Structured output: ReviewResult guarantees score is an int 1-5,
        # approved is a bool, and feedback is a string.  No string-parsing needed.
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
            # Acceptance path — set final_output so the router goes to END.
            # `forced=True` in the log means we accepted despite low score
            # because retries were exhausted (avoids infinite loops).
            updates["final_output"] = report
            updates["review_feedback"] = review.feedback or ""
            log.info(
                "reviewer_accepted",
                forced=not review.approved,  # True = accepted only because retries maxed out
                score=review.score,
            )
        else:
            # Rejection path — set feedback so the Writer receives it on retry.
            # Do NOT set final_output here; its absence tells the router to retry.
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
        # Fail-safe: if the Reviewer itself crashes, accept whatever the Writer
        # produced rather than leaving the pipeline stuck.  Record the error
        # for visibility but don't block delivery.
        return {
            "final_output": report,
            "errors": [f"Reviewer failed: {exc}"],
        }
