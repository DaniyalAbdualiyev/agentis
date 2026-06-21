"""
Writer Specialist Agent

Takes the analyst's structured data and writes the final polished report.
Can incorporate reviewer feedback on retries.
"""
from __future__ import annotations

import structlog

from app.llm.client import call_llm

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a Writing Specialist Agent. You produce polished,
professional research reports for business audiences.

Given structured analysis data, write a clear, well-organised report that:
- Opens with an Executive Summary
- Presents findings in logical, readable sections with headers
- Uses concrete details and data points (not vague generalities)
- Concludes with actionable recommendations or takeaways
- Is free of filler phrases ("it is important to note", "in conclusion", etc.)
- Markdown formatting is appropriate

{feedback_section}
"""

FEEDBACK_SECTION = """
IMPORTANT — Reviewer Feedback (incorporate this in your revised report):
{feedback}
"""


async def run_writer(
    subtask_description: str,
    structured_analysis: str,
    *,
    reviewer_feedback: str | None = None,
    task_id: str | None = None,
    subtask_id: str | None = None,
    db_session=None,
) -> str:
    """Execute the writer specialist."""
    log.info(
        "writer_start",
        subtask=subtask_description[:120],
        has_feedback=reviewer_feedback is not None,
    )

    feedback_section = (
        FEEDBACK_SECTION.format(feedback=reviewer_feedback)
        if reviewer_feedback
        else ""
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(feedback_section=feedback_section),
        },
        {
            "role": "user",
            "content": (
                f"Writing task: {subtask_description}\n\n"
                f"Structured analysis to work from:\n\n{structured_analysis}"
            ),
        },
    ]

    report: str = await call_llm(role="specialist", messages=messages)
    log.info("writer_done", chars=len(report))
    return report
