"""
Analyst Specialist Agent

Takes researcher's raw findings and produces a structured, analytical document.
May use execute_python for quantitative analysis.
"""
from __future__ import annotations

import structlog

from app.llm.client import call_llm
from app.tools.registry import registry

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are an Analysis Specialist Agent. You receive raw research
findings and your job is to:

1. Identify key themes, patterns, and data points
2. Structure the information into clear categories
3. Compare and contrast options/competitors/trends where relevant
4. Draw analytical insights (not just summaries)
5. Highlight any gaps or uncertainties in the data

You have access to these tools (use only if quantitative analysis is needed):
{tools}

Output a well-structured analytical document with:
- Executive Summary (3-5 sentences)
- Key Findings (bullet points, grouped by theme)
- Comparative Analysis (if applicable)
- Insights & Implications
- Data Gaps / Caveats
"""


async def run_analyst(
    subtask_description: str,
    raw_findings: str,
    *,
    task_id: str | None = None,
    subtask_id: str | None = None,
    db_session=None,
) -> str:
    """Execute the analyst specialist."""
    log.info("analyst_start", subtask=subtask_description[:120])

    tool_descriptions = registry.describe_for_agent("analyst")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools=tool_descriptions),
        },
        {
            "role": "user",
            "content": (
                f"Analysis task: {subtask_description}\n\n"
                f"Raw findings from researcher:\n\n{raw_findings}"
            ),
        },
    ]

    analysis: str = await call_llm(role="specialist", messages=messages)
    log.info("analyst_done", chars=len(analysis))
    return analysis
