"""
Researcher Specialist Agent

Has access to web_search. Gathers raw information based on the subtask description.
"""
from __future__ import annotations

import json

import structlog

from app.llm.client import call_llm
from app.tools.registry import registry

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a Research Specialist Agent. Your job is to gather
comprehensive, accurate information from the web to answer research questions.

You have access to the following tools:
{tools}

Instructions:
1. Analyse the research task
2. Formulate 2-3 targeted search queries to gather thorough information
3. Call web_search for each query (you will receive results inline)
4. Synthesise the results into a well-organised raw findings document
5. Include source URLs where relevant
6. Be factual — do not invent information not found in search results

Return ONLY the raw findings document (no meta-commentary).
"""


async def run_researcher(
    subtask_description: str,
    *,
    task_id: str | None = None,
    subtask_id: str | None = None,
    db_session=None,
) -> str:
    """Execute the researcher specialist for a given subtask."""
    log.info("researcher_start", subtask=subtask_description[:120])

    tool_descriptions = registry.describe_for_agent("researcher")

    # Step 1 — Ask LLM to plan queries
    plan_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools=tool_descriptions),
        },
        {
            "role": "user",
            "content": (
                f"Research task: {subtask_description}\n\n"
                "First, list 2-3 search queries you will use (one per line, "
                "prefixed with 'QUERY: '). Then I will run them and give you "
                "the results to synthesise."
            ),
        },
    ]

    query_plan: str = await call_llm(role="specialist", messages=plan_messages)

    # Extract queries
    queries = [
        line.replace("QUERY:", "").strip()
        for line in query_plan.splitlines()
        if line.strip().upper().startswith("QUERY:")
    ]
    if not queries:
        # Fallback: use the subtask description itself as the query
        queries = [subtask_description]

    log.info("researcher_queries", queries=queries)

    # Step 2 — Execute searches
    search_results_text = ""
    for q in queries[:3]:  # cap at 3
        try:
            result = await registry.invoke(
                "web_search",
                {"query": q, "max_results": 5},
                task_id=task_id,
                subtask_id=subtask_id,
                db_session=db_session,
            )
            search_results_text += f"\n\n=== Search: {q} ===\n"
            if result.get("answer"):
                search_results_text += f"AI Answer: {result['answer']}\n"
            for r in result.get("results", []):
                search_results_text += (
                    f"\n- [{r['title']}]({r['url']})\n  {r['content'][:400]}\n"
                )
        except Exception as exc:
            log.warning("researcher_search_failed", query=q, error=str(exc))
            search_results_text += f"\n\n=== Search: {q} ===\n[Search failed: {exc}]\n"

    # Step 3 — Synthesise
    synthesis_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools=tool_descriptions),
        },
        {
            "role": "user",
            "content": (
                f"Research task: {subtask_description}\n\n"
                f"Here are the search results:\n{search_results_text}\n\n"
                "Now synthesise these into a comprehensive raw findings document."
            ),
        },
    ]

    findings: str = await call_llm(role="specialist", messages=synthesis_messages)
    log.info("researcher_done", chars=len(findings))
    return findings
