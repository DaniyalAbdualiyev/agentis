"""
Web search tool — powered by Tavily API.
"""
from __future__ import annotations

import os

import structlog
from tavily import AsyncTavilyClient

from app.tools.registry import ToolSpec, registry

log = structlog.get_logger(__name__)


async def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using Tavily and return structured results.

    Returns
    -------
    dict with keys:
        - query: original query
        - results: list of {title, url, content, score}
        - answer: Tavily's AI-generated answer (if available)
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    client = AsyncTavilyClient(api_key=api_key)
    log.info("web_search_start", query=query, max_results=max_results)

    response = await client.search(
        query=query,
        max_results=max_results,
        include_answer=True,
        search_depth="advanced",
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "content": r.get("content", ""),
            "score":   r.get("score", 0.0),
        })

    log.info("web_search_done", query=query, num_results=len(results))

    return {
        "query":   query,
        "results": results,
        "answer":  response.get("answer", ""),
    }


# Register tool
registry.register(
    ToolSpec(
        name="web_search",
        description=(
            "Search the web for up-to-date information. "
            "Input: query (str), max_results (int, default 5). "
            "Returns a dict with 'results' (list of {title, url, content}) and 'answer'."
        ),
        fn=web_search,
        allowed_agents=["researcher"],
    )
)
