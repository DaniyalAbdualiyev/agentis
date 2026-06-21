"""
Unified LLM client — provider-agnostic wrapper.

To swap in Claude later, add a new branch in _build_client() and map
the role to the model name in MODEL_REGISTRY.  No agent code changes needed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Type

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Model registry — one-line swap per role
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "openai": {
        "supervisor": "gpt-4o-mini",
        "reviewer":   "gpt-4o-mini",
        "specialist": "gpt-4o-mini",   # use same tier; swap to nano when available
    },
    # "anthropic": {
    #     "supervisor": "claude-3-5-sonnet-20241022",
    #     "reviewer":   "claude-3-5-sonnet-20241022",
    #     "specialist": "claude-3-haiku-20240307",
    # },
}

PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")


def _build_client(role: str) -> ChatOpenAI:
    """Return a configured LangChain chat model for *role*."""
    models = MODEL_REGISTRY.get(PROVIDER, MODEL_REGISTRY["openai"])
    model_name = models.get(role, models["specialist"])

    if PROVIDER == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    # Future: elif PROVIDER == "anthropic": ...
    raise ValueError(f"Unknown LLM provider: {PROVIDER}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def call_llm(
    role: str,
    messages: list[dict[str, str]],
    response_model: Type[BaseModel] | None = None,
) -> Any:
    """
    Invoke the LLM for *role*.

    Parameters
    ----------
    role:           "supervisor" | "reviewer" | "specialist"
    messages:       list of {"role": "system"|"user"|"assistant", "content": str}
    response_model: optional Pydantic v2 model — triggers structured output

    Returns
    -------
    Pydantic model instance if *response_model* supplied, else raw string.
    """
    client = _build_client(role)

    lc_messages: list[BaseMessage] = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        else:
            lc_messages.append(HumanMessage(content=m["content"]))

    log.debug("llm_call", role=role, provider=PROVIDER, structured=response_model is not None)

    if response_model is not None:
        structured_client = client.with_structured_output(response_model)
        result = await structured_client.ainvoke(lc_messages)
        log.debug("llm_structured_response", role=role, model=response_model.__name__)
        return result

    result = await client.ainvoke(lc_messages)
    text = result.content
    log.debug("llm_text_response", role=role, chars=len(text))
    return text
