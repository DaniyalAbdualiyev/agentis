"""
llm/client.py — Unified, provider-agnostic LLM wrapper.

THE CORE PROBLEM THIS FILE SOLVES
-----------------------------------
Without this wrapper, every agent would directly import `ChatOpenAI` and
hardcode a model name.  That creates three problems:

1. PROVIDER LOCK-IN — switching from OpenAI to Anthropic would require
   editing every agent file.  With this wrapper, it is a one-line change
   to MODEL_REGISTRY and a new `elif PROVIDER == "anthropic"` branch.

2. INCONSISTENT RETRY LOGIC — without a shared wrapper, each agent would
   need its own retry handling, or none would have it.  A transient 429 rate
   limit from OpenAI would crash the whole graph run.

3. SCATTERED MODEL CONFIGURATION — model names and temperature settings
   would be duplicated across agent files.  Updating the Supervisor's model
   from gpt-5.4-mini to gpt-5.4 would require finding every occurrence.

HOW TO SWAP IN CLAUDE TOMORROW
--------------------------------
1. Add an "anthropic" entry to MODEL_REGISTRY.
2. Uncomment (or add) the `elif PROVIDER == "anthropic"` block in
   `_build_client()` using `from langchain_anthropic import ChatAnthropic`.
3. Set `LLM_PROVIDER=anthropic` in .env.
4. No changes to supervisor.py, reviewer.py, or any specialist file.

WHY WE USE LANGCHAIN'S CHAT MODELS (not the OpenAI SDK directly)
------------------------------------------------------------------
Two reasons:

1. LANGSMITH AUTO-TRACING — LangChain's ChatOpenAI class automatically
   emits traces to LangSmith when LANGCHAIN_TRACING_V2=true is set.
   If we called `openai.chat.completions.create()` directly, we would get
   no tracing without writing custom callback handlers.

2. PROVIDER ABSTRACTION — LangChain's `ChatModel` interface is the same
   for OpenAI, Anthropic, Google, etc.  `_build_client()` returns a
   `ChatOpenAI` today, but could return `ChatAnthropic` tomorrow and the
   rest of `call_llm` would not need to change a single line.

WHY temperature=0.2
-------------------
- 0.0 is fully deterministic — useful for structured output but can produce
  very repetitive text in the Writer's reports.
- 1.0 is highly creative — introduces hallucinations and inconsistency,
  especially in the Supervisor and Reviewer where we need reliable structure.
- 0.2 is a pragmatic middle ground: low enough for consistent structured
  outputs, high enough for varied, readable prose from the Writer.
  All roles currently share this temperature; it can be made per-role later.
"""
from __future__ import annotations

import os
from typing import Any, Type

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# MODEL_REGISTRY — the single place where provider + role → model is decided
#
# WHY A NESTED DICT (provider → role → model_name)
# -------------------------------------------------
# The outer key is the provider ("openai", "anthropic", …).
# The inner key is the agent role ("supervisor", "reviewer", "specialist").
# This two-level structure means:
# - Adding a new provider requires adding one top-level key — no conditionals.
# - Changing which model the Supervisor uses requires changing one string.
# - The role names are the same vocabulary used in every agent file, so
#   there is no translation layer between "what the agent says it is" and
#   "what model it gets".
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "openai": {
        "supervisor": "gpt-5.4-mini",
        "reviewer":   "gpt-5.4-mini",
        "specialist": "gpt-5.4-mini",
    },
    # Uncomment to enable Anthropic support:
    # "anthropic": {
    #     "supervisor": "claude-3-5-sonnet-20241022",
    #     "reviewer":   "claude-3-5-sonnet-20241022",
    #     "specialist": "claude-3-haiku-20240307",
    # },
}

# Read from environment at module load time.
# Defaulting to "openai" means the system works out of the box without
# setting LLM_PROVIDER explicitly.
PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")


def _model_for_role(role: str) -> str:
    """
    Resolve the concrete model name for a role.

    Extracted from _build_client so the tracing layer can label each LLM call
    with the exact model that ran (needed for correct per-model cost pricing)
    without re-implementing the provider/role fallback logic.
    """
    models = MODEL_REGISTRY.get(PROVIDER, MODEL_REGISTRY["openai"])
    return models.get(role, models["specialist"])


def _build_client(role: str) -> ChatOpenAI:
    """
    Instantiate and return a LangChain chat model for the given role.

    WHY THIS IS A SEPARATE FUNCTION (not inlined in call_llm)
    ----------------------------------------------------------
    Separating client construction from invocation makes it easy to:
    - Mock the client in tests: patch `_build_client` to return a mock.
    - Add per-role configuration (streaming, timeout, max_tokens) without
      making call_llm more complex.
    - Swap the provider by changing this one function's branching logic.

    WHY WE CREATE A NEW CLIENT PER CALL (not a singleton)
    ------------------------------------------------------
    LangChain's ChatOpenAI is cheap to construct.  Creating one per call
    avoids shared mutable state between concurrent requests — in an async
    FastAPI server, a singleton client with internal state could cause
    subtle race conditions.  If construction cost becomes a concern, a
    per-role LRU cache could be added here without changing call_llm.
    """
    # Fall back to the "specialist" model if an unknown role is passed.
    model_name = _model_for_role(role)

    if PROVIDER == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Future provider branches go here.  Example:
    # elif PROVIDER == "anthropic":
    #     from langchain_anthropic import ChatAnthropic
    #     return ChatAnthropic(model=model_name, temperature=0.2)

    raise ValueError(f"Unknown LLM provider: {PROVIDER!r}. Add a branch in _build_client().")


# ---------------------------------------------------------------------------
# call_llm — the only function agents should call
#
# WHY @retry IS ON THIS FUNCTION (not inside each agent)
# -------------------------------------------------------
# OpenAI rate limits (429) and transient network errors are common in
# production.  Putting retry logic here means every agent benefits
# automatically — no agent file needs to import tenacity.
#
# RETRY PARAMETERS EXPLAINED
# ---------------------------
# stop_after_attempt(3): try up to 3 times total (1 original + 2 retries).
#   More retries increase cost; fewer leave too little margin for transient errors.
# wait_exponential(multiplier=1, min=2, max=10):
#   Wait 2s after first failure, ~4s after second, capped at 10s.
#   Exponential backoff reduces thundering-herd pressure on the API.
# reraise=True: if all attempts fail, propagate the original exception up to
#   the calling agent, which then catches it and returns an error state update.
# ---------------------------------------------------------------------------

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
    Invoke the LLM for a given agent role.

    This is the only LLM entry point in the codebase.  All agents call this
    function; none import or construct chat models directly.

    Parameters
    ----------
    role : str
        Agent role key — "supervisor", "reviewer", or "specialist".
        Used to look up the correct model in MODEL_REGISTRY.

    messages : list[dict[str, str]]
        Conversation history in the standard format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        We use a plain list of dicts (not LangChain message objects) because:
        - It is a simpler, provider-agnostic data format.
        - Agents don't need to import from langchain_core just to build messages.
        - This function handles the translation to LangChain objects internally.

    response_model : Type[BaseModel] | None
        If provided, uses LangChain's `.with_structured_output()` to force
        the model to return a valid instance of this Pydantic class.
        Used by the Supervisor (→ ExecutionPlan) and Reviewer (→ ReviewResult).
        If None, returns the model's raw text response as a plain string.

    Returns
    -------
    BaseModel instance if response_model is provided, else str.

    HOW STRUCTURED OUTPUT WORKS UNDER THE HOOD
    -------------------------------------------
    `client.with_structured_output(response_model)` tells LangChain to:
    1. Generate a JSON schema from the Pydantic class.
    2. Pass that schema to OpenAI as a `tools` definition (function calling).
    3. Instruct OpenAI to always respond with a tool call matching the schema.
    4. Parse the JSON response into the Pydantic model, raising ValidationError
       if the model's output doesn't conform.

    This is more reliable than prompt-engineering alone ("return valid JSON")
    because the constraint is enforced at the API level, not just as a hint.
    """
    client = _build_client(role)
    model_name = _model_for_role(role)

    # Translate plain dicts into LangChain message objects.
    # WHY SystemMessage vs HumanMessage: LangChain (and the OpenAI API) treats
    # system messages differently from human/user messages — system messages
    # set the model's persona and rules, while human messages carry the actual
    # request.  Using the correct types ensures the API receives the right
    # `role` field in the underlying JSON payload.
    lc_messages: list[BaseMessage] = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        else:
            lc_messages.append(HumanMessage(content=m["content"]))

    log.debug("llm_call", role=role, provider=PROVIDER, structured=response_model is not None)

    # Phase 4: capture the outgoing prompt so the trace can show exactly what
    # was sent to the model.  We serialise the plain-dict messages (not the
    # LangChain objects) because they are already the human-readable form.
    prompt_text = _serialise_prompt(messages)

    if response_model is not None:
        # Structured output path — returns a Pydantic model instance.
        # LangSmith will trace this call automatically because we're using
        # LangChain's ChatOpenAI, which has built-in tracing callbacks.
        #
        # WHY include_raw=True (Phase 4)
        # -------------------------------
        # `.with_structured_output(model)` normally returns ONLY the parsed
        # Pydantic instance, discarding the underlying AIMessage — and with it
        # the token-usage metadata we need for cost accounting.  Passing
        # include_raw=True makes it return {"raw": AIMessage, "parsed": model,
        # "parsing_error": ...} so we can read usage from `raw` and still hand
        # the caller the `parsed` model exactly as before.
        structured_client = client.with_structured_output(
            response_model, include_raw=True
        )
        result = await structured_client.ainvoke(lc_messages)
        raw_msg = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        response_text = parsed.model_dump_json() if parsed is not None else ""
        _record_usage(model_name, raw_msg, prompt_text, response_text)
        log.debug("llm_structured_response", role=role, model=response_model.__name__)
        return parsed

    # Free-text path — returns the model's response content as a plain string.
    result = await client.ainvoke(lc_messages)
    text = result.content
    _record_usage(model_name, result, prompt_text, text if isinstance(text, str) else str(text))
    log.debug("llm_text_response", role=role, chars=len(text))
    return text


# ---------------------------------------------------------------------------
# Phase 4: usage-capture helpers
# ---------------------------------------------------------------------------

def _serialise_prompt(messages: list[dict[str, str]]) -> str:
    """Render the message list as a readable transcript for the trace record."""
    return "\n\n".join(f"[{m.get('role', '?')}]\n{m.get('content', '')}" for m in messages)


def _record_usage(model_name: str, message: Any, prompt_text: str, response_text: str) -> None:
    """
    Extract token usage from a LangChain AIMessage and forward it to the
    per-node tracing accumulator.

    WHY TWO USAGE SOURCES
    ---------------------
    Newer LangChain messages expose `.usage_metadata` (a normalised dict with
    input_tokens/output_tokens).  Older/edge responses only populate
    `.response_metadata["token_usage"]` (the raw OpenAI field names).  We try
    the normalised form first and fall back to the raw form so token counts are
    captured regardless of the exact response shape.
    """
    input_tokens = 0
    output_tokens = 0
    try:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
        else:
            meta = getattr(message, "response_metadata", {}) or {}
            token_usage = meta.get("token_usage", {}) or {}
            input_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    except Exception as exc:  # never let usage extraction break the LLM call
        log.debug("usage_extraction_failed", error=str(exc))

    try:
        from app.observability.tracing import record_llm_call
        record_llm_call(
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt_text,
            response=response_text,
        )
    except Exception as exc:
        log.debug("record_llm_call_failed", error=str(exc))
