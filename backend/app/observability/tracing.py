"""
observability/tracing.py — per-node execution tracing for Phase 4.

WHAT THIS MODULE DOES
---------------------
Every LangGraph node is wrapped by `traced_node(...)`.  The wrapper:
  1. Records the node's start time.
  2. Installs a fresh, per-node accumulator (a ContextVar list) that any LLM
     call made *inside* the node appends to via `record_llm_call(...)`.
  3. Installs a per-node tool-call counter that `registry.invoke(...)` bumps
     via `increment_tool_calls()`.
  4. On node exit, sums tokens/cost, measures latency, decides the node status
     (success / failed / escalated) and writes one ExecutionTrace row.

WHY CONTEXTVARS (not passing an object through the call stack)
--------------------------------------------------------------
The agents (supervisor, specialists, reviewer) call `call_llm(...)` several
layers deep and never see the graph state.  Threading a "trace collector"
argument through every function signature would be invasive and would touch
files that have nothing to do with observability.  A ContextVar is the
idiomatic asyncio-safe way to make per-execution context available to code
deep in the call stack without changing signatures.  Because LangGraph runs
each node (and everything it awaits) inside the same asyncio task, the value
set at the top of `traced_node` is visible to `call_llm` and reset cleanly
afterwards — even with concurrent tasks, each has its own ContextVar copy.

GRACEFUL DEGRADATION
--------------------
Trace persistence is wrapped in try/except.  A trace write failure logs a
warning but NEVER propagates — observability must not be able to crash a node.
"""
from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)

# Full prompt/response text is capped before it ever hits the DB.
TRUNCATE_AT = 10_000

# ---------------------------------------------------------------------------
# Token pricing — USD per 1,000 tokens.
#
# WHY A FLAT DICT KEYED BY MODEL NAME
# -----------------------------------
# call_llm knows the concrete model name it used (from MODEL_REGISTRY).  Keying
# pricing by that same string means there is exactly one lookup and no mapping
# layer between "what model ran" and "what it cost".  Unknown models fall back
# to (0.0, 0.0) so a mis-typed model name yields a visibly-zero cost rather than
# a crash — easy to spot in the dashboard.
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model_name: (input_usd_per_1k, output_usd_per_1k)
    "gpt-5.4-mini": (0.00015, 0.00060),
    "gpt-5.4-nano": (0.00005, 0.00020),
    # Embeddings are output-free; we store the input rate in both slots.
    "text-embedding-3-small": (0.00002, 0.00002),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a single LLM call given its token usage."""
    in_rate, out_rate = MODEL_PRICING.get(model, (0.0, 0.0))
    return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate


# ---------------------------------------------------------------------------
# Per-node accumulators (ContextVars)
# ---------------------------------------------------------------------------

@dataclass
class LLMCallRecord:
    """One LLM invocation captured inside a node."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    prompt: str = ""
    response: str = ""


# Default is None (not []) so we can cheaply detect "not inside a traced node"
# and skip recording without allocating a list on import.
_llm_calls: ContextVar[list[LLMCallRecord] | None] = ContextVar(
    "agentis_llm_calls", default=None
)
_tool_calls: ContextVar[int | None] = ContextVar(
    "agentis_tool_calls", default=None
)
# Per-node extra metadata a node can attach to its own trace (e.g. the
# specialist_execution node records its parallel_groups / speedup here).
_node_metadata: ContextVar[dict | None] = ContextVar(
    "agentis_node_metadata", default=None
)


def record_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prompt: str = "",
    response: str = "",
) -> None:
    """
    Append an LLM call's usage to the current node's accumulator.

    Called by app.llm.client.call_llm after every model invocation.  A no-op
    when called outside a traced node (e.g. from a unit test), so call_llm never
    has to know whether tracing is active.
    """
    calls = _llm_calls.get()
    if calls is None:
        return  # not inside a traced node — nothing to accumulate
    calls.append(
        LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
            prompt=prompt or "",
            response=response or "",
        )
    )


def increment_tool_calls(n: int = 1) -> None:
    """Bump the current node's tool-call counter (no-op outside a traced node)."""
    current = _tool_calls.get()
    if current is None:
        return
    _tool_calls.set(current + n)


def record_node_metadata(**fields: Any) -> None:
    """
    Attach extra key/value pairs to the CURRENT node's ExecutionTrace metadata.

    The wrapper always records generic fields (llm_call_count, models,
    retry_count).  A node that wants to surface node-specific structured data —
    e.g. specialist_execution recording parallel_groups / sequential_time_ms /
    parallel_time_ms / speedup_factor so GET /tasks/{id}/trace can show how much
    a parallel run saved — calls this instead of threading extra return values
    through the node's own dict (which is reserved for AgentState updates).

    No-op outside a traced node, mirroring record_llm_call's degrade-gracefully
    behaviour, so callers never need to check "am I inside a traced node?".
    """
    meta = _node_metadata.get()
    if meta is None:
        return
    meta.update(fields)


# ---------------------------------------------------------------------------
# Node wrapper
# ---------------------------------------------------------------------------

NodeFn = Callable[[dict], Awaitable[dict]]


def traced_node(node_name: str, fn: NodeFn) -> NodeFn:
    """
    Wrap a LangGraph node function so every execution writes an ExecutionTrace.

    The returned coroutine preserves the node's original return value and
    re-raises any exception it does not expect — but since the agents catch
    their own errors and return them via the `errors` accumulator, the common
    "failed" path is detected from the returned dict, not from an exception.
    """

    async def wrapped(state: dict) -> dict:
        calls_token = _llm_calls.set([])
        tools_token = _tool_calls.set(0)
        meta_token = _node_metadata.set({})

        started_at = datetime.now(timezone.utc)
        perf_start = time.perf_counter()

        status = "success"
        error_message: str | None = None
        result: dict = {}

        try:
            result = await fn(state)
            if isinstance(result, dict):
                # A node signals its own failure by appending to `errors`
                # (operator.add reducer) — anything here was added by THIS node.
                node_errors = result.get("errors")
                if node_errors:
                    status = "failed"
                    error_message = "; ".join(str(e) for e in node_errors)
                # check_escalation_node sets this when a HITL trigger fires.
                elif result.get("requires_human_review"):
                    status = "escalated"
            return result
        except BaseException as exc:  # noqa: BLE001
            # A GraphInterrupt is NOT a failure — it is the human_review node
            # pausing the graph via interrupt().  Record it as "escalated" (the
            # node is waiting on a human) and re-raise so LangGraph can suspend
            # the run.  When the graph resumes, the node runs again and writes a
            # second trace row with the real outcome.
            if _is_graph_interrupt(exc):
                status = "escalated"
            else:
                status = "failed"
                error_message = str(exc)
            raise
        finally:
            latency_ms = int((time.perf_counter() - perf_start) * 1000)
            completed_at = datetime.now(timezone.utc)

            calls = _llm_calls.get() or []
            tool_count = _tool_calls.get() or 0
            extra_metadata = _node_metadata.get() or {}
            _llm_calls.reset(calls_token)
            _tool_calls.reset(tools_token)
            _node_metadata.reset(meta_token)

            input_tokens = sum(c.input_tokens for c in calls)
            output_tokens = sum(c.output_tokens for c in calls)
            cost_usd = sum(c.cost_usd for c in calls)

            # Concatenate multi-call prompts/responses (e.g. researcher makes 2
            # LLM calls) with a clear separator so the UI can show the full flow.
            prompt = _join([c.prompt for c in calls])
            response = _join([c.response for c in calls])

            metadata: dict[str, Any] = {
                "llm_call_count": len(calls),
                "models": sorted({c.model for c in calls}) if calls else [],
            }
            retry_count = state.get("retry_count")
            if retry_count is not None:
                metadata["retry_count"] = retry_count
            # Node-specific extras attached via record_node_metadata (e.g.
            # specialist_execution's parallel_groups / speedup_factor).
            metadata.update(extra_metadata)

            await _save_trace(
                task_id=state.get("task_id"),
                node_name=node_name,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                tool_calls_count=tool_count,
                status=status,
                error_message=error_message,
                llm_prompt=prompt,
                llm_response=response,
                metadata=metadata,
            )

    wrapped.__name__ = f"traced_{node_name}"
    return wrapped


def _is_graph_interrupt(exc: BaseException) -> bool:
    """
    True if the exception is a LangGraph interrupt (the human_review pause),
    which must be treated as a normal control-flow event, not an error.

    We match by class name rather than importing GraphInterrupt directly so the
    tracing layer stays decoupled from LangGraph's exact exception module path
    (which has moved between versions).
    """
    return any(
        cls.__name__ in ("GraphInterrupt", "Interrupt")
        for cls in type(exc).__mro__
    )


def _join(parts: list[str]) -> str | None:
    """Join non-empty prompt/response fragments; return None if all empty."""
    non_empty = [p for p in parts if p]
    if not non_empty:
        return None
    return "\n\n----- next LLM call -----\n\n".join(non_empty)


async def _save_trace(
    *,
    task_id: str | None,
    node_name: str,
    started_at: datetime,
    completed_at: datetime,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    tool_calls_count: int,
    status: str,
    error_message: str | None,
    llm_prompt: str | None,
    llm_response: str | None,
    metadata: dict,
) -> None:
    """
    Persist a single ExecutionTrace row.  All failures are swallowed so a DB
    hiccup can never crash the node whose execution we are recording.
    """
    if not task_id:
        # No task_id (e.g. a bare Studio run without one) — nothing to link to.
        log.debug("trace_skipped_no_task_id", node=node_name)
        return
    try:
        from app.db.engine import get_db_session
        from app.db.queries import create_execution_trace

        async with get_db_session() as session:
            await create_execution_trace(
                session,
                task_id=task_id,
                node_name=node_name,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost_usd, 8),
                tool_calls_count=tool_calls_count,
                status=status,
                error_message=error_message,
                llm_prompt=(llm_prompt or None) and llm_prompt[:TRUNCATE_AT],
                llm_response=(llm_response or None) and llm_response[:TRUNCATE_AT],
                trace_metadata=metadata,
            )
    except Exception as exc:
        log.warning("execution_trace_save_failed", node=node_name, error=str(exc))
