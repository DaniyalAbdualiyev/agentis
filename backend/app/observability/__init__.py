"""
Phase 4 observability package.

Everything needed to make each LangGraph node execution fully inspectable —
latency, token usage, cost, tool calls, and the raw LLM prompt/response — lives
here so the instrumentation concern is isolated from the agent/graph logic.
"""
from app.observability.tracing import (  # noqa: F401
    MODEL_PRICING,
    calculate_cost,
    increment_tool_calls,
    record_llm_call,
    traced_node,
)
