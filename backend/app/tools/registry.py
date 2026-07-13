"""
Tool registry — central catalogue of all available tools.

Each tool registers with:
  - name
  - description (shown to LLM in agent prompts)
  - callable (async function)
  - allowed_agents (which agent roles may invoke it)

The registry logs every call to PostgreSQL via the DB session passed at call time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

log = structlog.get_logger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Awaitable[Any]]
    allowed_agents: list[str] = field(default_factory=lambda: ["researcher", "analyst", "writer"])


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        log.info("tool_registered", name=spec.name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_for_agent(self, agent_role: str) -> list[ToolSpec]:
        return [t for t in self._tools.values() if agent_role in t.allowed_agents]

    def describe_for_agent(self, agent_role: str) -> str:
        specs = self.list_for_agent(agent_role)
        if not specs:
            return "No tools available."
        lines = []
        for s in specs:
            lines.append(f"- **{s.name}**: {s.description}")
        return "\n".join(lines)

    async def invoke(
        self,
        name: str,
        inputs: dict[str, Any],
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        db_session: Any = None,   # AsyncSession, typed as Any to avoid circular import
    ) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise ValueError(f"Tool '{name}' not found in registry")

        # Phase 4: count this invocation toward the current node's tool_calls_count.
        # No-op outside a traced node, so tests / direct calls are unaffected.
        try:
            from app.observability.tracing import increment_tool_calls
            increment_tool_calls()
        except Exception:
            pass

        start = time.monotonic()
        error_msg: str | None = None
        result: Any = None

        try:
            result = await spec.fn(**inputs)
            return result
        except Exception as exc:
            error_msg = str(exc)
            log.error("tool_invocation_failed", tool=name, error=error_msg)
            raise
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "tool_invoked",
                tool=name,
                latency_ms=latency_ms,
                success=error_msg is None,
            )
            if db_session is not None:
                try:
                    from app.db.queries import log_tool_call  # local import to avoid circular
                    await log_tool_call(
                        session=db_session,
                        task_id=task_id,
                        subtask_id=subtask_id,
                        tool_name=name,
                        inputs=inputs,
                        outputs=result,
                        latency_ms=latency_ms,
                        success=error_msg is None,
                        error_message=error_msg,
                    )
                except Exception as log_exc:
                    log.warning("tool_log_failed", error=str(log_exc))


# Singleton registry
registry = ToolRegistry()
