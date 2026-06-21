"""
Supervisor Agent

Receives the user's task, produces an ExecutionPlan with ordered Subtasks,
and assigns each to the appropriate specialist agent.
"""
from __future__ import annotations

import structlog

from app.graph.state import AgentState, ExecutionPlan
from app.llm.client import call_llm

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a Supervisor Agent in a multi-agent research system.

Your job is to decompose a complex user task into a sequential execution plan
consisting of exactly 3 subtasks assigned to the following specialist agents:

1. researcher  — gathers raw information from the web
2. analyst     — takes raw findings and structures / analyzes them
3. writer      — takes structured analysis and writes the final report

Rules:
- Always produce exactly 3 subtasks in order: researcher → analyst → writer
- Each subtask must have a unique id: "subtask_1", "subtask_2", "subtask_3"
- depends_on for subtask_2 = ["subtask_1"], for subtask_3 = ["subtask_2"]
- Descriptions must be specific and actionable
- The reasoning field should briefly explain your decomposition strategy
"""


async def run_supervisor(state: AgentState) -> dict:
    """
    LangGraph node: supervisor_planning

    Reads `original_task` from state, returns updated state with `execution_plan`.
    """
    task = state["original_task"]
    log.info("supervisor_start", task=task[:120])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Decompose the following research task into an execution plan:\n\n{task}"
            ),
        },
    ]

    try:
        plan: ExecutionPlan = await call_llm(
            role="supervisor",
            messages=messages,
            response_model=ExecutionPlan,
        )
        log.info(
            "supervisor_done",
            num_subtasks=len(plan.subtasks),
            reasoning=plan.reasoning[:120],
        )
        return {
            "execution_plan": plan,
            "current_subtask_index": 0,
            "subtask_results": {},
            "retry_count": 0,
        }
    except Exception as exc:
        log.error("supervisor_failed", error=str(exc))
        return {"errors": [f"Supervisor failed: {exc}"]}
