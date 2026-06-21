"""
agents/supervisor.py — The Supervisor Agent: task decomposition and planning.

SINGLE RESPONSIBILITY
---------------------
The Supervisor has exactly one job: receive the user's raw task string and
return a structured ExecutionPlan that tells the rest of the pipeline what
to do.  It does not search the web, does not write reports, does not review
anything.  This narrow scope matters because:

- It is the only agent whose output is a Pydantic model (not a text string).
  Everything downstream depends on having a valid plan, so validation here
  must be strict.
- Keeping it focused means we can swap or upgrade the planning logic (e.g.
  add dynamic subtask counts, add cost estimation) without touching any other
  agent.

WHY THE SUPERVISOR USES STRUCTURED OUTPUT
------------------------------------------
The Supervisor calls `call_llm(..., response_model=ExecutionPlan)`.
This uses OpenAI's function-calling / JSON-mode feature under the hood:
LangChain generates a JSON schema from the Pydantic model and passes it to
the API as a tool definition.  The API is then constrained to return JSON
that matches that schema.

The alternative — asking the LLM to "return valid JSON" in the prompt and
then parsing it with json.loads — is unreliable.  LLMs frequently add
markdown fences, trailing commas, or extraneous keys.  Structured output
eliminates that entire class of failure at the cost of a slightly more
expensive API call (which is worth it for the Supervisor, since a bad plan
breaks the entire pipeline).

WHY THE SYSTEM PROMPT IS RIGID (exactly 3 subtasks, fixed IDs)
---------------------------------------------------------------
In principle the Supervisor could dynamically decide how many subtasks to
create.  We deliberately constrain it to exactly 3 (researcher → analyst →
writer) with fixed IDs because:

1. The specialist_execution node in graph.py looks up subtasks by
   `assigned_agent` key, so the three roles must always be present.
2. Fixed IDs ("subtask_1", "subtask_2", "subtask_3") make the
   `subtask_results` cache predictable — no subtask ID collision risk.
3. Phase 1 is explicitly sequential.  Allowing variable subtask counts
   would require a more complex scheduler that is out of scope here.

This rigidity is a Phase 1 trade-off, not a permanent design.  The
depends_on field in Subtask already supports arbitrary DAGs for Phase 2.
"""
from __future__ import annotations

import structlog

from app.graph.state import AgentState, ExecutionPlan
from app.llm.client import call_llm

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
#
# WHY THIS IS A MODULE-LEVEL CONSTANT (not built inside run_supervisor)
# ----------------------------------------------------------------------
# The prompt never changes between calls — it is the same for every task.
# Defining it at module level means:
# - It is loaded once at import time, not re-allocated on every call.
# - It is easy to review: anyone reading this file can immediately see
#   the full instructions given to the Supervisor LLM without tracing
#   through function logic.
# - It is easy to version-control: a git diff on this file clearly shows
#   when and how the Supervisor's instructions changed.
# ---------------------------------------------------------------------------

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
    LangGraph node: supervisor_planning.

    Reads `original_task` from state, calls the LLM with structured output,
    and returns a state update containing the validated ExecutionPlan.

    WHY THIS RETURNS A DICT (not an ExecutionPlan)
    -----------------------------------------------
    LangGraph node functions must return a plain dict of state updates.
    LangGraph merges that dict into the existing AgentState.  If we returned
    the ExecutionPlan directly, LangGraph would not know which state field
    to update.

    WHY ERRORS ARE RETURNED (not raised)
    -------------------------------------
    Raising an exception inside a LangGraph node crashes the entire graph run
    and leaves the task in an unrecoverable state in the database.  Instead,
    we catch exceptions, log them, and return an error message via the `errors`
    accumulator field.  This means:
    - The graph terminates gracefully (returns an empty final_output).
    - The FastAPI endpoint can inspect `errors` and report what went wrong.
    - The LangSmith trace shows a completed run with an error payload, rather
      than an aborted run with no useful information.

    WHY retry_count AND subtask_results ARE RESET HERE
    ---------------------------------------------------
    The Supervisor is the first real agent to run.  Resetting these fields
    here (rather than in the initial state dict in routers/tasks.py) makes
    the Supervisor the authoritative "start of pipeline" initialiser.
    If we ever add a pre-planning node that runs before the Supervisor, only
    this file needs to change.
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
        # response_model=ExecutionPlan tells call_llm to use OpenAI structured
        # output and parse the response directly into an ExecutionPlan instance.
        # If the model returns invalid JSON or a schema mismatch, this raises
        # immediately — we never proceed with a malformed plan.
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
            "execution_plan":        plan,
            "current_subtask_index": 0,   # reset for clean pipeline start
            "subtask_results":       {},  # empty cache — nothing has run yet
            "retry_count":           0,   # reset retry counter
        }
    except Exception as exc:
        log.error("supervisor_failed", error=str(exc))
        # Append to `errors` (not overwrite) — see AgentState.errors docstring.
        return {"errors": [f"Supervisor failed: {exc}"]}
