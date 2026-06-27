"""
graph/state.py — Shared state and data contracts for the entire pipeline.

WHY THIS FILE EXISTS
--------------------
LangGraph passes a single state object between every node in the graph.
Every agent reads what it needs from that object and writes its results
back into it — there are no direct function calls between agents.
This file is therefore the single source of truth for:

  1. What data moves through the pipeline (AgentState)
  2. What structured outputs agents must produce (ExecutionPlan, ReviewResult)
  3. What lifecycle stages a subtask can be in (SubtaskStatus)

Keeping all of this in one place means you can understand the full data
contract of the system without reading any agent code.

WHY PYDANTIC FOR THE STRUCTURED OUTPUT MODELS
----------------------------------------------
The Supervisor and Reviewer use OpenAI's structured-output / function-calling
feature to guarantee their responses conform to a schema.  Without Pydantic
models, you'd get raw JSON strings and have to validate them yourself.
With `with_structured_output(ExecutionPlan)`, LangChain forces the LLM to
emit valid JSON that maps exactly to the Pydantic model — the call raises an
exception rather than returning a malformed plan.  This matters because a
malformed plan would cause silent, hard-to-debug downstream failures.

WHY TypedDict FOR AgentState (NOT Pydantic)
-------------------------------------------
LangGraph requires the state to be a TypedDict (or dataclass), not a
Pydantic model.  The reason is that LangGraph needs to be able to merge
partial state updates: each node returns a plain dict with only the keys
it changed, and LangGraph merges those changes into the existing state.
Pydantic models make full replacement the default behavior, which would
wipe out fields that a node didn't touch.

WHY `Annotated[list[str], operator.add]` FOR errors
----------------------------------------------------
Most state fields use "last write wins" semantics — if two nodes both
set `final_output`, the second one wins.  That is wrong for errors:
we want to ACCUMULATE errors from every node so the caller can see the
full failure history.  The `Annotated[..., operator.add]` annotation
tells LangGraph to use `operator.add` (list concatenation) as the
reducer for this field instead of replacement.  Every node can safely
append to errors without knowing what previous nodes wrote.

WHY optional / nullable fields
-------------------------------
The graph is sequential, so early nodes cannot know what later nodes will
produce.  If `AgentState` required every field to be non-None at creation
time, we'd have to pre-fill the state with meaningless sentinel values
before the graph starts.  Making downstream fields Optional lets the initial
state be sparse and honest — None means "not yet produced", not "missing".
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
import operator

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# SubtaskStatus — lifecycle constants
# ---------------------------------------------------------------------------

class SubtaskStatus:
    """
    Simple namespace for subtask lifecycle string constants.

    WHY NOT an Enum: Pydantic v2 serialises string-valued Enums cleanly, but
    when this model is stored in PostgreSQL as JSON and read back, Enum members
    would need explicit coercion on load.  Plain string constants survive a
    JSON round-trip without any conversion logic.
    """
    PENDING   = "pending"    # created by Supervisor, not yet started
    RUNNING   = "running"    # specialist currently executing
    COMPLETED = "completed"  # specialist finished successfully
    FAILED    = "failed"     # specialist raised an exception


# ---------------------------------------------------------------------------
# Subtask — one unit of work inside an ExecutionPlan
# ---------------------------------------------------------------------------

class Subtask(BaseModel):
    """
    Represents a single atomic piece of work the Supervisor delegates to a
    specialist agent.

    WHY depends_on EXISTS
    ---------------------
    Even though Phase 1 always runs subtasks sequentially (researcher →
    analyst → writer), the depends_on field makes the dependency structure
    explicit in data rather than implicit in code.  This has two benefits:

    1. The graph node can check whether a subtask's dependencies are satisfied
       before running it, making the ordering self-documenting.
    2. In a future phase where subtasks can run in parallel, the scheduler
       already has the information it needs to figure out which subtasks are
       unblocked without any additional work.

    WHY status IS ON THE SUBTASK (not just in AgentState)
    ------------------------------------------------------
    Storing the status on the Subtask itself (and therefore inside the
    ExecutionPlan that lives in AgentState) means the entire plan — including
    live progress — is captured in one serialisable object.  When we persist
    the plan to PostgreSQL as JSON, a single write captures the full snapshot
    of what ran and what the outcome was.
    """
    id: str = Field(..., description="Unique subtask identifier, e.g. 'subtask_1'")
    description: str = Field(..., description="What this subtask should do")
    assigned_agent: str = Field(
        ...,
        description="One of: researcher, analyst, writer",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of subtask IDs that must complete first",
    )
    # Default to PENDING; the specialist node flips this to RUNNING / COMPLETED / FAILED
    status: str = Field(default=SubtaskStatus.PENDING)


# ---------------------------------------------------------------------------
# ExecutionPlan — the Supervisor's structured output
# ---------------------------------------------------------------------------

class ExecutionPlan(BaseModel):
    """
    The Supervisor's decomposition of the user's task into ordered subtasks.

    WHY THIS IS A PYDANTIC MODEL (not a plain dict)
    ------------------------------------------------
    The Supervisor calls `call_llm(..., response_model=ExecutionPlan)`.
    LangChain's `.with_structured_output()` uses this class to:

      a) Generate a JSON schema that is sent to the OpenAI API as a
         function/tool definition, constraining what the model can output.
      b) Parse and validate the response, raising a clear error if the
         model returns something that doesn't fit the schema.

    Without this, the LLM might return free text or a JSON blob with wrong
    field names, and we'd have a fragile string-parsing step instead of a
    type-safe object.

    WHY reasoning IS INCLUDED
    -------------------------
    Asking the model to produce a `reasoning` field is a prompting technique
    called "chain-of-thought with structured output".  By forcing the LLM to
    articulate WHY it chose this plan before outputting the subtasks, the model
    reasons more carefully and produces better-quality subtask descriptions.
    The field is also valuable for debugging: if the plan is wrong, the
    reasoning field usually tells you exactly where the model's thinking went
    off track.
    """
    subtasks: list[Subtask] = Field(
        ...,
        description="Ordered list of subtasks to execute",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this plan was chosen",
    )


# ---------------------------------------------------------------------------
# ReviewResult — the Reviewer's structured output
# ---------------------------------------------------------------------------

class ReviewResult(BaseModel):
    """
    The Reviewer's evaluation of the Writer's report.

    WHY A NUMERIC SCORE ALONGSIDE A BOOLEAN
    ----------------------------------------
    The `approved` boolean is what the graph router actually acts on — it
    decides whether to end or retry.  The `score` (1-5) is retained because:

    1. It gives the writer (on retry) a sense of how far off the report was,
       not just a binary pass/fail signal.
    2. It is stored in logs and future dashboards can visualise quality
       trends over time without having to re-evaluate old outputs.
    3. It forces the Reviewer LLM to make a more deliberate judgement rather
       than just outputting True/False, which tends to produce better feedback.

    WHY ge=1, le=5 CONSTRAINTS
    --------------------------
    Pydantic Field validators (`ge`, `le`) are enforced at parse time, which
    means if the LLM returns a score of 0 or 6, the structured output call
    will raise a ValidationError immediately rather than silently accepting a
    bad value that would confuse downstream logic.

    WHY feedback IS ALWAYS REQUIRED (even when approved)
    -----------------------------------------------------
    Making `feedback` a required string (not Optional) means the model always
    produces it.  When the report is approved the feedback is an empty string.
    This avoids a None-check every time the field is used and makes the
    contract with the Writer simpler: "feedback is always a string; act on it
    if non-empty."
    """
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Quality score 1-5 (5 = excellent)",
    )
    approved: bool = Field(..., description="True if the output is acceptable")
    feedback: str = Field(
        ...,
        description="Specific improvement feedback if not approved, else empty string",
    )


# ---------------------------------------------------------------------------
# AgentState — the single object that flows through every graph node
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The shared mutable state that LangGraph passes from node to node.

    Each node receives the full current state as a read-only input and
    returns a dict of only the keys it wants to update.  LangGraph merges
    those updates into the state before calling the next node.

    FIELD-BY-FIELD RATIONALE
    ------------------------
    task_id / original_task
        Immutable identity fields set at graph entry and never changed.
        Every node can look up the original question without having to
        carry a reference to external context.

    execution_plan
        Written once by the Supervisor, then mutated in place by the
        specialist_execution node as it flips subtask statuses to RUNNING /
        COMPLETED / FAILED.  Storing the whole plan (rather than just the
        subtask outputs) means the caller can inspect the full plan shape
        even before all subtasks finish.

    subtask_results
        A dict keyed by subtask_id, holding each specialist's text output.
        Keying by subtask_id (not agent name) means the graph correctly
        handles future plans where the same agent type appears more than once.
        This dict also acts as a simple cache: on a writer retry, only the
        writer's entry is evicted; researcher and analyst results survive and
        do not need to be re-computed.

    current_subtask_index
        Reserved for future use when subtasks become individually addressable.
        Not actively used in Phase 1 (specialist_execution drives sequencing
        internally) but keeping it in state costs nothing and avoids a
        schema migration later.

    review_feedback / retry_count
        Together these implement the review loop.  `review_feedback` is set
        by the Reviewer when it rejects a report; the Writer reads it on the
        next pass to know what to improve.  `retry_count` is the guard that
        prevents infinite loops — once it reaches MAX_RETRIES the graph exits
        regardless of score.

    final_output
        The canonical "done" signal.  The Reviewer writes this when it accepts
        a report (or when retries are exhausted).  The graph router checks this
        field: if it is set, the run ends; if None, a retry is attempted.
        The FastAPI endpoint reads this to determine task status and to return
        the result to the caller.

    errors
        Annotated with `operator.add` so that LangGraph concatenates error
        lists from multiple nodes rather than overwriting them.  This means
        a partial failure (e.g. the researcher fails but the graph continues)
        is fully visible in the final state, rather than being silently
        discarded when the next node writes its own error list.
    """
    task_id: str
    original_task: str

    # Written by supervisor_planning
    execution_plan: Optional[ExecutionPlan]

    # Written and cached by specialist_execution
    subtask_results: dict[str, Any]   # subtask_id → specialist output text
    current_subtask_index: int        # reserved for future per-subtask routing

    # Written by reviewer_validation; read by writer on retry
    review_feedback: Optional[str]
    retry_count: int

    # Written by reviewer_validation when the report is accepted
    final_output: Optional[str]

    # Accumulated (not overwritten) across all nodes via operator.add reducer
    errors: Annotated[list[str], operator.add]

    # Written by memory_retrieval_node, read by supervisor_planning.
    # Stored as a plain dict (MemoryContext.model_dump()) so it survives
    # LangGraph's JSON serialisation without importing memory models here.
    # None means either no relevant memories were found, or retrieval failed.
    memory_context: Optional[dict]  # MemoryContext.model_dump()
