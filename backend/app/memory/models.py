"""
memory/models.py — Pydantic data contracts for the Phase 2A memory system.

WHY THESE ARE SEPARATE FROM graph/state.py
-------------------------------------------
graph/state.py defines the contracts that flow through LangGraph nodes
(AgentState, ExecutionPlan, etc.).  This file defines the contracts for
the memory subsystem — what gets stored in ChromaDB and what gets returned
to callers.  Keeping them separate means:

1. Importing from app.memory.models never pulls in LangGraph or graph state
   machinery — memory components remain independently testable.
2. The two sets of models can evolve at different rates.  Adding a field
   to MemoryDocument does not require a migration of AgentState.

WHY Pydantic v2 (not dataclasses)
-----------------------------------
Pydantic v2 gives us:
- JSON serialisation/deserialisation for free (used when storing/retrieving
  from Redis as JSON strings).
- Field-level validation (e.g. `ge=0.0, le=1.0` for confidence_score).
- IDE-friendly type inference with no extra boilerplate.
- model_dump() / model_validate() for clean dict round-trips.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryDocument(BaseModel):
    """
    A single document retrieved from ChromaDB semantic memory.

    WHY similarity_score IS OPTIONAL
    ---------------------------------
    When fetching a memory by exact ID (get_memory_by_id), there is no query
    vector to compare against, so no similarity score is computed.  Making
    the field Optional avoids returning a meaningless sentinel (e.g. -1.0)
    that callers would have to filter out.
    """

    id: str = Field(..., description="Unique document identifier within ChromaDB")
    content: str = Field(..., description="The full text content of the memory")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs stored alongside the document",
    )
    collection: str = Field(
        ...,
        description="ChromaDB collection name this document lives in",
    )
    similarity_score: float | None = Field(
        default=None,
        description=(
            "Cosine similarity to the query vector (0–1). "
            "None when retrieved by exact ID."
        ),
    )


class TaskMemorySummary(BaseModel):
    """
    A condensed summary of a completed task, stored as a ChromaDB document.

    WHY NOT STORE THE FULL FINAL OUTPUT
    -------------------------------------
    Full outputs can be thousands of tokens.  Storing them in ChromaDB would:
    a) Inflate embedding costs (embedding very long texts degrades quality).
    b) Return huge payloads on every similarity query.
    We store a 500-character preview of the output plus structured metadata
    so the semantic search works well while the full output remains in
    PostgreSQL where it belongs.
    """

    task_description: str = Field(
        ...,
        description="The user's original task string",
    )
    execution_plan_summary: str = Field(
        ...,
        description="Brief summary of the plan the Supervisor produced",
    )
    outcome: str = Field(
        ...,
        description="'success' or 'failed' — the final task status",
    )
    agents_used: list[str] = Field(
        default_factory=list,
        description="List of agent names that participated (e.g. researcher, writer)",
    )
    key_learnings: str = Field(
        default="",
        description=(
            "Free-text summary of what worked, what failed, and notable insights. "
            "Used to improve future task planning via memory retrieval."
        ),
    )


class ApproachSummary(BaseModel):
    """
    A description of a strategy that was attempted for a class of problems.

    WHY confidence_score (not just outcome)
    ----------------------------------------
    A boolean success/failure flag would lose the nuance of partial successes
    (e.g. the approach worked but produced mediocre quality).  confidence_score
    lets the retrieval layer rank approaches by expected effectiveness rather
    than just returning everything that succeeded.
    """

    problem_type: str = Field(
        ...,
        description=(
            "A short label classifying what kind of problem this approach addresses, "
            "e.g. 'market-research', 'code-explanation', 'competitive-analysis'"
        ),
    )
    approach_description: str = Field(
        ...,
        description="A human-readable description of the strategy or approach taken",
    )
    outcome: str = Field(
        ...,
        description="'success', 'partial', or 'failed'",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that this approach will succeed on a similar problem",
    )


class MemoryContext(BaseModel):
    """
    The aggregated memory context returned to the Supervisor before planning.

    WHY has_relevant_memories AS A FLAG
    -------------------------------------
    Without this flag, the caller would have to inspect all four list fields
    and check if any are non-empty — four equality checks instead of one.
    has_relevant_memories=False is also a meaningful signal: the Supervisor
    should not adjust its plan based on memory if there is none to consider.

    WHY FOUR SEPARATE LISTS (not one flat list)
    ---------------------------------------------
    The Supervisor prompt needs to cite memories with context: "Previous tasks
    like this used approach X" is different from "User prefers Y".  Mixing
    everything into one flat list would lose that semantic structure and force
    the prompt to re-infer it.
    """

    similar_tasks: list[MemoryDocument] = Field(
        default_factory=list,
        description="Past tasks semantically similar to the current one",
    )
    effective_approaches: list[MemoryDocument] = Field(
        default_factory=list,
        description="Strategies that succeeded for similar problem types",
    )
    failed_approaches: list[MemoryDocument] = Field(
        default_factory=list,
        description="Strategies that failed — helps the Supervisor avoid repeating mistakes",
    )
    user_preferences: list[MemoryDocument] = Field(
        default_factory=list,
        description="Stored user preferences that should influence the plan",
    )
    domain_context: list[MemoryDocument] = Field(
        default_factory=list,
        description="Domain-specific facts relevant to the current task",
    )
    has_relevant_memories: bool = Field(
        default=False,
        description=(
            "True if at least one of the list fields is non-empty. "
            "Convenience flag so callers don't have to inspect all five lists."
        ),
    )


class MemoryStats(BaseModel):
    """
    Aggregate statistics about the contents of the semantic memory store.

    Useful for diagnostics, dashboards, and understanding how the memory
    system is being used over time.
    """

    total_memories: int = Field(
        ...,
        description="Total number of documents across all ChromaDB collections",
    )
    by_collection: dict[str, int] = Field(
        default_factory=dict,
        description="Per-collection document counts, e.g. {'task_memories': 42, ...}",
    )
    avg_importance: float = Field(
        default=0.0,
        description="Average importance_score across all documents (0–1 scale)",
    )
    most_accessed: list[str] = Field(
        default_factory=list,
        description="IDs of the most-frequently-accessed memory documents",
    )
