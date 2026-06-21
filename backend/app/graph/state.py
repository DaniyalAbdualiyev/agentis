"""
LangGraph shared state definition.

AgentState flows through every node in the graph.  All fields are optional
(or have defaults) so that early nodes don't need to pre-populate later ones.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import operator


# ---------------------------------------------------------------------------
# Pydantic models (also used by Supervisor structured output)
# ---------------------------------------------------------------------------

class SubtaskStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class Subtask(BaseModel):
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
    status: str = Field(default=SubtaskStatus.PENDING)


class ExecutionPlan(BaseModel):
    subtasks: list[Subtask] = Field(
        ...,
        description="Ordered list of subtasks to execute",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this plan was chosen",
    )


class ReviewResult(BaseModel):
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Quality score 1-5 (5 = excellent)",
    )
    approved: bool = Field(..., description="True if the output is acceptable")
    feedback: str = Field(
        ...,
        description="Specific improvement feedback if not approved, else empty",
    )


# ---------------------------------------------------------------------------
# LangGraph TypedDict state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    task_id: str
    original_task: str

    # Supervisor output
    execution_plan: Optional[ExecutionPlan]

    # Specialist execution tracking
    subtask_results: dict[str, Any]          # subtask_id -> output string
    current_subtask_index: int

    # Reviewer
    review_feedback: Optional[str]
    retry_count: int

    # Final
    final_output: Optional[str]

    # Error tracking — accumulated across nodes
    errors: Annotated[list[str], operator.add]
