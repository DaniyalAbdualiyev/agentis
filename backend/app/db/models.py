"""
SQLAlchemy ORM models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.engine import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Task(Base):
    __tablename__ = "tasks"

    # Possible status values:
    #   pending               — task submitted, not yet running
    #   running               — graph currently executing
    #   completed             — graph finished, final_output set
    #   failed                — graph crashed, no final_output
    #   awaiting_human_review — graph paused at HITL interrupt (Phase 3)
    #   rejected              — human rejected the output (Phase 3)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    original_task: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    execution_plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    langsmith_trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Phase 3: populated when check_escalation_node triggers human review.
    escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Phase 4: aggregated observability metrics, filled in when the task
    # completes (sum of all ExecutionTrace rows for this task).  Defaulting to
    # zero (not NULL) means the dashboard never has to special-case a missing
    # value — a task that never ran an LLM call simply shows 0 cost.
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    subtask_logs: Mapped[list["SubtaskLog"]] = relationship(
        "SubtaskLog", back_populates="task", cascade="all, delete-orphan"
    )
    tool_call_logs: Mapped[list["ToolCallLog"]] = relationship(
        "ToolCallLog", back_populates="task", cascade="all, delete-orphan"
    )
    human_review: Mapped["HumanReview | None"] = relationship(
        "HumanReview", back_populates="task", uselist=False, cascade="all, delete-orphan"
    )
    execution_traces: Mapped[list["ExecutionTrace"]] = relationship(
        "ExecutionTrace", back_populates="task", cascade="all, delete-orphan"
    )


class SubtaskLog(Base):
    __tablename__ = "subtask_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    subtask_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subtask_description: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_agent: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    task: Mapped["Task"] = relationship("Task", back_populates="subtask_logs")


class ToolCallLog(Base):
    __tablename__ = "tool_call_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True
    )
    subtask_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    inputs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    outputs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    task: Mapped["Task"] = relationship("Task", back_populates="tool_call_logs")


class HumanReview(Base):
    """
    Phase 3: records every human-in-the-loop review decision.

    Created by check_escalation_node when any escalation trigger fires.
    Updated by POST /reviews/{task_id}/decision when the human submits.

    WHY A SEPARATE TABLE (not columns on Task)
    ------------------------------------------
    Task already has 8 columns; adding 6 more HITL-specific columns would
    make it hard to read and harder to query for review dashboards.
    A separate table also makes it possible in the future to support
    multiple reviewers per task (one-to-many) without a Task schema change.
    """
    __tablename__ = "human_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    escalation_reason: Mapped[str] = mapped_column(Text, nullable=False)
    # Snapshot of the AI-generated output that will be shown to the reviewer.
    output_shown_to_human: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Filled in by the human after reviewing.
    human_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    human_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str] = mapped_column(String(128), default="manager")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    task: Mapped["Task"] = relationship("Task", back_populates="human_review")


class ExecutionTrace(Base):
    """
    Phase 4: one row per LangGraph node execution.

    WHY A SEPARATE TABLE (not columns on Task or SubtaskLog)
    --------------------------------------------------------
    A single task run produces one trace row per node — and on a writer retry
    the same node (specialist_execution / reviewer_validation) runs twice, so
    there is a genuine one-to-many relationship between a task and its node
    executions.  Storing these as rows lets us:
      - Reconstruct the full timeline in execution order (ORDER BY started_at).
      - Aggregate cost/latency per node across ALL tasks for the dashboard
        without parsing a JSON blob.
      - Keep large llm_prompt / llm_response text out of the hot Task row.

    WHY status IS A STRING (not an Enum)
    -------------------------------------
    Matches the existing convention used by SubtaskLog.status and Task.status:
    plain strings survive JSON round-trips and are trivially filterable in SQL.
    Allowed values: "success", "failed", "escalated".
    """
    __tablename__ = "execution_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    node_name: Mapped[str] = mapped_column(String(64), nullable=False)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    # 0 when the node made no LLM call (e.g. memory_retrieval, writer_retry).
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    tool_calls_count: Mapped[int] = mapped_column(Integer, default=0)

    # "success" | "failed" | "escalated"
    status: Mapped[str] = mapped_column(String(32), default="success")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full LLM prompt/response, truncated at 10000 chars before insert.
    llm_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_response: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Any extra node-specific info (models used, retry_count, etc.).
    trace_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    task: Mapped["Task"] = relationship("Task", back_populates="execution_traces")
