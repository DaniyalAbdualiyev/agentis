"""
graph/graph.py — LangGraph state machine: the orchestration backbone.

WHY LANGGRAPH INSTEAD OF PLAIN PYTHON ASYNC
--------------------------------------------
We could orchestrate agents with a simple `async def run_pipeline()` that
calls researcher, then analyst, then writer in sequence.  LangGraph adds
four things that plain Python does not give us for free:

1. AUTOMATIC LANGSMITH TRACING — every node execution appears as a named
   span in LangSmith automatically.  No manual instrumentation needed.

2. STATE CHECKPOINTING — LangGraph can persist state to a store between
   nodes, enabling resumable runs and human-in-the-loop pauses (Phase 3).

3. DECLARATIVE CONTROL FLOW — the conditional retry edge is declared once
   in `build_graph()` and is visible to anyone reading the graph definition.
   In plain Python the retry logic would be buried inside an if/else block
   inside `run_pipeline()`, invisible from the outside.

4. COMPOSABILITY — in Phase 2 we can add new nodes (e.g. a memory retrieval
   node, a parallel search node) by adding `graph.add_node()` and
   `graph.add_edge()` calls, without refactoring existing nodes.

FLOW DIAGRAM (Phase 3)
-----------------------
  task_intake
       │
       ▼
  memory_retrieval              ← Phase 2B: queries ChromaDB for similar past tasks
       │
       ▼
  supervisor_planning          ← Supervisor decomposes task → ExecutionPlan
       │
       ▼
  specialist_execution         ← researcher → analyst → writer (sequential)
       │
       ▼
  reviewer_validation          ← Reviewer scores 1-5, sets approved flag
       │
       ├─ rejected AND retries < 2
       │        │
       │        ▼
       │   writer_retry         ← evicts stale writer output from cache
       │        │
       │   specialist_execution ← re-runs ONLY the writer
       │        │
       │   reviewer_validation  ← second review
       │
       └─ approved (or retries >= 2)
                │
                ▼
         check_escalation      ← Phase 3: evaluate all 3 HITL triggers
                │
                ├─ no escalation ───────────────────────────────────────► END
                │
                └─ escalation needed
                         │
                         ▼
                   human_review         ← PAUSES via interrupt()
                         │               (resumes when POST /reviews/{id}/decision)
                         ▼
                        END

WHY check_escalation IS A SEPARATE NODE (not inside reviewer_validation)
------------------------------------------------------------------------
Keeping escalation detection separate means:
  - The Reviewer's job stays focused: score, approve/reject, give feedback.
  - The escalation policy (keywords, quality thresholds, agent flags) lives
    in one place and can be changed without touching the Reviewer.
  - LangSmith traces show a distinct "check_escalation" span so it's easy
    to see why a task was escalated vs simply approved.

WHY interrupt() IS USED IN human_review (not interrupt_before on the node)
--------------------------------------------------------------------------
`interrupt_before=["human_review"]` would pause EVERY run that reaches
`human_review`, but it gives no way to inject a payload (e.g. the draft
output, escalation reason) for the reviewer to see.  Using `interrupt(payload)`
inside the node passes structured context to the interrupt event, which
LangGraph Studio and the API can surface to the human reviewer.

WHY MAX_RETRIES = 2 (not 1 or 3)
---------------------------------
One retry gives the writer one chance to improve — that's too little margin
for edge cases where the first review is overly strict.  Three or more
retries risk a feedback loop that never converges and burns API budget.
Two is the pragmatic middle ground: enough to catch a genuinely improvable
report, not so many that a low-quality pipeline never terminates.
"""
from __future__ import annotations

import asyncio
import structlog
import time
import uuid
from datetime import datetime, timezone

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents.reviewer import run_reviewer
from app.agents.supervisor import run_supervisor
from app.agents.specialists.researcher import run_researcher
from app.agents.specialists.analyst import run_analyst
from app.agents.specialists.writer import run_writer
from app.graph.state import AgentState, SENSITIVE_KEYWORDS, Subtask, SubtaskStatus
from app.observability.tracing import record_node_metadata, traced_node

log = structlog.get_logger(__name__)

# Maximum number of writer retries before the graph accepts whatever it has.
# This is a hard ceiling to prevent infinite loops regardless of review scores.
MAX_RETRIES = 2

# Safety ceiling on how long a subtask will poll waiting for its dependencies
# before giving up and failing.  This should never trigger in practice — it
# only guards against a malformed plan (e.g. depends_on referencing a subtask
# id that can never complete) hanging the graph forever.
DEPENDENCY_WAIT_TIMEOUT_SEC = 300
DEPENDENCY_POLL_INTERVAL_SEC = 0.05


# ---------------------------------------------------------------------------
# Node implementations
#
# WHY EACH NODE IS A THIN WRAPPER
# --------------------------------
# Each graph node is a thin async function that delegates to a dedicated
# agent module (e.g. run_supervisor, run_researcher).  This separation keeps
# the graph file focused on wiring and control flow, while each agent module
# contains all the prompting and business logic for that role.
# If you want to understand what the Supervisor does, read supervisor.py —
# you do not need to understand the graph wiring to do so.
# ---------------------------------------------------------------------------

async def memory_retrieval_node(state: AgentState) -> dict:
    """
    Queries long-term memory for context relevant to the current task.
    Runs before the Supervisor so planning is informed by past experience.

    WHY THIS IS A SEPARATE NODE (not inside supervisor_planning)
    ------------------------------------------------------------
    Keeping memory retrieval in its own node means:
    - It has its own named span in LangSmith traces, so you can see exactly
      how long memory retrieval took and whether it found anything.
    - It can fail independently without affecting the Supervisor — if ChromaDB
      is down, the Supervisor still runs with memory_context=None.
    - It can be easily disabled/bypassed for testing by changing graph edges
      without touching supervisor.py at all.
    - Future phases can add more pre-planning nodes (e.g. a task classifier)
      without touching the Supervisor's code.

    GRACEFUL DEGRADATION
    --------------------
    All exceptions are caught here.  If retrieval fails for any reason (network
    timeout, ChromaDB unavailable, embedding API error), the node returns
    {"memory_context": None} and logs a warning.  The Supervisor proceeds
    without memory context — exactly as it did before Phase 2B was added.
    """
    task = state.get("original_task", "")
    if not task:
        return {"memory_context": None}

    try:
        from app.memory.retrieval import PlanningMemoryRetriever
        retriever = PlanningMemoryRetriever()
        context = await retriever.retrieve_context(task, user_id="default_user")
        if context.has_relevant_memories:
            log.info(
                "memory_context_found",
                similar_tasks=len(context.similar_tasks),
                effective_approaches=len(context.effective_approaches),
            )
            return {"memory_context": context.model_dump()}
        else:
            log.info("no_relevant_memories")
            return {"memory_context": None}
    except Exception as exc:
        log.warning("memory_retrieval_failed", error=str(exc))
        return {"memory_context": None}  # degrade gracefully


async def task_intake_node(state: AgentState) -> dict:
    """
    Entry point node — validates that a non-empty task was submitted and
    initialises any pipeline-start fields that were not provided by the caller.

    WHY THIS NODE EXISTS AS A SEPARATE STEP
    ----------------------------------------
    We could validate the task inside the supervisor node, but having a
    dedicated intake node means:
    - The validation logic has its own named span in LangSmith traces,
      making it easy to see where a run failed if the task was empty.
    - In Phase 2, this is the natural place to add input sanitisation,
      token-length checks, or a content-policy filter before spending any
      LLM budget.
    - It returns `{}` (no state changes) on success, which is the LangGraph
      idiom for "I ran, everything is fine, pass the state through unchanged."

    WHY task_id IS AUTO-GENERATED HERE
    -----------------------------------
    AgentState.task_id is NotRequired so LangGraph Studio's input form only
    shows the user the one field they care about (original_task).  If no
    task_id was supplied (the Studio path), this node generates a fresh UUID.
    The FastAPI path (routers/tasks.py) always supplies a task_id, so the
    branch below is never reached in production — it is purely a Studio
    convenience.  Similarly, subtask_results / retry_count / errors are
    seeded with their zero-values here so downstream nodes never see KeyError.
    """
    task = state.get("original_task", "").strip()
    if not task:
        # Writing to `errors` appends (not overwrites) because of the
        # operator.add reducer declared in AgentState.
        return {"errors": ["No task provided"]}

    # Seed fields that the FastAPI caller normally provides but Studio omits.
    updates: dict = {}
    if not state.get("task_id"):
        updates["task_id"] = str(uuid.uuid4())
    if "subtask_results" not in state:
        updates["subtask_results"] = {}
    if "retry_count" not in state:
        updates["retry_count"] = 0
    if "current_subtask_index" not in state:
        updates["current_subtask_index"] = 0
    # errors uses operator.add (list concat) — returning [] is safe: it appends
    # nothing, but it also guarantees the field exists for downstream readers.
    if "errors" not in state:
        updates["errors"] = []

    log.info("task_intake", task_id=updates.get("task_id") or state.get("task_id"), task=task[:80])
    return updates


async def supervisor_planning_node(state: AgentState) -> dict:
    """
    Thin wrapper — delegates fully to agents/supervisor.py.

    WHY THE WRAPPER EXISTS (rather than using run_supervisor directly as a node)
    ----------------------------------------------------------------------------
    LangGraph nodes can be any callable, so `run_supervisor` could be
    registered directly.  The wrapper is kept because:
    - It makes the graph file self-consistent: all five node names follow the
      same `*_node` naming convention, making them easy to spot.
    - It is a convenient place to add cross-cutting logic later (timing,
      per-node retries, auth checks) without touching the agent module.
    """
    return await run_supervisor(state)


# Maps assigned_agent -> the specialist coroutine function.  Extend this dict
# (and _dispatch_specialist below) when a new specialist type is added; the
# scheduler in specialist_execution_node itself stays generic and never needs
# to change.
_SPECIALIST_RUNNERS = {
    "researcher": run_researcher,
    "analyst":    run_analyst,
    "writer":     run_writer,
}


async def _dispatch_specialist(
    subtask: Subtask,
    dependency_output: str,
    review_feedback: str | None,
    task_id: str | None,
) -> str:
    """
    Call the right specialist function for `subtask.assigned_agent`, feeding it
    the concatenated output of whatever it depends on.

    WHY THIS IS SEPARATE FROM _run_subtask
    ----------------------------------------
    Each specialist has a different signature (researcher takes no upstream
    input, analyst takes raw_findings, writer takes structured_analysis +
    reviewer_feedback).  Isolating that mapping here keeps the concurrency
    logic in _run_subtask free of per-agent argument wiring.
    """
    runner = _SPECIALIST_RUNNERS[subtask.assigned_agent]
    if subtask.assigned_agent == "researcher":
        return await runner(
            subtask_description=subtask.description,
            task_id=task_id,
            subtask_id=subtask.id,
        )
    if subtask.assigned_agent == "analyst":
        return await runner(
            subtask_description=subtask.description,
            raw_findings=dependency_output,
            task_id=task_id,
            subtask_id=subtask.id,
        )
    # "writer" — the only role that ever receives reviewer_feedback (set on retries).
    return await runner(
        subtask_description=subtask.description,
        structured_analysis=dependency_output,
        reviewer_feedback=review_feedback,
        task_id=task_id,
        subtask_id=subtask.id,
    )


async def _run_subtask(
    subtask: Subtask,
    plan_ids: set[str],
    results: dict[str, str],
    completed: dict[str, str],
    failed: set[str],
    review_feedback: str | None,
    task_id: str | None,
    errors: list[str],
    durations: dict[str, float],
) -> None:
    """
    Run ONE subtask once its dependencies are satisfied, as part of an
    asyncio.gather() fan-out launched by specialist_execution_node.

    HOW THE DEPENDENCY WAIT WORKS
    ------------------------------
    Every subtask (regardless of depends_on) is launched into the gather()
    call at the same time.  A subtask with depends_on=[] passes its own wait
    check immediately (an empty list is vacuously satisfied) and starts
    running right away — this is what gives independent subtasks true
    concurrency.  A subtask with dependencies polls `completed`/`failed`
    (populated by whichever coroutine finishes each dependency) until every
    dependency id it needs has resolved one way or the other, then proceeds.
    This means subtask_2 -> depends_on=["subtask_1"] behaves identically to
    the old strictly-sequential code: it cannot start until subtask_1 is done.

    HOW FAILURES CASCADE (matching the old early-return behaviour)
    ----------------------------------------------------------------
    The old sequential code returned immediately when researcher/analyst
    failed, so a downstream specialist never spent LLM budget on doomed input.
    Here, a subtask whose dependency ends up in `failed` marks itself failed
    too (without calling its specialist) — so the "don't waste a call on
    input that will never arrive" property still holds even though everything
    is scheduled concurrently.

    HOW THE RETRY CACHE STILL WORKS
    ---------------------------------
    `results` (passed in from state) already contains outputs for
    already-completed subtasks (e.g. researcher + analyst survive a writer
    retry because writer_retry_node only evicts the writer's entry).  If this
    subtask's id is already in `results`, we skip straight to seeding
    `completed` and return — exactly as the old "if id not in results" guard did.
    """
    deps = subtask.depends_on
    waited = 0.0
    while not all(d in completed or d in failed or d not in plan_ids for d in deps):
        if waited >= DEPENDENCY_WAIT_TIMEOUT_SEC:
            failed.add(subtask.id)
            subtask.status = SubtaskStatus.FAILED
            errors.append(
                f"{subtask.assigned_agent} ({subtask.id}) timed out waiting on "
                f"dependencies {deps}"
            )
            return
        await asyncio.sleep(DEPENDENCY_POLL_INTERVAL_SEC)
        waited += DEPENDENCY_POLL_INTERVAL_SEC

    if any(d in failed for d in deps):
        # A dependency failed — this subtask can never produce valid output.
        failed.add(subtask.id)
        subtask.status = SubtaskStatus.FAILED
        return

    if subtask.id in results:
        # Cached from a previous pass (writer-retry path) — nothing to run.
        completed[subtask.id] = results[subtask.id]
        return

    dependency_output = "\n\n".join(completed[d] for d in deps if d in completed)

    log.info(f"running_{subtask.assigned_agent}", subtask_id=subtask.id)
    subtask.status = SubtaskStatus.RUNNING
    started = time.perf_counter()
    try:
        output = await _dispatch_specialist(subtask, dependency_output, review_feedback, task_id)
        subtask.status = SubtaskStatus.COMPLETED
        results[subtask.id] = output
        completed[subtask.id] = output
        log.info(f"{subtask.assigned_agent}_completed", subtask_id=subtask.id)
    except Exception as exc:
        subtask.status = SubtaskStatus.FAILED
        failed.add(subtask.id)
        log.error(f"{subtask.assigned_agent}_node_failed", subtask_id=subtask.id, error=str(exc))
        errors.append(f"{subtask.assigned_agent.capitalize()} failed: {exc}")
    finally:
        durations[subtask.id] = time.perf_counter() - started


def _parallel_groups(subtasks: list[Subtask]) -> list[list[str]]:
    """
    Compute the dependency "waves" of a plan purely for trace/observability
    purposes — group 0 is every subtask with no unresolved dependency, group 1
    is every subtask whose dependencies are all in group 0, and so on.

    This does NOT drive execution (that is fully dependency-driven inside
    _run_subtask); it only answers "what could have run together" so the
    specialist_execution trace can show parallel_groups, e.g.
    [["subtask_1"], ["subtask_2", ...], ["subtask_3"]] for a sequential chain,
    or [["subtask_1", "subtask_2"], ["subtask_3"]] when two subtasks are
    independent and only the third depends on both of them.
    """
    order = {s.id: i for i, s in enumerate(subtasks)}
    by_id = {s.id: s for s in subtasks}
    resolved: set[str] = set()
    remaining = set(by_id)
    groups: list[list[str]] = []

    while remaining:
        wave = [
            sid for sid in remaining
            if all(d in resolved or d not in by_id for d in by_id[sid].depends_on)
        ]
        if not wave:
            # Unresolvable (e.g. a dependency cycle) — dump the rest as a
            # final group rather than looping forever; this should never
            # happen with a well-formed plan.
            wave = list(remaining)
        wave.sort(key=lambda sid: order[sid])
        groups.append(wave)
        resolved.update(wave)
        remaining -= set(wave)

    return groups


async def specialist_execution_node(state: AgentState) -> dict:
    """
    Runs every subtask in the plan, launching all of them concurrently via
    asyncio.gather() and letting each one wait internally for its own
    depends_on to resolve (see _run_subtask).

    WHY THIS REPLACED THE OLD SEQUENTIAL researcher -> analyst -> writer LOOP
    ----------------------------------------------------------------------------
    The Supervisor can mark subtasks as independent by giving them
    depends_on=[].  Launching every subtask up front (instead of awaiting
    each one in turn) means two independent subtasks actually overlap in wall
    time — asyncio.gather() schedules them onto the same event loop and they
    make progress while each other is awaiting I/O (LLM calls, web search).
    When every subtask depends on the previous one (the original researcher ->
    analyst -> writer chain), each one's dependency wait blocks until its
    predecessor finishes, so the observable behaviour is identical to the old
    strictly-sequential code — this node is a strict superset, not a
    behavioural change, for existing plans.

    WHY subtask_results IS STILL THE CACHE (unchanged from Phase 1)
    -------------------------------------------------------------------
    `subtask_results` still acts as a cache keyed by subtask_id so a writer
    retry only re-runs the writer: researcher/analyst entries are already in
    `results`, so _run_subtask returns immediately for them without spending
    any LLM budget.
    """
    plan = state.get("execution_plan")
    if plan is None:
        return {"errors": ["specialist_execution: no execution plan"]}

    # Copy results so we can mutate safely — state is read-only in LangGraph nodes.
    results: dict = dict(state.get("subtask_results", {}))
    # review_feedback is None on the first pass; set on retries by the Reviewer.
    review_feedback = state.get("review_feedback")
    task_id = state.get("task_id")

    plan_ids = {s.id for s in plan.subtasks}
    completed: dict[str, str] = dict(results)  # seed with anything already cached
    failed: set[str] = set()
    errors: list[str] = []
    durations: dict[str, float] = {}

    wall_start = time.perf_counter()
    await asyncio.gather(*(
        _run_subtask(
            subtask, plan_ids, results, completed, failed,
            review_feedback, task_id, errors, durations,
        )
        for subtask in plan.subtasks
    ))
    parallel_time_ms = int((time.perf_counter() - wall_start) * 1000)

    if failed:
        return {"errors": errors, "subtask_results": results, "execution_plan": plan}

    # Record how much concurrency actually helped on this run.  sequential_time_ms
    # is the sum of each subtask's own duration — i.e. how long this node would
    # have taken if every subtask had run one after another instead of
    # concurrently.  parallel_time_ms is the real wall-clock time for the whole
    # node.  Both land in this node's ExecutionTrace.metadata via
    # record_node_metadata (see observability/tracing.py).
    sequential_time_ms = int(sum(durations.values()) * 1000)
    speedup_factor = (
        round(sequential_time_ms / parallel_time_ms, 2) if parallel_time_ms else 1.0
    )
    record_node_metadata(
        parallel_groups=_parallel_groups(plan.subtasks),
        sequential_time_ms=sequential_time_ms,
        parallel_time_ms=parallel_time_ms,
        speedup_factor=speedup_factor,
    )

    return {
        "subtask_results": results,
        # Return the mutated plan so updated subtask statuses are visible in state.
        "execution_plan": plan,
    }


async def reviewer_validation_node(state: AgentState) -> dict:
    """
    Thin wrapper — delegates fully to agents/reviewer.py.

    The Reviewer either:
    - Sets `final_output` in state → graph router will choose "end"
    - Sets `review_feedback` and increments `retry_count` → router will retry
    """
    return await run_reviewer(state)


async def writer_retry_node(state: AgentState) -> dict:
    """
    Prepare state for a writer retry WITHOUT re-running researcher or analyst.

    HOW THE CACHE EVICTION WORKS
    ----------------------------
    `subtask_results` is a dict mapping subtask_id → output text.
    This node removes the writer's entry from that dict, then returns the
    modified dict to state.  When `specialist_execution` runs next, it finds
    researcher and analyst entries present (skips them) but the writer entry
    absent (runs the writer again).

    WHY WE ALSO RESET writer_subtask.status TO PENDING
    ---------------------------------------------------
    The `execution_plan.subtasks` objects carry status flags that are persisted
    to the database.  Resetting the writer's status to PENDING before the
    retry ensures the DB log correctly shows: PENDING → RUNNING → COMPLETED
    for the second attempt, rather than showing COMPLETED immediately followed
    by RUNNING (which would be confusing).
    """
    plan = state.get("execution_plan")
    if plan is None:
        return {"errors": ["writer_retry: no execution plan"]}

    writer_subtask = next(
        (s for s in plan.subtasks if s.assigned_agent == "writer"), None
    )
    if writer_subtask is None:
        return {"errors": ["writer_retry: no writer subtask"]}

    # Evict the stale writer output so specialist_execution re-runs it.
    results = dict(state.get("subtask_results", {}))
    results.pop(writer_subtask.id, None)
    writer_subtask.status = SubtaskStatus.PENDING

    log.info("writer_retry_prepared", retry_count=state.get("retry_count"))
    return {"subtask_results": results, "execution_plan": plan}


# ---------------------------------------------------------------------------
# Phase 3: Human-in-the-Loop helpers and nodes
# ---------------------------------------------------------------------------

def should_escalate(state: AgentState) -> tuple[bool, str]:
    """
    Single authoritative function that evaluates all 3 HITL escalation triggers.

    Returns (True, reason) if any trigger fires, else (False, "").

    WHY A STANDALONE FUNCTION (not inline in the node)
    ---------------------------------------------------
    Having all three triggers in one function makes the escalation policy
    easy to audit: one diff to this function tells you exactly what changed.
    It is also unit-testable in isolation — tests can call should_escalate()
    with a fake state dict without running the full graph.

    TRIGGER PRIORITY ORDER
    ----------------------
    1. Sensitive keywords in original_task (quickest check, no state needed)
    2. Low reviewer score after retries exhausted (quality gate)
    3. Any agent explicitly set requires_human_review = True in state

    Priority is low-to-high — if trigger 3 is set AND trigger 1 also fires,
    trigger 1's reason is used (the more specific escalation message wins).
    Since we return on first match, ordering here IS the priority ordering.
    """
    # Trigger 1: sensitive topic detected in original task text.
    task_lower = state.get("original_task", "").lower()
    found = [kw for kw in SENSITIVE_KEYWORDS if kw in task_lower]
    if found:
        return True, f"Sensitive topic detected: {', '.join(sorted(found))}"

    # Trigger 2: reviewer gave a low score AND all retries were already used.
    # retry_count is incremented by the Reviewer on each rejection; when
    # retries are exhausted the Reviewer force-accepts (sets final_output) but
    # retry_count is still >= 2 from the last rejection cycle.
    score = state.get("review_score")
    retry_count = state.get("retry_count", 0)
    if score is not None and score < 3 and retry_count >= 2:
        return True, (
            f"Low quality output after retries exhausted "
            f"(reviewer score {score}/5, retries={retry_count})"
        )

    # Trigger 3: any agent explicitly flagged this task for human review.
    if state.get("requires_human_review"):
        reason = state.get("escalation_reason") or "Agent explicitly requested human review"
        return True, reason

    return False, ""


async def check_escalation_node(state: AgentState) -> dict:
    """
    Phase 3: evaluate all 3 HITL triggers and, if escalation is needed,
    update the task status in PostgreSQL and set the escalation fields
    so that the conditional edge can route to human_review.

    GRACEFUL DEGRADATION
    --------------------
    The DB update is wrapped in try/except.  If PostgreSQL is unreachable,
    the node still sets requires_human_review=True in state so the graph
    routes correctly — the only cost is that the DB status is stale.

    WHY WE CREATE THE HumanReview RECORD HERE
    ------------------------------------------
    We create it here (not in the router) so the GET /reviews/pending
    endpoint can read escalation_reason and output_shown_to_human directly
    from the human_reviews table without querying the checkpointer.
    """
    escalate, reason = should_escalate(state)

    if not escalate:
        return {}  # nothing to do — route_after_escalation will go to END

    task_id = state.get("task_id")
    draft_output = state.get("final_output", "")

    log.info(
        "escalation_triggered",
        task_id=task_id,
        reason=reason[:120],
    )

    # Persist escalation status + create the HumanReview audit record.
    try:
        from app.db.engine import get_db_session
        from app.db.queries import create_human_review, update_task
        async with get_db_session() as session:
            await update_task(
                session,
                task_id,
                status="awaiting_human_review",
                escalation_reason=reason,
            )
            await create_human_review(
                session,
                task_id=task_id,
                escalation_reason=reason,
                output_shown_to_human=draft_output,
            )
    except Exception as db_exc:
        log.warning(
            "check_escalation_db_failed",
            task_id=task_id,
            error=str(db_exc),
        )

    return {
        "requires_human_review": True,
        "escalation_reason": reason,
    }


async def human_review_node(state: AgentState) -> dict:
    """
    Phase 3: pause the graph and wait for a human decision.

    On FIRST execution: interrupt() pauses the graph and checkpoints state.
    The caller (run_task_background) catches GraphInterrupt and returns.

    On RESUME: the POST /reviews/{task_id}/decision endpoint calls
      graph.ainvoke(Command(resume={...}), config={"configurable": {"thread_id": task_id}})
    LangGraph re-executes this node; interrupt() returns the resume payload
    instead of raising.  The node then processes the decision and updates
    the task status in PostgreSQL.

    WHY interrupt() IS USED HERE (not interrupt_before)
    ----------------------------------------------------
    Passing a payload to interrupt() lets us surface the draft output and
    escalation reason to whoever calls aget_state() on the checkpointed thread.
    interrupt_before gives no such mechanism.
    """
    task_id = state.get("task_id")
    draft_output = state.get("final_output", "")
    escalation_reason = state.get("escalation_reason", "")

    # ── First execution: pause here. Resume execution: returns human payload. ──
    human_input: dict = interrupt({
        "task_id": task_id,
        "draft_output": draft_output,
        "escalation_reason": escalation_reason,
        "message": "Awaiting human review decision (approved / edited / rejected)",
    })

    # ── Post-resume: process the human's decision ───────────────────────────
    decision: str = human_input.get("decision", "approved")
    feedback: str = human_input.get("feedback", "") or ""

    log.info(
        "human_review_decision_received",
        task_id=task_id,
        decision=decision,
        has_feedback=bool(feedback),
    )

    updates: dict = {
        "human_decision": decision,
        "human_feedback": feedback,
        "human_reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    if decision == "edited":
        # Human replaced the output — use their text as final_output.
        updates["final_output"] = feedback
    elif decision == "rejected":
        # Human rejected — clear final_output so the caller knows.
        updates["final_output"] = None

    # Persist the updated task status and final output to PostgreSQL.
    try:
        from app.db.engine import get_db_session
        from app.db.queries import update_human_review, update_task
        from datetime import datetime as dt

        now = dt.now(timezone.utc)
        new_status = "rejected" if decision == "rejected" else "completed"
        new_output = updates.get("final_output", draft_output)

        async with get_db_session() as session:
            await update_task(
                session,
                task_id,
                status=new_status,
                final_output=new_output,
                completed_at=now,
            )
            await update_human_review(
                session,
                task_id,
                human_decision=decision,
                human_feedback=feedback or None,
                reviewed_at=now,
            )
    except Exception as db_exc:
        log.warning(
            "human_review_db_failed",
            task_id=task_id,
            error=str(db_exc),
        )

    return updates


# ---------------------------------------------------------------------------
# Routing functions
#
# WHY A SEPARATE FUNCTION (not inline lambda)
# --------------------------------------------
# LangGraph accepts any callable as a router, but a named function:
# - Is unit-testable in isolation (see tests/test_pipeline.py::TestRouteAfterReview)
# - Shows up with a meaningful name in LangSmith traces instead of "<lambda>"
# - Makes the routing logic readable without having to decode a one-liner
# ---------------------------------------------------------------------------

def route_after_review(state: AgentState) -> str:
    """
    Routing function called after every reviewer_validation run.

    Decision logic:
    - If `final_output` is set, the Reviewer accepted → "check_escalation"
      (Phase 3: we go to check_escalation first, not directly to END)
    - If `retry_count` is still below MAX_RETRIES → "retry_writer"
    - Otherwise (retries exhausted) → "check_escalation"

    WHY "check_escalation" INSTEAD OF "end"
    ----------------------------------------
    In Phase 3 the graph never goes directly from reviewer_validation to END.
    check_escalation evaluates whether human review is needed; if not it
    passes straight through to END via route_after_escalation.  This keeps
    the escalation decision out of the Reviewer and centralised in one node.
    """
    final_output = state.get("final_output")
    retry_count = state.get("retry_count", 0)

    if final_output is not None:
        # Reviewer accepted — pass to escalation check before ending.
        return "check_escalation"

    if retry_count < MAX_RETRIES:
        # Reviewer rejected but we still have retries available.
        return "retry_writer"

    # Retries exhausted — accept whatever we have (or fail gracefully).
    # The Reviewer's fail-safe in reviewer.py sets final_output before this
    # branch is even reached in practice, but this is the safety net.
    return "check_escalation"


def route_after_escalation(state: AgentState) -> str:
    """
    Routing function called after check_escalation_node.

    - If escalation was triggered (requires_human_review=True) → "human_review"
    - Otherwise → "end"

    WHY THIS IS SEPARATE FROM route_after_review
    ---------------------------------------------
    Keeping the two routers separate means each has exactly one concern:
    - route_after_review decides retry vs done
    - route_after_escalation decides human-needed vs auto-done
    Their coupling is only via the graph edge declarations in build_graph().
    """
    if state.get("requires_human_review"):
        return "human_review"
    return "end"


# ---------------------------------------------------------------------------
# Graph construction
#
# WHY build_graph() IS A FUNCTION (not module-level imperative code)
# -------------------------------------------------------------------
# Wrapping graph construction in a function makes it easy to call in tests
# to get a fresh, uncompiled graph for inspection or alternative compilation
# options (e.g. with a checkpointer).  Module-level imperative code runs
# on import and cannot be easily re-executed with different parameters.
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Declare all nodes and edges, return the uncompiled StateGraph.

    The compiled graph (`compiled_graph` below) is what actually runs.
    Keeping build and compile separate lets tests call `build_graph()` to
    inspect the graph structure without triggering compilation.
    """
    graph = StateGraph(AgentState)

    # Register nodes — order here does not affect execution order,
    # which is determined entirely by edges below.
    #
    # PHASE 4: every node is wrapped with `traced_node(name, fn)`.  The wrapper
    # times the node, captures any LLM token usage / prompts, counts tool calls,
    # and writes one ExecutionTrace row per execution — all transparently, so
    # the node functions themselves stay free of instrumentation code.  We use
    # the SAME string for the LangGraph node name and the trace node_name so the
    # timeline in the UI lines up 1:1 with the graph structure.
    graph.add_node("task_intake",           traced_node("task_intake",          task_intake_node))
    graph.add_node("memory_retrieval",      traced_node("memory_retrieval",     memory_retrieval_node))    # Phase 2B
    graph.add_node("supervisor_planning",   traced_node("supervisor",           supervisor_planning_node))
    graph.add_node("specialist_execution",  traced_node("specialist_execution", specialist_execution_node))
    graph.add_node("reviewer_validation",   traced_node("reviewer",             reviewer_validation_node))
    graph.add_node("writer_retry",          traced_node("writer_retry",         writer_retry_node))
    graph.add_node("check_escalation",      traced_node("check_escalation",     check_escalation_node))    # Phase 3
    graph.add_node("human_review",          traced_node("human_review",         human_review_node))         # Phase 3

    # Declare the entry point — LangGraph needs to know where to start.
    graph.set_entry_point("task_intake")

    # Linear forward edges — the happy path.
    # Phase 2B inserts memory_retrieval between task_intake and supervisor_planning
    # so the Supervisor can plan with awareness of past similar tasks.
    graph.add_edge("task_intake",          "memory_retrieval")
    graph.add_edge("memory_retrieval",     "supervisor_planning")
    graph.add_edge("supervisor_planning",  "specialist_execution")
    graph.add_edge("specialist_execution", "reviewer_validation")

    # Conditional fork after review.
    # Phase 3: "end" now routes to check_escalation instead of END directly.
    graph.add_conditional_edges(
        "reviewer_validation",
        route_after_review,
        {
            "check_escalation": "check_escalation",  # accepted → check HITL triggers
            "retry_writer":     "writer_retry",       # rejected → evict + re-run writer
        },
    )

    # After writer_retry evicts the stale output, jump back into
    # specialist_execution.  That node will skip researcher+analyst
    # (cached) and only run the writer.
    graph.add_edge("writer_retry", "specialist_execution")

    # Phase 3: after escalation check, either end or pause for human review.
    graph.add_conditional_edges(
        "check_escalation",
        route_after_escalation,
        {
            "end":          END,             # no escalation → terminate normally
            "human_review": "human_review",  # escalation → pause for human
        },
    )

    # After the human reviews (and the interrupt resolves), terminate.
    graph.add_edge("human_review", END)

    return graph


# Module-level compiled graph — imported by the task runner in routers/tasks.py.
# Compiled once at startup; reused for every incoming request.
# WHY COMPILE AT MODULE LEVEL: compilation is expensive (validates the graph,
# builds internal lookup tables).  Doing it once at import time means request
# handlers pay no compilation cost.
compiled_graph = build_graph().compile()

# `graph` is an alias required by LangGraph Studio / langgraph-cli.
# The CLI discovers graphs by looking for a module-level variable named `graph`
# at the path specified in langgraph.json.  Both names point to the same
# compiled object so existing imports of `compiled_graph` are unaffected.
graph = compiled_graph
