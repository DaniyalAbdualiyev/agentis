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
   nodes, enabling resumable runs and human-in-the-loop pauses (Phase 2).

3. DECLARATIVE CONTROL FLOW — the conditional retry edge is declared once
   in `build_graph()` and is visible to anyone reading the graph definition.
   In plain Python the retry logic would be buried inside an if/else block
   inside `run_pipeline()`, invisible from the outside.

4. COMPOSABILITY — in Phase 2 we can add new nodes (e.g. a memory retrieval
   node, a parallel search node) by adding `graph.add_node()` and
   `graph.add_edge()` calls, without refactoring existing nodes.

FLOW DIAGRAM
------------
  task_intake
       │
       ▼
  memory_retrieval              ← Phase 2B: queries ChromaDB for similar past tasks
       │
       ▼
  supervisor_planning          ← Supervisor decomposes task → ExecutionPlan
       │                          (now enriched with memory context when available)
       ▼
  specialist_execution         ← researcher → analyst → writer (sequential)
       │
       ▼
  reviewer_validation          ← Reviewer scores 1-5, sets approved flag
       │
       ├─ approved (or retries >= 2) ────────────────────────────► END
       │
       └─ rejected AND retries < 2
                │
                ▼
          writer_retry          ← evicts stale writer output from cache
                │
                ▼
  specialist_execution          ← re-runs ONLY the writer (researcher +
       │                           analyst outputs are still cached)
       ▼
  reviewer_validation           ← second review
       │
       └─ always → END          ← retries now == 2, router forces exit

WHY THE RETRY LOOP IS DESIGNED THIS WAY
----------------------------------------
A naive retry would send the task all the way back to the Supervisor, making
the researcher and analyst re-run (wasting time and API cost).  Instead:

  - `writer_retry_node` only evicts the writer's cached output from
    `subtask_results`, leaving the researcher's and analyst's outputs intact.
  - When `specialist_execution` runs again, it checks `subtask_id not in results`
    before running each specialist.  Researcher and analyst both already have
    entries in `results`, so they are silently skipped.
  - Only the writer runs again, this time with `review_feedback` set in state.

The result is a targeted retry: the writer gets a second chance with
specific feedback, but nothing expensive is repeated.

WHY MAX_RETRIES = 2 (not 1 or 3)
---------------------------------
One retry gives the writer one chance to improve — that's too little margin
for edge cases where the first review is overly strict.  Three or more
retries risk a feedback loop that never converges and burns API budget.
Two is the pragmatic middle ground: enough to catch a genuinely improvable
report, not so many that a low-quality pipeline never terminates.
"""
from __future__ import annotations

import structlog
import uuid

from langgraph.graph import END, StateGraph

from app.agents.reviewer import run_reviewer
from app.agents.supervisor import run_supervisor
from app.agents.specialists.researcher import run_researcher
from app.agents.specialists.analyst import run_analyst
from app.agents.specialists.writer import run_writer
from app.graph.state import AgentState, SubtaskStatus

log = structlog.get_logger(__name__)

# Maximum number of writer retries before the graph accepts whatever it has.
# This is a hard ceiling to prevent infinite loops regardless of review scores.
MAX_RETRIES = 2


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


async def specialist_execution_node(state: AgentState) -> dict:
    """
    Runs all three specialists in dependency order: researcher → analyst → writer.

    WHY ALL THREE SPECIALISTS SHARE ONE GRAPH NODE
    -----------------------------------------------
    An alternative design would give each specialist its own node
    (researcher_node → analyst_node → writer_node) with edges between them.
    That would look cleaner in a graph diagram but has a practical downside:
    the retry mechanism would need to re-enter the graph at the writer_node,
    requiring LangGraph's checkpointing feature to skip already-completed nodes.
    That feature is not available in Phase 1 without a persistent checkpointer.

    Instead, we handle the sequencing and caching inside a single node:
    - `subtask_results` acts as a cache keyed by subtask_id.
    - Before running each specialist, the node checks whether its result is
      already in the cache (`subtask_id not in results`).
    - On the first pass, all three run.
    - On a writer retry, only the writer's cache entry was evicted by
      `writer_retry_node`, so only the writer runs again.

    WHY subtasks_by_role LOOKUP (not index-based access)
    -----------------------------------------------------
    `subtasks_by_role = {s.assigned_agent: s for s in plan.subtasks}` builds
    a dict so we access subtasks by their role name rather than by position.
    This is more robust: if the Supervisor ever produces subtasks in a
    different order, or omits one, the code handles it gracefully rather than
    crashing with an IndexError or running the wrong agent for a subtask.
    """
    plan = state.get("execution_plan")
    if plan is None:
        return {"errors": ["specialist_execution: no execution plan"]}

    # Copy results so we can mutate safely — state is read-only in LangGraph nodes.
    results: dict = dict(state.get("subtask_results", {}))
    # review_feedback is None on the first pass; set on retries by the Reviewer.
    review_feedback = state.get("review_feedback")
    task_id = state.get("task_id")

    # Build a role → subtask lookup for clean, position-independent access.
    subtasks_by_role = {s.assigned_agent: s for s in plan.subtasks}

    # ── Researcher ─────────────────────────────────────────────────────────
    # Skip if the result is already cached (happens on writer retry).
    researcher_subtask = subtasks_by_role.get("researcher")
    if researcher_subtask and researcher_subtask.id not in results:
        log.info("running_researcher", subtask_id=researcher_subtask.id)
        try:
            researcher_subtask.status = SubtaskStatus.RUNNING
            raw_findings = await run_researcher(
                subtask_description=researcher_subtask.description,
                task_id=task_id,
                subtask_id=researcher_subtask.id,
            )
            results[researcher_subtask.id] = raw_findings
            researcher_subtask.status = SubtaskStatus.COMPLETED
            log.info("researcher_completed", subtask_id=researcher_subtask.id)
        except Exception as exc:
            researcher_subtask.status = SubtaskStatus.FAILED
            log.error("researcher_node_failed", error=str(exc))
            # Return early — analyst and writer cannot run without research.
            return {"errors": [f"Researcher failed: {exc}"], "subtask_results": results}

    # ── Analyst ────────────────────────────────────────────────────────────
    # Also skipped on retry (its cache entry is still present).
    analyst_subtask = subtasks_by_role.get("analyst")
    raw_findings = results.get(researcher_subtask.id, "") if researcher_subtask else ""

    if analyst_subtask and analyst_subtask.id not in results:
        log.info("running_analyst", subtask_id=analyst_subtask.id)
        try:
            analyst_subtask.status = SubtaskStatus.RUNNING
            analysis = await run_analyst(
                subtask_description=analyst_subtask.description,
                raw_findings=raw_findings,
                task_id=task_id,
                subtask_id=analyst_subtask.id,
            )
            results[analyst_subtask.id] = analysis
            analyst_subtask.status = SubtaskStatus.COMPLETED
            log.info("analyst_completed", subtask_id=analyst_subtask.id)
        except Exception as exc:
            analyst_subtask.status = SubtaskStatus.FAILED
            log.error("analyst_node_failed", error=str(exc))
            return {"errors": [f"Analyst failed: {exc}"], "subtask_results": results}

    # ── Writer ─────────────────────────────────────────────────────────────
    # Always runs (on first pass and on retry — its cache entry was evicted).
    # On retry, review_feedback is non-None, so the Writer receives it.
    writer_subtask = subtasks_by_role.get("writer")
    analysis = results.get(analyst_subtask.id, "") if analyst_subtask else ""

    if writer_subtask:
        log.info(
            "running_writer",
            subtask_id=writer_subtask.id,
            has_feedback=review_feedback is not None,
        )
        try:
            writer_subtask.status = SubtaskStatus.RUNNING
            report = await run_writer(
                subtask_description=writer_subtask.description,
                structured_analysis=analysis,
                reviewer_feedback=review_feedback,  # None on first pass, string on retry
                task_id=task_id,
                subtask_id=writer_subtask.id,
            )
            results[writer_subtask.id] = report
            writer_subtask.status = SubtaskStatus.COMPLETED
            log.info("writer_completed", subtask_id=writer_subtask.id)
        except Exception as exc:
            writer_subtask.status = SubtaskStatus.FAILED
            log.error("writer_node_failed", error=str(exc))
            return {"errors": [f"Writer failed: {exc}"], "subtask_results": results}

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
# Routing function
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
    - If `final_output` is set, the Reviewer accepted → return "end"
    - If `retry_count` is still below MAX_RETRIES → return "retry_writer"
    - Otherwise (retries exhausted) → return "end" regardless of score

    WHY WE CHECK final_output RATHER THAN approved FLAG
    ----------------------------------------------------
    The Reviewer sets `final_output` when it accepts the report.  Checking
    for `final_output is not None` is the canonical signal because it is the
    same field the FastAPI endpoint checks to determine task completion.
    Checking an `approved` flag separately would add a second source of
    truth that could drift out of sync.
    """
    final_output = state.get("final_output")
    retry_count = state.get("retry_count", 0)

    if final_output is not None:
        # Reviewer accepted the report — we're done.
        return "end"

    if retry_count < MAX_RETRIES:
        # Reviewer rejected but we still have retries available.
        return "retry_writer"

    # Retries exhausted — accept whatever we have (or fail gracefully).
    # The Reviewer's fail-safe in reviewer.py sets final_output before this
    # branch is even reached in practice, but this is the safety net.
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
    graph.add_node("task_intake",           task_intake_node)
    graph.add_node("memory_retrieval",      memory_retrieval_node)   # Phase 2B
    graph.add_node("supervisor_planning",   supervisor_planning_node)
    graph.add_node("specialist_execution",  specialist_execution_node)
    graph.add_node("reviewer_validation",   reviewer_validation_node)
    graph.add_node("writer_retry",          writer_retry_node)

    # Declare the entry point — LangGraph needs to know where to start.
    graph.set_entry_point("task_intake")

    # Linear forward edges — the happy path.
    # Phase 2B inserts memory_retrieval between task_intake and supervisor_planning
    # so the Supervisor can plan with awareness of past similar tasks.
    graph.add_edge("task_intake",          "memory_retrieval")
    graph.add_edge("memory_retrieval",     "supervisor_planning")
    graph.add_edge("supervisor_planning",  "specialist_execution")
    graph.add_edge("specialist_execution", "reviewer_validation")

    # Conditional fork after review — the only non-linear part of the graph.
    # route_after_review() inspects state and returns a string key; LangGraph
    # maps that key to the next node using the dict below.
    graph.add_conditional_edges(
        "reviewer_validation",
        route_after_review,
        {
            "end":          END,           # accepted → terminate
            "retry_writer": "writer_retry", # rejected → evict + re-run writer
        },
    )

    # After writer_retry evicts the stale output, jump back into
    # specialist_execution.  That node will skip researcher+analyst
    # (cached) and only run the writer.
    graph.add_edge("writer_retry", "specialist_execution")

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
