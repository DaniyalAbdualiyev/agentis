# Agentis — Architecture Guide

This document explains how a request flows through the system, why the graph
is shaped the way it is, how the unified LLM client works, and what each
agent is responsible for.  It assumes you understand LangGraph basics (nodes,
edges, state) but haven't read this codebase before.

---

## 1. How a request flows from POST /tasks to final output

### Step 1 — HTTP request arrives

```
curl -X POST /tasks -d '{"task": "Research the top 3 PM tools..."}'
```

`POST /tasks` in `routers/tasks.py` does two things synchronously:
1. Creates a row in the `tasks` table with `status = "pending"`.
2. Returns a `task_id` to the caller immediately (HTTP 202 Accepted).

The actual work is dispatched to a FastAPI `BackgroundTask`.  The caller
gets a response in milliseconds and polls `GET /tasks/{task_id}` for results.

**Why async / background task?**  LLM calls take 10–60 seconds per agent.
Blocking the HTTP connection for that duration would time out reverse proxies
and frustrate clients.  The fire-and-poll pattern is standard for long-running
AI workloads.

---

### Step 2 — The graph starts

`run_task_background()` in `routers/tasks.py` invokes the compiled LangGraph:

```python
final_state = await compiled_graph.ainvoke(initial_state, config=run_config)
```

`initial_state` is a sparse `AgentState` dict: `task_id`, `original_task`,
and empty/None values for everything else.  The graph fills in the rest.

`run_config` includes a `run_name` and metadata that LangSmith uses to label
the trace.  A `LangChainTracer` callback is attached so every node and every
LLM call appears as a named span in the LangSmith UI.

---

### Step 3 — task_intake node

The first node simply validates that `original_task` is a non-empty string.
It returns `{}` (no state changes) if valid, or appends to `errors` if not.

This is a deliberate early-exit gate: no LLM budget is spent if the input
is empty or malformed.

---

### Step 4 — supervisor_planning node

The Supervisor agent (`agents/supervisor.py`) calls the LLM with
`response_model=ExecutionPlan`.  This uses OpenAI's function-calling feature
to constrain the response to a valid JSON object matching the `ExecutionPlan`
Pydantic schema.

The LLM returns three `Subtask` objects:

```
subtask_1: assigned_agent=researcher, description="Search for..."
subtask_2: assigned_agent=analyst,    description="Analyse...", depends_on=["subtask_1"]
subtask_3: assigned_agent=writer,     description="Write...",   depends_on=["subtask_2"]
```

The plan is written into `AgentState.execution_plan`.  The Supervisor does
nothing else — it does not run any searches or write any content.

---

### Step 5 — specialist_execution node

This single node runs all three specialists in sequence.

**Why one node, not three?**  With three separate nodes (researcher_node →
analyst_node → writer_node), the retry mechanism would need LangGraph's
checkpointing feature to re-enter the graph at `writer_node` and skip the
already-completed nodes.  Phase 1 doesn't use a checkpointer, so we handle
sequencing and caching inside one node instead.

**The cache pattern:**
`subtask_results` is a dict in AgentState keyed by `subtask_id`.  Before
running each specialist, the node checks:

```python
if researcher_subtask.id not in results:
    raw_findings = await run_researcher(...)
    results[researcher_subtask.id] = raw_findings
```

On the first pass, the dict is empty — all three run.
On a writer retry, only the writer's entry was evicted — only the writer runs.

**The data chain:**
```
Researcher output  →  passed as raw_findings  →  Analyst
Analyst output     →  passed as structured_analysis  →  Writer
Writer output      →  stored in subtask_results[writer_subtask.id]
```

---

### Step 6 — reviewer_validation node

The Reviewer agent (`agents/reviewer.py`) reads the Writer's output from
`subtask_results` and calls the LLM with `response_model=ReviewResult`.

It returns a `ReviewResult` with:
- `score`: integer 1–5
- `approved`: True if score ≥ 4
- `feedback`: specific improvement notes if not approved

**Two paths:**

| Condition | What Reviewer writes to state |
|-----------|-------------------------------|
| `approved=True` OR `retry_count >= 2` | Sets `final_output = report` |
| `approved=False` AND `retry_count < 2` | Sets `review_feedback`, increments `retry_count` |

---

### Step 7 — conditional edge: route_after_review

After every reviewer run, LangGraph calls `route_after_review(state)`:

```python
if final_output is not None:  → "end"   → graph terminates
if retry_count < 2:           → "retry_writer" → writer gets another chance
else:                         → "end"   → retries exhausted, accept what we have
```

The key signal is `final_output`.  If the Reviewer set it, the graph ends.
If it's still `None`, there is a retry available.

---

### Step 8 — writer_retry node (retry path only)

If the router chose `"retry_writer"`, the `writer_retry` node:
1. Removes the writer's entry from `subtask_results` (cache eviction).
2. Resets the writer subtask's status to `PENDING`.

Then the graph loops back to `specialist_execution`, which skips the
researcher and analyst (their cache entries are intact) and only runs
the writer — this time with `review_feedback` set in state.

---

### Step 9 — graph terminates, results persisted

Once the graph returns, `run_task_background()` writes to PostgreSQL:
- Updates the `tasks` row: `status`, `execution_plan` (JSON), `final_output`,
  `langsmith_trace_id` (the trace URL).
- Creates one `subtask_logs` row per specialist showing what ran and its output.

`GET /tasks/{task_id}` then returns the full picture: status, plan, output,
and a clickable `langsmith_trace_url` that opens the trace in LangSmith.

---

## 2. Why the graph has the conditional retry edge

The naive alternative would be to loop the whole graph (back to the
Supervisor) on every rejection.  That would re-run the researcher and
analyst, costing time and API budget for no benefit — the research data
hasn't changed, only the presentation of it.

Instead, the retry edge is **targeted**:

```
reviewer_validation
       │
       └─ rejected ──► writer_retry ──► specialist_execution
                              │                │
                     evicts writer cache       only writer runs
                                               (researcher + analyst cached)
```

This means:
- A rejected Writer gets the Reviewer's specific feedback injected into its
  next prompt via `AgentState.review_feedback`.
- The Researcher and Analyst are never called twice in a single pipeline run.
- The maximum additional cost of two retries is: 2 × (1 Writer LLM call +
  1 Reviewer LLM call).  Everything else is reused.

The hard ceiling of `MAX_RETRIES = 2` prevents infinite loops.  After two
retries the graph always terminates, even if the Reviewer still scores the
report below 4.  A forced acceptance is logged (`forced=True`) so you can
identify quality issues in LangSmith without the pipeline getting stuck.

---

## 3. How the unified LLM client enables provider swapping

Every agent calls exactly one function:

```python
from app.llm.client import call_llm

result = await call_llm(role="supervisor", messages=[...], response_model=ExecutionPlan)
```

`call_llm` in `llm/client.py` does three things:
1. Looks up `role` in `MODEL_REGISTRY[PROVIDER]` to get a model name string.
2. Calls `_build_client(role)` to get a LangChain chat model object.
3. Invokes the model, optionally with structured output.

**To add Claude:**

```python
# In MODEL_REGISTRY:
"anthropic": {
    "supervisor": "claude-3-5-sonnet-20241022",
    "reviewer":   "claude-3-5-sonnet-20241022",
    "specialist": "claude-3-haiku-20240307",
}

# In _build_client():
elif PROVIDER == "anthropic":
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model_name, temperature=0.2)
```

```bash
# In .env:
LLM_PROVIDER=anthropic
```

Zero changes to supervisor.py, reviewer.py, researcher.py, analyst.py, or
writer.py.  The agent files don't import any provider-specific library —
they only know about `call_llm`.

**Why LangChain's chat models rather than the raw OpenAI SDK?**
LangChain's `ChatOpenAI` automatically emits LangSmith traces when
`LANGCHAIN_TRACING_V2=true` is set.  The raw SDK does not.  Getting the
same tracing with the raw SDK would require writing custom callback handlers.

---

## 4. What each agent is responsible for

The system follows a strict single-responsibility principle.  Each agent
knows what it does and explicitly does not do the other agents' jobs.

### Supervisor
**Does:** Receive the user's raw task string → return a structured
`ExecutionPlan` with 3 ordered `Subtask` objects.

**Does not:** Search the web, analyse data, write prose, or evaluate quality.

**Why it matters:** The plan is the contract that all downstream agents follow.
If the Supervisor produces a malformed plan, everything breaks.  That's why
it's the only agent that uses structured output with Pydantic validation —
a plan that doesn't parse is rejected immediately rather than causing a
confusing failure three steps later.

---

### Researcher (Specialist)
**Does:** Run 2–3 targeted web searches via Tavily, synthesise the raw
results into a factual findings document.

**Does not:** Structure the data, form analytical opinions, or write the report.

**Why it matters:** Raw search results are noisy — multiple sources, repeated
information, varying formats.  The Researcher's job is to gather and
consolidate, not to interpret.  Keeping gathering and analysis separate
means the Analyst always receives clean, well-sourced input.

---

### Analyst (Specialist)
**Does:** Receive the Researcher's raw findings → produce a structured
analytical document with an executive summary, grouped findings,
comparative analysis, and data gaps.

**Does not:** Search the web or write the final polished report.

**Why it matters:** The Writer needs structured input to produce structured
output.  If the Writer received raw search snippets directly, it would spend
tokens reorganising data instead of writing — producing lower-quality prose.
The Analyst's output is intentionally "analytical but not pretty" — that's
the Writer's job.

---

### Writer (Specialist)
**Does:** Receive the Analyst's structured analysis (and, on retries, the
Reviewer's feedback) → produce a polished, professional research report in
Markdown.

**Does not:** Search for new information, re-analyse data, or evaluate itself.

**Why it matters:** The Writer's prompt is tuned for prose quality — no filler
phrases, logical sections, concrete data points.  On a retry, `review_feedback`
is injected into the system prompt so the Writer knows exactly what to fix
without re-reading the entire history of the conversation.

---

### Reviewer
**Does:** Receive the Writer's report and the original task → score 1–5 →
approve (score ≥ 4) or reject with specific, actionable feedback.

**Does not:** Generate new content, revise the report itself, or change the
execution plan.

**Why a separate agent exists for this:** A model cannot reliably evaluate
its own output.  A separate Reviewer prompt, given only the task and the
finished report (no knowledge of the writing process), produces more honest
and consistent quality judgements.  The explicit scoring rubric in the prompt
anchors the 1–5 scale to concrete criteria so scores are reproducible across
runs.

---

## 5. How LangSmith tracing works

LangSmith tracing is enabled entirely through environment variables:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=agentis
```

When these are set, **every** LangChain operation emits a trace span
automatically — including every `ChatOpenAI.ainvoke()` call inside `call_llm`
and every LangGraph node execution.  No manual instrumentation is needed.

Additionally, `run_task_background()` attaches a `LangChainTracer` callback
to the graph run:

```python
tracer = LangChainTracer(project_name="agentis")
run_config = {"run_name": f"agentis-task-{task_id}", "callbacks": [tracer]}
final_state = await compiled_graph.ainvoke(initial_state, config=run_config)
trace_url = tracer.get_run_url()  # direct link to this specific run
```

The trace URL is stored in the `tasks.langsmith_trace_id` column and
returned in `GET /tasks/{task_id}` as `langsmith_trace_url`.

**To view traces:**
1. Log in to [smith.langchain.com](https://smith.langchain.com).
2. Select the **agentis** project.
3. Each submitted task appears as one root run named `agentis-task-<task_id>`.
4. Expand any run to see the full tree: `task_intake` → `supervisor_planning`
   → `specialist_execution` (with nested LLM calls and tool invocations) →
   `reviewer_validation` → (optional) `writer_retry` → `specialist_execution`
   → `reviewer_validation`.
