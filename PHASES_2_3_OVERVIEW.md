# Agentis — Phase 2 & Phase 3 Overview

This document explains what was built in Phase 2 and Phase 3, which files to read,
what new files were created, and how the whole system works today.

---

## Table of Contents

1. [Project Structure at a Glance](#1-project-structure-at-a-glance)
2. [How the System Works (End-to-End)](#2-how-the-system-works-end-to-end)
3. [Phase 2A — Working Memory & Semantic Memory](#3-phase-2a--working-memory--semantic-memory)
4. [Phase 2B — Memory Retrieval & Planning Integration](#4-phase-2b--memory-retrieval--planning-integration)
5. [Phase 3 — Human-in-the-Loop (HITL)](#5-phase-3--human-in-the-loop-hitl)
6. [File Directory: What to Read and Why](#6-file-directory-what-to-read-and-why)
7. [New Files Created (Phases 2 & 3)](#7-new-files-created-phases-2--3)
8. [Modified Files (Phases 2 & 3)](#8-modified-files-phases-2--3)
9. [Infrastructure Services](#9-infrastructure-services)
10. [API Reference](#10-api-reference)

---

## 1. Project Structure at a Glance

```
agentis/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── supervisor.py          # Plans the task (Phase 1 + 2B)
│   │   │   ├── reviewer.py            # Scores the output (Phase 1 + 3)
│   │   │   └── specialists/
│   │   │       ├── researcher.py      # Searches the web (Phase 1)
│   │   │       ├── analyst.py         # Analyses findings (Phase 1)
│   │   │       └── writer.py          # Writes the report (Phase 1)
│   │   ├── graph/
│   │   │   ├── graph.py               # LangGraph state machine (Phase 1 + 2B + 3)
│   │   │   └── state.py               # All shared data types (Phase 1 + 2B + 3)
│   │   ├── memory/                    # *** NEW in Phase 2 ***
│   │   │   ├── config.py              # Settings for Redis + ChromaDB
│   │   │   ├── models.py              # Memory data models
│   │   │   ├── working_memory.py      # Redis short-term scratchpad
│   │   │   ├── semantic_memory.py     # ChromaDB long-term vector store
│   │   │   ├── retrieval.py           # Queries memory before planning
│   │   │   └── management.py          # Memory health (decay, consolidate)
│   │   ├── db/
│   │   │   ├── engine.py              # SQLAlchemy engine + session
│   │   │   ├── models.py              # DB tables (Phase 1 + 3)
│   │   │   └── queries.py             # DB query helpers (Phase 1 + 3)
│   │   ├── routers/
│   │   │   ├── tasks.py               # POST /tasks, GET /tasks/:id (Phase 1 + 2 + 3)
│   │   │   ├── memory.py              # Memory dashboard API (Phase 2B)  ***NEW***
│   │   │   └── reviews.py             # HITL review API (Phase 3)         ***NEW***
│   │   ├── checkpointer.py            # LangGraph checkpointing (Phase 3) ***NEW***
│   │   ├── main.py                    # FastAPI app entry point
│   │   └── llm/
│   │       └── client.py              # OpenAI wrapper
│   ├── .env.studio                    # Env overrides for LangGraph Studio  ***NEW***
│   └── requirements.txt
└── langgraph.json                     # Studio config
```

---

## 2. How the System Works (End-to-End)

A single task goes through this exact flow today:

```
User
 │
 │  POST /tasks  {"task": "Research X"}
 ▼
FastAPI (routers/tasks.py)
 │  Creates DB record, starts background coroutine
 ▼
run_task_background()
 │  Gets the checkpointer-compiled graph from checkpointer.py
 │  Passes thread_id = task_id so LangGraph can checkpoint this run
 ▼
LangGraph Graph (graph/graph.py)
 │
 ├─ task_intake_node
 │    Generates task_id if missing, initialises all state fields
 │
 ├─ memory_retrieval_node          ← Phase 2B
 │    Queries ChromaDB for similar past tasks, effective approaches,
 │    user preferences.  Builds a MemoryContext dict and stores it
 │    in state.memory_context.  Gracefully skips if ChromaDB is down.
 │
 ├─ supervisor_planning_node
 │    Calls OpenAI with structured output → returns ExecutionPlan.
 │    If memory_context is set, appends it to the prompt so the
 │    Supervisor knows what worked before.
 │
 ├─ specialist_execution_node  (loops until all 3 are done)
 │    ├─ researcher: 3 web searches → raw findings
 │    ├─ analyst:    analyses findings → structured insight
 │    └─ writer:     writes final report
 │    Results are cached in state.subtask_results by subtask_id.
 │
 ├─ reviewer_validation_node
 │    Scores the report 1-5 and decides approved / rejected.
 │    Sets state.review_score for later use by check_escalation.
 │
 │    ┌─ rejected AND retries < 2 ──────────────────────────────┐
 │    │                                                          ▼
 │    │                                              writer_retry_node
 │    │                                    Evicts only the writer's cached output.
 │    │                                    Loops back to specialist_execution_node.
 │    │                                    Only the writer re-runs (researcher +
 │    │                                    analyst results are still cached).
 │    └──────────────────────────────────────────────────────────┘
 │
 ├─ check_escalation_node              ← Phase 3
 │    Checks 3 triggers (see Phase 3 section).
 │    If triggered: updates DB to "awaiting_human_review",
 │                  creates HumanReview record in DB,
 │                  sets state.requires_human_review = True.
 │
 │    ┌─ no escalation ──────────────────────────────────────────► END
 │    │
 │    └─ escalation triggered
 │              │
 │              ▼
 │        human_review_node            ← Phase 3
 │              │  Calls interrupt() → graph PAUSES here.
 │              │  State is saved to PostgreSQL checkpoint tables.
 │              │  run_task_background() detects requires_human_review=True
 │              │  and returns WITHOUT overwriting the DB status.
 │              │
 │              │  ... human reviews at GET /reviews/pending ...
 │              │
 │              │  POST /reviews/{task_id}/decision  {"decision": "approved"}
 │              │  LangGraph resumes via Command(resume={...})
 │              │  interrupt() returns the human payload.
 │              │  Node updates DB (completed / rejected) and state.
 │              ▼
 │             END
 ▼
DB record updated: status = "completed" | "failed" | "rejected"
Semantic memory stored in ChromaDB (for future tasks)  ← Phase 2A
```

---

## 3. Phase 2A — Working Memory & Semantic Memory

### What problem does it solve?

Before Phase 2, the system had no memory. Every task was treated as if it were the
first one ever. Two memory layers were added:

**Working Memory (Redis)** — a per-task scratchpad that lives only for the duration
of one task run. It gives any node a fast key-value store to stash intermediate data
without bloating AgentState. It is automatically wiped after the task finishes.

**Semantic Memory (ChromaDB)** — long-term storage that persists across tasks. When a
task completes, a summary of it (what the task was, what approach worked, what the
output looked like) is converted to a vector embedding and stored. Future tasks can
search this store to find similar past experience.

### How it works in practice

After every successful task, `run_task_background()` calls:
```python
await memory_mgr.store_task_completion(
    task_id=task_id,
    task_summary={"original_task": ..., "plan_reasoning": ..., "final_output_preview": ...},
    execution_data={"subtask_count": 3, "retry_count": 0, "agents_used": [...]}
)
```
This creates two ChromaDB documents:
- one in `task_memories` collection (what was asked and what was produced)
- one in `approach_memories` collection (which agents ran and how many retries it took)

Both are embedded with OpenAI's `text-embedding-3-small` model so they can be found
by cosine similarity for future related tasks.

### Key files for Phase 2A

| File | What it does |
|------|-------------|
| `app/memory/config.py` | Settings: Redis URL, ChromaDB host/port, embedding model, TTL |
| `app/memory/models.py` | Data models: MemoryDocument, MemoryContext, MemoryStats |
| `app/memory/working_memory.py` | Redis CRUD: store/retrieve/clear per-task scratchpad data |
| `app/memory/semantic_memory.py` | ChromaDB CRUD: embed and store/query task memories |
| `app/memory/management.py` | Health ops: importance scoring, decay stale memories, consolidate duplicates |

---

## 4. Phase 2B — Memory Retrieval & Planning Integration

### What problem does it solve?

Phase 2A stored memories but nothing used them yet. Phase 2B connects the memory store
to the planning step so the Supervisor can see what worked before.

### How it works

A new graph node — `memory_retrieval_node` — runs **before** the Supervisor. It:

1. Takes the original task text
2. Queries ChromaDB for up to 5 similar past tasks
3. Queries for effective approaches (strategies that produced high scores)
4. Queries for failed approaches (things to avoid)
5. Bundles everything into a `MemoryContext` dict stored in state

The Supervisor then reads `state.memory_context` and, if present, appends it to the
user message it sends to the LLM:

```
## Past Context (from memory)

### Similar Past Tasks:
- Task: "What is the recommended ibuprofen dosage?" → success, used researcher,analyst,writer

### Effective Approaches:
- For general research: three-stage pipeline (researcher→analyst→writer) works well
```

The Supervisor is instructed (via its system prompt) to reference past experience and
adapt its plan accordingly — e.g. reuse a pattern that worked, or avoid an approach
that failed.

### Memory Management API

A new router (`routers/memory.py`) exposes endpoints for inspecting and managing the
memory store. These are mostly for developers/admins.

| Endpoint | What it does |
|----------|-------------|
| `GET /api/memory/dashboard/{user_id}` | Rich view: top accessed memories, recent activity |
| `GET /api/memory/stats/{user_id}` | Aggregate counts by collection |
| `GET /api/memory/search?q=...` | Semantic search across all collections |
| `DELETE /api/memory/user/{user_id}` | Delete all memories for a user |
| `POST /api/memory/consolidate/{user_id}` | Merge near-duplicate memories |

### Key files for Phase 2B

| File | What it does |
|------|-------------|
| `app/memory/retrieval.py` | Queries all 5 memory categories, builds MemoryContext |
| `app/routers/memory.py` | Memory management REST API |
| `app/graph/graph.py` | Added `memory_retrieval_node` as first real step after intake |
| `app/graph/state.py` | Added `memory_context: NotRequired[Optional[dict]]` field |
| `app/agents/supervisor.py` | Reads `state.memory_context`, injects it into the LLM prompt |
| `app/routers/tasks.py` | Stores completed-task data to ChromaDB after each run |

---

## 5. Phase 3 — Human-in-the-Loop (HITL)

### What problem does it solve?

Some tasks are too sensitive, too low-quality, or too uncertain to publish automatically.
Phase 3 adds the ability to **pause the graph mid-run**, show the draft output to a human,
and let them decide: approve it, edit it, or reject it entirely.

### The 3 escalation triggers

`check_escalation_node` (in `graph.py`) evaluates all 3 triggers after the Reviewer
accepts a report:

| # | Trigger | How it fires |
|---|---------|--------------|
| 1 | **Sensitive keyword** | The original task text contains any word from `SENSITIVE_KEYWORDS` in `state.py` (e.g. "medication", "financial", "legal", "diagnosis") |
| 2 | **Low quality after retries** | `review_score < 3` AND `retry_count >= 2` (Reviewer never approved even after 2 retry attempts) |
| 3 | **Agent explicit flag** | Any node sets `state.requires_human_review = True` directly — useful if the Supervisor or a specialist detects something unusual |

### How the pause/resume works

**Pausing:**
1. `check_escalation_node` fires → updates DB status to `awaiting_human_review` → creates a `HumanReview` DB record with the draft output
2. `human_review_node` calls LangGraph's `interrupt()` → the graph suspends
3. LangGraph saves the full state snapshot to PostgreSQL (via `AsyncPostgresSaver`)
4. `run_task_background()` detects `requires_human_review=True` in the returned state and returns early without overwriting the DB status

**Resuming:**
1. A human calls `POST /reviews/{task_id}/decision` with `{"decision": "approved"}`
2. The router calls `graph.ainvoke(Command(resume={...}), config={"configurable": {"thread_id": task_id}})`
3. LangGraph loads the checkpoint from PostgreSQL using `thread_id`
4. `human_review_node` re-runs; `interrupt()` now returns the human's payload instead of pausing
5. The node updates the DB: `completed` (approved/edited) or `rejected`
6. The graph reaches END

### The 3 possible decisions

| Decision | What happens |
|----------|-------------|
| `approved` | AI output is kept as-is. Task status → `completed`. |
| `edited` | Human's `feedback` text replaces the AI output. Task status → `completed`. |
| `rejected` | Output is discarded. Task status → `rejected`. |

### Checkpointing

LangGraph needs a persistence layer to save state when the graph pauses. We use
`AsyncPostgresSaver` from `langgraph-checkpoint-postgres`, which writes checkpoint
data to a `checkpoints` table in the same PostgreSQL database we already use.

`checkpointer.py` manages this as a singleton:
- Opened once at app startup via the `lifespan` context manager in `main.py`
- Uses `psycopg3` (NOT `asyncpg`) because that is what LangGraph requires
- Every task run passes `"configurable": {"thread_id": task_id}` in the run config
  so LangGraph can identify the right checkpoint for each task

### New database tables (Phase 3)

**`human_reviews`** — audit record created for every escalated task:

| Column | What it stores |
|--------|---------------|
| `task_id` | Which task was escalated |
| `escalation_reason` | Why it was flagged ("Sensitive topic detected: dosage") |
| `output_shown_to_human` | The exact AI draft the human sees |
| `human_decision` | approved / edited / rejected |
| `human_feedback` | Replacement text (if edited) or reviewer comment |
| `reviewed_by` | Who reviewed (defaults to "manager") |
| `reviewed_at` | When the decision was submitted |

**New task statuses:**
- `awaiting_human_review` — graph is paused, waiting for a human
- `rejected` — human rejected the output

### Key files for Phase 3

| File | What it does |
|------|-------------|
| `app/checkpointer.py` | psycopg3 pool + AsyncPostgresSaver singleton; `setup()` / `teardown()` |
| `app/graph/graph.py` | `should_escalate()`, `check_escalation_node`, `human_review_node`, `route_after_escalation` |
| `app/graph/state.py` | HITL fields: `review_score`, `requires_human_review`, `escalation_reason`, `human_decision`, `human_feedback`, `human_reviewed_at` |
| `app/routers/reviews.py` | 3 REST endpoints for listing, deciding, and fetching reviews |
| `app/db/models.py` | `HumanReview` table, `Task.escalation_reason` column |
| `app/db/queries.py` | `create_human_review`, `update_human_review`, `list_pending_reviews` |
| `app/agents/reviewer.py` | Now always writes `review_score` to state (needed by trigger 2) |
| `app/routers/tasks.py` | Uses checkpointer graph; detects pause via `requires_human_review` flag |
| `app/main.py` | Calls `checkpointer.setup()` on startup, `checkpointer.teardown()` on shutdown |

---

## 6. File Directory: What to Read and Why

Start here to understand the system, in this order:

### To understand the data model (start here)
**`app/graph/state.py`** — Read this first. It defines every piece of data that
moves through the pipeline. All fields are documented inline. You will see Phase 1
fields (task_id, execution_plan, final_output), Phase 2B additions (memory_context),
and Phase 3 HITL fields (review_score, requires_human_review, human_decision, etc.).

### To understand the graph flow
**`app/graph/graph.py`** — The full orchestration. The module docstring at the top
has a flow diagram. Read the node functions top-to-bottom: `task_intake_node`,
`memory_retrieval_node`, `supervisor_planning_node`, `specialist_execution_node`,
`reviewer_validation_node`, `writer_retry_node`, then the Phase 3 nodes:
`check_escalation_node` and `human_review_node`. The `build_graph()` function at the
bottom shows exactly which nodes and edges exist.

### To understand the agents
| File | What you learn |
|------|---------------|
| `app/agents/supervisor.py` | How tasks are decomposed into 3-subtask plans; how memory context is injected |
| `app/agents/reviewer.py` | How reports are scored 1-5; approval logic; retry trigger |
| `app/agents/specialists/researcher.py` | Web search loop |
| `app/agents/specialists/analyst.py` | How raw findings become structured insight |
| `app/agents/specialists/writer.py` | How analysis becomes the final report |

### To understand the memory system (Phase 2)
| File | What you learn |
|------|---------------|
| `app/memory/config.py` | What env vars configure the memory system |
| `app/memory/models.py` | What a MemoryDocument, MemoryContext look like |
| `app/memory/working_memory.py` | Redis scratchpad API (store/retrieve/clear) |
| `app/memory/semantic_memory.py` | ChromaDB API (embed/store/query) |
| `app/memory/retrieval.py` | How memory is queried and formatted for the Supervisor |
| `app/memory/management.py` | Memory lifecycle (importance decay, consolidation) |

### To understand the HITL system (Phase 3)
| File | What you learn |
|------|---------------|
| `app/checkpointer.py` | How LangGraph state is persisted to PostgreSQL; psycopg3 pool setup |
| `app/routers/reviews.py` | All 3 HITL API endpoints; how graph.ainvoke(Command(resume=...)) works |
| `app/db/models.py` | HumanReview table schema |

### To understand the API surface
| File | What you learn |
|------|---------------|
| `app/routers/tasks.py` | POST /tasks (submit), GET /tasks/:id (poll) |
| `app/routers/memory.py` | Memory dashboard and management endpoints |
| `app/routers/reviews.py` | HITL review endpoints |
| `app/main.py` | How all routers are registered; startup/shutdown logic |

---

## 7. New Files Created (Phases 2 & 3)

These files did not exist in Phase 1:

| File | Phase | Purpose |
|------|-------|---------|
| `app/memory/config.py` | 2A | Pydantic settings for Redis, ChromaDB, embeddings |
| `app/memory/models.py` | 2A | Data models: MemoryDocument, MemoryContext, MemoryStats |
| `app/memory/working_memory.py` | 2A | Redis-backed per-task scratchpad |
| `app/memory/semantic_memory.py` | 2A | ChromaDB vector store for long-term memories |
| `app/memory/management.py` | 2A | Memory importance scoring, decay, consolidation |
| `app/memory/retrieval.py` | 2B | Queries memory, formats context for Supervisor prompt |
| `app/routers/memory.py` | 2B | REST API: dashboard, stats, search, delete, consolidate |
| `app/checkpointer.py` | 3 | AsyncPostgresSaver singleton; graph compiled with checkpointing |
| `app/routers/reviews.py` | 3 | REST API: list pending, submit decision, fetch review record |
| `backend/.env.studio` | 3 | Localhost env overrides so `langgraph dev` can reach Docker services |

---

## 8. Modified Files (Phases 2 & 3)

These files existed in Phase 1 and were extended:

| File | What changed |
|------|-------------|
| `app/graph/state.py` | Added `memory_context` (2B); added 6 HITL fields (3) |
| `app/graph/graph.py` | Added `memory_retrieval_node` (2B); added `check_escalation_node`, `human_review_node`, `should_escalate`, `route_after_escalation`, updated routing (3) |
| `app/agents/supervisor.py` | Reads `state.memory_context` and injects it into the LLM prompt (2B) |
| `app/agents/reviewer.py` | Writes `review_score` to state after every review (3) |
| `app/db/models.py` | Added `HumanReview` table; added `Task.escalation_reason` column (3) |
| `app/db/queries.py` | Added `create_human_review`, `update_human_review`, `list_pending_reviews` (3) |
| `app/routers/tasks.py` | Added semantic memory store-after-complete (2A); added checkpointer graph + `requires_human_review` interrupt detection (3) |
| `app/main.py` | Added Redis/ChromaDB health checks (2A); migrated to `lifespan` context manager; added checkpointer setup/teardown and reviews router (3) |
| `requirements.txt` | Added redis, chromadb (2A); langgraph-checkpoint-postgres, psycopg[binary,pool], fastapi>=0.116.0 (3) |
| `langgraph.json` | Points to `backend/.env.studio` so LangGraph Studio dev server uses localhost addresses |

---

## 9. Infrastructure Services

The system uses 4 services at runtime:

| Service | Port | Used for | Required? |
|---------|------|----------|-----------|
| PostgreSQL | 5432 | Task records, subtask logs, tool call logs, HITL review records, LangGraph checkpoints | Yes |
| Redis | 6379 | Per-task working memory scratchpad (Phase 2A) | Soft (graceful degradation) |
| ChromaDB | 8100 | Long-term semantic memory vector store (Phase 2A/2B) | Soft (graceful degradation) |
| OpenAI API | — | All LLM calls (supervisor, specialists, reviewer) + embeddings | Yes |

"Soft" means: if the service is unavailable at startup, the app logs a warning and
continues. Tasks still complete — they just run without memory enrichment.

### Environment configuration

For local development, copy `backend/.env.studio`:
```env
DATABASE_URL=postgresql+asyncpg://agentis:agentis123@localhost:5432/agentisdb
REDIS_URL=redis://localhost:6379/0
CHROMADB_HOST=localhost
CHROMADB_PORT=8100
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
```

---

## 10. API Reference

### Task API

```
POST /tasks
Body: {"task": "your research question"}
Returns: {"task_id": "...", "status": "pending"}

GET /tasks/{task_id}
Returns: {"task_id": "...", "status": "completed|failed|awaiting_human_review|rejected", "final_output": "..."}
```

### Memory API (Phase 2B)

```
GET  /api/memory/dashboard/{user_id}     — rich overview of stored memories
GET  /api/memory/stats/{user_id}         — count by collection
GET  /api/memory/search?q=...&user_id=.. — semantic search
DELETE /api/memory/user/{user_id}        — delete all user memories
POST /api/memory/consolidate/{user_id}   — merge near-duplicate memories
```

### Review API (Phase 3)

```
GET  /reviews/pending
     Lists all tasks currently paused waiting for human review.
     Returns: task_id, original_task, escalation_reason, draft_output

POST /reviews/{task_id}/decision
     Body: {"decision": "approved"|"edited"|"rejected", "feedback": "optional text"}
     Resumes the paused graph and updates task status.
     Returns: {"task_id": "...", "decision": "...", "status": "completed|rejected"}

GET  /reviews/{task_id}
     Full audit record: escalation reason, output shown, human decision, timestamps.
```

### Health check

```
GET /health
Returns: {"status": "ok"}
```
