# Agentis — Multi-Agent Research Orchestration System

A production-ready multi-agent system where a Supervisor decomposes research tasks, specialized agents execute them with tools, and a Reviewer validates the output before delivery.

## Architecture

```
POST /tasks
     │
     ▼
task_intake ──► supervisor_planning ──► specialist_execution ──► reviewer_validation
                                            │                           │
                                   researcher → analyst → writer    approved?
                                                                        │
                                                          yes ──► final_output ──► END
                                                          no (retry < 2) ──► writer_retry ──► writer
```

### Agent Hierarchy

| Agent | Model | Role |
|-------|-------|------|
| Supervisor | gpt-5.4-mini | Decomposes task into ExecutionPlan |
| Researcher | gpt-5.4-mini | Web search via Tavily, raw findings |
| Analyst | gpt-5.4-mini | Structures findings, comparative analysis |
| Writer | gpt-5.4-mini | Polished final report |
| Reviewer | gpt-5.4-mini | Scores 1-5, approves or sends back with feedback |

### LLM Provider Swap

To switch to Claude (or any other provider), edit `backend/app/llm/client.py`:
1. Add a new `"anthropic"` entry in `MODEL_REGISTRY`
2. Implement the `elif PROVIDER == "anthropic"` branch in `_build_client()`
3. Set `LLM_PROVIDER=anthropic` in `.env`

No agent code changes needed.

## Quick Start

### 1. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and fill in:
# OPENAI_API_KEY=sk-...
# TAVILY_API_KEY=tvly-...
# LANGCHAIN_API_KEY=ls__...
```

### 2. Start with Docker Compose

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### 3. Submit a task

```bash
curl -s -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Research the top 3 project management tools for small teams and write a one-paragraph comparison."}' \
  | python3 -m json.tool
```

Response:
```json
{
  "task_id": "abc123...",
  "status": "pending",
  "message": "Task submitted. Poll GET /tasks/{task_id} for status."
}
```

### 4. Poll for results

```bash
curl -s http://localhost:8000/tasks/<task_id> | python3 -m json.tool
```

Response when complete:
```json
{
  "task_id": "abc123...",
  "status": "completed",
  "original_task": "...",
  "execution_plan": {...},
  "final_output": "# Research Report\n...",
  "langsmith_trace_url": "https://smith.langchain.com/..."
}
```

## LangSmith Tracing

Every LLM call and every LangGraph node execution is traced automatically.

**To view traces:**
1. Log in to [smith.langchain.com](https://smith.langchain.com)
2. Select the **agentis** project from the left sidebar
3. Each submitted task creates one root run named `agentis-task-<task_id>`
4. Click any run to see the full execution tree: supervisor → researcher → analyst → writer → reviewer

The `langsmith_trace_url` field in `GET /tasks/{task_id}` response links directly to the specific run.

**Configuration:**
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=agentis
```

## LangGraph Studio (visual debugger)

Studio lets you step through the graph node-by-node, inspect state at every
step, replay runs, and edit inputs — all without touching the FastAPI layer.

### 1. Install the CLI (once, dev machine only)

```bash
pip install -r backend/dev-requirements.txt
# installs langgraph-cli[inmem] — NOT added to the production image
```

### 2. Run the local Studio server

```bash
cd backend
langgraph dev
```

The CLI starts a local API server on `http://127.0.0.1:2024` and prints:

```
Ready!
- API: http://127.0.0.1:2024
- Docs: http://127.0.0.1:2024/docs
- LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Open the printed Studio URL in **Chrome or Firefox** (Safari blocks localhost
connections — use `langgraph dev --tunnel` if you must use Safari).

### 3. What you can do in Studio

- **Visualise** the full graph topology (nodes, edges, conditional retry edge)
- **Submit a run** directly from the UI and watch state update at each node
- **Inspect state** after every node — see `execution_plan`, `subtask_results`,
  `review_feedback`, `retry_count`, `final_output` as they are written
- **Replay** a previous run from any checkpoint
- **Edit state** mid-run to test what happens when the Reviewer rejects

### Important tradeoffs — Studio vs the FastAPI app

| Aspect | `langgraph dev` (Studio) | `docker-compose up` (production) |
|--------|--------------------------|----------------------------------|
| Postgres / DB logging | **No** — Studio uses its own in-memory runtime; `Task`, `SubtaskLog`, `ToolCallLog` tables are **not written** | Yes — full DB persistence |
| FastAPI layer | **Bypassed** — Studio talks directly to the LangGraph API server, not to `POST /tasks` | In use |
| State persistence | In-memory, pickled to `.langgraph_api/` — **lost on server restart** | Postgres (when checkpointer added in Phase 2) |
| LangSmith tracing | Yes — same `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` env vars apply | Yes |
| Hot reload | Yes — saves on restart per code change | No (volume mount auto-reloads uvicorn) |
| Needs Postgres running | **No** | Yes |

**Bottom line:** Studio is the right tool for iterating on agent prompts, graph
topology, and control flow. Switch back to `docker-compose up` when you need to
verify the full end-to-end path including DB writes and the REST API.

## Local Development (without Docker)

```bash
# Start Postgres separately, then:
cd backend
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload
```

## Project Structure

```
agentis/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI app factory
│   │   ├── llm/
│   │   │   └── client.py               # Unified LLM client (provider-agnostic)
│   │   ├── agents/
│   │   │   ├── supervisor.py           # Task decomposition → ExecutionPlan
│   │   │   ├── reviewer.py             # Quality scoring + feedback loop
│   │   │   └── specialists/
│   │   │       ├── researcher.py       # Web search + synthesis
│   │   │       ├── analyst.py          # Structured analysis
│   │   │       └── writer.py           # Final report generation
│   │   ├── graph/
│   │   │   ├── state.py                # AgentState TypedDict + Pydantic models
│   │   │   └── graph.py                # LangGraph StateGraph
│   │   ├── tools/
│   │   │   ├── registry.py             # Tool registry with DB logging
│   │   │   ├── web_search.py           # Tavily web search
│   │   │   └── code_execution.py       # Sandboxed Python subprocess
│   │   ├── db/
│   │   │   ├── engine.py               # SQLAlchemy async engine
│   │   │   ├── models.py               # Task, SubtaskLog, ToolCallLog
│   │   │   └── queries.py              # Async query helpers
│   │   └── routers/
│   │       └── tasks.py                # POST /tasks, GET /tasks/{id}
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
└── docker-compose.yml
```

## API Reference

### `POST /tasks`
Submit a research task.

**Body:** `{"task": "your research question"}`

**Returns:** `{"task_id": "...", "status": "pending", "message": "..."}`

### `GET /tasks/{task_id}`
Get task status and results.

**Returns:**
- `status`: `pending | running | completed | failed`
- `execution_plan`: Supervisor's decomposition plan
- `final_output`: Final written report (when completed)
- `langsmith_trace_url`: Direct link to the LangSmith trace

### `GET /health`
Health check. Returns `{"status": "ok"}`.

## Database Models

| Table | Purpose |
|-------|---------|
| `tasks` | One row per task: status, plan, output, trace ID |
| `subtask_logs` | One row per specialist execution |
| `tool_call_logs` | Every tool invocation with latency and I/O |
