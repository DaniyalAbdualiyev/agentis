# Contributing to Agentis

## Run locally

```bash
docker-compose up --build
```

Open http://localhost:3000

## Run tests

```bash
pytest tests/ --ignore=tests/test_e2e.py
```

## Add a new specialist agent

1. Create `backend/app/agents/specialists/your_agent.py`
2. Implement `run_your_agent(subtask, context, state) -> str`
3. Register in `specialist_execution_node` dispatch table
4. Add to tool registry if new tools needed

## Add a new tool

1. Create `backend/app/tools/your_tool.py`
2. Implement with Pydantic input/output schemas
3. Register in `tools/registry.py` with name, description, allowed_agents list
