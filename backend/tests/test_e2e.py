"""
tests/test_e2e.py — end-to-end integration tests against a RUNNING stack.

These make real HTTP calls to the Agentis backend (which in turn makes real
LLM + tool calls), so they require:
  - `docker compose up` running (postgres, redis, chromadb, backend)
  - valid OPENAI_API_KEY / TAVILY_API_KEY in backend/.env

They are marked `integration` (module-level) so the default unit run skips them:
    pytest tests/ -m "not integration"      # unit only
    pytest tests/test_e2e.py -m integration  # these tests

API base URL comes from AGENTIS_API_URL (default http://localhost:8000).  The
docker-compose `test` service sets it to http://backend:8000.

NOTE ON TIMEOUTS
----------------
A full pipeline run (researcher → analyst → writer → reviewer) takes ~45-90s
because it makes several real LLM calls and web searches.  The timeouts below
are deliberately generous to avoid false failures on a slow model/network; they
are ceilings, not expected durations (polling returns as soon as the target
status is reached).
"""
from __future__ import annotations

import os
import time

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE_URL = os.getenv("AGENTIS_API_URL", "http://localhost:8000")

# Non-sensitive topic — must contain none of the SENSITIVE_KEYWORDS so it runs
# straight through to completion without a human-review escalation.
NORMAL_TASK = (
    "Summarize the main benefits of using version control for software teams"
)
# Contains "legal" → triggers the sensitive-topic escalation.
SENSITIVE_TASK = "I need legal advice for a contract dispute"

COMPLETION_TIMEOUT = 180   # normal task → completed
ESCALATION_TIMEOUT = 180   # sensitive task → awaiting_human_review
RESUME_TIMEOUT = 90        # after a human decision → completed/rejected
POLL_INTERVAL = 3


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        # Fail fast with a clear message if the stack isn't up.
        try:
            c.get("/health")
        except httpx.HTTPError as exc:  # pragma: no cover
            pytest.skip(f"Agentis stack not reachable at {BASE_URL}: {exc}")
        yield c


def _submit(client: httpx.Client, task: str) -> str:
    resp = client.post("/tasks", json={"task": task})
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _poll_status(client: httpx.Client, task_id: str, targets: set[str], timeout: int) -> dict:
    """Poll GET /tasks/{id} until its status is in `targets` or timeout."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in targets:
            return last
        time.sleep(POLL_INTERVAL)
    raise AssertionError(
        f"Task {task_id} did not reach {targets} within {timeout}s "
        f"(last status: {last['status'] if last else 'unknown'})"
    )


def _decision(client: httpx.Client, task_id: str, decision: str, feedback=None) -> dict:
    body = {"decision": decision}
    if feedback is not None:
        body["feedback"] = feedback
    resp = client.post(f"/reviews/{task_id}/decision", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Shared fixture: one completed normal task reused by several tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def completed_task(client) -> str:
    task_id = _submit(client, NORMAL_TASK)
    task = _poll_status(client, task_id, {"completed"}, COMPLETION_TIMEOUT)
    assert task["status"] == "completed"
    return task_id


# ---------------------------------------------------------------------------
# Test 1 — Normal task completion
# ---------------------------------------------------------------------------

def test_normal_task_completion(client, completed_task):
    task = client.get(f"/tasks/{completed_task}").json()

    assert task["status"] == "completed"
    assert task["final_output"] is not None
    assert len(task["final_output"]) > 0

    plan = task["execution_plan"]
    assert plan is not None
    subtasks = plan["subtasks"]
    assert len(subtasks) == 3
    assert all(s["status"] == "completed" for s in subtasks), subtasks


# ---------------------------------------------------------------------------
# Test 2 — Trace is recorded correctly
# ---------------------------------------------------------------------------

def test_trace_recorded(client, completed_task):
    trace = client.get(f"/tasks/{completed_task}/trace").json()
    nodes = trace["nodes"]

    assert len(nodes) > 0
    # Every node records a non-negative latency.  (task_intake can be sub-ms and
    # round to 0, so we require >= 0 for all and > 0 for the LLM-bearing nodes.)
    assert all(n["latency_ms"] >= 0 for n in nodes)

    by_name = {n["node_name"]: n for n in nodes}

    assert "supervisor" in by_name
    assert by_name["supervisor"]["input_tokens"] > 0
    assert by_name["supervisor"]["latency_ms"] > 0

    assert "specialist_execution" in by_name
    assert by_name["specialist_execution"]["tool_calls_count"] >= 1

    assert trace["total_cost_usd"] > 0
    assert trace["total_latency_ms"] > 0


# ---------------------------------------------------------------------------
# Test 3 — Sensitive topic escalation + approve
# ---------------------------------------------------------------------------

def test_sensitive_escalation_and_approve(client):
    task_id = _submit(client, SENSITIVE_TASK)
    task = _poll_status(client, task_id, {"awaiting_human_review"}, ESCALATION_TIMEOUT)
    assert task["status"] == "awaiting_human_review"

    pending = client.get("/reviews/pending").json()
    assert any(item["task_id"] == task_id for item in pending), pending

    _decision(client, task_id, "approved")

    final = _poll_status(client, task_id, {"completed"}, RESUME_TIMEOUT)
    assert final["status"] == "completed"
    assert final["final_output"] is not None


# ---------------------------------------------------------------------------
# Test 4 — Human edits output
# ---------------------------------------------------------------------------

def test_human_edits_output(client):
    edited_text = "This is the human-corrected version."
    task_id = _submit(client, SENSITIVE_TASK)
    _poll_status(client, task_id, {"awaiting_human_review"}, ESCALATION_TIMEOUT)

    _decision(client, task_id, "edited", feedback=edited_text)

    final = _poll_status(client, task_id, {"completed"}, RESUME_TIMEOUT)
    assert final["final_output"] == edited_text


# ---------------------------------------------------------------------------
# Test 5 — Human rejects output
# ---------------------------------------------------------------------------

def test_human_rejects_output(client):
    task_id = _submit(client, SENSITIVE_TASK)
    _poll_status(client, task_id, {"awaiting_human_review"}, ESCALATION_TIMEOUT)

    _decision(client, task_id, "rejected")

    final = _poll_status(client, task_id, {"rejected"}, RESUME_TIMEOUT)
    assert final["status"] == "rejected"


# ---------------------------------------------------------------------------
# Test 6 — Memory is saved after completion
# ---------------------------------------------------------------------------

def test_memory_saved_after_completion(client, completed_task):
    # Completed tasks are stored under user_id="default_user" by the task runner.
    resp = client.get("/api/memory/dashboard/default_user")
    assert resp.status_code == 200, resp.text
    dashboard = resp.json()
    assert dashboard["total_memories"] > 0, dashboard


# ---------------------------------------------------------------------------
# Test 7 — Replay creates a new, independently-traced task
# ---------------------------------------------------------------------------

def test_replay_creates_new_task(client, completed_task):
    resp = client.post(f"/tasks/{completed_task}/replay", json={})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    new_id = body["new_task_id"]
    assert new_id and new_id != completed_task

    _poll_status(client, new_id, {"completed"}, COMPLETION_TIMEOUT)

    trace = client.get(f"/tasks/{new_id}/trace").json()
    assert len(trace["nodes"]) > 0
    assert trace["total_cost_usd"] > 0


# ---------------------------------------------------------------------------
# Test 8 — Performance stats are populated
# ---------------------------------------------------------------------------

def test_performance_stats_populated(client, completed_task):
    stats = client.get("/stats/performance").json()
    assert stats["total_tasks"] > 0
    assert stats["avg_cost_usd"] > 0
    assert stats["slowest_node"] is not None
    assert stats["most_expensive_node"] is not None
