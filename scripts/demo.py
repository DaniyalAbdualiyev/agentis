#!/usr/bin/env python3
"""
scripts/demo.py — automated end-to-end showcase of the Agentis system.

Usage:
    cd agentis && python scripts/demo.py

Requires:
    - the docker-compose stack running (postgres, redis, chromadb, backend)
    - backend/.env present with valid OPENAI_API_KEY / TAVILY_API_KEY

The script talks to the backend over HTTP using ONLY the Python standard library
(urllib), so it needs no third-party packages installed on the host.  Override
the backend URL with the AGENTIS_API_URL environment variable if needed.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = os.getenv("AGENTIS_API_URL", "http://localhost:8000")

RESEARCH_TASK = (
    "Research the top 3 benefits of regular exercise for mental health "
    "and write a concise summary"
)
SENSITIVE_TASK = (
    "What are the legal requirements for starting a business in Kazakhstan?"
)

POLL_INTERVAL = 5
COMPLETION_TIMEOUT = 240
ESCALATION_TIMEOUT = 240


# ---------------------------------------------------------------------------
# Tiny stdlib HTTP helpers
# ---------------------------------------------------------------------------

def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def get(path: str) -> dict:
    return _request("GET", path)


def post(path: str, body: dict | None = None) -> dict:
    return _request("POST", path, body or {})


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def hr(char: str = "-", width: int = 68) -> None:
    print(char * width)


def poll_until(task_id: str, targets: set[str], timeout: int) -> dict:
    """Poll a task, printing elapsed progress, until its status is in targets."""
    start = time.time()
    while True:
        task = get(f"/tasks/{task_id}")
        status = task["status"]
        if status in targets:
            return task
        elapsed = int(time.time() - start)
        if elapsed >= timeout:
            raise TimeoutError(
                f"Task {task_id} stuck at '{status}' after {elapsed}s "
                f"(waiting for {targets})"
            )
        print(f"    ... still running ({elapsed}s elapsed, status={status})")
        time.sleep(POLL_INTERVAL)


def print_trace_summary(task_id: str) -> float:
    """Print the trace cost/latency breakdown; return the task's total cost."""
    trace = get(f"/tasks/{task_id}/trace")
    print()
    print(f"  Trace summary for task {task_id}")
    print(f"    Total cost   : ${trace['total_cost_usd']:.6f}")
    print(f"    Total latency: {trace['total_latency_ms'] / 1000:.1f} s")
    print(f"    Total tokens : {trace['total_tokens']}")
    print()
    print(f"    {'NODE':<24}{'LATENCY':>12}{'COST (USD)':>16}")
    print(f"    {'-' * 24}{'-' * 12:>12}{'-' * 16:>16}")
    for n in trace["nodes"]:
        print(
            f"    {n['node_name']:<24}"
            f"{n['latency_ms'] / 1000:>10.1f}s"
            f"{n['cost_usd']:>16.6f}"
        )
    return trace["total_cost_usd"]


# ---------------------------------------------------------------------------
# Demo flow
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    hr("=")
    print("=== Agentis Demo ===")
    hr("=")
    print(f"Backend: {BASE_URL}")

    tasks_run = 0
    total_cost = 0.0
    escalation_triggered = False

    # ── Scenario 1: normal research task ────────────────────────────────────
    print()
    print("[1/2] Submitting a normal research task:")
    print(f'      "{RESEARCH_TASK}"')
    task_id = post("/tasks", {"task": RESEARCH_TASK})["task_id"]
    tasks_run += 1
    print(f"      task_id = {task_id}")
    print()

    task = poll_until(task_id, {"completed", "failed"}, COMPLETION_TIMEOUT)
    if task["status"] != "completed":
        print(f"  ! Task did not complete (status={task['status']}).")
        return 1

    print()
    print("  Completed. Final output:")
    hr()
    print(task["final_output"])
    hr()

    total_cost += print_trace_summary(task_id)

    # ── Scenario 2: sensitive task → human-in-the-loop ──────────────────────
    print()
    print("[2/2] Submitting a SENSITIVE task (should escalate to human review):")
    print(f'      "{SENSITIVE_TASK}"')
    sens_id = post("/tasks", {"task": SENSITIVE_TASK})["task_id"]
    tasks_run += 1
    print(f"      task_id = {sens_id}")
    print()

    task = poll_until(sens_id, {"awaiting_human_review", "completed"}, ESCALATION_TIMEOUT)
    if task["status"] == "awaiting_human_review":
        escalation_triggered = True
        print()
        print("  ⚠ Task escalated to AWAITING HUMAN REVIEW.")
        pending = get("/reviews/pending")
        for item in pending:
            if item["task_id"] == sens_id:
                print(f"    Escalation reason: {item.get('escalation_reason')}")
        print("  Auto-approving on behalf of the human reviewer...")
        post(f"/reviews/{sens_id}/decision", {"decision": "approved"})
        task = poll_until(sens_id, {"completed", "rejected"}, ESCALATION_TIMEOUT)
        print(f"  Task resolved with status: {task['status']}")
    else:
        print("  (Task completed without escalation.)")

    total_cost += print_trace_summary(sens_id)

    # ── Wrap-up ─────────────────────────────────────────────────────────────
    print()
    hr("=")
    print("=== Demo Complete ===")
    hr("=")
    print(f"Tasks run           : {tasks_run}")
    print(f"Total cost          : ${total_cost:.6f}")
    print(f"Escalation triggered: {'Yes' if escalation_triggered else 'No'}")
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"\nERROR: could not reach the Agentis backend at {BASE_URL}.")
        print(f"       Is the docker-compose stack running?  ({exc})")
        raise SystemExit(2)
