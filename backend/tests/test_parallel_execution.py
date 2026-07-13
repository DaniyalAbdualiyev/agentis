"""
tests/test_parallel_execution.py — unit tests for Phase 5's parallel
specialist execution (asyncio.gather over dependency-aware subtasks).

These tests mock the specialist functions with artificial sleeps so we can
assert on WALL-CLOCK behaviour (independent subtasks overlap; dependent
subtasks still wait) without making real LLM/tool calls.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.graph import graph as graph_module
from app.graph.state import ExecutionPlan, Subtask, SubtaskStatus

SLEEP = 0.3


async def _slow_researcher(*, subtask_description, task_id=None, subtask_id=None):
    await asyncio.sleep(SLEEP)
    return f"findings for {subtask_id}"


async def _slow_analyst(*, subtask_description, raw_findings, task_id=None, subtask_id=None):
    await asyncio.sleep(SLEEP)
    return f"analysis of: {raw_findings}"


async def _slow_writer(
    *, subtask_description, structured_analysis, reviewer_feedback=None, task_id=None, subtask_id=None
):
    await asyncio.sleep(SLEEP)
    return f"report using: {structured_analysis}"


class TestParallelSpecialistExecution(unittest.IsolatedAsyncioTestCase):
    async def test_independent_subtasks_run_concurrently(self):
        """Two depends_on=[] research subtasks should overlap in wall time."""
        plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research A", assigned_agent="researcher"),
                Subtask(id="subtask_2", description="Research B", assigned_agent="researcher"),
                Subtask(
                    id="subtask_3",
                    description="Write combined summary",
                    assigned_agent="writer",
                    depends_on=["subtask_1", "subtask_2"],
                ),
            ],
            reasoning="Two independent research streams feeding one writer",
        )

        recorded_metadata: dict = {}

        state = {
            "task_id": "parallel-test-1",
            "execution_plan": plan,
            "subtask_results": {},
            "review_feedback": None,
        }

        with (
            patch.dict(
                graph_module._SPECIALIST_RUNNERS,
                {"researcher": _slow_researcher, "writer": _slow_writer},
            ),
            patch.object(
                graph_module, "record_node_metadata", side_effect=recorded_metadata.update
            ),
        ):
            start = time.perf_counter()
            result = await graph_module.specialist_execution_node(state)
            wall = time.perf_counter() - start

        # Two 0.3s researchers run concurrently, then a 0.3s writer waits for
        # both: total wall time should be close to 2*SLEEP, not 3*SLEEP as a
        # fully sequential run would take.
        self.assertLess(wall, 2.5 * SLEEP)

        self.assertEqual(result["subtask_results"]["subtask_1"], "findings for subtask_1")
        self.assertEqual(result["subtask_results"]["subtask_2"], "findings for subtask_2")
        self.assertIn("subtask_3", result["subtask_results"])
        self.assertNotIn("errors", result)

        self.assertEqual(
            recorded_metadata["parallel_groups"], [["subtask_1", "subtask_2"], ["subtask_3"]]
        )
        self.assertGreater(recorded_metadata["sequential_time_ms"], recorded_metadata["parallel_time_ms"])
        self.assertGreater(recorded_metadata["speedup_factor"], 1.0)

    async def test_sequential_chain_matches_old_behaviour(self):
        """A fully sequential plan (each subtask depends on the previous one)
        must still run in dependency order and produce the same outputs the
        old researcher -> analyst -> writer loop produced."""
        plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research", assigned_agent="researcher"),
                Subtask(
                    id="subtask_2", description="Analyse", assigned_agent="analyst",
                    depends_on=["subtask_1"],
                ),
                Subtask(
                    id="subtask_3", description="Write", assigned_agent="writer",
                    depends_on=["subtask_2"],
                ),
            ],
            reasoning="Standard pipeline",
        )

        state = {
            "task_id": "parallel-test-2",
            "execution_plan": plan,
            "subtask_results": {},
            "review_feedback": None,
        }

        with patch.dict(
            graph_module._SPECIALIST_RUNNERS,
            {
                "researcher": _slow_researcher,
                "analyst": _slow_analyst,
                "writer": _slow_writer,
            },
        ):
            result = await graph_module.specialist_execution_node(state)

        self.assertEqual(result["subtask_results"]["subtask_1"], "findings for subtask_1")
        self.assertEqual(
            result["subtask_results"]["subtask_2"], "analysis of: findings for subtask_1"
        )
        self.assertEqual(
            result["subtask_results"]["subtask_3"],
            "report using: analysis of: findings for subtask_1",
        )
        for s in result["execution_plan"].subtasks:
            self.assertEqual(s.status, SubtaskStatus.COMPLETED)

    async def test_dependency_failure_cascades_without_wasting_calls(self):
        """If a dependency fails, dependents must fail too WITHOUT calling
        their own specialist (matching the old early-return behaviour)."""
        plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research", assigned_agent="researcher"),
                Subtask(
                    id="subtask_2", description="Analyse", assigned_agent="analyst",
                    depends_on=["subtask_1"],
                ),
            ],
            reasoning="Pipeline with a failing researcher",
        )

        async def failing_researcher(**kwargs):
            raise RuntimeError("search API down")

        analyst_mock = AsyncMock(side_effect=AssertionError("analyst must not be called"))

        state = {
            "task_id": "parallel-test-3",
            "execution_plan": plan,
            "subtask_results": {},
            "review_feedback": None,
        }

        with patch.dict(
            graph_module._SPECIALIST_RUNNERS,
            {"researcher": failing_researcher, "analyst": analyst_mock},
        ):
            result = await graph_module.specialist_execution_node(state)

        analyst_mock.assert_not_called()
        self.assertIn("errors", result)
        self.assertTrue(any("Researcher failed" in e for e in result["errors"]))
        self.assertEqual(plan.subtasks[0].status, SubtaskStatus.FAILED)
        self.assertEqual(plan.subtasks[1].status, SubtaskStatus.FAILED)

    async def test_writer_retry_cache_skips_researcher_and_analyst(self):
        """Cached subtask_results (the writer-retry path) must not re-invoke
        already-completed specialists, even under the new concurrent scheduler."""
        plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research", assigned_agent="researcher"),
                Subtask(
                    id="subtask_2", description="Analyse", assigned_agent="analyst",
                    depends_on=["subtask_1"],
                ),
                Subtask(
                    id="subtask_3", description="Write", assigned_agent="writer",
                    depends_on=["subtask_2"],
                ),
            ],
            reasoning="Retry pipeline",
        )

        researcher_mock = AsyncMock(side_effect=AssertionError("researcher must not re-run"))
        analyst_mock = AsyncMock(side_effect=AssertionError("analyst must not re-run"))

        state = {
            "task_id": "parallel-test-4",
            "execution_plan": plan,
            # Simulates writer_retry_node having evicted only the writer's entry.
            "subtask_results": {"subtask_1": "cached findings", "subtask_2": "cached analysis"},
            "review_feedback": "Please add more detail",
        }

        with patch.dict(
            graph_module._SPECIALIST_RUNNERS,
            {"researcher": researcher_mock, "analyst": analyst_mock, "writer": _slow_writer},
        ):
            result = await graph_module.specialist_execution_node(state)

        researcher_mock.assert_not_called()
        analyst_mock.assert_not_called()
        self.assertEqual(result["subtask_results"]["subtask_3"], "report using: cached analysis")

    async def test_parallel_metadata_reaches_execution_trace(self):
        """record_node_metadata's output must land in the ExecutionTrace row
        that traced_node persists — i.e. what GET /tasks/{id}/trace returns."""
        from app.observability.tracing import traced_node

        plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research A", assigned_agent="researcher"),
                Subtask(id="subtask_2", description="Research B", assigned_agent="researcher"),
            ],
            reasoning="Two independent research streams",
        )

        state = {
            "task_id": "parallel-test-5",
            "execution_plan": plan,
            "subtask_results": {},
            "review_feedback": None,
        }

        wrapped = traced_node("specialist_execution", graph_module.specialist_execution_node)

        with (
            patch.dict(graph_module._SPECIALIST_RUNNERS, {"researcher": _slow_researcher}),
            patch("app.observability.tracing._save_trace", new_callable=AsyncMock) as save,
        ):
            await wrapped(state)

        saved_metadata = save.await_args.kwargs["metadata"]
        self.assertEqual(saved_metadata["parallel_groups"], [["subtask_1", "subtask_2"]])
        self.assertIn("sequential_time_ms", saved_metadata)
        self.assertIn("parallel_time_ms", saved_metadata)
        self.assertIn("speedup_factor", saved_metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
