"""
Integration tests for the Agentis pipeline.

Tests that can run without API keys verify structure and logic.
Tests that require keys are marked and skipped when keys are absent.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.graph.state import (
    AgentState,
    ExecutionPlan,
    ReviewResult,
    Subtask,
    SubtaskStatus,
)
from app.graph.graph import build_graph, route_after_review
from app.tools.registry import registry, ToolSpec


class TestStateModels(unittest.TestCase):
    """Verify Pydantic models serialise correctly."""

    def test_subtask_defaults(self):
        s = Subtask(
            id="subtask_1",
            description="Search for PM tools",
            assigned_agent="researcher",
        )
        self.assertEqual(s.status, SubtaskStatus.PENDING)
        self.assertEqual(s.depends_on, [])

    def test_execution_plan(self):
        plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research", assigned_agent="researcher"),
                Subtask(id="subtask_2", description="Analyse", assigned_agent="analyst", depends_on=["subtask_1"]),
                Subtask(id="subtask_3", description="Write", assigned_agent="writer", depends_on=["subtask_2"]),
            ],
            reasoning="Standard research pipeline",
        )
        self.assertEqual(len(plan.subtasks), 3)
        d = plan.model_dump()
        self.assertIn("subtasks", d)
        self.assertIn("reasoning", d)

    def test_review_result_validation(self):
        r = ReviewResult(score=5, approved=True, feedback="")
        self.assertTrue(r.approved)
        with self.assertRaises(Exception):
            ReviewResult(score=6, approved=True, feedback="")  # score > 5 should fail


class TestToolRegistry(unittest.TestCase):
    """Verify tools registered on import."""

    def test_web_search_registered(self):
        import app.tools  # noqa: F401
        self.assertIn("web_search", registry._tools)

    def test_execute_python_registered(self):
        import app.tools  # noqa: F401
        self.assertIn("execute_python", registry._tools)

    def test_researcher_sees_web_search(self):
        import app.tools  # noqa: F401
        tools = registry.list_for_agent("researcher")
        names = [t.name for t in tools]
        self.assertIn("web_search", names)
        self.assertNotIn("execute_python", names)

    def test_analyst_sees_execute_python(self):
        import app.tools  # noqa: F401
        tools = registry.list_for_agent("analyst")
        names = [t.name for t in tools]
        self.assertIn("execute_python", names)
        self.assertNotIn("web_search", names)


class TestRouteAfterReview(unittest.TestCase):
    """Test the conditional routing logic."""

    def _make_state(self, final_output=None, retry_count=0):
        return {
            "task_id": "test",
            "original_task": "test task",
            "execution_plan": None,
            "subtask_results": {},
            "current_subtask_index": 0,
            "review_feedback": None,
            "retry_count": retry_count,
            "final_output": final_output,
            "errors": [],
        }

    def test_routes_to_end_when_approved(self):
        # Phase 3+: an accepted report no longer routes straight to END.  It
        # routes to check_escalation, which then decides END vs human_review.
        state = self._make_state(final_output="Great report")
        self.assertEqual(route_after_review(state), "check_escalation")

    def test_routes_to_retry_when_rejected_first_time(self):
        state = self._make_state(final_output=None, retry_count=0)
        self.assertEqual(route_after_review(state), "retry_writer")

    def test_routes_to_end_when_max_retries_reached(self):
        # Phase 3+: retries exhausted also routes to check_escalation (the
        # escalation node is the single exit point toward END/human_review).
        state = self._make_state(final_output=None, retry_count=2)
        self.assertEqual(route_after_review(state), "check_escalation")


class TestGraphStructure(unittest.TestCase):
    """Verify the graph has the expected nodes and edges."""

    def test_graph_compiles(self):
        from app.graph.graph import compiled_graph
        self.assertIsNotNone(compiled_graph)

    def test_expected_nodes_present(self):
        # Node set reflects the current graph: Phase 2B added memory_retrieval,
        # Phase 3 added check_escalation and human_review.
        from app.graph.graph import compiled_graph
        node_names = set(compiled_graph.nodes.keys())
        expected = {
            "__start__",
            "task_intake",
            "memory_retrieval",
            "supervisor_planning",
            "specialist_execution",
            "reviewer_validation",
            "writer_retry",
            "check_escalation",
            "human_review",
        }
        self.assertEqual(expected, node_names)


class TestCodeExecution(unittest.IsolatedAsyncioTestCase):
    """Test the sandboxed code execution tool (no API key needed)."""

    async def test_simple_print(self):
        from app.tools.code_execution import execute_python
        result = await execute_python("print('hello agentis')")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello agentis", result["stdout"])
        self.assertFalse(result["timed_out"])

    async def test_math_import(self):
        from app.tools.code_execution import execute_python
        result = await execute_python("import math; print(math.pi)")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("3.14", result["stdout"])

    async def test_timeout(self):
        from app.tools.code_execution import execute_python
        import app.tools.code_execution as ce
        original = ce.EXECUTION_TIMEOUT
        ce.EXECUTION_TIMEOUT = 1
        try:
            result = await execute_python("import time; time.sleep(10)")
            self.assertTrue(result["timed_out"])
        finally:
            ce.EXECUTION_TIMEOUT = original


class TestGraphMockedRun(unittest.IsolatedAsyncioTestCase):
    """Test the full graph with all LLM/tool calls mocked out."""

    async def test_happy_path(self):
        """Graph should produce final_output when all agents succeed."""
        from app.graph.graph import compiled_graph
        from app.graph.state import ExecutionPlan, Subtask

        mock_plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research PM tools", assigned_agent="researcher"),
                Subtask(id="subtask_2", description="Analyse findings", assigned_agent="analyst", depends_on=["subtask_1"]),
                Subtask(id="subtask_3", description="Write report", assigned_agent="writer", depends_on=["subtask_2"]),
            ],
            reasoning="Standard 3-step research pipeline",
        )

        with (
            patch("app.agents.supervisor.call_llm", new_callable=AsyncMock) as mock_supervisor,
            patch("app.agents.specialists.researcher.call_llm", new_callable=AsyncMock) as mock_researcher_llm,
            patch("app.tools.web_search.AsyncTavilyClient") as mock_tavily_cls,
            patch("app.agents.specialists.analyst.call_llm", new_callable=AsyncMock) as mock_analyst,
            patch("app.agents.specialists.writer.call_llm", new_callable=AsyncMock) as mock_writer,
            patch("app.agents.reviewer.call_llm", new_callable=AsyncMock) as mock_reviewer,
        ):
            # Supervisor returns ExecutionPlan
            mock_supervisor.return_value = mock_plan

            # Researcher LLM returns query plan then synthesis
            mock_researcher_llm.side_effect = [
                "QUERY: top project management tools 2024\nQUERY: PM software comparison small teams",
                "Raw findings: Asana, Trello, and Notion are top PM tools...",
            ]

            # Mock Tavily client
            mock_tavily = AsyncMock()
            mock_tavily.search.return_value = {
                "results": [{"title": "Best PM Tools", "url": "https://example.com", "content": "Asana is great", "score": 0.9}],
                "answer": "Asana, Trello, Notion are top PM tools",
            }
            mock_tavily_cls.return_value = mock_tavily

            # Analyst and writer return text
            mock_analyst.return_value = "Structured analysis: Asana leads, Trello is simple, Notion is flexible."
            mock_writer.return_value = "# PM Tools Report\n\nAsana, Trello, and Notion are the top 3..."

            # Reviewer approves on first try
            mock_reviewer.return_value = ReviewResult(
                score=5,
                approved=True,
                feedback="",
            )

            initial_state = {
                "task_id": "test-001",
                "original_task": "Research top 3 PM tools for small teams",
                "execution_plan": None,
                "subtask_results": {},
                "current_subtask_index": 0,
                "review_feedback": None,
                "retry_count": 0,
                "final_output": None,
                "errors": [],
            }

            result = await compiled_graph.ainvoke(initial_state)

            self.assertIsNotNone(result.get("final_output"), "final_output should be set")
            self.assertIn("PM Tools", result["final_output"])
            self.assertIsNotNone(result.get("execution_plan"))
            self.assertEqual(len(result["execution_plan"].subtasks), 3)

    async def test_reviewer_retry_path(self):
        """Graph should retry writer once on rejection, then accept."""
        from app.graph.graph import compiled_graph
        from app.graph.state import ExecutionPlan, Subtask

        mock_plan = ExecutionPlan(
            subtasks=[
                Subtask(id="subtask_1", description="Research", assigned_agent="researcher"),
                Subtask(id="subtask_2", description="Analyse", assigned_agent="analyst", depends_on=["subtask_1"]),
                Subtask(id="subtask_3", description="Write", assigned_agent="writer", depends_on=["subtask_2"]),
            ],
            reasoning="Standard pipeline",
        )

        call_count = {"writer": 0, "reviewer": 0}

        async def writer_side_effect(*args, **kwargs):
            call_count["writer"] += 1
            return f"Report v{call_count['writer']}: PM tools..."

        async def reviewer_side_effect(*args, **kwargs):
            call_count["reviewer"] += 1
            if call_count["reviewer"] == 1:
                return ReviewResult(score=2, approved=False, feedback="Too shallow, add more detail")
            return ReviewResult(score=4, approved=True, feedback="")

        with (
            patch("app.agents.supervisor.call_llm", new_callable=AsyncMock, return_value=mock_plan),
            patch("app.agents.specialists.researcher.call_llm", new_callable=AsyncMock, side_effect=[
                "QUERY: PM tools",
                "Raw findings about PM tools",
            ]),
            patch("app.tools.web_search.AsyncTavilyClient") as mock_tc,
            patch("app.agents.specialists.analyst.call_llm", new_callable=AsyncMock, return_value="Analysis"),
            patch("app.agents.specialists.writer.call_llm", new_callable=AsyncMock, side_effect=writer_side_effect),
            patch("app.agents.reviewer.call_llm", new_callable=AsyncMock, side_effect=reviewer_side_effect),
        ):
            mock_tc.return_value.search = AsyncMock(return_value={
                "results": [{"title": "T", "url": "u", "content": "c", "score": 0.9}],
                "answer": "answer",
            })

            initial_state = {
                "task_id": "test-retry",
                "original_task": "Research PM tools",
                "execution_plan": None,
                "subtask_results": {},
                "current_subtask_index": 0,
                "review_feedback": None,
                "retry_count": 0,
                "final_output": None,
                "errors": [],
            }

            result = await compiled_graph.ainvoke(initial_state)

            self.assertIsNotNone(result.get("final_output"))
            self.assertEqual(call_count["writer"], 2, "Writer should be called twice (initial + 1 retry)")
            self.assertEqual(call_count["reviewer"], 2, "Reviewer should run twice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
