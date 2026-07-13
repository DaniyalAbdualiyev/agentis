"""
tests/test_tracing.py — unit tests for the Phase 4 observability layer.

These tests use mocks only; they do NOT need Docker, PostgreSQL, or API keys.
They verify:
  - cost math (calculate_cost) for every priced model,
  - the traced_node wrapper's latency measurement,
  - status classification (success / failed / escalated),
  - token accumulation via record_llm_call / increment_tool_calls,
  - LLM usage extraction in app.llm.client._record_usage,
  - that a DB write failure inside tracing never crashes the node.
"""
from __future__ import annotations

import os
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

# Make the backend package importable when run from the repo root or backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.observability import tracing
from app.observability.tracing import (
    MODEL_PRICING,
    calculate_cost,
    increment_tool_calls,
    record_llm_call,
    traced_node,
)


# ---------------------------------------------------------------------------
# calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost(unittest.TestCase):
    def test_gpt_mini_input_and_output(self):
        # gpt-5.4-mini: $0.00015 / 1K input, $0.00060 / 1K output.
        # 1000 input + 1000 output => 0.00015 + 0.00060 = 0.00075
        self.assertAlmostEqual(
            calculate_cost("gpt-5.4-mini", 1000, 1000), 0.00075, places=8
        )

    def test_gpt_mini_input_only(self):
        # 2000 input tokens only => 2 * 0.00015 = 0.00030
        self.assertAlmostEqual(
            calculate_cost("gpt-5.4-mini", 2000, 0), 0.00030, places=8
        )

    def test_gpt_nano(self):
        # gpt-5.4-nano: $0.00005 / 1K input, $0.00020 / 1K output.
        self.assertAlmostEqual(
            calculate_cost("gpt-5.4-nano", 1000, 1000), 0.00025, places=8
        )

    def test_embedding_model(self):
        # text-embedding-3-small: $0.00002 / 1K (input side).
        self.assertAlmostEqual(
            calculate_cost("text-embedding-3-small", 1000, 0), 0.00002, places=8
        )

    def test_every_priced_model_matches_registry(self):
        # For each model, 1000/1000 tokens should equal in_rate + out_rate.
        for model, (in_rate, out_rate) in MODEL_PRICING.items():
            self.assertAlmostEqual(
                calculate_cost(model, 1000, 1000), in_rate + out_rate, places=10
            )

    def test_unknown_model_is_free(self):
        self.assertEqual(calculate_cost("does-not-exist", 5000, 5000), 0.0)

    def test_zero_tokens(self):
        self.assertEqual(calculate_cost("gpt-5.4-mini", 0, 0), 0.0)


# ---------------------------------------------------------------------------
# traced_node
# ---------------------------------------------------------------------------

class TestTracedNode(unittest.IsolatedAsyncioTestCase):
    """
    Every test patches tracing._save_trace with an AsyncMock so the trace that
    the wrapper *would* persist is captured without touching a database.
    """

    def _saved(self, save_mock: AsyncMock) -> dict:
        """Return the kwargs the wrapper passed to _save_trace."""
        self.assertTrue(save_mock.await_count >= 1, "trace was never saved")
        return save_mock.await_args.kwargs

    async def test_records_latency_from_perf_counter(self):
        async def node(_state):
            return {}

        with (
            patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save,
            # perf_counter is called exactly twice: start, then in finally.
            patch.object(tracing.time, "perf_counter", side_effect=[1000.0, 1002.5]),
        ):
            wrapped = traced_node("unit", node)
            await wrapped({"task_id": "t1"})

        saved = self._saved(save)
        self.assertEqual(saved["latency_ms"], 2500)  # (1002.5 - 1000.0) * 1000

    async def test_success_status_on_normal_completion(self):
        async def node(_state):
            return {"final_output": "done"}

        with patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save:
            await traced_node("unit", node)({"task_id": "t1"})

        self.assertEqual(self._saved(save)["status"], "success")
        self.assertIsNone(self._saved(save)["error_message"])

    async def test_failed_status_when_node_returns_errors(self):
        # A node that reports its own error via the `errors` list is "failed"
        # but does NOT raise (this is the common in-graph failure path).
        async def node(_state):
            return {"errors": ["Researcher failed: boom"]}

        with patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save:
            result = await traced_node("unit", node)({"task_id": "t1"})

        self.assertEqual(result, {"errors": ["Researcher failed: boom"]})
        saved = self._saved(save)
        self.assertEqual(saved["status"], "failed")
        self.assertIn("boom", saved["error_message"])

    async def test_failed_status_when_node_raises_and_reraises(self):
        # ACTUAL BEHAVIOR: an unexpected exception is recorded as "failed" and
        # RE-RAISED (tracing does not swallow node exceptions).  The trace is
        # still written from the finally block before the exception propagates.
        async def node(_state):
            raise ValueError("kaboom")

        with patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save:
            with self.assertRaises(ValueError):
                await traced_node("unit", node)({"task_id": "t1"})

        saved = self._saved(save)
        self.assertEqual(saved["status"], "failed")
        self.assertIn("kaboom", saved["error_message"])

    async def test_escalated_status_when_requires_human_review(self):
        async def node(_state):
            return {"requires_human_review": True, "escalation_reason": "sensitive"}

        with patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save:
            await traced_node("check_escalation", node)({"task_id": "t1"})

        self.assertEqual(self._saved(save)["status"], "escalated")

    async def test_token_counts_accumulated_from_record_llm_call(self):
        # The node makes two "LLM calls"; the wrapper must sum tokens and cost.
        async def node(_state):
            record_llm_call("gpt-5.4-mini", 1000, 500, prompt="p1", response="r1")
            record_llm_call("gpt-5.4-mini", 200, 100, prompt="p2", response="r2")
            return {}

        with patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save:
            await traced_node("supervisor", node)({"task_id": "t1"})

        saved = self._saved(save)
        self.assertEqual(saved["input_tokens"], 1200)
        self.assertEqual(saved["output_tokens"], 600)
        expected_cost = calculate_cost("gpt-5.4-mini", 1200, 600)
        self.assertAlmostEqual(saved["cost_usd"], expected_cost, places=10)
        # Prompts/responses are concatenated across calls.
        self.assertIn("p1", saved["llm_prompt"])
        self.assertIn("p2", saved["llm_prompt"])
        self.assertEqual(saved["metadata"]["llm_call_count"], 2)

    async def test_tool_calls_counted(self):
        async def node(_state):
            increment_tool_calls()
            increment_tool_calls(2)
            return {}

        with patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save:
            await traced_node("specialist_execution", node)({"task_id": "t1"})

        self.assertEqual(self._saved(save)["tool_calls_count"], 3)

    async def test_no_llm_call_yields_zero_tokens(self):
        async def node(_state):
            return {}

        with patch.object(tracing, "_save_trace", new_callable=AsyncMock) as save:
            await traced_node("memory_retrieval", node)({"task_id": "t1"})

        saved = self._saved(save)
        self.assertEqual(saved["input_tokens"], 0)
        self.assertEqual(saved["output_tokens"], 0)
        self.assertEqual(saved["cost_usd"], 0.0)
        self.assertIsNone(saved["llm_prompt"])

    async def test_record_llm_call_is_noop_outside_traced_node(self):
        # Calling record_llm_call with no active node context must not raise.
        record_llm_call("gpt-5.4-mini", 10, 10)  # should be a silent no-op
        increment_tool_calls()

    async def test_db_write_failure_does_not_crash_node(self):
        # Simulate a real _save_trace path where the DB layer raises.  The node
        # must still return its state and no exception should escape.
        @asynccontextmanager
        async def fake_session():
            yield MagicMock()

        async def node(_state):
            return {"final_output": "ok"}

        with (
            patch("app.db.engine.get_db_session", fake_session),
            patch(
                "app.db.queries.create_execution_trace",
                new_callable=AsyncMock,
                side_effect=RuntimeError("db down"),
            ),
        ):
            result = await traced_node("unit", node)({"task_id": "t1"})

        # Node output is intact despite the trace-write failure.
        self.assertEqual(result, {"final_output": "ok"})


# ---------------------------------------------------------------------------
# app.llm.client._record_usage — token extraction from an LLM message
# ---------------------------------------------------------------------------

class TestRecordUsageExtraction(unittest.TestCase):
    def test_extracts_from_usage_metadata(self):
        from app.llm import client

        msg = MagicMock()
        msg.usage_metadata = {"input_tokens": 123, "output_tokens": 45}

        with patch("app.observability.tracing.record_llm_call") as rec:
            client._record_usage("gpt-5.4-mini", msg, "prompt", "response")

        rec.assert_called_once()
        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["input_tokens"], 123)
        self.assertEqual(kwargs["output_tokens"], 45)
        self.assertEqual(kwargs["model"], "gpt-5.4-mini")

    def test_falls_back_to_response_metadata(self):
        from app.llm import client

        # No usage_metadata → fall back to response_metadata.token_usage.
        msg = MagicMock()
        msg.usage_metadata = None
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 77, "completion_tokens": 22}
        }

        with patch("app.observability.tracing.record_llm_call") as rec:
            client._record_usage("gpt-5.4-nano", msg, "p", "r")

        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["input_tokens"], 77)
        self.assertEqual(kwargs["output_tokens"], 22)

    def test_missing_usage_is_zero_not_error(self):
        from app.llm import client

        msg = MagicMock()
        msg.usage_metadata = None
        msg.response_metadata = {}

        with patch("app.observability.tracing.record_llm_call") as rec:
            client._record_usage("gpt-5.4-mini", msg, "p", "r")

        kwargs = rec.call_args.kwargs
        self.assertEqual(kwargs["input_tokens"], 0)
        self.assertEqual(kwargs["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
