"""
tests/test_stats.py — unit tests for the Phase 4 stats endpoints.

These run the REAL stats queries against an in-memory SQLite database (via
aiosqlite), so they validate the actual SQL aggregation / sorting logic without
needing PostgreSQL or Docker.

HOW THE DB IS SWAPPED
---------------------
`app.db.engine.get_db_session` opens sessions from the module-level
`AsyncSessionFactory`.  We monkeypatch that global to a factory bound to a
shared in-memory SQLite engine (StaticPool keeps the single :memory: connection
alive across sessions), create the tables from the ORM metadata, seed rows, and
then call the endpoint coroutines directly.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Make the backend package importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app.db.engine as engine_module
from app.db.engine import Base
from app.db.models import ExecutionTrace, Task
from app.routers.stats import cost_by_task, performance_stats, stats_by_node


def _dt(offset_seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)


class _StatsTestBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Shared in-memory SQLite — StaticPool ensures every session reuses the
        # same connection so the schema/data persist across get_db_session calls.
        self._engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._orig_factory = engine_module.AsyncSessionFactory
        engine_module.AsyncSessionFactory = self._factory

        await self._seed()

    async def asyncTearDown(self) -> None:
        engine_module.AsyncSessionFactory = self._orig_factory
        await self._engine.dispose()

    async def _seed(self) -> None:
        # 3 tasks: two completed, one escalated (awaiting_human_review).
        tasks = [
            Task(
                id="task-A", original_task="Cheapest task", status="completed",
                total_cost_usd=0.010, total_latency_ms=30000,
                total_input_tokens=1000, total_output_tokens=500, created_at=_dt(0),
            ),
            Task(
                id="task-B", original_task="Mid task", status="completed",
                total_cost_usd=0.005, total_latency_ms=20000,
                total_input_tokens=800, total_output_tokens=400, created_at=_dt(10),
            ),
            Task(
                id="task-C", original_task="Most expensive task",
                status="awaiting_human_review",
                total_cost_usd=0.020, total_latency_ms=40000,
                total_input_tokens=2000, total_output_tokens=1000, created_at=_dt(20),
            ),
        ]
        # Traces spanning multiple nodes.  specialist_execution is both the
        # slowest (highest avg latency) and most expensive (highest total cost).
        traces = [
            ExecutionTrace(id="tr1", task_id="task-A", node_name="supervisor",
                           latency_ms=3000, input_tokens=1000, output_tokens=500,
                           cost_usd=0.001, tool_calls_count=0, status="success",
                           started_at=_dt(1)),
            ExecutionTrace(id="tr2", task_id="task-A", node_name="specialist_execution",
                           latency_ms=40000, input_tokens=5000, output_tokens=3000,
                           cost_usd=0.020, tool_calls_count=3, status="success",
                           started_at=_dt(2)),
            ExecutionTrace(id="tr3", task_id="task-B", node_name="reviewer",
                           latency_ms=900, input_tokens=200, output_tokens=100,
                           cost_usd=0.0005, tool_calls_count=0, status="success",
                           started_at=_dt(3)),
            ExecutionTrace(id="tr4", task_id="task-B", node_name="memory_retrieval",
                           latency_ms=100, input_tokens=0, output_tokens=0,
                           cost_usd=0.0, tool_calls_count=0, status="success",
                           started_at=_dt(4)),
            ExecutionTrace(id="tr5", task_id="task-C", node_name="check_escalation",
                           latency_ms=5, input_tokens=0, output_tokens=0,
                           cost_usd=0.0, tool_calls_count=0, status="escalated",
                           started_at=_dt(5)),
        ]
        async with self._factory() as session:
            session.add_all(tasks + traces)
            await session.commit()


class TestPerformanceStats(_StatsTestBase):
    async def test_structure_and_values(self):
        stats = await performance_stats()

        self.assertEqual(stats.total_tasks, 3)
        self.assertGreater(stats.avg_cost_usd, 0)
        self.assertAlmostEqual(stats.total_cost_usd, 0.035, places=6)
        self.assertAlmostEqual(stats.avg_cost_usd, 0.035 / 3, places=6)
        self.assertAlmostEqual(stats.avg_latency_ms, 30000.0, places=1)

        # avg tokens = avg(input+output) over 3 tasks = (1500+1200+3000)/3 = 1900
        self.assertAlmostEqual(stats.avg_tokens_per_task, 1900.0, places=1)

        # Hot spots.
        self.assertEqual(stats.slowest_node, "specialist_execution")
        self.assertEqual(stats.most_expensive_node, "specialist_execution")

        # Status breakdown.
        self.assertEqual(stats.tasks_by_status.get("completed"), 2)
        self.assertEqual(stats.tasks_by_status.get("awaiting_human_review"), 1)

        # One escalated task out of three.
        self.assertAlmostEqual(stats.escalation_rate, 1 / 3, places=4)

    async def test_hot_nodes_not_none(self):
        stats = await performance_stats()
        self.assertIsNotNone(stats.slowest_node)
        self.assertIsNotNone(stats.most_expensive_node)


class TestCostByTask(_StatsTestBase):
    async def test_sorted_by_cost_descending(self):
        rows = await cost_by_task()
        self.assertEqual(len(rows), 3)
        costs = [r.total_cost_usd for r in rows]
        self.assertEqual(costs, sorted(costs, reverse=True))
        # Most expensive first.
        self.assertEqual(rows[0].task_id, "task-C")
        self.assertEqual(rows[-1].task_id, "task-B")

    async def test_item_structure(self):
        rows = await cost_by_task()
        item = rows[0]
        for field in (
            "task_id", "original_task", "total_cost_usd",
            "total_latency_ms", "status", "created_at",
        ):
            self.assertTrue(hasattr(item, field))


class TestStatsByNode(_StatsTestBase):
    async def test_per_node_aggregates(self):
        rows = await stats_by_node()
        by_name = {r.node_name: r for r in rows}

        # All five distinct nodes present.
        self.assertEqual(
            set(by_name),
            {"supervisor", "specialist_execution", "reviewer",
             "memory_retrieval", "check_escalation"},
        )
        # specialist_execution aggregates.
        spec = by_name["specialist_execution"]
        self.assertAlmostEqual(spec.total_cost_usd, 0.020, places=6)
        self.assertAlmostEqual(spec.avg_latency_ms, 40000.0, places=1)
        self.assertEqual(spec.executions, 1)

        # Sorted by total cost descending → most expensive node first.
        self.assertEqual(rows[0].node_name, "specialist_execution")


if __name__ == "__main__":
    unittest.main(verbosity=2)
