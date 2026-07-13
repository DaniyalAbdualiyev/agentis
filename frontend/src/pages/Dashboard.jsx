import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, fmtCost, fmtLatency, truncate } from "../api.js";
import MetricCard from "../components/MetricCard.jsx";
import StatusBadge from "../components/StatusBadge.jsx";

const PIE_COLORS = {
  completed: "#22c55e",
  awaiting_human_review: "#eab308",
  rejected: "#ef4444",
  failed: "#ef4444",
  running: "#3b82f6",
  pending: "#94a3b8",
};
const FALLBACK_COLORS = ["#6366f1", "#0ea5e9", "#f97316", "#14b8a6", "#a855f7"];

export default function Dashboard() {
  const [perf, setPerf] = useState(null);
  const [byNode, setByNode] = useState([]);
  const [costByTask, setCostByTask] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.performance(), api.statsByNode(), api.costByTask()])
      .then(([p, n, c]) => {
        setPerf(p);
        setByNode(n);
        setCostByTask(c);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error)
    return <div className="rounded-md bg-red-50 p-4 text-red-700">{error}</div>;
  if (!perf) return <div className="text-slate-500">Loading…</div>;

  const statusData = Object.entries(perf.tasks_by_status).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold text-slate-900">Performance Dashboard</h1>

      {/* Metric cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Total Tasks" value={perf.total_tasks} />
        <MetricCard label="Avg Cost / Task" value={fmtCost(perf.avg_cost_usd)} />
        <MetricCard label="Avg Latency" value={fmtLatency(perf.avg_latency_ms)} />
        <MetricCard
          label="Escalation Rate"
          value={`${(perf.escalation_rate * 100).toFixed(1)}%`}
          sub={`${perf.avg_tokens_per_task.toFixed(0)} avg tokens/task`}
        />
      </div>

      {/* Bar charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ChartCard title="Cost by Node (USD)">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={byNode} margin={{ left: 10, right: 10 }}>
              <XAxis
                dataKey="node_name"
                tick={{ fontSize: 11 }}
                angle={-20}
                textAnchor="end"
                height={60}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmtCost(v)} />
              <Bar dataKey="total_cost_usd" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Avg Latency by Node (ms)">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={byNode} margin={{ left: 10, right: 10 }}>
              <XAxis
                dataKey="node_name"
                tick={{ fontSize: 11 }}
                angle={-20}
                textAnchor="end"
                height={60}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${v} ms`} />
              <Bar dataKey="avg_latency_ms" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Status pie + hot nodes */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ChartCard title="Tasks by Status">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={statusData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={90}
                label
              >
                {statusData.map((entry, i) => (
                  <Cell
                    key={entry.name}
                    fill={
                      PIE_COLORS[entry.name] ||
                      FALLBACK_COLORS[i % FALLBACK_COLORS.length]
                    }
                  />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Hot Spots">
          <div className="space-y-3 p-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Slowest node</span>
              <span className="font-medium">{perf.slowest_node || "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Most expensive node</span>
              <span className="font-medium">{perf.most_expensive_node || "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Total spend</span>
              <span className="font-medium">{fmtCost(perf.total_cost_usd)}</span>
            </div>
          </div>
        </ChartCard>
      </div>

      {/* Recent tasks table */}
      <div>
        <h2 className="mb-3 text-lg font-semibold text-slate-900">Recent Tasks</h2>
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3">Task</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Cost</th>
                <th className="px-4 py-3">Latency</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {costByTask.slice(0, 8).map((t) => (
                <tr key={t.task_id} className="hover:bg-slate-50">
                  <td className="px-4 py-3">{truncate(t.original_task)}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={t.status} />
                  </td>
                  <td className="px-4 py-3">{fmtCost(t.total_cost_usd)}</td>
                  <td className="px-4 py-3">{fmtLatency(t.total_latency_ms)}</td>
                  <td className="px-4 py-3 text-right">
                    <Link
                      to={`/tasks/${t.task_id}/trace`}
                      className="text-indigo-600 hover:underline"
                    >
                      View trace
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="mb-4 text-sm font-semibold text-slate-700">{title}</h3>
      {children}
    </div>
  );
}
