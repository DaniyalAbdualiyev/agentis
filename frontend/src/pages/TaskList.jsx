import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtCost, fmtDate, fmtLatency, truncate } from "../api.js";
import StatusBadge from "../components/StatusBadge.jsx";

export default function TaskList() {
  const [tasks, setTasks] = useState(null);
  const [error, setError] = useState(null);

  const load = () =>
    api
      .listTasks()
      .then(setTasks)
      .catch((e) => setError(e.message));

  useEffect(() => {
    load();
  }, []);

  if (error)
    return <div className="rounded-md bg-red-50 p-4 text-red-700">{error}</div>;
  if (!tasks) return <div className="text-slate-500">Loading…</div>;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-900">Tasks</h1>
        <button
          onClick={load}
          className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Refresh
        </button>
      </div>

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
            <tr>
              <th className="px-4 py-3">Task</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Cost</th>
              <th className="px-4 py-3">Latency</th>
              <th className="px-4 py-3">Created</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {tasks.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-slate-400">
                  No tasks yet.
                </td>
              </tr>
            )}
            {tasks.map((t) => (
              <tr key={t.task_id} className="hover:bg-slate-50">
                <td className="px-4 py-3 font-medium text-slate-800">
                  {truncate(t.original_task)}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={t.status} />
                </td>
                <td className="px-4 py-3">{fmtCost(t.total_cost_usd)}</td>
                <td className="px-4 py-3">{fmtLatency(t.total_latency_ms)}</td>
                <td className="px-4 py-3 text-slate-500">{fmtDate(t.created_at)}</td>
                <td className="px-4 py-3 text-right">
                  <Link
                    to={`/tasks/${t.task_id}/trace`}
                    className="text-indigo-600 hover:underline"
                  >
                    Trace →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
