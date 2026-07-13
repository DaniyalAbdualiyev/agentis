import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, fmtCost, fmtLatency } from "../api.js";
import StatusBadge from "../components/StatusBadge.jsx";

const BAR_COLORS = {
  success: "bg-green-500",
  failed: "bg-red-500",
  escalated: "bg-yellow-500",
};

export default function TraceExplorer() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [trace, setTrace] = useState(null);
  const [error, setError] = useState(null);
  const [replayMsg, setReplayMsg] = useState(null);
  const [replaying, setReplaying] = useState(false);

  useEffect(() => {
    setTrace(null);
    setError(null);
    setReplayMsg(null);
    api
      .getTrace(id)
      .then(setTrace)
      .catch((e) => setError(e.message));
  }, [id]);

  const handleReplay = async () => {
    setReplaying(true);
    setReplayMsg(null);
    try {
      const res = await api.replayTask(id, {});
      setReplayMsg(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setReplaying(false);
    }
  };

  if (error)
    return <div className="rounded-md bg-red-50 p-4 text-red-700">{error}</div>;
  if (!trace) return <div className="text-slate-500">Loading…</div>;

  const maxLatency = Math.max(1, ...trace.nodes.map((n) => n.latency_ms));

  return (
    <div className="space-y-8">
      {/* Header / summary */}
      <div>
        <button
          onClick={() => navigate("/tasks")}
          className="mb-3 text-sm text-slate-500 hover:underline"
        >
          ← Back to tasks
        </button>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Trace Explorer</h1>
            <p className="mt-1 max-w-3xl text-slate-600">{trace.original_task}</p>
            <div className="mt-2 font-mono text-xs text-slate-400">{trace.task_id}</div>
          </div>
          <button
            onClick={handleReplay}
            disabled={replaying}
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 disabled:opacity-50"
          >
            {replaying ? "Replaying…" : "Replay task"}
          </button>
        </div>

        <div className="mt-4 flex flex-wrap gap-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <Summary label="Status" value={<StatusBadge status={trace.status} />} />
          <Summary label="Total cost" value={fmtCost(trace.total_cost_usd)} />
          <Summary label="Total latency" value={fmtLatency(trace.total_latency_ms)} />
          <Summary label="Total tokens" value={trace.total_tokens.toLocaleString()} />
          <Summary label="Nodes" value={trace.nodes.length} />
        </div>

        {replayMsg && (
          <div className="mt-3 rounded-md bg-green-50 p-3 text-sm text-green-800">
            Replay started as new task{" "}
            <button
              className="font-mono font-semibold underline"
              onClick={() => navigate(`/tasks/${replayMsg.new_task_id}/trace`)}
            >
              {replayMsg.new_task_id}
            </button>
            . It may take a moment to complete.
          </div>
        )}
      </div>

      {/* Timeline */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-slate-900">Timeline</h2>
        <div className="space-y-2 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          {trace.nodes.map((n, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="w-44 shrink-0 truncate text-sm font-medium text-slate-700">
                {n.node_name}
              </div>
              <div className="relative h-6 flex-1 rounded bg-slate-100">
                <div
                  className={`h-6 rounded ${BAR_COLORS[n.status] || "bg-slate-400"}`}
                  style={{
                    width: `${Math.max(2, (n.latency_ms / maxLatency) * 100)}%`,
                  }}
                  title={`${n.node_name}: ${n.latency_ms} ms`}
                />
              </div>
              <div className="w-20 shrink-0 text-right text-xs text-slate-500">
                {fmtLatency(n.latency_ms)}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Node detail cards */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-slate-900">Nodes</h2>
        <div className="space-y-3">
          {trace.nodes.map((n, i) => (
            <NodeCard key={i} node={n} />
          ))}
        </div>
      </section>
    </div>
  );
}

function Summary({ label, value }) {
  return (
    <div>
      <div className="text-xs uppercase text-slate-400">{label}</div>
      <div className="mt-1 text-base font-semibold text-slate-800">{value}</div>
    </div>
  );
}

function NodeCard({ node }) {
  const [open, setOpen] = useState(false);
  const hasLLM = node.llm_prompt || node.llm_response;

  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-slate-800">{node.node_name}</span>
          <StatusBadge status={node.status} />
        </div>
        <div className="flex flex-wrap gap-5 text-sm text-slate-600">
          <Stat label="Latency" value={fmtLatency(node.latency_ms)} />
          <Stat
            label="Tokens"
            value={`${node.input_tokens} in / ${node.output_tokens} out`}
          />
          <Stat label="Cost" value={fmtCost(node.cost_usd)} />
          <Stat label="Tools" value={node.tool_calls_count} />
        </div>
      </div>

      {node.error_message && (
        <div className="mx-4 mb-4 rounded bg-red-50 p-3 text-sm text-red-700">
          {node.error_message}
        </div>
      )}

      {hasLLM && (
        <div className="border-t border-slate-100 px-4 py-3">
          <button
            onClick={() => setOpen((o) => !o)}
            className="text-sm font-medium text-indigo-600 hover:underline"
          >
            {open ? "Hide" : "Show"} LLM prompt & response
          </button>
          {open && (
            <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-2">
              <PrePanel title="Prompt" text={node.llm_prompt} />
              <PrePanel title="Response" text={node.llm_response} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="text-center">
      <div className="text-xs uppercase text-slate-400">{label}</div>
      <div className="font-medium text-slate-700">{value}</div>
    </div>
  );
}

function PrePanel({ title, text }) {
  return (
    <div>
      <div className="mb-1 text-xs font-semibold uppercase text-slate-400">
        {title}
      </div>
      <pre className="max-h-96 overflow-auto rounded bg-slate-900 p-3 text-xs leading-relaxed text-slate-100">
        {text || "—"}
      </pre>
    </div>
  );
}
