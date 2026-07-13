import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import { api, fmtCost, truncate } from "../api.js";
import StatusBadge from "../components/StatusBadge.jsx";

const POLL_INTERVAL_MS = 3000;

// Statuses that mean "the graph is still working" — keep polling & show
// the live progress banner. The backend never writes an explicit "running"
// status (it goes straight from "pending" to a terminal state), so
// "pending" IS the in-progress state for a submitted task.
const IN_PROGRESS_STATUSES = new Set(["pending", "running"]);

export default function Chat() {
  const [taskText, setTaskText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const [activeTaskId, setActiveTaskId] = useState(null);
  const [activeTask, setActiveTask] = useState(null); // full TaskStatusResponse
  const [elapsedSec, setElapsedSec] = useState(0);

  const [recentTasks, setRecentTasks] = useState([]);
  const [recentError, setRecentError] = useState(null);

  const pollRef = useRef(null);
  const startedAtRef = useRef(null);

  const loadRecentTasks = () =>
    api
      .listTasks()
      .then((tasks) =>
        setRecentTasks(tasks.filter((t) => t.status === "completed").slice(0, 5))
      )
      .catch((e) => setRecentError(e.message));

  useEffect(() => {
    loadRecentTasks();
  }, []);

  // Elapsed-time ticker while a task is in progress.
  useEffect(() => {
    if (!activeTask || !IN_PROGRESS_STATUSES.has(activeTask.status)) return;
    const tick = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 1000);
    return () => clearInterval(tick);
  }, [activeTask?.status]);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  // Cleanup polling on unmount.
  useEffect(() => stopPolling, []);

  const pollTask = async (taskId) => {
    try {
      const task = await api.getTask(taskId);
      setActiveTask(task);
      if (!IN_PROGRESS_STATUSES.has(task.status)) {
        stopPolling();
        loadRecentTasks();
      }
    } catch (e) {
      setSubmitError(e.message);
      stopPolling();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!taskText.trim() || submitting) return;

    setSubmitting(true);
    setSubmitError(null);
    stopPolling();

    try {
      const res = await api.submitTask(taskText.trim());
      setActiveTaskId(res.task_id);
      setActiveTask({
        task_id: res.task_id,
        status: res.status,
        original_task: taskText.trim(),
        final_output: null,
      });
      startedAtRef.current = Date.now();
      setElapsedSec(0);

      pollRef.current = setInterval(() => pollTask(res.task_id), POLL_INTERVAL_MS);
    } catch (e) {
      setSubmitError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const openRecentTask = async (taskId) => {
    stopPolling();
    setSubmitError(null);
    try {
      const task = await api.getTask(taskId);
      setActiveTaskId(taskId);
      setActiveTask(task);
    } catch (e) {
      setSubmitError(e.message);
    }
  };

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold text-slate-900">New Task</h1>

      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          value={taskText}
          onChange={(e) => setTaskText(e.target.value)}
          placeholder="Ask anything to research..."
          rows={4}
          className="w-full rounded-md border border-slate-300 p-3 text-sm shadow-sm focus:border-slate-400 focus:outline-none"
        />
        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={submitting || !taskText.trim()}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700 disabled:opacity-50"
          >
            {submitting ? "Submitting…" : "Submit"}
          </button>
          {submitError && <span className="text-sm text-red-600">{submitError}</span>}
        </div>
      </form>

      {activeTask && (
        <TaskResult taskId={activeTaskId} task={activeTask} elapsedSec={elapsedSec} />
      )}

      <div>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Recent Tasks</h2>
          <button
            onClick={loadRecentTasks}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Refresh
          </button>
        </div>

        {recentError && (
          <div className="rounded-md bg-red-50 p-4 text-red-700">{recentError}</div>
        )}

        {!recentError && recentTasks.length === 0 && (
          <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-slate-400 shadow-sm">
            No completed tasks yet.
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {recentTasks.map((t) => (
            <button
              key={t.task_id}
              onClick={() => openRecentTask(t.task_id)}
              className="rounded-lg border border-slate-200 bg-white p-4 text-left shadow-sm transition-colors hover:border-slate-300 hover:bg-slate-50"
            >
              <div className="text-sm font-medium text-slate-800">
                {truncate(t.original_task, 80)}
              </div>
              <div className="mt-3 flex items-center justify-between">
                <StatusBadge status={t.status} />
                <span className="text-sm text-slate-500">{fmtCost(t.total_cost_usd)}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function TaskResult({ taskId, task, elapsedSec }) {
  const status = task.status;

  if (IN_PROGRESS_STATUSES.has(status)) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700" />
          <span className="text-sm font-medium text-slate-700">
            Researching... ({elapsedSec}s elapsed)
          </span>
        </div>
      </div>
    );
  }

  if (status === "awaiting_human_review") {
    return (
      <div className="rounded-md border border-yellow-200 bg-yellow-50 p-4 text-yellow-800">
        This task requires human review.{" "}
        <Link to="/reviews" className="font-semibold underline hover:text-yellow-900">
          Go to Reviews to approve.
        </Link>
      </div>
    );
  }

  if (status === "rejected") {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-red-700">
        Task was rejected
      </div>
    );
  }

  if (status === "failed") {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-red-700">
        Task failed{task.errors?.length ? `: ${task.errors.join("; ")}` : ""}
      </div>
    );
  }

  if (status === "completed") {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-700">Result</h3>
          <StatusBadge status={status} />
        </div>
        <div className="markdown-body">
          <ReactMarkdown>{task.final_output || "*No output.*"}</ReactMarkdown>
        </div>
      </div>
    );
  }

  return null;
}
