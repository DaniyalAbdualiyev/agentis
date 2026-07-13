import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtDate, truncate } from "../api.js";

export default function Reviews() {
  const [reviews, setReviews] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null); // task_id currently submitting

  const load = () =>
    api
      .pendingReviews()
      .then(setReviews)
      .catch((e) => setError(e.message));

  useEffect(() => {
    load();
  }, []);

  const decide = async (taskId, decision, feedback) => {
    setBusy(taskId);
    setError(null);
    try {
      await api.submitDecision(taskId, decision, feedback);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  if (error)
    return <div className="rounded-md bg-red-50 p-4 text-red-700">{error}</div>;
  if (!reviews) return <div className="text-slate-500">Loading…</div>;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-900">Pending Reviews</h1>
        <button
          onClick={load}
          className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
        >
          Refresh
        </button>
      </div>

      {reviews.length === 0 && (
        <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-slate-400 shadow-sm">
          No tasks are awaiting human review.
        </div>
      )}

      <div className="space-y-4">
        {reviews.map((r) => (
          <ReviewCard
            key={r.task_id}
            review={r}
            busy={busy === r.task_id}
            onDecide={decide}
          />
        ))}
      </div>
    </div>
  );
}

function ReviewCard({ review, busy, onDecide }) {
  const [feedback, setFeedback] = useState("");

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="font-semibold text-slate-800">
            {truncate(review.original_task, 90)}
          </div>
          <div className="mt-1 text-sm text-yellow-700">
            {review.escalation_reason}
          </div>
          <div className="mt-1 text-xs text-slate-400">
            {fmtDate(review.created_at)}
          </div>
        </div>
        <Link
          to={`/tasks/${review.task_id}/trace`}
          className="shrink-0 text-sm text-indigo-600 hover:underline"
        >
          View trace →
        </Link>
      </div>

      {review.draft_output && (
        <pre className="mt-3 max-h-56 overflow-auto rounded bg-slate-50 p-3 text-xs leading-relaxed text-slate-700">
          {review.draft_output}
        </pre>
      )}

      <div className="mt-4 space-y-3">
        <textarea
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="Optional feedback (used as replacement text when editing)…"
          className="w-full rounded-md border border-slate-300 p-2 text-sm"
          rows={2}
        />
        <div className="flex gap-2">
          <button
            disabled={busy}
            onClick={() => onDecide(review.task_id, "approved", feedback)}
            className="rounded-md bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-500 disabled:opacity-50"
          >
            Approve
          </button>
          <button
            disabled={busy}
            onClick={() => onDecide(review.task_id, "edited", feedback)}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-50"
          >
            Edit
          </button>
          <button
            disabled={busy}
            onClick={() => onDecide(review.task_id, "rejected", feedback)}
            className="rounded-md bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-500 disabled:opacity-50"
          >
            Reject
          </button>
        </div>
      </div>
    </div>
  );
}
