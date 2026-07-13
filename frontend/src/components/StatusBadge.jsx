// Small colored pill for a task or node status.
const STYLES = {
  completed: "bg-green-100 text-green-800",
  success: "bg-green-100 text-green-800",
  running: "bg-blue-100 text-blue-800",
  pending: "bg-slate-200 text-slate-700",
  failed: "bg-red-100 text-red-800",
  rejected: "bg-red-100 text-red-800",
  escalated: "bg-yellow-100 text-yellow-800",
  awaiting_human_review: "bg-yellow-100 text-yellow-800",
};

export default function StatusBadge({ status }) {
  const cls = STYLES[status] || "bg-slate-200 text-slate-700";
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${cls}`}
    >
      {status?.replace(/_/g, " ") || "unknown"}
    </span>
  );
}
