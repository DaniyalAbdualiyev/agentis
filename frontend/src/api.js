// Central API client. The base URL comes from the VITE_API_URL build-time env
// variable (see docker-compose / .env), falling back to localhost for `npm dev`.
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  // 202/204 responses may have no body.
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

export const api = {
  // Tasks
  listTasks: () => request("/tasks"),
  getTask: (id) => request(`/tasks/${id}`),
  submitTask: (task) =>
    request("/tasks", { method: "POST", body: JSON.stringify({ task }) }),

  // Traces + replay
  getTrace: (id) => request(`/tasks/${id}/trace`),
  replayTask: (id, body = {}) =>
    request(`/tasks/${id}/replay`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Stats
  performance: () => request("/stats/performance"),
  costByTask: () => request("/stats/cost-by-task"),
  statsByNode: () => request("/stats/by-node"),

  // Reviews (Phase 3 HITL queue)
  pendingReviews: () => request("/reviews/pending"),
  submitDecision: (id, decision, feedback) =>
    request(`/reviews/${id}/decision`, {
      method: "POST",
      body: JSON.stringify({ decision, feedback }),
    }),
};

// ── Formatting helpers shared across pages ──────────────────────────────────
export const fmtCost = (v) => `$${(v ?? 0).toFixed(4)}`;
export const fmtLatency = (ms) => {
  const n = ms ?? 0;
  if (n < 1000) return `${n} ms`;
  return `${(n / 1000).toFixed(1)} s`;
};
export const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString() : "—");
export const truncate = (s, n = 70) =>
  s && s.length > n ? `${s.slice(0, n)}…` : s || "";
