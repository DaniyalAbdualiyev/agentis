"""
memory/config.py — Centralised configuration for the Phase 2A memory system.

WHY pydantic-settings HERE
---------------------------
pydantic-settings reads environment variables and validates them at import
time.  This catches misconfiguration early (before the first Redis or
ChromaDB call fails at runtime) and documents every tunable knob in one
place.  Compare this with scattered `os.getenv("REDIS_URL", "redis://..."))`
calls across multiple files — those would silently diverge if a default
was updated in one place but not another.

WHY NO PREFIX (env_prefix="")
------------------------------
The existing `.env` file uses bare variable names like `REDIS_URL` and
`CHROMADB_HOST` — no `AGENTIS_` or `MEMORY_` prefix.  Keeping the prefix
empty means no change is needed to the `.env` file or docker-compose env
blocks.  If a prefix is added later, one-line change here propagates
everywhere.

WHY working_memory_ttl_seconds = 86400 (24 hours)
---------------------------------------------------
Tasks in this system rarely exceed a few minutes.  24 hours gives ample
headroom for debugging long-running tasks while ensuring Redis doesn't
accumulate data from abandoned or crashed tasks indefinitely.
A shorter TTL (e.g. 1 hour) risks evicting keys mid-task on a slow run;
a longer TTL (e.g. 7 days) risks filling Redis with stale scratchpad data.

WHY memory_relevance_threshold = 0.5
--------------------------------------
ChromaDB returns cosine similarity scores in [0, 1] using cosine distance
(similarity = 1 - distance).  The original 0.7 threshold was too strict in
practice: empirical measurements on real task queries show that closely
related research topics (e.g. "exercise for mental health" vs "mindfulness
for wellness") produce inter-document similarities of ~0.54–0.76, meaning
a 0.7 floor would reject genuinely relevant memories.

Measured distribution of same-domain research task similarities:
  - Very similar tasks (same topic, different angle): ~0.74–0.80
  - Related tasks (adjacent topic, shared vocabulary): ~0.55–0.65
  - Unrelated tasks (different domain entirely): ~0.10–0.40

A threshold of 0.5 reliably includes same-domain and closely-related
memories while still excluding unrelated ones.  It can be overridden
without code changes by setting MEMORY_RELEVANCE_THRESHOLD in the
environment (e.g. raise to 0.65 for stricter retrieval).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings


class MemorySettings(BaseSettings):
    """
    All configuration for the working memory (Redis) and semantic memory
    (ChromaDB) subsystems.

    Values are read from environment variables matching the field names
    exactly (case-insensitive).  Defaults allow the system to start in a
    Docker Compose context where redis and chromadb are service hostnames.
    """

    redis_url: str = "redis://redis:6379/0"
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    embedding_model: str = "text-embedding-3-small"

    # How long each task's working-memory keys live in Redis before auto-expiry.
    # Even if clear() is never called (e.g. the process crashes), keys will
    # evict themselves, preventing unbounded Redis growth.
    working_memory_ttl_seconds: int = 86400  # 24 hours

    # Minimum cosine similarity score a ChromaDB result must have to be
    # included in MemoryContext.  Results below this threshold are discarded
    # before being returned to the caller.
    memory_relevance_threshold: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # No prefix — env vars like REDIS_URL, CHROMADB_HOST map directly.
        env_prefix = ""
        # Allow extra fields so that unrelated .env vars don't cause errors.
        extra = "ignore"


# Module-level singleton.
# WHY NOT re-instantiate on every call: MemorySettings reads and validates
# env vars at construction time.  Constructing it once at import time is
# cheaper and ensures all consumers see the same configuration values.
memory_settings = MemorySettings()
