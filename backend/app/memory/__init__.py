"""
memory/__init__.py — Phase 2A memory system package.

WHY THIS PACKAGE EXISTS
-----------------------
Phase 1 of Agentis was stateless: every task started with a blank slate.
The memory system introduced in Phase 2A gives agents two new capabilities:

1. WORKING MEMORY (Redis, short-term)
   A per-task scratchpad that survives across the LangGraph node boundaries
   within a single task run.  Nodes can stash intermediate results, error
   logs, and agent messages here without bloating AgentState.  Keys expire
   after 24 hours so Redis never accumulates stale data.

2. SEMANTIC MEMORY (ChromaDB, long-term)
   A vector store of past task completions, approach summaries, and user
   preferences.  When a new task arrives, the supervisor can query ChromaDB
   to retrieve similar past tasks and reuse proven approaches — avoiding
   repeated work and improving output quality over time.

WHY TWO DIFFERENT STORES
-------------------------
Redis and ChromaDB serve different access patterns:
- Redis answers "what did agent X produce 3 seconds ago for task Y?" in
  sub-millisecond time using exact key lookups.  It's wrong for similarity
  search because it has no embedding layer.
- ChromaDB answers "what tasks in the past were similar to this one?" via
  ANN (approximate nearest neighbour) search on embeddings.  It would be
  too slow and expensive for per-request scratchpad writes.

Using each store for the problem it was designed for avoids the pitfall of
forcing one store to do both jobs poorly.
"""
