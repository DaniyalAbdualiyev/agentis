"""
memory/retrieval.py — Planning-time memory retrieval for the Supervisor agent.

WHY THIS MODULE EXISTS
----------------------
Without memory retrieval, the Supervisor plans every task from scratch, with
zero knowledge of what worked (or failed) on similar tasks in the past.  This
module queries ChromaDB for five categories of relevant past experience and
packages them into a MemoryContext that the Supervisor can consume before
producing an ExecutionPlan.

WHY THE RETRIEVER IS SEPARATE FROM SemanticMemoryManager
---------------------------------------------------------
SemanticMemoryManager is a low-level I/O adapter: it speaks ChromaDB and
knows how to query individual collections.  This module is a higher-level
orchestrator: it knows *which* queries to run, *how many* results to fetch
per category, *how* to filter by relevance threshold, and *how* to format
the aggregated result for an LLM prompt.

Keeping these two concerns separate means:
- SemanticMemoryManager can be tested in isolation without any notion of
  "planning context".
- PlanningMemoryRetriever can be tested by mocking SemanticMemoryManager,
  without requiring a live ChromaDB instance.
- Future phases can add a different retriever (e.g. ContextualRetriever for
  mid-task retrieval) without touching SemanticMemoryManager.

WHY GRACEFUL DEGRADATION IS THE DEFAULT
-----------------------------------------
Memory retrieval happens before the Supervisor's LLM call.  If ChromaDB is
slow or unavailable, blocking the whole task is unacceptable.  Every method
that can raise is wrapped in try/except so a failing retrieval returns an
empty MemoryContext — the Supervisor still runs, just without memory context.
"""
from __future__ import annotations

import structlog

from app.memory.config import memory_settings
from app.memory.models import MemoryContext, MemoryDocument
from app.memory.semantic_memory import SemanticMemoryManager

log = structlog.get_logger(__name__)


class PlanningMemoryRetriever:
    """
    Retrieves relevant memories from long-term storage to inform the
    Supervisor's planning step.

    WHY THIS EXISTS
    ---------------
    Without memory retrieval, the Supervisor plans each task from scratch
    with zero knowledge of what worked before.  This retriever queries
    ChromaDB for similar past tasks, effective approaches, and user
    preferences, then packages them into a MemoryContext that the
    Supervisor can use to make better planning decisions.

    INSTANTIATION
    -------------
    Instantiation is cheap — no network calls happen in __init__.  The
    SemanticMemoryManager is created lazily here, or the caller can inject
    a pre-constructed instance (useful in tests for mocking).
    """

    def __init__(
        self, semantic_memory: SemanticMemoryManager | None = None
    ) -> None:
        """
        Initialise the retriever.

        WHY DEPENDENCY INJECTION FOR semantic_memory
        ---------------------------------------------
        Tests can inject a mock SemanticMemoryManager so that unit tests
        for PlanningMemoryRetriever do not require a live ChromaDB service.
        Production code calls PlanningMemoryRetriever() with no args, which
        creates a real SemanticMemoryManager internally.
        """
        self._semantic_memory = semantic_memory or SemanticMemoryManager()

    async def retrieve_context(
        self,
        task_description: str,
        user_id: str = "default",
    ) -> MemoryContext:
        """
        Query long-term memory across five categories and assemble a MemoryContext.

        WHY FIVE SEPARATE QUERIES (not one broad search)
        -------------------------------------------------
        A single broad query would return a mixed list of tasks, approaches,
        and preferences sorted only by similarity.  Five targeted queries
        preserve semantic structure: the Supervisor prompt can reference
        "similar past tasks" and "effective approaches" separately, which
        produces better planning decisions than a flat list with no context
        about what each memory represents.

        RELEVANCE THRESHOLD
        -------------------
        Only memories with similarity_score >= memory_settings.memory_relevance_threshold
        (default 0.7) are included.  This is enforced inside
        SemanticMemoryManager._parse_results(), so results returned from
        the manager are already filtered.  We perform no additional filtering
        here — the threshold is a single source of truth in MemorySettings.

        Parameters
        ----------
        task_description : str
            The user's raw task string.  Used as the query vector for all
            similarity searches.
        user_id : str
            User identifier for multi-tenant filtering.  Defaults to "default"
            until multi-user support is implemented.

        Returns
        -------
        MemoryContext
            Populated with up to 3 similar tasks, 3 effective approaches,
            2 failed approaches, user preferences, and 3 domain facts.
            has_relevant_memories is True if at least one list is non-empty.
        """
        similar_tasks: list[MemoryDocument] = []
        effective_approaches: list[MemoryDocument] = []
        failed_approaches: list[MemoryDocument] = []
        user_preferences: list[MemoryDocument] = []
        domain_context: list[MemoryDocument] = []

        # 1. Query similar past tasks (top 3)
        try:
            similar_tasks = await self._semantic_memory.query_similar_tasks(
                task_description=task_description,
                user_id=user_id,
                n_results=3,
            )
            log.debug("retrieved_similar_tasks", count=len(similar_tasks))
        except Exception as exc:
            log.warning("retrieve_similar_tasks_failed", error=str(exc))

        # 2. Query effective approaches (top 3) — outcome=success only
        try:
            all_approaches = await self._semantic_memory.query_approaches(
                problem_type=task_description,
                user_id=user_id,
            )
            effective_approaches = [
                doc for doc in all_approaches
                if doc.metadata.get("outcome") in ("success", "partial")
            ][:3]
            log.debug("retrieved_effective_approaches", count=len(effective_approaches))
        except Exception as exc:
            log.warning("retrieve_effective_approaches_failed", error=str(exc))

        # 3. Query failed approaches to avoid (top 2)
        try:
            if not all_approaches:
                # Re-fetch if the earlier approach query failed
                all_approaches_for_failed = await self._semantic_memory.query_approaches(
                    problem_type=task_description,
                    user_id=user_id,
                )
            else:
                all_approaches_for_failed = all_approaches
            failed_approaches = [
                doc for doc in all_approaches_for_failed
                if doc.metadata.get("outcome") == "failed"
            ][:2]
            log.debug("retrieved_failed_approaches", count=len(failed_approaches))
        except Exception as exc:
            log.warning("retrieve_failed_approaches_failed", error=str(exc))

        # 4. Query user preferences (no similarity — fetch all for user)
        try:
            user_preferences = await self._semantic_memory.query_user_preferences(
                user_id=user_id,
            )
            log.debug("retrieved_user_preferences", count=len(user_preferences))
        except Exception as exc:
            log.warning("retrieve_user_preferences_failed", error=str(exc))

        # 5. Query relevant domain facts (top 3) via broad semantic search
        try:
            all_results = await self._semantic_memory.query(
                query_text=task_description,
                user_id=user_id,
                n_results=3,
                filters={"collection": "domain_facts"} if False else None,
                # NOTE: ChromaDB does not support filtering by collection name in
                # a metadata where clause — collections are already separate
                # namespaces.  We use query_similar_tasks for domain facts via
                # the domain_facts collection below.
            )
            # Filter to only domain_facts collection results
            domain_context = [
                doc for doc in all_results
                if doc.collection == "domain_facts"
            ][:3]
            log.debug("retrieved_domain_context", count=len(domain_context))
        except Exception as exc:
            log.warning("retrieve_domain_context_failed", error=str(exc))

        has_relevant_memories = any([
            similar_tasks,
            effective_approaches,
            failed_approaches,
            user_preferences,
            domain_context,
        ])

        if has_relevant_memories:
            log.info(
                "memory_context_assembled",
                similar_tasks=len(similar_tasks),
                effective_approaches=len(effective_approaches),
                failed_approaches=len(failed_approaches),
                user_preferences=len(user_preferences),
                domain_context=len(domain_context),
            )
        else:
            log.info("no_relevant_memories_found", user_id=user_id)

        return MemoryContext(
            similar_tasks=similar_tasks,
            effective_approaches=effective_approaches,
            failed_approaches=failed_approaches,
            user_preferences=user_preferences,
            domain_context=domain_context,
            has_relevant_memories=has_relevant_memories,
        )

    def format_for_prompt(self, context: MemoryContext) -> str:
        """
        Format a MemoryContext into a readable text block for injection into
        the Supervisor's LLM prompt.

        WHY ONLY NON-EMPTY SECTIONS ARE INCLUDED
        ------------------------------------------
        Including empty section headers ("### Similar Past Tasks:\n(none)")
        wastes tokens and may confuse the model by drawing attention to absent
        information.  We only emit sections that have content.

        WHY THIS IS A PLAIN TEXT BLOCK (not JSON)
        -------------------------------------------
        The LLM reads this as part of its user-message context.  Plain
        markdown is more natural for the model and cheaper in tokens than
        JSON.  The structured MemoryDocument data has already served its
        purpose (filtering, sorting); the formatted text is the "last mile"
        before the LLM sees it.

        Parameters
        ----------
        context : MemoryContext
            The assembled context from retrieve_context().

        Returns
        -------
        str
            A formatted markdown block, or empty string if no memories.
        """
        if not context.has_relevant_memories:
            return ""

        sections: list[str] = ["## Past Context (from memory)\n"]

        if context.similar_tasks:
            sections.append("### Similar Past Tasks:")
            for doc in context.similar_tasks:
                meta = doc.metadata
                outcome = "success" if meta.get("success", True) else "failed"
                agents = meta.get("agents_used", "")
                # Content starts with "Task: <description>\n..."
                task_line = doc.content.split("\n")[0].replace("Task: ", "")
                entry = f"- Task: \"{task_line[:120]}\" → {outcome}"
                if agents:
                    entry += f", used {agents}"
                sections.append(entry)
            sections.append("")

        if context.effective_approaches:
            sections.append("### Effective Approaches:")
            for doc in context.effective_approaches:
                meta = doc.metadata
                problem_type = meta.get("problem_type", "general")
                confidence = meta.get("confidence_score", 0.0)
                # Content starts with "Problem type: <type>\nApproach: <desc>\n..."
                lines = doc.content.split("\n")
                approach_line = ""
                for line in lines:
                    if line.startswith("Approach:"):
                        approach_line = line.replace("Approach: ", "")
                        break
                if approach_line:
                    sections.append(
                        f"- For {problem_type} tasks (confidence {confidence:.0%}): "
                        f"{approach_line[:150]}"
                    )
                else:
                    sections.append(f"- {doc.content[:150]}")
            sections.append("")

        if context.failed_approaches:
            sections.append("### Approaches to Avoid:")
            for doc in context.failed_approaches:
                meta = doc.metadata
                problem_type = meta.get("problem_type", "general")
                lines = doc.content.split("\n")
                approach_line = ""
                for line in lines:
                    if line.startswith("Approach:"):
                        approach_line = line.replace("Approach: ", "")
                        break
                if approach_line:
                    sections.append(
                        f"- For {problem_type} tasks (failed): {approach_line[:150]}"
                    )
                else:
                    sections.append(f"- {doc.content[:150]}")
            sections.append("")

        if context.user_preferences:
            sections.append("### User Preferences:")
            for doc in context.user_preferences:
                sections.append(f"- {doc.content[:200]}")
            sections.append("")

        if context.domain_context:
            sections.append("### Relevant Domain Facts:")
            for doc in context.domain_context:
                sections.append(f"- {doc.content[:200]}")
            sections.append("")

        return "\n".join(sections).strip()
