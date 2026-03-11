"""BenchmarkRuntime — wraps a Synix Release with policy-gated search and context.

The runtime is the interface between the agent and the sealed artifact bank.
It enforces policy visibility: which artifacts the agent can search/see,
what context gets injected into the system prompt, and what tools are available.

A runtime is scoped to a single (scope, checkpoint, policy) triple.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from bench.schemas import PolicyManifest

logger = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    """Strip thinking preamble from Qwen3.5 output.

    Handles <think>...</think> XML tags and plain-text "Thinking Process:"
    sections that Qwen3.5 sometimes produces even with thinking disabled.
    """
    # XML think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Plain-text thinking preamble — often starts with "Thinking Process:" or
    # "Thinking:" followed by numbered steps, then the actual content
    # Look for the actual content after the thinking section
    m = re.match(r"(?:Thinking(?:\s+Process)?:\s*\n(?:.*?\n)*?\n)(.*)", text, re.DOTALL)
    if m and m.group(1).strip():
        text = m.group(1).strip()
    return text


# Tool definitions for OpenAI function-calling format
_MEMORY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Search the memory bank for information relevant to a query. "
            "Returns ranked results from episodes and chunks with relevance scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information.",
                },
            },
            "required": ["query"],
        },
    },
}


class BenchmarkRuntime:
    """Policy-gated access to a Synix Release.

    Wraps a Release object and filters search/retrieve/context access
    according to the policy manifest. The agent interacts with the bank
    exclusively through this runtime.
    """

    def __init__(
        self,
        release: Any,  # synix.Release — Any to avoid import at module level
        policy: PolicyManifest,
    ) -> None:
        self._release = release
        self._policy = policy

    @property
    def policy(self) -> PolicyManifest:
        return self._policy

    def search(self, query: str, *, limit: int | None = None) -> list[dict]:
        """Search the bank, gated by policy query_surfaces.

        Returns empty list if no query surfaces are allowed (null policy).
        """
        if not self._policy.query_surfaces:
            return []

        max_results = limit or self._policy.retrieval_caps.max_results

        # Use hybrid mode if both surfaces available, else first available
        surfaces = self._policy.query_surfaces
        if "keyword" in surfaces and "semantic" in surfaces:
            mode = "hybrid"
        elif "semantic" in surfaces:
            mode = "semantic"
        elif "keyword" in surfaces:
            mode = "keyword"
        else:
            return []

        results = self._release.search(query, mode=mode, limit=max_results)

        # Convert SDK results to dicts for the agent
        return [
            {
                "content": r.content,
                "label": r.label,
                "score": r.score,
                "layer": r.layer,
                "provenance": r.provenance,
            }
            for r in results
        ]

    def get_context(self) -> str | None:
        """Get derived artifact content for system prompt injection.

        Returns core-memory or summary content based on policy, or None
        for null/base policies that don't inject derived context.
        """
        families = self._policy.visible_artifact_families

        if "core_memory" in families:
            try:
                art = self._release.artifact("core-memory")
                return _strip_think(art.content)
            except Exception:
                logger.warning("core-memory artifact not found in release")
                return None

        if "summary" in families:
            try:
                art = self._release.artifact("summary")
                return _strip_think(art.content)
            except Exception:
                logger.warning("summary artifact not found in release")
                return None

        return None

    def get_tools(self) -> list[dict]:
        """Return OpenAI function-calling tool definitions based on policy.

        null policy → no tools
        All other policies → memory_search
        """
        if not self._policy.query_surfaces:
            return []
        return [_MEMORY_SEARCH_TOOL]

    def dispatch_tool(self, name: str, args: dict) -> str:
        """Execute a tool call and return the result as a string.

        Raises ValueError for unknown tools or tools not allowed by policy.
        """
        if name != "memory_search":
            raise ValueError(f"Unknown tool: {name!r}")

        if not self._policy.query_surfaces:
            raise ValueError(f"Tool {name!r} not allowed by policy {self._policy.policy_id!r}")

        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query parameter is required"})

        results = self.search(query)

        # Format results for the agent — truncate content to prevent
        # full-episode dumps (episodes can be 30-48K chars) from blowing
        # up the agent's context window
        MAX_RESULT_CHARS = 1500
        formatted = []
        for r in results:
            content = r["content"]
            if len(content) > MAX_RESULT_CHARS:
                content = content[:MAX_RESULT_CHARS] + f"\n[... truncated, {len(r['content'])} chars total]"
            formatted.append({
                "content": content,
                "label": r["label"],
                "score": round(r["score"], 4) if r["score"] else None,
                "source": r.get("provenance", []),
            })
        return json.dumps(formatted)
