"""Compaction baseline adapter for LENS.

The "naive memory" baseline from the literature: summarize-then-answer.
All buffered episodes are re-summarized from scratch at each checkpoint
via a single LLM call. The summary is returned as the sole search result.

This establishes the floor for what counts as a useful memory system —
any system that doesn't beat compaction is doing worse than a simple
"compress everything and search the summary" approach.

Environment variables:
    LENS_LLM_API_KEY / OPENAI_API_KEY   — API key for the compaction LLM
    LENS_LLM_API_BASE / OPENAI_BASE_URL — Base URL for the LLM API
    COMPACTION_MAX_TOKENS               — Max tokens for summary (default 2000)
"""
from __future__ import annotations

import logging
import os
import re

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.core.errors import AdapterError

logger = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"\[([^\]]+)\]")

_COMPACTION_SYSTEM = (
    "You are a memory compaction agent. You will be given a series of sequential "
    "episode logs. Your task is to compress them into a single summary document "
    "that preserves the most important information for future analytical queries."
)

_COMPACTION_USER_TMPL = """\
EPISODES ({n} total, {first_ts} to {last_ts}):

{episodes_block}

COMPRESSION OBJECTIVE:
Compress these episodes into a summary. Cite [episode_id] for specific data points. \
Preserve numeric values exactly. Focus on patterns and changes across episodes rather \
than repeating each entry. Prioritise information that reveals trends, anomalies, or \
cause-and-effect relationships.

Max output: approximately {max_tokens} tokens.

SUMMARY:"""


def _strip_provider_prefix(model: str) -> str:
    """Convert 'together/Qwen/Qwen3-...' → 'Qwen/Qwen3-...' for OpenAI API."""
    if "/" in model and model.startswith(("together/", "openai/")):
        return model.split("/", 1)[1]
    return model


@register_adapter("compaction")
class CompactionAdapter(MemoryAdapter):
    """Compaction baseline — summarize all episodes, search the summary.

    At each checkpoint, all buffered episodes are re-summarized from scratch.
    Search returns the summary as a single result. Retrieve supports both the
    summary document and original episode IDs (for evidence grounding via
    citations in the summary).
    """

    requires_metering: bool = True  # LLM calls in prepare()

    def __init__(self) -> None:
        self._episodes: list[dict] = []
        self._summary: str = ""
        self._cited_episode_ids: list[str] = []
        self._scope_id: str | None = None
        self._max_tokens = int(os.environ.get("COMPACTION_MAX_TOKENS", "2000"))
        # Track episodes by scope for cross-scope isolation
        self._scope_episodes: dict[str, list[dict]] = {}

    def reset(self, scope_id: str) -> None:
        # Remove episodes for this scope only
        self._scope_episodes.pop(scope_id, None)
        self._episodes = []
        self._summary = ""
        self._cited_episode_ids = []
        self._scope_id = scope_id
        # Rebuild episodes from remaining scopes
        for eps in self._scope_episodes.values():
            self._episodes.extend(eps)

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Buffer episode — instant, no I/O."""
        ep = {
            "episode_id": episode_id,
            "scope_id": scope_id,
            "timestamp": timestamp,
            "text": text,
            "meta": meta or {},
        }
        self._episodes.append(ep)
        self._scope_episodes.setdefault(scope_id, []).append(ep)

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Re-summarize ALL buffered episodes from scratch."""
        if not self._episodes:
            return

        if _OpenAI is None:
            logger.error("openai package required for compaction adapter")
            return

        # Build episode block
        lines: list[str] = []
        for ep in self._episodes:
            lines.append(f"[{ep['episode_id']}] {ep['timestamp']}: {ep['text']}")
        episodes_block = "\n\n".join(lines)

        first_ts = self._episodes[0]["timestamp"]
        last_ts = self._episodes[-1]["timestamp"]

        user_msg = _COMPACTION_USER_TMPL.format(
            n=len(self._episodes),
            first_ts=first_ts,
            last_ts=last_ts,
            episodes_block=episodes_block,
            max_tokens=self._max_tokens,
        )

        # Resolve API credentials
        api_key = (
            os.environ.get("LENS_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        base_url = (
            os.environ.get("LENS_LLM_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        model_raw = os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini")
        model = _strip_provider_prefix(model_raw)

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        oai = _OpenAI(**client_kwargs)
        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _COMPACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            self._summary = resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("Compaction LLM call failed: %s", e)
            # Non-fatal — continue with empty summary
            self._summary = ""
            return

        # Parse cited episode IDs from summary
        self._cited_episode_ids = _EP_ID_RE.findall(self._summary)

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if self._summary:
            return [SearchResult(
                ref_id="compaction_summary",
                text=self._summary[:500],
                score=1.0,
                metadata={"type": "compaction_summary", "cited_episodes": len(self._cited_episode_ids)},
            )]
        # Fallback: return individual episode results if no summary but episodes exist
        if self._episodes:
            cap = limit or 10
            results: list[SearchResult] = []
            for ep in self._episodes[:cap]:
                results.append(SearchResult(
                    ref_id=ep["episode_id"],
                    text=ep["text"][:500],
                    score=0.5,
                    metadata=ep.get("meta", {}),
                ))
            return results
        return []

    def retrieve(self, ref_id: str) -> Document | None:
        if ref_id == "compaction_summary":
            if self._summary:
                return Document(ref_id="compaction_summary", text=self._summary)
            return None
        if ref_id == "compaction_fallback":
            if self._episodes:
                concat = "\n".join(
                    f"[{ep['episode_id']}] {ep['timestamp']}: {ep['text']}"
                    for ep in self._episodes
                )
                return Document(ref_id="compaction_fallback", text=concat)
            return None
        # Support original episode IDs (for evidence grounding via citations)
        for ep in self._episodes:
            if ep["episode_id"] == ref_id:
                return Document(
                    ref_id=ref_id,
                    text=ep["text"],
                    metadata=ep.get("meta", {}),
                )
        return None

    def get_synthetic_refs(self) -> list[tuple[str, str]]:
        if self._summary:
            return [("compaction_summary", self._summary)]
        return []

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["compaction"],
            max_results_per_search=1,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple documents by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times. "
                        "Valid ref_ids: 'compaction_summary' (the full summary), or original "
                        "episode IDs cited in the summary (e.g. 'scope_01_ep_005')."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "ref_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reference IDs to retrieve.",
                            },
                        },
                        "required": ["ref_ids"],
                    },
                ),
            ],
        )

    def call_extended_tool(self, tool_name: str, arguments: dict) -> object:
        if tool_name == "batch_retrieve":
            ref_ids = arguments.get("ref_ids", [])
            docs = []
            for ref_id in ref_ids:
                doc = self.retrieve(ref_id)
                if doc is not None:
                    docs.append(doc.to_dict())
            return {"documents": docs, "count": len(docs)}
        return super().call_extended_tool(tool_name, arguments)

    def get_cache_state(self) -> dict | None:
        """Return state needed to restore compaction summary."""
        if not self._summary:
            return None
        return {
            "summary": self._summary,
            "episodes": self._episodes,
            "cited_episode_ids": self._cited_episode_ids,
            "scope_id": self._scope_id,
        }

    def restore_cache_state(self, state: dict) -> bool:
        """Restore summary and episodes from cached state."""
        try:
            self._summary = state["summary"]
            self._episodes = state.get("episodes", [])
            self._cited_episode_ids = state.get("cited_episode_ids", [])
            self._scope_id = state.get("scope_id")
            # Rebuild scope_episodes index
            self._scope_episodes = {}
            for ep in self._episodes:
                sid = ep.get("scope_id", self._scope_id)
                self._scope_episodes.setdefault(sid, []).append(ep)
            logger.info(
                "Restored Compaction cache: %d episodes, summary=%d chars",
                len(self._episodes),
                len(self._summary),
            )
            return True
        except Exception as e:
            logger.warning("Failed to restore Compaction cache: %s", e)
            return False
