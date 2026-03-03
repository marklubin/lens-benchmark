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
        """Summarize buffered episodes with incremental batching.

        If all episodes fit in one LLM call, summarize in a single pass.
        Otherwise, split into batches that fit within the context window,
        summarize each batch, then merge the batch summaries into a final
        summary. This prevents context overflow with large narrative scopes.
        """
        if not self._episodes:
            return

        if _OpenAI is None:
            logger.error("openai package required for compaction adapter")
            return

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
        model_raw = os.environ.get("LENS_LLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
        model = _strip_provider_prefix(model_raw)

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        oai = _OpenAI(**client_kwargs)

        # Estimate tokens: ~1.3 tokens per word, ~4 chars per token
        max_input_chars = int(os.environ.get(
            "COMPACTION_MAX_INPUT_CHARS", "800000"
        ))  # ~200K tokens, leaves room for system prompt + output

        # Build per-episode text blocks
        ep_blocks: list[tuple[dict, str]] = []
        for ep in self._episodes:
            block = f"[{ep['episode_id']}] {ep['timestamp']}: {ep['text']}"
            ep_blocks.append((ep, block))

        # Check if everything fits in one call
        total_chars = sum(len(b) for _, b in ep_blocks)
        if total_chars <= max_input_chars:
            # Single-pass: all episodes fit
            self._summary = self._compact_batch(
                oai, model, ep_blocks,
            )
        else:
            # Multi-pass: split into batches, summarize each, then merge
            batches: list[list[tuple[dict, str]]] = []
            current_batch: list[tuple[dict, str]] = []
            current_chars = 0
            for ep, block in ep_blocks:
                if current_chars + len(block) > max_input_chars and current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_chars = 0
                current_batch.append((ep, block))
                current_chars += len(block)
            if current_batch:
                batches.append(current_batch)

            logger.info(
                "Compaction: splitting %d episodes into %d batches",
                len(self._episodes), len(batches),
            )

            # Summarize each batch
            batch_summaries: list[str] = []
            for i, batch in enumerate(batches):
                summary = self._compact_batch(oai, model, batch)
                if summary:
                    batch_summaries.append(summary)
                    logger.info(
                        "Compaction batch %d/%d: %d episodes -> %d chars",
                        i + 1, len(batches), len(batch), len(summary),
                    )

            if not batch_summaries:
                self._summary = ""
                return

            if len(batch_summaries) == 1:
                self._summary = batch_summaries[0]
            else:
                # Merge batch summaries into final summary
                self._summary = self._merge_summaries(
                    oai, model, batch_summaries,
                )

        # Parse cited episode IDs from summary
        self._cited_episode_ids = _EP_ID_RE.findall(self._summary)

    def _compact_batch(
        self,
        oai: _OpenAI,
        model: str,
        ep_blocks: list[tuple[dict, str]],
    ) -> str:
        """Summarize a batch of episodes into a single summary."""
        episodes_block = "\n\n".join(block for _, block in ep_blocks)
        first_ts = ep_blocks[0][0]["timestamp"]
        last_ts = ep_blocks[-1][0]["timestamp"]

        user_msg = _COMPACTION_USER_TMPL.format(
            n=len(ep_blocks),
            first_ts=first_ts,
            last_ts=last_ts,
            episodes_block=episodes_block,
            max_tokens=self._max_tokens,
        )

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
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("Compaction LLM call failed: %s", e)
            return ""

    def _merge_summaries(
        self,
        oai: _OpenAI,
        model: str,
        summaries: list[str],
    ) -> str:
        """Merge multiple batch summaries into a single final summary."""
        summaries_block = "\n\n---\n\n".join(
            f"BATCH {i+1}:\n{s}" for i, s in enumerate(summaries)
        )
        user_msg = (
            f"PARTIAL SUMMARIES ({len(summaries)} batches):\n\n"
            f"{summaries_block}\n\n"
            "MERGE OBJECTIVE:\n"
            "Merge these partial summaries into a single coherent summary. "
            "Preserve all [episode_id] citations. Focus on cross-batch patterns, "
            "trends, and relationships. Remove redundancy.\n\n"
            f"Max output: approximately {self._max_tokens} tokens.\n\n"
            "MERGED SUMMARY:"
        )
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
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("Compaction merge failed: %s", e)
            # Fall back to concatenating summaries
            return "\n\n".join(summaries)

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
