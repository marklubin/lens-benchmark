"""Cognee (GraphRAG) memory adapter for LENS.

Uses Cognee's graph-aware knowledge pipeline: ingests episodes as text,
builds a knowledge graph via cognify(), and retrieves chunks via vector
search. All embedded databases (SQLite, LanceDB, Kuzu) — no extra container.

Requires:
    uv add cognee

Environment variables:
    COGNEE_LLM_API_KEY     LLM API key (required)
    COGNEE_LLM_MODEL       LLM model (default: Qwen/Qwen3-235B-A22B-Instruct-2507-tput)
    COGNEE_LLM_ENDPOINT    LLM API base URL (default: https://api.together.xyz/v1)
    COGNEE_EMBED_API_KEY   Embedding API key (required)
    COGNEE_EMBED_MODEL     Embedding model (default: Alibaba-NLP/gte-modernbert-base)
    COGNEE_EMBED_ENDPOINT  Embedding API base URL (default: https://api.together.xyz/v1)
    COGNEE_EMBED_DIMS      Embedding dimensions (default: 768)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import uuid

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.core.errors import AdapterError

log = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")


def _parse_ep_id(text: str) -> str | None:
    """Extract episode_id from '[ep_id] ...' prefixed text."""
    m = _EP_ID_RE.match(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Thread-hosted event loop — avoids asyncio.run() cross-loop issues
# ---------------------------------------------------------------------------


class _AsyncRunner:
    """Hosts a persistent event loop in a daemon thread for sync→async bridging."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 600.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


_RUNNER: _AsyncRunner | None = None
_RUNNER_LOCK = threading.Lock()


def _get_runner() -> _AsyncRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = _AsyncRunner()
    return _RUNNER


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter("cognee")
class CogneeAdapter(MemoryAdapter):
    """Cognee GraphRAG adapter for LENS.

    Ingests episodes as raw text, builds a knowledge graph with LLM entity
    extraction (cognify), and retrieves episodes via CHUNKS vector search.

    Scope isolation: reset() calls cognee.prune() to clear all data, then
    assigns a fresh dataset name. Episodes are buffered in ingest() and
    processed in prepare() (before agent's budget clock).

    Note: cognee stores databases inside the package directory by default.
    Set COGNEE_DATA_DIR to override.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._llm_api_key = os.environ.get("COGNEE_LLM_API_KEY", "")
        self._llm_model = os.environ.get(
            "COGNEE_LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
        )
        self._llm_endpoint = os.environ.get(
            "COGNEE_LLM_ENDPOINT", "https://api.together.xyz/v1"
        )
        self._embed_api_key = os.environ.get("COGNEE_EMBED_API_KEY", "")
        self._embed_model = os.environ.get(
            "COGNEE_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base"
        )
        self._embed_endpoint = os.environ.get(
            "COGNEE_EMBED_ENDPOINT", "https://api.together.xyz/v1"
        )
        self._embed_dims = int(os.environ.get("COGNEE_EMBED_DIMS", "768"))

        # Propagate embedding config via env vars so pydantic settings pick them up
        self._apply_env_config()

        # Per-run state
        self._dataset_id: str | None = None
        self._text_cache: dict[str, str] = {}
        self._pending_episodes: list[dict] = []

    def _apply_env_config(self) -> None:
        """Propagate COGNEE_* env vars to the names cognee's pydantic settings read."""
        if self._embed_api_key:
            os.environ.setdefault("EMBEDDING_API_KEY", self._embed_api_key)
        if self._embed_model:
            os.environ.setdefault("EMBEDDING_MODEL", self._embed_model)
        if self._embed_dims:
            os.environ.setdefault("EMBEDDING_DIMENSIONS", str(self._embed_dims))
        if self._embed_endpoint:
            os.environ.setdefault("EMBEDDING_ENDPOINT", self._embed_endpoint)
        os.environ.setdefault("EMBEDDING_PROVIDER", "openai")

    def _get_cognee(self):
        """Lazy import cognee and configure LLM settings."""
        try:
            import cognee  # noqa: PLC0415
        except ImportError as e:
            raise AdapterError(
                "cognee not installed. Run: uv add cognee"
            ) from e

        # Configure LLM programmatically (persists for process lifetime)
        try:
            cognee.config.set_llm_config({
                "llm_provider": "openai",
                "llm_model": self._llm_model,
                "llm_endpoint": self._llm_endpoint,
                "llm_api_key": self._llm_api_key,
            })
        except Exception as e:
            log.warning("cognee.config.set_llm_config failed: %s", e)

        return cognee

    def reset(self, scope_id: str) -> None:
        """Clear all cognee data and set a fresh dataset name.

        Calls cognee.prune() to wipe all graphs, vectors, and relational
        data — ensuring clean isolation between benchmark runs.
        """
        suffix = uuid.uuid4().hex[:8]
        safe_scope = "".join(c if c.isalnum() or c == "_" else "_" for c in scope_id)
        self._dataset_id = f"lens_{safe_scope}_{suffix}"
        self._text_cache = {}
        self._pending_episodes = []

        cognee = self._get_cognee()
        try:
            _get_runner().run(cognee.prune(), timeout=120.0)
        except Exception as e:
            raise AdapterError(
                f"cognee.prune() failed during reset for scope '{scope_id}': {e}"
            ) from e

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Buffer an episode for batch processing in prepare().

        Instant (<1ms) — no LLM or I/O calls. Episodes accumulate here and
        are flushed to cognee in prepare() before the agent's budget clock.
        """
        if self._dataset_id is None:
            raise AdapterError("reset() must be called before ingest()")

        content = f"[{episode_id}] {timestamp}: {text}"
        self._pending_episodes.append(
            {"episode_id": episode_id, "content": content}
        )
        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Flush buffered episodes to cognee and build the knowledge graph.

        Calls cognee.add() for each episode (stores raw chunks) then
        cognee.cognify() to extract entities and build the graph.
        This is where LLM processing happens — before agent's budget clock.
        """
        if not self._pending_episodes or self._dataset_id is None:
            return

        cognee = self._get_cognee()

        # Add each episode as raw text to cognee
        for item in self._pending_episodes:
            try:
                _get_runner().run(
                    cognee.add(
                        data=item["content"],
                        dataset_name=self._dataset_id,
                    ),
                    timeout=120.0,
                )
            except Exception as e:
                raise AdapterError(
                    f"cognee.add() failed for episode '{item['episode_id']}' "
                    f"at checkpoint {checkpoint}: {e}"
                ) from e

        # Build knowledge graph from added episodes
        try:
            _get_runner().run(
                cognee.cognify(datasets=[self._dataset_id]),
                timeout=600.0,
            )
        except Exception as e:
            raise AdapterError(
                f"cognee.cognify() failed at checkpoint {checkpoint}: {e}"
            ) from e

        self._pending_episodes = []

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search cognee using CHUNKS vector search, returning episode-grounded results.

        Uses cognee's vector search over chunked episode text. Each chunk is
        prefixed with [episode_id], which we parse to build the ref_id for
        evidence grounding.
        """
        if not query or not query.strip() or self._dataset_id is None:
            return []

        cap = limit or 10
        cognee = self._get_cognee()
        SearchType = cognee.SearchType

        try:
            raw = _get_runner().run(
                cognee.search(
                    query_text=query,
                    query_type=SearchType.CHUNKS,
                    top_k=cap * 2,  # over-fetch for dedup
                ),
                timeout=60.0,
            )
        except Exception as e:
            log.warning("cognee.search() failed: %s", e)
            return []

        search_results: list[SearchResult] = []
        seen_episode_ids: set[str] = set()

        for item in raw or []:
            if len(search_results) >= cap:
                break

            # Unwrap cognee SearchResult → payload dicts or plain strings
            chunk_list = self._extract_chunks(item)
            for chunk_text in chunk_list:
                if len(search_results) >= cap:
                    break
                if not chunk_text:
                    continue

                ep_id = _parse_ep_id(chunk_text)
                if ep_id is None:
                    # Best-effort: try matching against known episode text prefixes
                    ep_id = self._match_episode(chunk_text)

                if ep_id is None or ep_id in seen_episode_ids:
                    continue
                seen_episode_ids.add(ep_id)

                search_results.append(
                    SearchResult(
                        ref_id=ep_id,
                        text=chunk_text[:500],
                        score=0.5,
                    )
                )

        return search_results

    def _extract_chunks(self, item) -> list[str]:
        """Unpack a cognee search result item into a list of chunk text strings."""
        # cognee.SearchResult has .search_result which may be:
        # - a list of payload dicts {"text": ...}
        # - a string (joined context when only_context=True)
        # - a list of strings

        search_result = getattr(item, "search_result", item)

        if isinstance(search_result, str):
            return [search_result]

        if isinstance(search_result, list):
            texts = []
            for entry in search_result:
                if isinstance(entry, str):
                    texts.append(entry)
                elif isinstance(entry, dict):
                    text = entry.get("text") or entry.get("content") or ""
                    if text:
                        texts.append(str(text))
            return texts

        return []

    def _match_episode(self, chunk_text: str) -> str | None:
        """Attempt to match a chunk back to an episode via substring matching."""
        chunk_lower = chunk_text.lower()[:200]
        for ep_id, ep_text in self._text_cache.items():
            # Check if the chunk overlaps significantly with the episode text
            ep_prefix = ep_text[:100].lower()
            if ep_prefix and ep_prefix[:50] in chunk_lower:
                return ep_id
        return None

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve full episode text by episode_id from local cache."""
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", "graph"],
            max_results_per_search=10,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple full episodes by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times — it uses only "
                        "one tool call instead of one per document. "
                        "After memory_search, pass all ref_ids you want to read to this tool."
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
