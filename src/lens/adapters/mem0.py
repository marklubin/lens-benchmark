"""Mem0 memory adapters: raw vector search and LLM-based extraction.

Requires: pip install mem0ai
Local setup: docker run -d -p 6333:6333 qdrant/qdrant
"""
from __future__ import annotations

import os

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.core.errors import AdapterError


def _check_mem0_available():
    """Raise AdapterError if mem0ai is not installed."""
    try:
        import mem0  # noqa: F401
    except ImportError:
        raise AdapterError(
            "mem0ai package not installed. Install with: pip install mem0ai"
        )


def _build_mem0_config() -> dict:
    """Build Mem0 config from environment variables.

    Embedding env vars:
        MEM0_EMBED_PROVIDER  - embedder provider (default: openai)
        MEM0_EMBED_MODEL     - embedding model name
        MEM0_EMBED_API_KEY   - API key for embedder
        MEM0_EMBED_BASE_URL  - base URL for embedder (OpenAI-compatible)
        MEM0_EMBED_DIMS      - embedding dimensions

    LLM env vars (Mem0 always initializes an LLM, even for infer=False):
        MEM0_LLM_API_KEY     - API key for Mem0's internal LLM
        MEM0_LLM_BASE_URL    - base URL for Mem0's internal LLM
        MEM0_LLM_MODEL       - model name for Mem0's internal LLM
    """
    qdrant_url = os.environ.get("MEM0_QDRANT_URL", "http://localhost:6333")

    # Embedder config
    embed_provider = os.environ.get("MEM0_EMBED_PROVIDER", "openai")
    embed_model = os.environ.get("MEM0_EMBED_MODEL", "text-embedding-3-small")
    embed_api_key = os.environ.get("MEM0_EMBED_API_KEY")
    embed_base_url = os.environ.get("MEM0_EMBED_BASE_URL")
    embed_dims = os.environ.get("MEM0_EMBED_DIMS")

    embedder_config: dict = {"model": embed_model}
    if embed_api_key:
        embedder_config["api_key"] = embed_api_key
    if embed_base_url:
        embedder_config["openai_base_url"] = embed_base_url
    if embed_dims:
        embedder_config["embedding_dims"] = int(embed_dims)

    # LLM config (Mem0 always initializes this, even for infer=False)
    llm_api_key = os.environ.get("MEM0_LLM_API_KEY")
    llm_base_url = os.environ.get("MEM0_LLM_BASE_URL")
    llm_model = os.environ.get("MEM0_LLM_MODEL")

    llm_config: dict = {}
    if llm_model:
        llm_config["model"] = llm_model
    if llm_api_key:
        llm_config["api_key"] = llm_api_key
    if llm_base_url:
        llm_config["openai_base_url"] = llm_base_url

    # Qdrant needs to know the embedding dimensions for collection creation
    qdrant_dims = int(embed_dims) if embed_dims else 1536

    config: dict = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": qdrant_url,
                "collection_name": "lens_benchmark",
                "embedding_model_dims": qdrant_dims,
            },
        },
        "embedder": {
            "provider": embed_provider,
            "config": embedder_config,
        },
    }

    if llm_config:
        config["llm"] = {
            "provider": "openai",
            "config": llm_config,
        }

    return config


def _patch_embed_no_dims(client: object) -> None:
    """Monkey-patch Mem0's embedder to omit the `dimensions` kwarg.

    Some OpenAI-compatible providers (Together AI, etc.) reject the
    `dimensions` parameter for models with fixed output dimensions
    (e.g. BGE).  Mem0 always sends it, so we wrap the embed method.
    """
    embedder = getattr(client, "embedding_model", None)
    if embedder is None:
        return
    original_embed = embedder.embed

    def _embed_no_dims(text, memory_action=None):
        # Call the OpenAI client directly without dimensions
        text = text.replace("\n", " ")
        return (
            embedder.client.embeddings.create(
                input=[text], model=embedder.config.model
            )
            .data[0]
            .embedding
        )

    embedder.embed = _embed_no_dims


class _Mem0Base(MemoryAdapter):
    """Shared base for Mem0 adapters."""

    requires_metering: bool = False

    def __init__(self) -> None:
        _check_mem0_available()
        from mem0 import Memory

        config = _build_mem0_config()
        try:
            self._client = Memory.from_config(config)
        except Exception as e:
            raise AdapterError(
                f"Failed to initialize Mem0 client. Is Qdrant running at "
                f"{config['vector_store']['config']['url']}? Error: {e}"
            ) from e

        # Patch: strip `dimensions` from embed calls when the provider
        # doesn't support it (e.g. Together AI serving BGE models).
        if os.environ.get("MEM0_EMBED_NO_DIMS"):
            _patch_embed_no_dims(self._client)

        # Local index: episode_id -> list of memory IDs for retrieval
        self._ep_index: dict[str, list[str]] = {}
        # Track current scope for search (mem0 requires user_id)
        self._current_scope_id: str | None = None

    def reset(self, scope_id: str) -> None:
        """Clear all memories for a scope."""
        self._current_scope_id = scope_id
        try:
            self._client.delete_all(user_id=scope_id)
        except Exception:
            pass  # Best effort -- collection may not exist yet
        # Clear local index entries for this scope
        self._ep_index = {
            k: v for k, v in self._ep_index.items()
            if not k.startswith(f"{scope_id}:")
        }

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        limit = limit or 10

        # mem0 requires user_id for search — use filters or stored scope_id
        user_id = None
        if filters and "scope_id" in filters:
            user_id = filters["scope_id"]
        elif self._current_scope_id:
            user_id = self._current_scope_id

        if not user_id:
            return []  # Can't search without user_id in mem0

        kwargs: dict = {"query": query, "limit": limit, "user_id": user_id}

        try:
            raw = self._client.search(**kwargs)
        except Exception:
            return []

        # mem0 returns {"results": [...]} dict, not a bare list
        if isinstance(raw, dict):
            results = raw.get("results", [])
        elif isinstance(raw, list):
            results = raw
        else:
            results = []

        search_results = []
        for r in results:
            # Each result is a dict with id, memory, score, metadata
            memory_text = r.get("memory", "") if isinstance(r, dict) else str(r)
            score = r.get("score", 0.0) if isinstance(r, dict) else 0.0
            metadata = r.get("metadata", {}) if isinstance(r, dict) else {}
            mem_id = r.get("id", "") if isinstance(r, dict) else ""

            # Map back to episode_id if possible
            ep_id = metadata.get("episode_id", mem_id)

            search_results.append(SearchResult(
                ref_id=ep_id,
                text=memory_text[:500],
                score=score,
                metadata=metadata,
            ))

        return search_results[:limit]

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve by episode_id using local index."""
        for key, mem_ids in self._ep_index.items():
            if key.endswith(f":{ref_id}"):
                for mem_id in mem_ids:
                    try:
                        result = self._client.get(mem_id)
                        if result:
                            text = result.get("memory", "") if isinstance(result, dict) else str(result)
                            metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
                            return Document(ref_id=ref_id, text=text, metadata=metadata)
                    except Exception:
                        continue
        return None

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic"],
            max_results_per_search=10,
        )


def _extract_mem_ids(result: object) -> list[str]:
    """Extract memory IDs from a Mem0 add() result."""
    mem_ids = []
    if isinstance(result, dict) and "results" in result:
        for r in result["results"]:
            if isinstance(r, dict) and "id" in r:
                mem_ids.append(r["id"])
    elif isinstance(result, list):
        for r in result:
            if isinstance(r, dict) and "id" in r:
                mem_ids.append(r["id"])
    return mem_ids


@register_adapter("mem0-raw")
class Mem0RawAdapter(_Mem0Base):
    """Mem0 adapter with raw vector search (no LLM extraction).

    Stores episode text directly in Mem0 with infer=False,
    bypassing Mem0's fact extraction LLM pipeline.
    """

    requires_metering = False

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        self._current_scope_id = scope_id
        metadata = dict(meta or {})
        metadata["episode_id"] = episode_id
        metadata["timestamp"] = timestamp

        result = self._client.add(
            text,
            user_id=scope_id,
            metadata=metadata,
            infer=False,
        )

        self._ep_index[f"{scope_id}:{episode_id}"] = _extract_mem_ids(result)

    def get_cache_state(self) -> dict | None:
        """Return state needed to skip re-ingest on cache hit."""
        import logging

        log = logging.getLogger(__name__)
        if not self._ep_index:
            return None
        log.info(
            "Caching mem0-raw state: %d ep_index entries, scope=%s",
            len(self._ep_index),
            self._current_scope_id,
        )
        return {
            "ep_index": self._ep_index,
            "current_scope_id": self._current_scope_id,
        }

    def restore_cache_state(self, state: dict) -> bool:
        """Restore ep_index and scope; Qdrant data persists across runs."""
        import logging

        log = logging.getLogger(__name__)
        try:
            self._ep_index = state["ep_index"]
            self._current_scope_id = state.get("current_scope_id")
            log.info(
                "Restored mem0-raw cache: %d ep_index entries, scope=%s",
                len(self._ep_index),
                self._current_scope_id,
            )
            return True
        except Exception as e:
            log.warning("Failed to restore mem0-raw cache: %s", e)
            return False


@register_adapter("mem0-extract")
class Mem0ExtractAdapter(_Mem0Base):
    """Mem0 adapter with LLM-based memory extraction.

    Buffers episodes during ingest, then processes them through Mem0's
    LLM extraction pipeline during prepare(). This extracts key facts
    and relationships from the raw text.
    """

    requires_metering = True

    def __init__(self) -> None:
        super().__init__()
        # Staged episodes waiting for prepare() to process
        self._staged: dict[str, list[tuple[str, str, str, str, dict]]] = {}

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Buffer episode locally -- no LLM calls during ingest."""
        self._current_scope_id = scope_id
        self._staged.setdefault(scope_id, []).append(
            (episode_id, scope_id, timestamp, text, dict(meta or {}))
        )

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Process staged episodes through Mem0's LLM extraction."""
        staged = self._staged.pop(scope_id, [])
        for episode_id, sid, timestamp, text, meta in staged:
            metadata = dict(meta)
            metadata["episode_id"] = episode_id
            metadata["timestamp"] = timestamp

            result = self._client.add(
                text,
                user_id=scope_id,
                metadata=metadata,
                infer=True,  # LLM extraction -- metered via proxy
            )

            self._ep_index[f"{scope_id}:{episode_id}"] = _extract_mem_ids(result)
