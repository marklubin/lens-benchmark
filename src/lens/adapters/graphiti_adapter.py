"""Graphiti (temporal knowledge graph) memory adapter for LENS.

Uses Graphiti's bi-temporal knowledge graph with FalkorDB backend for
longitudinal episode storage. Extracts entities and relationships from
episodes and supports graph-aware semantic search.

Requires:
    uv add graphiti-core[falkordb]
    FalkorDB running locally (default: localhost:6379)
    podman run -d -p 6379:6379 --name falkordb falkordb/falkordb

Environment variables:
    GRAPHITI_LLM_API_KEY      LLM provider API key (required)
    GRAPHITI_LLM_MODEL        LLM model name (default: Qwen/Qwen3-235B-A22B-Instruct-2507-tput)
    GRAPHITI_LLM_BASE_URL     LLM API base URL (default: https://api.together.xyz/v1)
    GRAPHITI_EMBED_API_KEY    Embedding API key (required)
    GRAPHITI_EMBED_MODEL      Embedding model (default: Alibaba-NLP/gte-modernbert-base)
    GRAPHITI_EMBED_BASE_URL   Embedding API base URL (default: https://api.together.xyz/v1)
    GRAPHITI_EMBED_DIM        Embedding dimensions (default: 768)
    GRAPHITI_FALKORDB_HOST    FalkorDB host (default: localhost)
    GRAPHITI_FALKORDB_PORT    FalkorDB port (default: 6379)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from datetime import datetime, timezone

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


# ---------------------------------------------------------------------------
# Thread-hosted event loop — avoids asyncio.run() cross-loop issues
# ---------------------------------------------------------------------------


class _AsyncRunner:
    """Hosts a persistent event loop in a daemon thread for sync→async bridging."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 300.0):
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


@register_adapter("graphiti")
class GraphitiAdapter(MemoryAdapter):
    """Graphiti temporal knowledge graph adapter for LENS.

    Stores episodes in FalkorDB as a knowledge graph with entity extraction.
    Each reset() creates a unique FalkorDB database for scope isolation.
    Episodes are buffered in ingest() and added to the graph in prepare()
    (where LLM entity extraction occurs, before the agent's budget clock).

    Search uses EDGE_HYBRID_SEARCH_EPISODE_MENTIONS which ranks extracted
    entity-relationship edges by episode provenance, returning edges that
    are most relevant to the query and most mentioned across episodes.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._llm_api_key = os.environ.get("GRAPHITI_LLM_API_KEY", "")
        self._llm_model = os.environ.get(
            "GRAPHITI_LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
        )
        self._llm_base_url = os.environ.get(
            "GRAPHITI_LLM_BASE_URL", "https://api.together.xyz/v1"
        )
        self._embed_api_key = os.environ.get("GRAPHITI_EMBED_API_KEY", "")
        self._embed_model = os.environ.get(
            "GRAPHITI_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base"
        )
        self._embed_base_url = os.environ.get(
            "GRAPHITI_EMBED_BASE_URL", "https://api.together.xyz/v1"
        )
        self._embed_dim = int(os.environ.get("GRAPHITI_EMBED_DIM", "768"))
        self._falkordb_host = os.environ.get("GRAPHITI_FALKORDB_HOST", "localhost")
        self._falkordb_port = int(os.environ.get("GRAPHITI_FALKORDB_PORT", "6379"))

        # Per-run state
        self._graphiti = None
        self._db_name: str | None = None
        self._text_cache: dict[str, str] = {}
        self._ep_uuid_to_id: dict[str, str] = {}  # episode node UUID → episode_id
        self._pending_episodes: list[dict] = []

    def _make_graphiti(self, db_name: str):
        """Create a Graphiti instance connected to a specific FalkorDB database."""
        try:
            from graphiti_core import Graphiti  # noqa: PLC0415
            from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: PLC0415
            from graphiti_core.embedder.openai import (  # noqa: PLC0415
                OpenAIEmbedder,
                OpenAIEmbedderConfig,
            )
            from graphiti_core.llm_client.config import LLMConfig  # noqa: PLC0415
            from graphiti_core.llm_client.openai_client import OpenAIClient  # noqa: PLC0415
        except ImportError as e:
            raise AdapterError(
                "graphiti-core[falkordb] not installed. Run: uv add graphiti-core[falkordb]"
            ) from e

        llm_client = OpenAIClient(
            LLMConfig(
                api_key=self._llm_api_key,
                model=self._llm_model,
                base_url=self._llm_base_url,
            )
        )
        embedder = OpenAIEmbedder(
            OpenAIEmbedderConfig(
                embedding_model=self._embed_model,
                api_key=self._embed_api_key,
                base_url=self._embed_base_url,
                embedding_dim=self._embed_dim,
            )
        )
        # FalkorDB database parameter provides scope isolation at the DB level
        driver = FalkorDriver(
            host=self._falkordb_host,
            port=self._falkordb_port,
            database=db_name,
        )
        return Graphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder)

    def reset(self, scope_id: str) -> None:
        """Create a fresh FalkorDB database with a unique run-scoped name.

        Each reset produces a distinct database name, ensuring complete
        isolation between benchmark runs without needing to delete old data.
        """
        suffix = uuid.uuid4().hex[:8]
        # FalkorDB names: alphanumeric + underscores only
        safe_scope = "".join(c if c.isalnum() or c == "_" else "_" for c in scope_id)
        self._db_name = f"lens_{safe_scope}_{suffix}"
        self._text_cache = {}
        self._ep_uuid_to_id = {}
        self._pending_episodes = []

        try:
            self._graphiti = self._make_graphiti(self._db_name)
            _get_runner().run(
                self._graphiti.build_indices_and_constraints(), timeout=60.0
            )
        except AdapterError:
            raise
        except Exception as e:
            raise AdapterError(
                f"Failed to initialize Graphiti with FalkorDB at "
                f"{self._falkordb_host}:{self._falkordb_port} (db={self._db_name!r}). "
                f"Is FalkorDB running? "
                f"Start with: podman run -d -p 6379:6379 --name falkordb falkordb/falkordb. "
                f"Error: {e}"
            ) from e

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Buffer an episode for batch graph construction in prepare().

        Instant (<1ms) — no LLM calls. Episodes are flushed in prepare()
        before the agent's question-answering budget clock starts.
        """
        if not self._graphiti:
            raise AdapterError("reset() must be called before ingest()")

        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        content = f"[{episode_id}] {timestamp}: {text}"
        self._pending_episodes.append(
            {"episode_id": episode_id, "content": content, "timestamp": ts}
        )
        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Add buffered episodes to the Graphiti knowledge graph.

        LLM entity extraction happens here, before the agent's budget clock.
        Populates _ep_uuid_to_id for reverse-lookup during search().
        """
        if not self._pending_episodes or not self._graphiti:
            return

        try:
            from graphiti_core.nodes import EpisodeType  # noqa: PLC0415
        except ImportError as e:
            raise AdapterError("graphiti-core not installed") from e

        for item in self._pending_episodes:
            episode_id = item["episode_id"]
            try:
                result = _get_runner().run(
                    self._graphiti.add_episode(
                        name=episode_id,
                        episode_body=item["content"],
                        source_description="LENS longitudinal operational log episode",
                        reference_time=item["timestamp"],
                        source=EpisodeType.text,
                    ),
                    timeout=300.0,
                )
                # Track episode UUID for ref_id → episode_id reverse lookup in search()
                if result and result.episode:
                    ep_uuid = str(result.episode.uuid)
                    self._ep_uuid_to_id[ep_uuid] = episode_id
                else:
                    log.warning(
                        "add_episode returned no episode node for %r", episode_id
                    )
            except Exception as e:
                raise AdapterError(
                    f"Graphiti add_episode failed for episode '{episode_id}' "
                    f"at checkpoint {checkpoint}: {e}"
                ) from e

        self._pending_episodes = []

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search the knowledge graph using hybrid edge search with episode reranking.

        Uses EDGE_HYBRID_SEARCH_EPISODE_MENTIONS which combines BM25 + cosine
        similarity on edges and reranks by number of episode mentions.
        Maps each edge back to its source episode for evidence grounding.
        """
        if not query or not query.strip() or not self._graphiti:
            return []

        cap = limit or 10
        try:
            from graphiti_core.search.search_config_recipes import (  # noqa: PLC0415
                EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
            )
            results = _get_runner().run(
                self._graphiti._search(query, config=EDGE_HYBRID_SEARCH_EPISODE_MENTIONS),
                timeout=60.0,
            )
            edges = results.edges or []
        except Exception as e:
            log.warning("Graphiti _search failed: %s", e)
            return []

        # Map edges to episodes — deduplicate by episode_id
        search_results: list[SearchResult] = []
        seen_episode_ids: set[str] = set()

        for edge in edges:
            if len(search_results) >= cap:
                break

            # Find source episode via the edge's episode UUID references
            ep_id: str | None = None
            ep_uuids = getattr(edge, "episodes", None) or []
            for ep_uuid in ep_uuids:
                candidate = self._ep_uuid_to_id.get(str(ep_uuid))
                if candidate:
                    ep_id = candidate
                    break

            if ep_id is None:
                continue  # Skip edges with no traceable episode

            if ep_id in seen_episode_ids:
                continue  # One result per episode (retrieve() gives full text)
            seen_episode_ids.add(ep_id)

            fact = getattr(edge, "fact", "") or ""
            search_results.append(
                SearchResult(
                    ref_id=ep_id,
                    text=fact[:500],
                    score=0.5,
                    metadata={"edge_uuid": str(getattr(edge, "uuid", ""))},
                )
            )

        return search_results

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve full episode text by episode_id from local cache."""
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", "graph", "keyword"],
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
