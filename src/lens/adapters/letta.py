"""Letta (formerly MemGPT) memory adapter for LENS.

Uses Letta's archival memory (semantic vector search) as the memory backend.
One Letta agent per scope — the agent is the isolation/namespace unit.

Requires:
    pip install letta-client
    Letta server running locally (default: http://localhost:8283)
    Embedding proxy running locally (scripts/letta_embed_proxy.py)

Setup:
    podman run -d -p 8283:8283 --name letta \\
        -e TOGETHER_API_KEY=<key> letta/letta:latest
    uv run python scripts/letta_embed_proxy.py &
    # Register together provider and configure together-oai embedding provider
    # (see scripts/letta_setup.py or docs/letta_setup.md)

Environment variables:
    LETTA_BASE_URL       Server URL (default: http://localhost:8283)
    LETTA_LLM_MODEL      LLM model handle (default: together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput)
    LETTA_EMBED_MODEL    Embedding model handle (default: together-oai/text-embedding-3-small)
"""
from __future__ import annotations

import os
import re

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.core.errors import AdapterError

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")

_PERSONA = (
    "I am an archival memory store for the LENS benchmark. "
    "My role is to accurately store and retrieve sequential episode logs "
    "for longitudinal analysis. I do not editorialize or add context."
)

_DEFAULT_LLM = "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
_DEFAULT_EMBED = "together-oai/text-embedding-3-small"


def _parse_ep_id(content: str) -> str:
    """Extract episode_id from '[ep_id] text' content format."""
    m = _EP_ID_RE.match(content)
    return m.group(1) if m else content[:32]


@register_adapter("letta")
class LettaAdapter(MemoryAdapter):
    """Letta archival memory adapter.

    Uses Letta's vector-search archival memory as the memory backend.
    Each scope gets its own Letta agent (the isolation unit in Letta).
    Episodes are stored as archival passages with the episode_id prepended.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self._llm_model = os.environ.get("LETTA_LLM_MODEL", _DEFAULT_LLM)
        self._embed_model = os.environ.get("LETTA_EMBED_MODEL", _DEFAULT_EMBED)
        self._client = None

        # State per scope
        self._agent_id: str | None = None
        self._scope_id: str | None = None
        # Local text cache for retrieve(): episode_id -> full text
        self._text_cache: dict[str, str] = {}

    def _get_client(self):
        if self._client is None:
            try:
                from letta_client import Letta  # noqa: PLC0415
            except ImportError as e:
                raise AdapterError(
                    "letta-client not installed. Run: pip install letta-client"
                ) from e
            # Default httpx timeout is 60s which is too short for slow LLM
            # providers (Together AI ~120-180s on large contexts). Increase
            # to 300s to handle worst-case latency.
            self._client = Letta(
                base_url=self._base_url, api_key="dummy", timeout=300.0
            )
        return self._client

    def reset(self, scope_id: str) -> None:
        """Delete existing agent for this scope and create a fresh one."""
        client = self._get_client()

        # Delete existing agent for this scope if present
        if self._agent_id is not None:
            try:
                client.agents.delete(agent_id=self._agent_id)
            except Exception:
                pass

        # Also scan for any stale agents from a previous run
        try:
            for agent in client.agents.list():
                if agent.name == f"lens-{scope_id}":
                    try:
                        client.agents.delete(agent_id=agent.id)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            agent = client.agents.create(
                name=f"lens-{scope_id}",
                model=self._llm_model,
                embedding=self._embed_model,
                memory_blocks=[
                    {
                        "label": "human",
                        "value": f"LENS benchmark scope: {scope_id}",
                    },
                    {
                        "label": "persona",
                        "value": _PERSONA,
                    },
                ],
            )
        except Exception as e:
            raise AdapterError(
                f"Failed to create Letta agent. Is the server running at "
                f"{self._base_url}? Error: {e}"
            ) from e

        self._agent_id = agent.id
        self._scope_id = scope_id
        self._text_cache = {}

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Store an episode as an archival passage."""
        if not self._agent_id:
            raise AdapterError("reset() must be called before ingest()")

        # Prepend episode_id so search results can be mapped back
        content = f"[{episode_id}] {timestamp}: {text}"

        client = self._get_client()
        client.agents.passages.create(
            agent_id=self._agent_id,
            text=content,
        )

        # Cache text for retrieve()
        self._text_cache[episode_id] = text

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if not self._agent_id:
            return []

        limit = limit or 10
        client = self._get_client()

        try:
            response = client.agents.passages.search(
                agent_id=self._agent_id,
                query=query,
            )
        except Exception:
            return []

        # PassageSearchResponse has .results: list[Result(content, timestamp, id, tags)]
        raw = getattr(response, "results", None)
        if raw is None:
            # Fallback: response might be a plain list
            raw = response if isinstance(response, list) else []

        results = []
        for item in raw[:limit]:
            content = getattr(item, "content", None) or getattr(item, "text", "")
            ep_id = _parse_ep_id(content)
            results.append(SearchResult(
                ref_id=ep_id,
                text=content[:500],
                score=0.0,
                metadata={"timestamp": getattr(item, "timestamp", "")},
            ))

        return results

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve a full episode by episode_id using local cache."""
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic"],
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
