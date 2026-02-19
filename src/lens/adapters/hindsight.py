"""Hindsight (vectorize.io) memory adapter for LENS.

Uses Hindsight's TEMPR retrieval (semantic + BM25 + graph + temporal, RRF-fused)
and exposes reflect() as a memory_reflect ExtraTool for longitudinal synthesis.

Requires:
    pip install hindsight-client
    Hindsight server running locally (default: http://localhost:8888)

Setup:
    TK=$(grep '^TOGETHER_API_KEY=' .env | cut -d= -f2 | tr -d '"')
    podman run -d -p 8888:8888 --name hindsight \\
        -e HINDSIGHT_API_LLM_PROVIDER=openai \\
        -e HINDSIGHT_API_LLM_API_KEY="$TK" \\
        -e HINDSIGHT_API_LLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-tput \\
        -e HINDSIGHT_API_LLM_BASE_URL=https://api.together.xyz/v1 \\
        -e HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai \\
        -e HINDSIGHT_API_EMBEDDINGS_API_KEY="$TK" \\
        -e HINDSIGHT_API_EMBEDDINGS_MODEL=Alibaba-NLP/gte-modernbert-base \\
        -e HINDSIGHT_API_EMBEDDINGS_BASE_URL=https://api.together.xyz/v1 \\
        ghcr.io/vectorize-io/hindsight:latest

Environment variables:
    HINDSIGHT_BASE_URL    Server URL (default: http://localhost:8888)
    HINDSIGHT_API_KEY     Optional API key (default: none, local server)
    HINDSIGHT_TIMEOUT     Request timeout in seconds (default: 60.0)
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

_BANK_MISSION = (
    "I store sequential operational episode logs for longitudinal analysis. "
    "Each entry is a time-stamped telemetry snapshot prefixed with its episode ID. "
    "Queries ask me to identify patterns, trends, and causal signals across many episodes."
)


def _parse_ep_id(content: str) -> str:
    """Extract episode_id from '[ep_id] text' content format."""
    m = _EP_ID_RE.match(content)
    return m.group(1) if m else content[:32]


@register_adapter("hindsight")
class HindsightAdapter(MemoryAdapter):
    """Hindsight memory adapter.

    Uses Hindsight's TEMPR retrieval (semantic + BM25 + graph + temporal, fused
    with RRF) and exposes reflect() as a memory_reflect ExtraTool for
    longitudinal synthesis.

    Each benchmark run gets a unique bank_id (scope_id + random suffix) to
    ensure clean isolation. Old banks persist in the server's database; this is
    acceptable in a benchmark context.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._base_url = os.environ.get("HINDSIGHT_BASE_URL", "http://localhost:8888")
        self._api_key = os.environ.get("HINDSIGHT_API_KEY", None)
        self._timeout = float(os.environ.get("HINDSIGHT_TIMEOUT", "60.0"))
        self._client = None

        # State per run
        self._bank_id: str | None = None
        # Local text cache for retrieve(): episode_id -> full text
        self._text_cache: dict[str, str] = {}

    def _get_client(self):
        if self._client is None:
            try:
                from hindsight_client import Hindsight  # noqa: PLC0415
            except ImportError as e:
                raise AdapterError(
                    "hindsight-client not installed. Run: pip install hindsight-client"
                ) from e
            self._client = Hindsight(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def reset(self, scope_id: str) -> None:
        """Create a fresh bank with a unique run-scoped ID."""
        import uuid

        run_suffix = uuid.uuid4().hex[:8]
        self._bank_id = f"{scope_id}-{run_suffix}"
        self._text_cache = {}

        client = self._get_client()
        try:
            client.create_bank(
                bank_id=self._bank_id,
                name=f"LENS {scope_id}",
                mission=_BANK_MISSION,
            )
        except Exception as e:
            raise AdapterError(
                f"Failed to create Hindsight bank '{self._bank_id}'. "
                f"Is the server running at {self._base_url}? Error: {e}"
            ) from e

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Store an episode as a Hindsight memory unit."""
        if not self._bank_id:
            raise AdapterError("reset() must be called before ingest()")

        content = f"[{episode_id}] {timestamp}: {text}"
        client = self._get_client()

        try:
            # Convert ISO timestamp string to datetime for Hindsight's temporal index
            from datetime import datetime  # noqa: PLC0415

            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = None

            # Pass episode_id as document_id so it's preserved in recall results.
            # Hindsight reformats content text, so the [ep_id] prefix may not survive.
            client.retain(
                bank_id=self._bank_id,
                content=content,
                timestamp=ts,
                document_id=episode_id,
            )
        except Exception as e:
            raise AdapterError(f"Failed to retain episode {episode_id!r}: {e}") from e

        # Cache full text for retrieve()
        self._text_cache[episode_id] = text

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if not self._bank_id:
            return []

        # Hindsight uses max_tokens to control result verbosity, not a hard limit.
        # We use budget='mid' (balanced quality/speed) and allow up to 4096 tokens.
        client = self._get_client()
        try:
            response = client.recall(
                bank_id=self._bank_id,
                query=query,
                budget="mid",
                max_tokens=4096,
            )
        except Exception:
            return []

        raw = getattr(response, "results", None) or []
        cap = limit or 10

        results = []
        for item in raw[:cap]:
            content = getattr(item, "text", "") or ""
            # Prefer document_id (set during retain) for reliable episode ID mapping.
            # Fall back to parsing [ep_id] prefix for synthesized memory units
            # (type='observation') that don't carry a document_id.
            doc_id = getattr(item, "document_id", None)
            ep_id = doc_id if doc_id else _parse_ep_id(content)
            memory_type = getattr(item, "type", "") or ""
            results.append(
                SearchResult(
                    ref_id=ep_id,
                    text=content[:500],
                    score=0.0,
                    metadata={"memory_type": memory_type},
                )
            )
        return results

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve a full episode by episode_id using local cache."""
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", "keyword", "graph", "temporal"],
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
                ExtraTool(
                    name="memory_reflect",
                    description=(
                        "Generate a longitudinal synthesis across all stored episodes using "
                        "Hindsight's TEMPR graph traversal. Use this for questions asking about "
                        "trends, root causes, patterns that emerge over time, or what changed "
                        "between early and late episodes. Returns a narrative synthesis grounded "
                        "in the memory graph. Use BEFORE forming your final answer on temporal "
                        "pattern questions."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The synthesis question to answer.",
                            },
                        },
                        "required": ["query"],
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

        if tool_name == "memory_reflect":
            query = arguments.get("query", "")
            if not query or not self._bank_id:
                return {"synthesis": "", "error": "No query or bank not initialized"}
            client = self._get_client()
            try:
                response = client.reflect(
                    bank_id=self._bank_id,
                    query=query,
                    budget="high",
                )
                text = getattr(response, "text", None) or str(response)
                return {"synthesis": text}
            except Exception as e:
                return {"synthesis": f"reflect() failed: {e}", "error": True}

        return super().call_extended_tool(tool_name, arguments)
