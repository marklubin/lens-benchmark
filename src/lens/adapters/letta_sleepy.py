"""Letta with checkpoint sleep consolidation for LENS.

Extends the base Letta adapter by running a consolidation ("sleep") cycle
in prepare() before each checkpoint's questions. The sleep cycle fetches all
archival passages ingested so far, calls an LLM to synthesize them, and stores
the result in a _synthesis string that is prepended to every search result.

The synthesis is produced by the same LLM that runs the benchmark agent, on the
same data the agent has access to, without any knowledge of the specific questions
that will be asked. The only advantage over the base letta adapter is that
cross-episode patterns have been pre-organised into a single document.

Variants (LETTA_SLEEP_VARIANT env var, default 2):
    0  no sleep — identical behaviour to the base letta adapter (control)
    1  minimal: summarise all episodes, cite episode IDs
    2  actionable filter: prioritise patterns/anomalies, de-emphasise baselines
    3  delta/causal: focus on what changed, when, and why

Requires:
    pip install letta-client openai
    Letta server running locally (default: http://localhost:8283)
    Embedding proxy running locally (scripts/letta_embed_proxy.py)

Environment variables (same as letta adapter plus):
    LETTA_BASE_URL         Server URL (default: http://localhost:8283)
    LETTA_LLM_MODEL        LLM model handle (default: together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput)
    LETTA_EMBED_MODEL      Embedding model handle (default: together-oai/text-embedding-3-small)
    LETTA_SLEEP_VARIANT    Prompt variant 0-3 (default: 2)
    LENS_LLM_API_KEY       API key for sleep LLM call (falls back to OPENAI_API_KEY)
    LENS_LLM_API_BASE      Base URL for sleep LLM call (falls back to OPENAI_BASE_URL)
"""
from __future__ import annotations

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

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")

_PERSONA = (
    "I am an archival memory store for the LENS benchmark. "
    "My role is to accurately store and retrieve sequential episode logs "
    "for longitudinal analysis. I do not editorialize or add context."
)

_DEFAULT_LLM = "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
_DEFAULT_EMBED = "together-oai/text-embedding-3-small"

# ---------------------------------------------------------------------------
# Sleep prompt variants
# ---------------------------------------------------------------------------
# All variants share the same system prompt and passage listing; only the
# CONSOLIDATION OBJECTIVE section differs.  Prompts are deliberately generic:
# no domain hints (no mention of latency, services, failures, etc.), so the
# sleep cycle cannot exploit foreknowledge of what the benchmark questions ask.

_SLEEP_SYSTEM = (
    "You are a memory consolidation agent. You have been given a set of "
    "sequential operational episode logs stored in a memory system. "
    "Your task is to analyse these episodes and produce a structured synthesis "
    "that will help answer future analytical questions about the data. "
    "Be specific and cite episode IDs (e.g. [ep_001]) whenever you reference "
    "particular data points."
)

_SLEEP_VARIANTS: dict[int, str | None] = {
    # 0 = control, no sleep
    0: None,

    # V1: minimal — just consolidate, no opinion on what matters
    1: (
        "Write a comprehensive summary of all stored episodes. "
        "Include the key information from each episode and cite episode IDs."
    ),

    # V2: actionable filter — prioritise signal over noise
    2: (
        "Organise this information for efficient future retrieval. "
        "Prioritise information that reveals patterns, anomalies, or changes "
        "over time. De-emphasise stable baselines and routine readings that are "
        "unlikely to distinguish one episode from another. "
        "Write a structured synthesis focusing on what is most likely to be "
        "analytically significant."
    ),

    # V3: delta/causal — focus on dynamics and causality
    3: (
        "Identify what changed over time, when transitions occurred, "
        "and what correlations exist between different components or metrics. "
        "Focus on cause and effect, progression, and turning points rather than "
        "describing the contents of each individual episode. "
        "Which episodes mark significant state changes, and what drove them?"
    ),
}

_SLEEP_USER_TMPL = """\
You have {n} episodes in memory spanning {first_ts} to {last_ts}.

EPISODES:
{passages}

CONSOLIDATION OBJECTIVE:
{objective}

Write your synthesis below. Cite episode IDs (e.g. [ep_001]) when referencing specific data points.

SYNTHESIS:"""

# Max chars per passage included in sleep prompt (keeps total context reasonable)
_MAX_PASSAGE_CHARS = 1200
# Max chars of synthesis stored / returned in search results
_MAX_SYNTHESIS_CHARS = 3000


def _parse_ep_id(content: str) -> str:
    m = _EP_ID_RE.match(content)
    return m.group(1) if m else content[:32]


def _strip_provider_prefix(letta_model: str) -> str:
    """Convert 'together/Qwen/Qwen3-...' → 'Qwen/Qwen3-...' for OpenAI API."""
    if "/" in letta_model:
        return letta_model.split("/", 1)[1]
    return letta_model


@register_adapter("letta-sleepy")
class LettaSleepyAdapter(MemoryAdapter):
    """Letta archival memory adapter with checkpoint sleep consolidation.

    Identical to the base letta adapter for ingest/search/retrieve, but adds
    a prepare() sleep cycle that synthesises all passages into a single
    consolidated document, which is then prepended to every search result.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self._llm_model = os.environ.get("LETTA_LLM_MODEL", _DEFAULT_LLM)
        self._embed_model = os.environ.get("LETTA_EMBED_MODEL", _DEFAULT_EMBED)
        self._variant = int(os.environ.get("LETTA_SLEEP_VARIANT", "2"))
        self._client = None

        # State per scope
        self._agent_id: str | None = None
        self._scope_id: str | None = None
        self._text_cache: dict[str, str] = {}
        self._synthesis: str = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from letta_client import Letta  # noqa: PLC0415
            except ImportError as e:
                raise AdapterError(
                    "letta-client not installed. Run: pip install letta-client"
                ) from e
            self._client = Letta(base_url=self._base_url, api_key="dummy")
        return self._client

    def _fetch_passages(self) -> list[str]:
        """Return all passage texts for the current agent, newest-first."""
        client = self._get_client()
        try:
            page = client.agents.passages.list(agent_id=self._agent_id, limit=500)
            return [p.text for p in page if getattr(p, "text", None)]
        except Exception:
            return []

    def _run_sleep_cycle(self, passages: list[str], checkpoint: int) -> str:
        """Call the LLM to synthesise passages and return synthesis text."""
        objective = _SLEEP_VARIANTS.get(self._variant)
        if not objective or not passages:
            return ""

        # Build passage block — truncate each to keep context manageable
        passage_lines = []
        for p in passages:
            passage_lines.append(p[:_MAX_PASSAGE_CHARS])
        passage_block = "\n\n".join(passage_lines)

        # Extract timestamp range from first/last passage (best-effort)
        def _ts(text: str) -> str:
            # Passages are "[ep_id] TIMESTAMP: ..." — extract second token
            parts = text.split(" ", 2)
            return parts[1].rstrip(":") if len(parts) >= 2 else "unknown"

        first_ts = _ts(passages[-1])  # oldest first (list may be newest-first)
        last_ts = _ts(passages[0])

        user_msg = _SLEEP_USER_TMPL.format(
            n=len(passages),
            first_ts=first_ts,
            last_ts=last_ts,
            passages=passage_block,
            objective=objective,
        )

        # Resolve API credentials — same sources as the benchmark runner
        api_key = (
            os.environ.get("LENS_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        base_url = (
            os.environ.get("LENS_LLM_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        model = _strip_provider_prefix(self._llm_model)

        if _OpenAI is None:
            raise AdapterError(
                "openai package required for sleep cycle. Run: pip install openai"
            )

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        oai = _OpenAI(**client_kwargs)
        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SLEEP_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            synthesis = resp.choices[0].message.content or ""
        except Exception:
            # Sleep failure is non-fatal — fall back to no synthesis
            return ""

        return synthesis[:_MAX_SYNTHESIS_CHARS]

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def reset(self, scope_id: str) -> None:
        """Delete existing agent for this scope and create a fresh one."""
        client = self._get_client()
        self._synthesis = ""

        # Clean up existing agent
        if self._agent_id is not None:
            try:
                client.agents.delete(agent_id=self._agent_id)
            except Exception:
                pass

        try:
            for agent in client.agents.list():
                if agent.name == f"lens-sleepy-{scope_id}":
                    try:
                        client.agents.delete(agent_id=agent.id)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            agent = client.agents.create(
                name=f"lens-sleepy-{scope_id}",
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

        content = f"[{episode_id}] {timestamp}: {text}"
        client = self._get_client()
        client.agents.passages.create(
            agent_id=self._agent_id,
            text=content,
        )
        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Run sleep consolidation cycle before each checkpoint's questions."""
        if self._variant == 0 or not self._agent_id:
            return
        passages = self._fetch_passages()
        if not passages:
            return
        self._synthesis = self._run_sleep_cycle(passages, checkpoint)

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

        cap = limit or 10
        client = self._get_client()

        # Reserve one slot for synthesis so total stays within cap
        passage_cap = cap - 1 if (self._synthesis and cap > 1) else cap

        try:
            response = client.agents.passages.search(
                agent_id=self._agent_id,
                query=query,
            )
        except Exception:
            response = None

        raw = getattr(response, "results", None) if response else None
        if raw is None:
            raw = response if isinstance(response, list) else []

        results: list[SearchResult] = []

        # Prepend synthesis as first result if available
        if self._synthesis:
            results.append(SearchResult(
                ref_id="synthesis",
                text=self._synthesis,
                score=0.5,
                metadata={"type": "consolidated_synthesis"},
            ))

        for item in raw[:passage_cap]:
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
        """Retrieve a full episode (or the synthesis document) by ref_id."""
        if ref_id == "synthesis":
            if self._synthesis:
                return Document(ref_id="synthesis", text=self._synthesis)
            return None
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        variant_label = {0: "none", 1: "minimal", 2: "actionable-filter", 3: "delta-causal"}
        return CapabilityManifest(
            search_modes=["semantic", f"sleep-consolidated-v{self._variant}"],
            max_results_per_search=10,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple full episodes by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times — it uses only "
                        "one tool call instead of one per document. "
                        "After memory_search, pass all ref_ids you want to read to this tool. "
                        "Note: ref_id='synthesis' returns the pre-consolidated memory synthesis."
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
