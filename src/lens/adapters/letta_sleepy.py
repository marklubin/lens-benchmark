"""Letta with native sleep-time compute for LENS.

Uses Letta's built-in sleep-time agent feature (enable_sleeptime=True).
The sleep-time agent runs in the background and consolidates the primary
agent's memory blocks — exactly as Letta intended.

All interactions go through Letta's agent interface:
  - ingest: passages.create() for bulk data loading into archival memory
  - search: messages.create() — ask the Letta agent directly, it uses its
    own archival_memory_search internally and responds with synthesised info
  - retrieve: local text cache for full episode text

The sleep-time agent automatically runs every N conversation steps (default 5)
and rewrites the primary agent's core memory blocks with "learned context" —
organised, consolidated insights from archival memory.

Requires:
    pip install letta-client
    Letta server >=0.16 running locally (default: http://localhost:8283)
    Embedding proxy running locally (scripts/letta_embed_proxy.py)

Environment variables:
    LETTA_BASE_URL         Server URL (default: http://localhost:8283)
    LETTA_LLM_MODEL        LLM model handle (default: letta/letta-free)
    LETTA_EMBED_MODEL      Embedding model handle (default: embed-proxy/text-embedding-3-small)
    LETTA_SLEEP_FREQ       Sleep agent frequency in steps (default: 5)
"""
from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")

_PERSONA = (
    "I am an archival memory store for the LENS benchmark. "
    "My role is to accurately store and retrieve sequential episode logs "
    "for longitudinal analysis. When asked a question, I search my archival "
    "memory and provide detailed answers citing episode IDs."
)

_DEFAULT_LLM = "openai-proxy/Qwen/Qwen3.5-35B-A3B"  # modal-llm provider
_DEFAULT_EMBED = "embed-proxy/text-embedding-3-small"


def _parse_ep_id(content: str) -> str:
    m = _EP_ID_RE.match(content)
    return m.group(1) if m else content[:32]


def _extract_assistant_text(response: object) -> str:
    """Extract the assistant's response text from a LettaResponse.

    Letta agents respond via send_message tool calls (ToolCallMessage),
    not AssistantMessage. We also check for AssistantMessage as a fallback.
    """
    import json as _json

    messages = getattr(response, "messages", [])
    parts = []
    for m in messages:
        mtype = type(m).__name__

        # Letta agents use send_message tool to communicate
        if mtype == "ToolCallMessage":
            tool_call = getattr(m, "tool_call", None)
            if tool_call and getattr(tool_call, "name", None) == "send_message":
                args_str = getattr(tool_call, "arguments", "")
                try:
                    args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                    msg = args.get("message", "")
                    if msg:
                        parts.append(msg)
                except (ValueError, AttributeError):
                    log.warning("Failed to parse send_message args: %s", args_str[:200])

        # Also check for direct AssistantMessage (older Letta versions)
        elif mtype == "AssistantMessage":
            content = getattr(m, "content", None)
            if content:
                parts.append(content)

    return "\n".join(parts)


@register_adapter("letta-sleepy")
class LettaSleepyAdapter(MemoryAdapter):
    """Letta adapter with native sleep-time memory consolidation.

    Uses enable_sleeptime=True so Letta's built-in sleep agent
    automatically consolidates memory blocks in the background.
    All queries go through the Letta agent's message interface.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self._llm_model = os.environ.get("LETTA_LLM_MODEL", _DEFAULT_LLM)
        self._embed_model = os.environ.get("LETTA_EMBED_MODEL", _DEFAULT_EMBED)
        self._sleep_freq = int(os.environ.get("LETTA_SLEEP_FREQ", "1"))
        self._client = None

        # State per scope
        self._agent_id: str | None = None
        self._scope_id: str | None = None
        self._text_cache: dict[str, str] = {}
        # Initial message IDs captured after agent creation — used to
        # reset the conversation buffer between episodes so history
        # doesn't accumulate (core memory + archival are preserved).
        self._initial_msg_ids: list[str] = []

    def _get_client(self):
        if self._client is None:
            try:
                from letta_client import Letta
            except ImportError as e:
                raise AdapterError(
                    "letta-client not installed. Run: pip install letta-client"
                ) from e
            self._client = Letta(
                base_url=self._base_url, api_key="dummy", timeout=300.0
            )
        return self._client

    def _ask_agent(self, message: str, max_steps: int = 10) -> str:
        """Send a message to the Letta agent and return its text response."""
        if not self._agent_id:
            return ""
        client = self._get_client()
        try:
            resp = client.agents.messages.create(
                agent_id=self._agent_id,
                input=message,
                max_steps=max_steps,
            )
            return _extract_assistant_text(resp)
        except Exception as e:
            log.warning("Letta agent message failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Message buffer management
    # ------------------------------------------------------------------

    def _snapshot_initial_messages(self) -> None:
        """Capture the agent's initial in-context message IDs.

        Called once after agent creation. The captured IDs are used by
        _reset_message_buffer() to restore the conversation to its
        pristine state between episodes.
        """
        import httpx

        try:
            resp = httpx.get(
                f"{self._base_url}/v1/agents/{self._agent_id}",
                headers={"Authorization": "Bearer dummy"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                self._initial_msg_ids = resp.json().get("message_ids", [])
                log.info(
                    "Captured %d initial message IDs for buffer resets",
                    len(self._initial_msg_ids),
                )
            else:
                log.warning(
                    "Failed to snapshot initial messages: HTTP %d",
                    resp.status_code,
                )
                self._initial_msg_ids = []
        except Exception as e:
            log.warning("Failed to snapshot initial messages: %s", e)
            self._initial_msg_ids = []

    def _reset_message_buffer(self) -> None:
        """Reset the agent's in-context messages to the initial state.

        Preserves core memory blocks and archival passages — only the
        conversation history is rolled back so it doesn't accumulate
        across episodes and overflow the LLM context window.
        """
        if not self._agent_id or not self._initial_msg_ids:
            return

        import httpx

        try:
            resp = httpx.patch(
                f"{self._base_url}/v1/agents/{self._agent_id}",
                headers={
                    "Authorization": "Bearer dummy",
                    "Content-Type": "application/json",
                },
                json={"message_ids": self._initial_msg_ids},
                timeout=30.0,
            )
            if resp.status_code < 300:
                log.debug(
                    "Reset message buffer to %d initial messages",
                    len(self._initial_msg_ids),
                )
            else:
                log.warning(
                    "Failed to reset message buffer: HTTP %d — %s",
                    resp.status_code,
                    resp.text[:300],
                )
        except Exception as e:
            log.warning("Failed to reset message buffer: %s", e)

    # ------------------------------------------------------------------
    # Letta configuration helpers
    # ------------------------------------------------------------------

    def _attach_memory_tools(self, agent_id: str) -> None:
        """Attach core_memory_append and core_memory_replace to the agent."""
        import httpx

        for tool_name in ("core_memory_append", "core_memory_replace"):
            try:
                # Find the tool ID
                resp = httpx.get(
                    f"{self._base_url}/v1/tools/",
                    params={"name": tool_name},
                    headers={"Authorization": "Bearer dummy"},
                    timeout=10.0,
                )
                tools = resp.json()
                if not tools:
                    log.warning("Tool %s not found in Letta", tool_name)
                    continue
                tool_id = tools[0]["id"] if isinstance(tools, list) else tools["id"]

                # Attach to agent
                resp = httpx.patch(
                    f"{self._base_url}/v1/agents/{agent_id}/tools/attach/{tool_id}",
                    headers={"Authorization": "Bearer dummy"},
                    timeout=10.0,
                )
                if resp.status_code < 300:
                    log.info("Attached %s to agent %s", tool_name, agent_id)
                else:
                    log.warning("Failed to attach %s: %s", tool_name, resp.text[:200])
            except Exception as e:
                log.warning("Error attaching %s: %s", tool_name, e)

    def _set_sleep_frequency(self, agent_id: str) -> None:
        """Set sleeptime_agent_frequency on the agent's group via DB.

        Letta only puts the sleeptime agent (not the primary) in groups_agents,
        so we look up the group via the sleeptime agent's name convention.
        """
        import subprocess

        sleeptime_name = f"lens-sleepy-{self._scope_id}-sleeptime"
        try:
            result = subprocess.run(
                [
                    "podman", "exec", "letta",
                    "psql", "-U", "letta", "-d", "letta", "-t", "-c",
                    f"UPDATE groups SET sleeptime_agent_frequency = {self._sleep_freq} "
                    f"WHERE id = ("
                    f"  SELECT ga.group_id FROM groups_agents ga "
                    f"  JOIN agents a ON a.id = ga.agent_id "
                    f"  WHERE a.name = '{sleeptime_name}' LIMIT 1"
                    f") RETURNING id;",
                ],
                capture_output=True, text=True, timeout=10,
            )
            group_id = result.stdout.strip()
            if group_id:
                log.info(
                    "Set sleeptime_agent_frequency=%d on group %s",
                    self._sleep_freq, group_id,
                )
            else:
                log.warning("No group found for sleeptime agent %s (stdout=%s stderr=%s)",
                            sleeptime_name, result.stdout.strip(), result.stderr.strip())
        except Exception as e:
            log.warning("Error setting sleep frequency: %s", e)

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def reset(self, scope_id: str) -> None:
        """Delete existing agent and create a fresh one with sleep-time enabled."""
        client = self._get_client()

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
                enable_sleeptime=True,
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

        # Capture initial message state before any interactions —
        # used to reset the conversation buffer between episodes.
        self._snapshot_initial_messages()

        # Give the primary agent core memory editing tools so it can
        # consolidate insights during prepare() — not just search.
        self._attach_memory_tools(agent.id)

        # Set sleep-time frequency on the group so the sleep agent runs
        # more often (default 5 is too infrequent for our 5 checkpoints).
        self._set_sleep_frequency(agent.id)

        log.info(
            "Created Letta sleepy agent %s (sleeptime=True, freq=%d)",
            agent.id, self._sleep_freq,
        )

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Ingest an episode: store in archival + send as agent message.

        Two-pronged approach:
        1. passages.create() — guarantees the episode is in archival memory
           for retrieval (the LLM agent may not reliably call
           insert_archival_memory on its own).
        2. messages.create() — lets the agent process the episode, update
           core memory, and trigger the sleep-time agent for consolidation.

        After the agent finishes processing, the conversation buffer is
        reset so messages don't accumulate across episodes.
        """
        if not self._agent_id:
            raise AdapterError("reset() must be called before ingest()")

        client = self._get_client()

        # 1) Direct archival storage — guaranteed retrieval
        tagged_text = f"[{episode_id}] {timestamp}\n\n{text}"
        try:
            client.agents.passages.create(
                agent_id=self._agent_id,
                text=tagged_text,
            )
        except Exception as e:
            log.warning("passages.create() failed for %s: %s", episode_id, e)

        # 2) Agent message — triggers sleep-time consolidation
        self._ask_agent(
            f"New episode [{episode_id}] has been stored in your archival memory. "
            f"Key context: {text[:500]}... "
            f"Update your core memory with any significant patterns or developments.",
            max_steps=10,
        )

        # Wipe conversation buffer — core memory + archival persist.
        self._reset_message_buffer()

        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Signal that all episodes for this checkpoint are ingested.

        Sends a consolidation prompt so the agent (and its sleep-time
        companion) can do any final processing.  The conversation buffer
        is reset afterwards to keep the next round clean.
        """
        if not self._agent_id:
            return
        log.info("Triggering consolidation at checkpoint %d", checkpoint)
        self._ask_agent(
            f"All episodes up to checkpoint {checkpoint} have been ingested. "
            f"Review your archival memory and consolidate the key patterns, "
            f"trends, and developments into your core memory blocks.",
            max_steps=15,
        )
        self._reset_message_buffer()

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search archival memory and include sleep-consolidated core memory.

        Uses direct passage search (like base letta adapter) for archival
        retrieval, plus includes the sleep agent's consolidated core memory
        blocks as the first result — this is where the sleep-time value lives.
        """
        if not query or not query.strip():
            return []
        if not self._agent_id:
            return []

        limit = limit or 5
        client = self._get_client()
        results: list[SearchResult] = []

        # 1) Include sleep-consolidated core memory as first result
        try:
            blocks = client.agents.blocks.list(agent_id=self._agent_id)
            core_parts = []
            for block in blocks:
                label = getattr(block, "label", "")
                value = getattr(block, "value", "")
                if value and label not in ("human",):
                    core_parts.append(f"[{label}] {value}")
            if core_parts:
                core_text = "\n\n".join(core_parts)
                results.append(SearchResult(
                    ref_id="sleep_memory",
                    text=core_text[:2000],
                    score=1.0,
                    metadata={"type": "sleep_consolidated"},
                ))
        except Exception as e:
            log.warning("Failed to read core memory blocks: %s", e)

        # 2) Semantic search on archival passages
        try:
            response = client.agents.passages.search(
                agent_id=self._agent_id,
                query=query,
            )
        except Exception as e:
            log.warning("Archival passage search failed: %s", e)
            return results

        raw = getattr(response, "results", None)
        if raw is None:
            raw = response if isinstance(response, list) else []

        for item in raw[: max(1, limit - len(results))]:
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
        """Retrieve a full episode by ref_id from local cache.

        Also supports 'sleep_memory' to retrieve the consolidated core memory.
        """
        if ref_id == "sleep_memory":
            # Return core memory blocks
            try:
                client = self._get_client()
                blocks = client.agents.blocks.list(agent_id=self._agent_id)
                parts = []
                for block in blocks:
                    label = getattr(block, "label", "")
                    value = getattr(block, "value", "")
                    if value and label not in ("human",):
                        parts.append(f"[{label}] {value}")
                if parts:
                    return Document(ref_id="sleep_memory", text="\n\n".join(parts))
            except Exception as e:
                log.warning("Failed to retrieve core memory: %s", e)
            return None

        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", "sleep-time-compute"],
            max_results_per_search=5,
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
