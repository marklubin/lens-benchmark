"""Letta Entity — Two-agent architecture with dynamic per-entity core memory.

Two agents per scope sharing dynamic entity blocks:

  Ingest agent:  Receives episodes via messages.create(). Extracts entities,
                 canonicalises names, and creates/updates/deletes individual
                 core memory blocks — one per tracked entity. Archives raw
                 evidence to archival memory.

  Q&A agent:     Shares entity blocks with ingest agent. Uses
                 archival_memory_search to find detailed evidence. Synthesises
                 answers from entity histories + archival evidence.

No sleep agent — all memory management happens at ingest time via dynamic
block operations.

Requires:
    pip install letta-client
    Letta server >=0.16 running locally (default: http://localhost:8283)

Environment variables:
    LETTA_BASE_URL         Server URL (default: http://localhost:8283)
    LETTA_LLM_MODEL        LLM model handle
    LETTA_EMBED_MODEL      Embedding model handle
"""
from __future__ import annotations

import logging
import os
import time

import httpx

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.letta_v4 import (
    _api_headers,
    _extract_assistant_text,
    _extract_inline_refs,
)
from lens.adapters.registry import register_adapter
from lens.core.errors import AdapterError
from lens.core.models import AgentAnswer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENTITY_BLOCK_LIMIT = 3000

_DEFAULT_LLM = "openai-proxy/Qwen/Qwen3.5-35B-A3B"
_DEFAULT_EMBED = "embed-proxy/text-embedding-3-small"

_PERSONA = (
    "I am an analytical system that tracks entities across sequential data "
    "episodes. I maintain individual memory blocks for important entities, "
    "tracking their state changes, relationships, and evidence over time.\n\n"
    "ENTITY BLOCKS: Dynamic, one per tracked entity. My working memory.\n"
    "ARCHIVAL MEMORY: Unlimited. All detailed evidence lives here.\n\n"
    "I cite episode IDs for every claim."
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_INGEST_SYSTEM = (
    "You are an analytical agent processing sequential data episodes.\n"
    "You maintain individual memory blocks for important entities.\n\n"

    "## MEMORY TIERS\n\n"

    "ENTITY BLOCKS — one block per tracked entity (~3K chars each), always in context.\n"
    "Each block tracks one entity's full history, relationships, and evidence.\n\n"

    "ARCHIVAL MEMORY — unlimited, searchable. Store detailed episode summaries here.\n\n"

    "## HOW TO PROCESS EACH EPISODE\n\n"

    "1. Archive a structured summary to archival memory (always do this)\n"
    "2. Extract entities — use full proper names, resolve pronouns\n"
    "3. For each important entity:\n"
    "   a) If entity has no block yet AND appears significant (2+ mentions, behavioral change, central to pattern):\n"
    "      → create_block with label=entity_name_lowercase, value=structured template\n"
    "   b) If entity has a block:\n"
    "      → core_memory_replace to update STATUS, append to HISTORY, update RELATIONSHIPS\n"
    "   c) If entity is no longer relevant (no mentions for many episodes, resolved):\n"
    "      → delete_block to free space\n"
    "4. Track at most ~15-20 entities. Be selective — not every mentioned name deserves a block.\n"
    "5. send_message with a 1-line acknowledgment\n\n"

    "## BLOCK FORMAT\n\n"

    "NAME: Full Name\n"
    "TYPE: ACTOR|OBJECT|PLACE|CONCEPT|EVENT\n"
    "STATUS: current state\n"
    "HISTORY:\n"
    "• YYYY-MM-DD: what changed [ep_X]\n"
    "RELATIONSHIPS:\n"
    "• related_entity: nature of relationship [ep_X]\n"
    "EVIDENCE: [ep_X, ep_Y, ep_Z]\n\n"

    "## CRITICAL RULES\n\n"

    "- Most episodes are ROUTINE. Only create/update entity blocks when meaningful.\n"
    "- An entity earns a block when: appears 2+ times, shows behavioral change, or is central to an emerging pattern.\n"
    "- Use episode IDs and dates. Never \"recently\" or \"today\".\n"
    "- When a block is full, condense: merge history entries, archive old details."
)

_QA_SYSTEM = (
    "You answer questions about patterns in sequential data episodes.\n\n"

    "## YOUR MEMORY\n\n"

    "ENTITY BLOCKS (visible now): Individual blocks for tracked entities, each with\n"
    "full history, relationships, and evidence citations. Scan these first.\n\n"

    "ARCHIVAL MEMORY (searchable): Detailed per-episode summaries. Use\n"
    "archival_memory_search to find specific evidence.\n\n"

    "## STRATEGY\n\n"

    "1. Scan entity blocks — which entities are relevant to the question?\n"
    "2. Note their histories, status changes, and relationships\n"
    "3. Search archival for detailed evidence from cited episodes\n"
    "4. Search archival for additional relevant episodes\n"
    "5. Synthesize: connect entity histories with specific evidence\n"
    "6. Cite episode IDs for every factual claim\n\n"

    "Your value is SYNTHESIS — connecting entity histories across episodes.\n"
    "If evidence is insufficient, say so."
)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter("letta-entity")
class LettaEntityAdapter(MemoryAdapter):
    """Letta Entity: two-agent architecture with dynamic per-entity blocks.

    Ingest agent processes episodes and manages entity blocks.
    Q&A agent answers questions using shared entity blocks + archival search.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self._llm_model = os.environ.get("LETTA_LLM_MODEL", _DEFAULT_LLM)
        self._embed_model = os.environ.get("LETTA_EMBED_MODEL", _DEFAULT_EMBED)
        self._client = None

        self._ingest_agent_id: str | None = None
        self._qa_agent_id: str | None = None
        self._scope_id: str | None = None
        self._text_cache: dict[str, str] = {}
        self._entity_blocks: dict[str, str] = {}  # label -> block_id
        self._persona_block_id: str | None = None

    # ------------------------------------------------------------------
    # Client helpers (same patterns as letta_v4)
    # ------------------------------------------------------------------

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

    def _send_message(self, agent_id: str, message: str, max_steps: int = 10) -> str:
        """Send a message to a Letta agent and return its text response."""
        client = self._get_client()
        try:
            resp = client.agents.messages.create(
                agent_id=agent_id,
                input=message,
                max_steps=max_steps,
            )
            return _extract_assistant_text(resp)
        except Exception as e:
            log.warning("Letta agent %s message failed: %s", agent_id[:12], e)
            return ""

    # ------------------------------------------------------------------
    # Agent lifecycle helpers
    # ------------------------------------------------------------------

    def _delete_agents_by_prefix(self, prefix: str) -> None:
        """Delete all agents whose name starts with prefix."""
        client = self._get_client()
        try:
            for agent in client.agents.list():
                if agent.name and agent.name.startswith(prefix):
                    try:
                        client.agents.delete(agent_id=agent.id)
                        log.info("Deleted agent %s (%s)", agent.name, agent.id[:12])
                    except Exception as e:
                        log.warning("Failed to delete agent %s: %s", agent.name, e)
        except Exception as e:
            log.warning("Error listing agents for cleanup: %s", e)

    def _create_block(self, label: str, value: str, description: str, limit: int = _ENTITY_BLOCK_LIMIT) -> str | None:
        """Create a memory block and return its ID."""

        resp = httpx.post(
            f"{self._base_url}/v1/blocks/",
            headers=_api_headers(),
            json={
                "label": label,
                "value": value,
                "description": description,
                "limit": limit,
            },
            timeout=10.0,
        )
        if resp.status_code >= 300:
            log.warning("Failed to create %s block: %s", label, resp.text[:200])
            return None
        block_id = resp.json().get("id")
        if block_id:
            log.info("Created block %s (%s)", label, block_id[:12])
        return block_id

    def _attach_block(self, agent_id: str, block_id: str) -> bool:
        """Attach an existing block to an agent."""

        resp = httpx.patch(
            f"{self._base_url}/v1/agents/{agent_id}/core-memory/blocks/attach/{block_id}",
            headers=_api_headers(),
            timeout=10.0,
        )
        if resp.status_code < 300:
            log.info("Attached block %s to agent %s", block_id[:12], agent_id[:12])
            return True
        log.warning("Failed to attach block %s: %s", block_id[:12], resp.text[:200])
        return False

    def _detach_block(self, agent_id: str, block_id: str) -> bool:
        """Detach a block from an agent."""

        resp = httpx.patch(
            f"{self._base_url}/v1/agents/{agent_id}/core-memory/blocks/detach/{block_id}",
            headers=_api_headers(),
            timeout=10.0,
        )
        if resp.status_code < 300:
            log.info("Detached block %s from agent %s", block_id[:12], agent_id[:12])
            return True
        log.warning("Failed to detach block %s: %s", block_id[:12], resp.text[:200])
        return False

    def _delete_block(self, block_id: str) -> bool:
        """Delete a block entirely."""

        resp = httpx.delete(
            f"{self._base_url}/v1/blocks/{block_id}",
            headers=_api_headers(),
            timeout=10.0,
        )
        if resp.status_code < 300:
            log.info("Deleted block %s", block_id[:12])
            return True
        log.warning("Failed to delete block %s: %s", block_id[:12], resp.text[:200])
        return False

    def _update_system_prompt(self, agent_id: str, system: str) -> bool:
        """Update an agent's system prompt."""

        resp = httpx.patch(
            f"{self._base_url}/v1/agents/{agent_id}",
            headers=_api_headers(),
            json={"system": system},
            timeout=10.0,
        )
        if resp.status_code < 300:
            log.info("Updated system prompt for agent %s", agent_id[:12])
            return True
        log.warning("Failed to update system prompt: %s", resp.text[:200])
        return False

    def _get_agent_blocks(self, agent_id: str) -> list[dict]:
        """Get an agent's core memory blocks."""

        resp = httpx.get(
            f"{self._base_url}/v1/agents/{agent_id}/core-memory/blocks",
            headers=_api_headers(),
            timeout=10.0,
        )
        return resp.json() if resp.status_code < 300 else []

    def _attach_tools(self, agent_id: str, tool_names: list[str]) -> None:
        """Attach server-side tools to an agent by name."""

        resp = httpx.get(
            f"{self._base_url}/v1/tools/",
            headers=_api_headers(),
            timeout=30.0,
        )
        if resp.status_code >= 300:
            log.warning("Failed to list tools: %s", resp.text[:200])
            return

        all_tools = {t["name"]: t["id"] for t in resp.json()}

        for name in tool_names:
            tool_id = all_tools.get(name)
            if not tool_id:
                log.warning("Tool %s not found on server", name)
                continue
            resp = httpx.patch(
                f"{self._base_url}/v1/agents/{agent_id}/tools/attach/{tool_id}",
                headers=_api_headers(),
                timeout=10.0,
            )
            if resp.status_code < 300:
                log.info("Attached tool %s to agent %s", name, agent_id[:12])
            elif resp.status_code == 409:
                pass  # already attached
            else:
                log.warning("Failed to attach tool %s: %s", name, resp.text[:200])

    # ------------------------------------------------------------------
    # Entity block sync
    # ------------------------------------------------------------------

    def _sync_entity_blocks(self) -> None:
        """Sync entity blocks between ingest and Q&A agents.

        After each ingest, the ingest agent may have created or deleted entity
        blocks. This method detects changes and mirrors them on the Q&A agent.
        """
        if not self._ingest_agent_id or not self._qa_agent_id:
            return

        current_blocks = self._get_agent_blocks(self._ingest_agent_id)
        current_labels: dict[str, str] = {}  # label -> block_id
        for block in current_blocks:
            label = block.get("label", "")
            block_id = block.get("id", "")
            if label == "persona":
                continue
            current_labels[label] = block_id

        # New blocks: on ingest agent but not tracked
        for label, block_id in current_labels.items():
            if label not in self._entity_blocks:
                self._attach_block(self._qa_agent_id, block_id)
                self._entity_blocks[label] = block_id
                log.info("Synced new entity block %s to Q&A agent", label)

        # Deleted blocks: tracked but no longer on ingest agent
        removed = [l for l in self._entity_blocks if l not in current_labels]
        for label in removed:
            block_id = self._entity_blocks.pop(label)
            self._detach_block(self._qa_agent_id, block_id)
            log.info("Removed entity block %s from Q&A agent", label)

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def reset(self, scope_id: str) -> None:
        """Create two agents: ingest + Q&A, sharing a persona block."""
        client = self._get_client()

        prefix = f"lens-entity-{scope_id}"
        self._delete_agents_by_prefix(prefix)

        self._scope_id = scope_id
        self._text_cache = {}
        self._entity_blocks = {}

        # Create persona block
        self._persona_block_id = self._create_block(
            label="persona",
            value=_PERSONA,
            description="Shared context about the entity tracking task.",
        )
        if not self._persona_block_id:
            raise AdapterError("Failed to create persona block")

        # Create ingest agent
        try:
            ingest_agent = client.agents.create(
                name=f"lens-entity-{scope_id}",
                model=self._llm_model,
                embedding=self._embed_model,
                enable_sleeptime=False,
                memory_blocks=[],
            )
        except Exception as e:
            raise AdapterError(
                f"Failed to create ingest agent. Is the server running at "
                f"{self._base_url}? Error: {e}"
            ) from e

        self._ingest_agent_id = ingest_agent.id
        self._attach_block(self._ingest_agent_id, self._persona_block_id)
        self._attach_tools(self._ingest_agent_id, [
            "core_memory_append",
            "core_memory_replace",
            "archival_memory_insert",
            "create_block",
            "delete_block",
        ])
        self._update_system_prompt(self._ingest_agent_id, _INGEST_SYSTEM)

        # Create Q&A agent
        try:
            qa_agent = client.agents.create(
                name=f"lens-entity-{scope_id}-qa",
                model=self._llm_model,
                embedding=self._embed_model,
                enable_sleeptime=False,
                memory_blocks=[],
            )
        except Exception as e:
            raise AdapterError(
                f"Failed to create Q&A agent: {e}"
            ) from e

        self._qa_agent_id = qa_agent.id
        self._attach_block(self._qa_agent_id, self._persona_block_id)
        self._attach_tools(self._qa_agent_id, ["archival_memory_search"])
        self._update_system_prompt(self._qa_agent_id, _QA_SYSTEM)

        log.info(
            "Created Letta Entity pair for %s: ingest=%s qa=%s",
            scope_id,
            self._ingest_agent_id[:12],
            self._qa_agent_id[:12],
        )

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Send episode to the ingest agent for processing."""
        if not self._ingest_agent_id:
            raise AdapterError("reset() must be called before ingest()")

        self._text_cache[episode_id] = text

        message = f"[{episode_id}] {timestamp}:\n\n{text}"
        response = self._send_message(self._ingest_agent_id, message, max_steps=10)
        if response:
            log.info("Ingest agent processed %s: %s", episode_id, response[:120])
        else:
            log.warning("Ingest agent returned empty response for %s", episode_id)

        self._sync_entity_blocks()

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Sync entity blocks at checkpoint (management happens at ingest time)."""
        self._sync_entity_blocks()

    def answer_question(self, prompt: str, question_id: str = "") -> AgentAnswer:
        """Send question to the Q&A agent."""
        if not self._qa_agent_id:
            return AgentAnswer(
                question_id=question_id,
                answer_text="Adapter not initialized.",
            )

        t0 = time.monotonic()
        answer_text = self._send_message(self._qa_agent_id, prompt, max_steps=15)
        wall_ms = (time.monotonic() - t0) * 1000

        refs = _extract_inline_refs(answer_text, scope_id=self._scope_id)

        ep_map: dict[str, str] = {}
        for r in refs:
            ep_map[r] = r

        return AgentAnswer(
            question_id=question_id,
            answer_text=answer_text,
            turns=[{"role": "letta_qa_agent", "content": answer_text}],
            tool_calls_made=0,
            total_tokens=0,
            wall_time_ms=wall_ms,
            budget_violations=[],
            refs_cited=refs,
            ref_episode_map=ep_map,
        )

    # ------------------------------------------------------------------
    # Stubs — Q&A goes through Letta
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        return []

    def retrieve(self, ref_id: str) -> Document | None:
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["letta-native"],
            max_results_per_search=5,
        )
