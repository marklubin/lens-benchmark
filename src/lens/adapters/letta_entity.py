"""Letta Entity — Two-agent architecture with entity-focused core memory.

Two agents per scope:

  Ingest agent:  Receives episode summaries via messages.create().
                 Maintains a large "entities" core memory block tracking
                 important actors, objects, and patterns. Raw episode text
                 is stored deterministically via passages.create().

  Q&A agent:     Shares the entities block. Uses archival_memory_search
                 to find detailed evidence. Synthesises answers from
                 entity histories + archival evidence.

Key design: episode text goes to archival via passages.create() (deterministic,
never lost). The ingest agent's job is solely to update the entity tracker block.
This follows the lesson from letta-sleepy: never rely on LLM agents to call
specific tools for critical data operations.

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

_ENTITY_BLOCK_LIMIT = 20000  # Large block for entity tracking

_DEFAULT_LLM = "openai-proxy/Qwen/Qwen3.5-35B-A3B"
_DEFAULT_EMBED = "embed-proxy/text-embedding-3-small"

_PERSONA = (
    "I am an analytical system that tracks entities across sequential data "
    "episodes. I maintain an entity tracker in core memory and use archival "
    "memory for detailed evidence.\n\n"
    "ENTITY TRACKER BLOCK: My working memory — tracks important entities, "
    "their state changes, relationships, and evidence over time.\n"
    "ARCHIVAL MEMORY: Unlimited. All full episode texts live here.\n\n"
    "I cite episode IDs for every claim."
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_INGEST_SYSTEM = (
    "You are an analytical agent processing sequential data episodes.\n"
    "You maintain an entity tracker in your 'entities' core memory block.\n\n"

    "## MEMORY TIERS\n\n"

    "ENTITIES BLOCK — your working memory (~20K chars), always in context.\n"
    "Tracks important entities with their histories, relationships, evidence.\n\n"

    "ARCHIVAL MEMORY — unlimited, searchable. Full episode texts are already\n"
    "stored there automatically. You do NOT need to archive episodes yourself.\n\n"

    "## HOW TO PROCESS EACH EPISODE\n\n"

    "1. Read the episode summary provided in the message\n"
    "2. Extract entities — use full proper names, resolve pronouns\n"
    "3. Update the 'entities' block using core_memory_replace:\n"
    "   - Add new entities that appear significant (2+ mentions, behavioral change, central to pattern)\n"
    "   - Update existing entity entries with new status, history, relationships\n"
    "   - Remove entities no longer relevant (no mentions for many episodes)\n"
    "4. send_message with a 1-line acknowledgment\n\n"

    "## ENTITY FORMAT (in entities block)\n\n"

    "=== ENTITY_NAME ===\n"
    "TYPE: ACTOR|OBJECT|PLACE|CONCEPT|EVENT\n"
    "STATUS: current state\n"
    "HISTORY:\n"
    "- what changed [ep_X]\n"
    "- what changed [ep_Y]\n"
    "RELATIONSHIPS:\n"
    "- related_entity: nature [ep_X]\n"
    "EVIDENCE: [ep_X, ep_Y, ep_Z]\n\n"

    "## CRITICAL RULES\n\n"

    "- Most episodes are ROUTINE. Only update entities block when meaningful.\n"
    "- Track at most ~15-20 entities. Be selective.\n"
    "- Use episode IDs. Never 'recently' or 'today'.\n"
    "- When the block fills up, condense: merge history entries, drop stale entities.\n"
    "- Use core_memory_replace to update the entities block — find the section "
    "to change and replace it with the updated version."
)

_QA_SYSTEM = (
    "You answer questions about patterns in sequential data episodes.\n\n"

    "## YOUR MEMORY\n\n"

    "ENTITY TRACKER (visible now): A block tracking important entities with\n"
    "full histories, relationships, and evidence citations. Scan this first.\n\n"

    "ARCHIVAL MEMORY (searchable): Full episode texts. Use\n"
    "archival_memory_search to find specific evidence.\n\n"

    "## STRATEGY\n\n"

    "1. Scan the entities block — which entities are relevant to the question?\n"
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
    """Letta Entity: two-agent architecture with entity-focused core memory.

    Ingest agent processes episodes and maintains entity tracker block.
    Q&A agent answers questions using shared entity block + archival search.
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
        self._entities_block_id: str | None = None

        # For message buffer reset
        self._ingest_initial_msg_ids: list[str] = []

    # ------------------------------------------------------------------
    # Client helpers
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

    def _capture_initial_messages(self, agent_id: str) -> list[str]:
        """Capture initial message IDs for buffer reset."""
        try:
            resp = httpx.get(
                f"{self._base_url}/v1/agents/{agent_id}",
                headers=_api_headers(),
                timeout=30.0,
            )
            if resp.status_code == 200:
                msg_ids = resp.json().get("message_ids", [])
                log.info("Captured %d initial message IDs for %s", len(msg_ids), agent_id[:12])
                return msg_ids
        except Exception as e:
            log.warning("Failed to capture initial messages for %s: %s", agent_id[:12], e)
        return []

    def _reset_message_buffer(self, agent_id: str, initial_msg_ids: list[str]) -> None:
        """Reset an agent's conversation to initial state. Preserves core memory + archival."""
        if not agent_id or not initial_msg_ids:
            return
        try:
            resp = httpx.patch(
                f"{self._base_url}/v1/agents/{agent_id}",
                headers=_api_headers(),
                json={"message_ids": initial_msg_ids},
                timeout=30.0,
            )
            if resp.status_code < 300:
                log.debug("Reset message buffer for %s", agent_id[:12])
            else:
                log.warning("Failed to reset buffer for %s: HTTP %d", agent_id[:12], resp.status_code)
        except Exception as e:
            log.warning("Failed to reset message buffer for %s: %s", agent_id[:12], e)

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def reset(self, scope_id: str) -> None:
        """Create two agents: ingest + Q&A, sharing an entities block."""
        client = self._get_client()

        prefix = f"lens-entity-{scope_id}"
        self._delete_agents_by_prefix(prefix)

        self._scope_id = scope_id
        self._text_cache = {}

        # Create shared entities block (large, for entity tracking)
        self._entities_block_id = self._create_block(
            label="entities",
            value="(no entities tracked yet)",
            description="Entity tracker: important actors, objects, patterns with histories and evidence.",
            limit=_ENTITY_BLOCK_LIMIT,
        )
        if not self._entities_block_id:
            raise AdapterError("Failed to create entities block")

        # Create persona block
        persona_block_id = self._create_block(
            label="persona",
            value=_PERSONA,
            description="Agent persona and task description.",
            limit=5000,
        )
        if not persona_block_id:
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
        self._attach_block(self._ingest_agent_id, persona_block_id)
        self._attach_block(self._ingest_agent_id, self._entities_block_id)
        self._attach_tools(self._ingest_agent_id, [
            "core_memory_append",
            "core_memory_replace",
        ])
        self._update_system_prompt(self._ingest_agent_id, _INGEST_SYSTEM)
        self._ingest_initial_msg_ids = self._capture_initial_messages(self._ingest_agent_id)

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
        self._attach_block(self._qa_agent_id, persona_block_id)
        self._attach_block(self._qa_agent_id, self._entities_block_id)
        self._attach_tools(self._qa_agent_id, ["archival_memory_search"])
        self._update_system_prompt(self._qa_agent_id, _QA_SYSTEM)

        log.info(
            "Created Letta Entity pair for %s: ingest=%s qa=%s entities_block=%s",
            scope_id,
            self._ingest_agent_id[:12],
            self._qa_agent_id[:12],
            self._entities_block_id[:12],
        )

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Ingest an episode: store in archival + ask agent to update entity tracker.

        Two-pronged approach (same as letta-sleepy):
        1. passages.create() — guarantees the episode is in archival memory
        2. messages.create() — lets the agent update the entities block
        """
        if not self._ingest_agent_id:
            raise AdapterError("reset() must be called before ingest()")

        client = self._get_client()
        self._text_cache[episode_id] = text

        # 1) Deterministic archival storage — guaranteed retrieval.
        # Store in BOTH ingest and QA agents: Letta archival is per-agent,
        # so the QA agent needs its own copy to search during answer_question().
        tagged_text = f"[{episode_id}] {timestamp}\n\n{text}"
        for agent_id, label in [
            (self._ingest_agent_id, "ingest"),
            (self._qa_agent_id, "qa"),
        ]:
            if not agent_id:
                continue
            try:
                client.agents.passages.create(
                    agent_id=agent_id,
                    text=tagged_text,
                )
            except Exception as e:
                log.warning("passages.create() failed for %s (%s): %s", episode_id, label, e)

        # 2) Agent message — update entity tracker (send condensed summary, not full text)
        # Truncate to ~2000 chars to keep messages manageable
        summary = text[:2000]
        if len(text) > 2000:
            summary += f"... [truncated, {len(text)} chars total]"

        response = self._send_message(
            self._ingest_agent_id,
            f"Episode [{episode_id}] ({timestamp}) stored in archival. "
            f"Update entity tracker as needed.\n\n{summary}",
            max_steps=10,
        )
        if response:
            log.info("Ingest agent processed %s: %s", episode_id, response[:120])
        else:
            log.warning("Ingest agent returned empty response for %s", episode_id)

        # Reset conversation buffer to prevent context overflow
        self._reset_message_buffer(self._ingest_agent_id, self._ingest_initial_msg_ids)

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """No special preparation needed — entity tracker is always up to date."""
        pass

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
