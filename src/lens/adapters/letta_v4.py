"""Letta V4 — Three-agent architecture: Ingest + Sleep + Q&A.

Three agents per scope, all sharing the same core memory blocks:

  Ingest agent:  Receives episodes via messages.create(). Extracts key
                 information — entities, events, metrics, behavioral changes —
                 and updates core memory. Focuses on what might impact future
                 decisions or change understanding of entities involved.

  Sleep agent:   Runs automatically every N conversation steps on the ingest
                 agent. Reconciles disparate information, updates predictions
                 and hypotheses, prunes irrelevant data.

  Q&A agent:     Separate agent sharing the same memory blocks. Receives
                 questions via messages.create() and answers using core memory
                 and archival search. Never sees raw episodes — only the
                 distilled knowledge the other agents built.

All agents are domain-agnostic. No "human" block. The persona block carries
the shared context about the task: learning from sequential data and
synthesising patterns for future questions.

Requires:
    pip install letta-client
    Letta server >=0.16 running locally (default: http://localhost:8283)
    Embedding proxy running locally (scripts/letta_embed_proxy.py)

Environment variables:
    LETTA_BASE_URL         Server URL (default: http://localhost:8283)
    LETTA_LLM_MODEL        LLM model handle
    LETTA_EMBED_MODEL      Embedding model handle
    LETTA_SLEEP_FREQ       Sleep agent frequency in steps (default: 5)
"""
from __future__ import annotations

import logging
import os
import re
import time

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.core.errors import AdapterError
from lens.core.models import AgentAnswer

log = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")

# ---------------------------------------------------------------------------
# Shared persona block — all three agents see this
# ---------------------------------------------------------------------------

_PERSONA = (
    "I am an analytical system that finds patterns across sequential data "
    "episodes. Most episodes are routine — my value comes from spotting "
    "the exceptions and connecting them into theories.\n\n"
    "CORE MEMORY: Small, precious. Only patterns, hypotheses, and the "
    "entities/events that matter to them.\n"
    "ARCHIVAL MEMORY: Unlimited. All detailed evidence lives here.\n\n"
    "I cite episode IDs for every claim."
)

# ---------------------------------------------------------------------------
# Four separate core memory blocks — 5K chars each (tight working memory)
# Detailed evidence overflows to archival memory.
# ---------------------------------------------------------------------------

_BLOCK_LIMIT = 5000

_BLOCKS = {
    "patterns": {
        "init": "(no patterns yet)",
        "description": (
            "MOST IMPORTANT BLOCK. Cross-episode trends, recurring behaviors, "
            "escalations, anomalies. Format: '• pattern_name (N eps): summary "
            "[ep_X, ep_Y]'. Only promote after 2+ observations. This is what "
            "makes you useful — connections across episodes."
        ),
    },
    "hypotheses": {
        "init": "(no hypotheses yet)",
        "description": (
            "Working theories connecting patterns. Format: '• HYPOTHESIS: claim "
            "| status: forming/supported/confirmed/refuted | evidence: [ep_X, "
            "ep_Y]'. Update as evidence accumulates. Remove refuted ones."
        ),
    },
    "entities": {
        "init": "(no entities yet)",
        "description": (
            "Only actors/systems RELEVANT TO A PATTERN. Skip one-off mentions. "
            "Format: '• name: role, notable behavior [ep_X]'. Max ~50 chars "
            "per entry. Not a roster — only entities that matter to your "
            "hypotheses."
        ),
    },
    "events": {
        "init": "(no events yet)",
        "description": (
            "Only TURNING POINTS and ANOMALIES. Skip routine events. "
            "Format: '• YYYY-MM-DD: what changed [ep_X]'. If an event is "
            "routine/expected, archive it but don't put it here. This is for "
            "surprises and inflection points."
        ),
    },
}

# ---------------------------------------------------------------------------
# System prompts — override Letta defaults
# ---------------------------------------------------------------------------

_INGEST_SYSTEM = (
    "You are an analytical agent processing sequential episodes of raw data. "
    "You will later answer questions requiring synthesis across episodes.\n\n"

    "## MEMORY TIERS\n\n"

    "CORE MEMORY — 4 small blocks (~5K chars each), always in your context:\n"
    "  patterns: Cross-episode trends (MOST IMPORTANT)\n"
    "  hypotheses: Working theories\n"
    "  entities: Only actors relevant to a pattern\n"
    "  events: Only turning points and anomalies\n\n"

    "ARCHIVAL MEMORY — unlimited, searchable. Use archival_memory_insert "
    "to store detailed observations, metrics, quotes, evidence.\n\n"

    "## HOW TO PROCESS EACH EPISODE\n\n"

    "1. Archive a structured summary to archival memory (always do this)\n"
    "2. Ask: does this episode CHANGE anything? Options:\n"
    "   a) ROUTINE — nothing new. Archive only. Do NOT update core memory.\n"
    "   b) NEW SIGNAL — new actor, behavior, or anomaly. Add to core.\n"
    "   c) REINFORCES PATTERN — an existing pattern gains evidence. "
    "Update the episode count in patterns block.\n"
    "   d) CONTRADICTS — something unexpected. Update hypotheses.\n"
    "3. If updating core, use the tightest format possible:\n"
    "   '• name: role [ep_X]' for entities (max 50 chars)\n"
    "   '• YYYY-MM-DD: what changed [ep_X]' for events\n"
    "   '• pattern (N eps): summary [ep_X, ep_Y]' for patterns\n"
    "4. send_message with a 1-line acknowledgment\n\n"

    "## CRITICAL RULES\n\n"

    "- Most episodes are ROUTINE. The default is to archive and move on.\n"
    "- Core memory is EXPENSIVE. Only write there when the episode changes "
    "your understanding or reinforces a multi-episode pattern.\n"
    "- patterns and hypotheses are your most valuable blocks. entities and "
    "events are supporting context only.\n"
    "- When a block is full, CONDENSE: merge entries, drop singletons, "
    "archive details. Never fail to process an episode.\n"
    "- Use episode IDs and dates. Never 'recently' or 'today'."
)

_SLEEP_SYSTEM = (
    "You are a memory consolidation agent. Your job: keep core memory "
    "SMALL, SHARP, and USEFUL.\n\n"

    "## PRIORITIES (in order)\n\n"

    "1. HYPOTHESIZE — form working theories that connect patterns. This "
    "is the most valuable thing you do.\n"
    "2. PROMOTE — if recurring events exist in the events block, create "
    "a pattern entry and remove the individual events.\n"
    "3. PRUNE — remove any entity that isn't connected to a pattern or "
    "hypothesis. Remove single-observation events.\n"
    "4. CONDENSE — if a block is over ~4K chars, merge entries and "
    "archive overflow via archival_memory_insert.\n"
    "5. CONNECT — look for relationships between patterns. Two patterns "
    "about the same entity probably tell one story.\n\n"

    "## RULES\n\n"

    "- patterns and hypotheses blocks are PRIMARY. entities and events "
    "only exist to support them.\n"
    "- An entity with no connection to a pattern should be pruned.\n"
    "- A pattern seen once is noise; 3+ times is signal.\n"
    "- Use episode IDs and dates, never 'recently' or 'today'."
)

_QA_SYSTEM = (
    "You answer questions about patterns in sequential data episodes.\n\n"

    "## YOUR MEMORY\n\n"

    "CORE MEMORY (visible now): patterns, hypotheses, key entities/events. "
    "This is your starting point — it tells you what matters and where "
    "to look.\n\n"

    "ARCHIVAL MEMORY (searchable): Detailed per-episode observations. "
    "Use archival_memory_search to find specific evidence.\n\n"

    "## STRATEGY\n\n"

    "1. Check core memory — do patterns/hypotheses already address the "
    "question?\n"
    "2. Search archival for specific evidence cited in core memory\n"
    "3. Search archival for additional relevant episodes\n"
    "4. Synthesize: connect patterns with specific evidence\n"
    "5. Cite episode IDs for every factual claim\n\n"

    "Your value is SYNTHESIS — connecting dots across episodes, not "
    "retrieving individual facts. If evidence is insufficient, say so."
)

_DEFAULT_LLM = "openai-proxy/Qwen/Qwen3.5-35B-A3B"
_DEFAULT_EMBED = "embed-proxy/text-embedding-3-small"


def _extract_inline_refs(text: str, scope_id: str | None = None) -> list[str]:
    """Extract episode citations from answer text.

    Matches full refs like ``[scope_ep_001]`` **and** bare refs like
    ``ep_001``, ``[ep_001]``, or ``ep_001,``.  When *scope_id* is provided
    bare refs are expanded to ``{scope_id}_ep_NNN``; otherwise they are
    returned as-is.
    """
    refs: list[str] = []

    # Full-qualified: [tutoring_jailbreak_07_ep_001]
    refs.extend(re.findall(r'\[([a-z][a-z0-9_]*_ep_\d+)\]', text))
    refs.extend(re.findall(r'\(ref_id:\s*([a-z][a-z0-9_]*_ep_\d+)\)', text))

    # Bare: ep_001, [ep_001], (ep_001)
    bare = re.findall(r'(?<![a-z0-9_])ep_(\d+)', text)
    for num in bare:
        full = f"{scope_id}_ep_{num}" if scope_id else f"ep_{num}"
        refs.append(full)

    return list(dict.fromkeys(refs))


def _extract_assistant_text(response: object) -> str:
    """Extract the assistant's response text from a LettaResponse."""
    import json as _json

    messages = getattr(response, "messages", [])
    parts = []
    for m in messages:
        mtype = type(m).__name__

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

        elif mtype == "AssistantMessage":
            content = getattr(m, "content", None)
            if content:
                parts.append(content)

    return "\n".join(parts)


def _api_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer dummy",
        "Content-Type": "application/json",
    }


@register_adapter("letta-v4")
class LettaV4Adapter(MemoryAdapter):
    """Letta V4: three-agent architecture with shared core memory.

    Ingest agent processes episodes and builds observations.
    Sleep agent consolidates in the background.
    Q&A agent answers questions using accumulated knowledge.
    All share the same persona + observations blocks.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self._llm_model = os.environ.get("LETTA_LLM_MODEL", _DEFAULT_LLM)
        self._embed_model = os.environ.get("LETTA_EMBED_MODEL", _DEFAULT_EMBED)
        self._client = None

        # Agent IDs per scope
        self._ingest_agent_id: str | None = None
        self._sleep_agent_id: str | None = None
        self._qa_agent_id: str | None = None
        self._scope_id: str | None = None
        self._text_cache: dict[str, str] = {}

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

    def _create_block(self, label: str, value: str, description: str, limit: int = _BLOCK_LIMIT) -> str | None:
        """Create a memory block and return its ID."""
        import httpx

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
        import httpx

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
        import httpx

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

    def _update_block_value(self, block_id: str, value: str) -> bool:
        """Update a block's value."""
        import httpx

        resp = httpx.patch(
            f"{self._base_url}/v1/blocks/{block_id}",
            headers=_api_headers(),
            json={"value": value},
            timeout=10.0,
        )
        if resp.status_code < 300:
            return True
        log.warning("Failed to update block %s: %s", block_id[:12], resp.text[:200])
        return False

    def _find_agent_by_name(self, name: str) -> dict | None:
        """Find an agent by exact name, returns raw dict."""
        import httpx

        agents = []
        after = None
        while True:
            params = {}
            if after:
                params["after"] = after
            resp = httpx.get(
                f"{self._base_url}/v1/agents/",
                headers=_api_headers(),
                params=params,
                timeout=30.0,
            )
            page = resp.json()
            if not page:
                break
            agents.extend(page)
            after = page[-1]["id"]
            if len(page) < 50:
                break

        for a in agents:
            if a.get("name") == name:
                return a
        return None

    def _get_agent_blocks(self, agent_id: str) -> list[dict]:
        """Get an agent's core memory blocks."""
        import httpx

        resp = httpx.get(
            f"{self._base_url}/v1/agents/{agent_id}/core-memory/blocks",
            headers=_api_headers(),
            timeout=10.0,
        )
        return resp.json() if resp.status_code < 300 else []

    def _configure_sleep_agent(self, sleep_name: str) -> None:
        """Update sleep agent's system prompt and memory_persona block."""
        sleep_agent = self._find_agent_by_name(sleep_name)
        if not sleep_agent:
            log.warning("Sleep agent %s not found", sleep_name)
            return

        sleep_id = sleep_agent["id"]

        # Update system prompt
        self._update_system_prompt(sleep_id, _SLEEP_SYSTEM)

        # Update memory_persona block
        blocks = self._get_agent_blocks(sleep_id)
        for block in blocks:
            if block.get("label") == "memory_persona":
                self._update_block_value(block["id"], _SLEEP_SYSTEM)
                log.info("Updated sleep memory_persona for %s", sleep_name)
                break
        else:
            log.warning("memory_persona block not found on %s", sleep_name)

    def _disable_auto_sleep(self, sleep_name: str) -> None:
        """Disable automatic sleep triggering — we trigger manually at checkpoints."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "podman", "exec", "letta",
                    "psql", "-U", "letta", "-d", "letta", "-t", "-c",
                    f"UPDATE groups SET sleeptime_agent_frequency = 9999 "
                    f"WHERE id = ("
                    f"  SELECT ga.group_id FROM groups_agents ga "
                    f"  JOIN agents a ON a.id = ga.agent_id "
                    f"  WHERE a.name = '{sleep_name}' LIMIT 1"
                    f") RETURNING id;",
                ],
                capture_output=True, text=True, timeout=10,
            )
            group_id = result.stdout.strip()
            if group_id:
                log.info("Disabled auto-sleep on group %s", group_id)
            else:
                log.warning(
                    "No group found for %s (stderr=%s)",
                    sleep_name, result.stderr.strip()[:200],
                )
        except Exception as e:
            log.warning("Error disabling auto-sleep: %s", e)

    def _create_qa_agent(self, scope_id: str, shared_block_ids: list[str]) -> str:
        """Create the Q&A agent that shares blocks with the ingest agent."""
        client = self._get_client()
        qa_name = f"lens-v4-{scope_id}-qa"

        # Clean up existing
        try:
            for agent in client.agents.list():
                if agent.name == qa_name:
                    client.agents.delete(agent_id=agent.id)
        except Exception:
            pass

        # Create with a minimal placeholder block (we'll attach shared blocks after)
        agent = client.agents.create(
            name=qa_name,
            model=self._llm_model,
            embedding=self._embed_model,
            enable_sleeptime=False,
            memory_blocks=[],
        )

        qa_id = agent.id

        # Update system prompt
        self._update_system_prompt(qa_id, _QA_SYSTEM)

        # Attach the shared blocks (persona + 4 knowledge blocks)
        for block_id in shared_block_ids:
            self._attach_block(qa_id, block_id)

        # Q&A agent needs archival_memory_search to query detailed evidence
        self._attach_tools(qa_id, ["archival_memory_search"])

        log.info("Created Q&A agent %s (%s)", qa_name, qa_id[:12])
        return qa_id

    # ------------------------------------------------------------------
    # MemoryAdapter interface
    # ------------------------------------------------------------------

    def _attach_tools(self, agent_id: str, tool_names: list[str]) -> None:
        """Attach server-side tools to an agent by name."""
        import httpx

        # Get all available tools
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

    def reset(self, scope_id: str) -> None:
        """Create three agents: ingest (with sleep) + Q&A, sharing memory blocks."""
        client = self._get_client()

        # Clean up all existing V4 agents for this scope
        prefix = f"lens-v4-{scope_id}"
        self._delete_agents_by_prefix(prefix)

        self._scope_id = scope_id
        self._text_cache = {}

        # --- 1. Create shared blocks (persona + 4 knowledge blocks) ---
        persona_block_id = self._create_block(
            label="persona",
            value=_PERSONA,
            description="Shared context about the learning task.",
        )
        if not persona_block_id:
            raise AdapterError("Failed to create persona block")

        shared_block_ids = [persona_block_id]

        for label, cfg in _BLOCKS.items():
            block_id = self._create_block(
                label=label,
                value=cfg["init"],
                description=cfg["description"],
            )
            if not block_id:
                raise AdapterError(f"Failed to create {label} block")
            shared_block_ids.append(block_id)

        # --- 2. Create ingest agent (with sleeptime) ---
        try:
            ingest_agent = client.agents.create(
                name=f"lens-v4-{scope_id}",
                model=self._llm_model,
                embedding=self._embed_model,
                enable_sleeptime=True,
                memory_blocks=[],
            )
        except Exception as e:
            raise AdapterError(
                f"Failed to create ingest agent. Is the server running at "
                f"{self._base_url}? Error: {e}"
            ) from e

        self._ingest_agent_id = ingest_agent.id

        # Attach shared blocks to ingest agent
        for block_id in shared_block_ids:
            self._attach_block(self._ingest_agent_id, block_id)

        # Attach core memory tools (not auto-added when blocks=[] at creation)
        self._attach_tools(self._ingest_agent_id, [
            "core_memory_append",
            "core_memory_replace",
            "archival_memory_insert",
        ])

        # Set ingest agent system prompt
        self._update_system_prompt(self._ingest_agent_id, _INGEST_SYSTEM)

        # Configure the auto-created sleep agent
        sleep_name = f"lens-v4-{scope_id}-sleeptime"
        self._configure_sleep_agent(sleep_name)
        self._disable_auto_sleep(sleep_name)

        # Store sleep agent ID for manual triggering
        sleep_agent = self._find_agent_by_name(sleep_name)
        if sleep_agent:
            self._sleep_agent_id = sleep_agent["id"]
            # Attach shared blocks (409 expected for auto-shared ones)
            for block_id in shared_block_ids:
                self._attach_block(sleep_agent["id"], block_id)
        else:
            log.warning("Sleep agent %s not found — consolidation disabled", sleep_name)

        # --- 3. Create Q&A agent (no sleeptime) ---
        self._qa_agent_id = self._create_qa_agent(scope_id, shared_block_ids)

        log.info(
            "Created Letta V4 trio for %s: ingest=%s sleep=%s qa=%s (auto-sleep disabled)",
            scope_id,
            self._ingest_agent_id[:12],
            self._sleep_agent_id[:12] if self._sleep_agent_id else "NONE",
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

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Trigger the sleep agent to consolidate memory at this checkpoint."""
        if not self._sleep_agent_id:
            log.warning("No sleep agent — skipping consolidation at checkpoint %d", checkpoint)
            return

        log.info("Checkpoint %d — triggering sleep agent for consolidation", checkpoint)
        self._send_message(
            self._sleep_agent_id,
            f"Checkpoint {checkpoint}: consolidate the memory blocks. "
            f"Reconcile disparate information, update hypotheses, "
            f"prune irrelevant entries, and strengthen well-supported patterns.",
            max_steps=15,
        )

    # ------------------------------------------------------------------
    # Direct Q&A via the dedicated Q&A agent
    # ------------------------------------------------------------------

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

        # Build episode map so scorer can validate refs against the vault
        ep_map: dict[str, str] = {}
        for r in refs:
            # Refs are already fully qualified by _extract_inline_refs
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
    # MemoryAdapter abstract methods (stubs — Q&A goes through Letta)
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
