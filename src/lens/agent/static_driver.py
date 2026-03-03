"""Static driver for LENS benchmark.

Executes pre-computed query plans instead of LLM-driven search.
Removes the "agent writes bad queries" variable by using predetermined,
human-designed search strategies. The LLM is only called once at the end
for synthesis.

Normal flow:  LLM → formulate query → search → LLM → maybe retrieve → LLM → answer
Static flow:  query plan → search × N → auto-retrieve top K → LLM → answer (one call)
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from lens.agent.llm_client import AgentTurn, BaseLLMClient, ToolCall, ToolResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from lens.agent.llm_client import ToolDefinition

log = logging.getLogger(__name__)


class StaticLLMClient(BaseLLMClient):
    """Executes pre-computed query plans instead of LLM-driven search.

    For each question:
    1. Look up query plan by question text (fuzzy match)
    2. Execute predetermined search queries via tool_executor
    3. Auto-retrieve top K results from each search
    4. Call OpenAI API once for synthesis
    """

    def __init__(
        self,
        plans: dict[str, dict],
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for StaticLLMClient. "
                "Install it with: pip install 'lens-bench[openai]'"
            ) from exc

        self._plans = plans  # question_text -> {"searches": [...], "retrieve_top_k": N}
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._oai = OpenAI(**kwargs)
        self._model = model
        self._temperature = temperature
        self._seed = seed

    def _find_plan(self, user_message: str) -> dict:
        """Find the query plan matching this question.

        Matching strategy:
        1. Exact match
        2. Normalized exact match (strip, lowercase)
        3. Substring containment (either direction)
        4. Prefix match after stripping punctuation
        5. Default single-query fallback
        """
        # Exact match
        if user_message in self._plans:
            return self._plans[user_message]

        # Normalized exact match
        norm_msg = user_message.strip().lower()
        for key, plan in self._plans.items():
            if key.strip().lower() == norm_msg:
                return plan

        # Substring match — check if a plan key is contained in the message
        # or the message is contained in a plan key
        for key, plan in self._plans.items():
            if key in user_message or user_message in key:
                return plan

        # Prefix match — strip trailing punctuation and compare
        norm_msg_stripped = norm_msg.rstrip("?.! ")
        for key, plan in self._plans.items():
            norm_key = key.strip().lower().rstrip("?.! ")
            if norm_key in norm_msg_stripped or norm_msg_stripped in norm_key:
                return plan

        log.warning(
            "No query plan found for question (using default): %.80s...",
            user_message,
        )
        return {"searches": [user_message], "retrieve_top_k": 3}

    def _completions_with_retry(self, max_retries: int = 5, **kwargs):
        """Call chat.completions.create with exponential backoff."""
        import openai as _openai

        for attempt in range(max_retries + 1):
            try:
                return self._oai.chat.completions.create(**kwargs)
            except (
                _openai.InternalServerError,
                _openai.RateLimitError,
                _openai.APIConnectionError,
                _openai.APITimeoutError,
            ) as e:
                if attempt == max_retries:
                    raise
                wait = min(2 ** attempt * 2, 60)
                log.warning(
                    "Static driver LLM error (attempt %d/%d, retry in %ds): %s",
                    attempt + 1, max_retries + 1, wait, e,
                )
                time.sleep(wait)

    def run_agent_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
        tool_executor: Callable[[ToolCall], ToolResult],
        max_turns: int = 10,
        turn_callback: Callable[[AgentTurn], None] | None = None,
    ) -> list[AgentTurn]:
        turns: list[AgentTurn] = []
        plan = self._find_plan(user_message)
        searches = plan.get("searches", [user_message])
        retrieve_top_k = plan.get("retrieve_top_k", 3)

        # Phase 1: Execute search queries
        all_ref_ids: list[str] = []
        for i, query in enumerate(searches):
            tc = ToolCall(
                id=f"static-search-{i}",
                name="memory_search",
                arguments={"query": query},
            )
            assistant_turn = AgentTurn(
                role="assistant",
                content=None,
                tool_calls=[tc],
                tokens_used=0,
            )
            turns.append(assistant_turn)
            if turn_callback:
                turn_callback(assistant_turn)

            result = tool_executor(tc)
            turns.append(AgentTurn(
                role="tool",
                tool_results=[result],
                tokens_used=0,
            ))

            # Parse ref_ids from search results
            ref_ids = self._extract_ref_ids(result.content, retrieve_top_k)
            all_ref_ids.extend(ref_ids)

        # Phase 2: Auto-retrieve unique ref_ids
        seen: set[str] = set()
        unique_ref_ids: list[str] = []
        for rid in all_ref_ids:
            if rid not in seen:
                seen.add(rid)
                unique_ref_ids.append(rid)

        collected_evidence: list[str] = []
        for j, ref_id in enumerate(unique_ref_ids):
            tc = ToolCall(
                id=f"static-retrieve-{j}",
                name="memory_retrieve",
                arguments={"ref_id": ref_id},
            )
            assistant_turn = AgentTurn(
                role="assistant",
                content=None,
                tool_calls=[tc],
                tokens_used=0,
            )
            turns.append(assistant_turn)
            if turn_callback:
                turn_callback(assistant_turn)

            result = tool_executor(tc)
            turns.append(AgentTurn(
                role="tool",
                tool_results=[result],
                tokens_used=0,
            ))

            if not result.is_error and result.content:
                collected_evidence.append(f"[{ref_id}]\n{result.content}")

        # Phase 3: Synthesize answer with single LLM call
        # Cap evidence to ~50K tokens (~40K words) to leave room for
        # system prompt + question + output within model context window
        MAX_EVIDENCE_CHARS = 400_000  # ~100K tokens at ~4 chars/token
        if collected_evidence:
            truncated: list[str] = []
            total_chars = 0
            for ev in collected_evidence:
                if total_chars + len(ev) > MAX_EVIDENCE_CHARS:
                    remaining = MAX_EVIDENCE_CHARS - total_chars
                    if remaining > 500:
                        truncated.append(ev[:remaining] + "\n[...truncated]")
                    break
                truncated.append(ev)
                total_chars += len(ev)
            evidence_block = "\n\n---\n\n".join(truncated)
            if len(truncated) < len(collected_evidence):
                log.warning(
                    "Truncated evidence from %d to %d pieces (%.0fK chars)",
                    len(collected_evidence), len(truncated),
                    total_chars / 1000,
                )
        else:
            evidence_block = "(No evidence retrieved)"

        synthesis_msg = (
            f"QUESTION:\n{user_message}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            "Based on the evidence above, provide a thorough answer to the question. "
            "Cite specific episode IDs (e.g. [episode_id]) for your claims."
        )

        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": synthesis_msg},
            ],
            "temperature": self._temperature,
            "max_tokens": 4096,
        }
        if self._seed is not None:
            kwargs["seed"] = self._seed

        try:
            resp = self._completions_with_retry(**kwargs)
        except Exception as e:
            # If context overflows, retry with reduced max_tokens
            if "context length" in str(e) or "input_tokens" in str(e):
                log.warning(
                    "Context overflow, retrying with max_tokens=1024: %s", e
                )
                kwargs["max_tokens"] = 1024
                try:
                    resp = self._completions_with_retry(**kwargs)
                except Exception as e2:
                    log.error("Synthesis failed even with reduced tokens: %s", e2)
                    final_turn = AgentTurn(
                        role="assistant",
                        content="Unable to synthesize — evidence too large for context.",
                        tokens_used=0,
                    )
                    turns.append(final_turn)
                    if turn_callback:
                        turn_callback(final_turn)
                    return turns
            else:
                raise
        answer = resp.choices[0].message.content or ""
        usage = resp.usage
        tokens = (usage.total_tokens if usage else 0) or 0

        final_turn = AgentTurn(
            role="assistant",
            content=answer,
            tokens_used=tokens,
        )
        turns.append(final_turn)
        if turn_callback:
            turn_callback(final_turn)

        return turns

    @staticmethod
    def _extract_ref_ids(search_result_content: str, top_k: int) -> list[str]:
        """Extract ref_ids from a search result JSON string."""
        try:
            data = json.loads(search_result_content)
        except (json.JSONDecodeError, TypeError):
            log.warning("Could not parse search result as JSON: %.100s", search_result_content)
            return []

        results = []
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data

        ref_ids = []
        for item in results[:top_k]:
            if isinstance(item, dict) and "ref_id" in item:
                ref_ids.append(item["ref_id"])
        return ref_ids
