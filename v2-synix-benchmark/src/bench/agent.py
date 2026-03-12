"""Agent harness — tool-use loop for answering benchmark questions.

Adapted from V1 AgentHarness pattern. Uses ModalBroker for LLM calls,
BenchmarkRuntime for tool dispatch, and produces structured answers
with citations and cost accounting.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from bench.broker import ModalBroker
from bench.runtime import BenchmarkRuntime

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_WITH_TOOLS = """\
You are a research assistant with access to a memory system. Your task is to \
answer the user's question by searching and retrieving information from memory.

Instructions:
- Use memory_search to find relevant information for the question.
- Synthesize your findings into a clear, concise answer.
- IMPORTANT: For each claim in your answer, cite the source by its label \
exactly as it appears in the search results, using square brackets. \
For example: [t-text-signal_003] or [chunks-t-text-signal_003-a1b2c3d4]. \
Every factual statement must have at least one citation.
- If you cannot find sufficient information, say so clearly.
- You have a limited number of turns and tool calls. Use them efficiently.
- Do NOT wrap your answer in <tool_call> tags. Only use tool calls through \
the function calling interface."""

SYSTEM_PROMPT_NO_TOOLS = """\
You are a research assistant. Your task is to answer the user's question \
based on your general knowledge and any context provided.

Instructions:
- Answer the question directly and concisely.
- If you lack sufficient information, say so clearly.
- Do NOT attempt to call any functions or tools. Just provide your answer."""

# Budget defaults
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_TOOL_CALLS = 20
DEFAULT_MAX_TOKENS = 4096


@dataclass
class AgentAnswer:
    """Result from a single question answering run."""

    question_id: str
    answer_text: str
    cited_refs: list[str]
    turns: list[dict[str, Any]]
    tool_calls_made: int
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_time_ms: float = 0.0


def _extract_inline_refs(text: str) -> list[str]:
    """Extract [ref_id] citations from answer text.

    Matches Synix artifact labels in various formats the model produces:
      - Full labels: [t-text-signal_001], [chunks-t-text-signal_001-abcdef12]
      - Shortened by model: [signal_001-abcdef12], [chunk-signal_020-fdbabcf0]
      - Legacy episode IDs (from V1 format)
    """
    patterns = [
        # Synix artifact labels: t-text-signal_001, t-text-distractor_theme_001
        r"\[(t-text-[a-z][a-z0-9_]*)\]",
        # Synix chunk labels: chunks-t-text-signal_001-abcdef12
        r"\[(chunks-t-text-[a-z0-9_]+-[a-f0-9]+)\]",
        # Model-shortened chunk refs: chunk-signal_020-fdbabcf0, signal_007-cc5b5840
        r"\[(chunk-[a-z][a-z0-9_]+-[a-f0-9]+)\]",
        r"\[(signal_\d+-[a-f0-9]+)\]",
        r"\[(distractor_[a-z_]+_\d+-[a-f0-9]+)\]",
        # Plain signal/distractor refs: [signal_003], [distractor_customer_growth_002]
        r"\[(signal_\d+)\]",
        r"\[(distractor_[a-z_]+_\d+)\]",
        # Legacy episode IDs (from V1 format)
        r"\[([a-z][a-z0-9_]*_ep_\d+)\]",
        r"\[([a-z][a-z0-9_]*_distractor_[a-z_]+_\d+)\]",
        r"\[([a-z][a-z0-9_]*_signal_\d+)\]",
    ]
    refs: list[str] = []
    for pat in patterns:
        refs.extend(re.findall(pat, text))
    return list(dict.fromkeys(refs))


class AgentHarness:
    """Runs the tool-use agent loop for a single question."""

    def __init__(
        self,
        broker: ModalBroker,
        runtime: BenchmarkRuntime,
        *,
        model: str = "Qwen/Qwen3.5-35B-A3B",
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self._broker = broker
        self._runtime = runtime
        self._model = model
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls
        self._max_tokens = max_tokens
        self._temperature = temperature

    def answer(self, question_prompt: str, question_id: str = "") -> AgentAnswer:
        """Run the agent to answer a single question.

        Builds system prompt (with optional context injection), enters
        the tool-use loop, and extracts the final answer with citations.
        """
        # Build system prompt — different prompts for tool vs no-tool policies
        tools = self._runtime.get_tools()
        system = SYSTEM_PROMPT_WITH_TOOLS if tools else SYSTEM_PROMPT_NO_TOOLS

        context = self._runtime.get_context()
        if context:
            policy_id = self._runtime.policy.policy_id
            if "core" in policy_id:
                label = "Working Memory"
            elif "summary" in policy_id:
                label = "Summary"
            else:
                label = "Context"
            system += f"\n\n## {label}\n{context}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question_prompt},
        ]
        turns: list[dict[str, Any]] = []
        tool_calls_made = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        wall_start = time.monotonic()

        for turn_idx in range(self._max_turns):
            # Call LLM
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if tools:
                kwargs["tools"] = tools

            response = self._broker.chat_completion(**kwargs)

            # Track tokens
            usage = getattr(response, "usage", None)
            if usage:
                total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            choice = response.choices[0]
            message = choice.message

            # Record assistant turn
            turn_record: dict[str, Any] = {"role": "assistant"}
            if message.content:
                turn_record["content"] = message.content

            # Check for tool calls
            if message.tool_calls:
                turn_record["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in message.tool_calls
                ]
                turns.append(turn_record)

                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Dispatch each tool call
                for tc in message.tool_calls:
                    tool_calls_made += 1
                    if tool_calls_made > self._max_tool_calls:
                        tool_result = json.dumps({
                            "error": "Tool call budget exceeded. Provide your answer now."
                        })
                    else:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        try:
                            tool_result = self._runtime.dispatch_tool(
                                tc.function.name, args
                            )
                        except Exception as exc:
                            logger.error(
                                "Tool dispatch error: %s(%s): %s",
                                tc.function.name, args, exc,
                            )
                            tool_result = json.dumps({"error": str(exc)})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

                    turns.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": tool_result,
                    })
            else:
                # No tool calls — assistant provided final answer
                turns.append(turn_record)
                break

            # Check finish reason
            if choice.finish_reason == "stop":
                break

        wall_ms = (time.monotonic() - wall_start) * 1000

        # Extract final answer from last assistant message with content
        answer_text = ""
        for turn in reversed(turns):
            if turn.get("role") == "assistant" and turn.get("content"):
                answer_text = turn["content"]
                break

        # Strip <think>...</think> blocks (Qwen3.5 chain-of-thought)
        answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

        # Extract citations
        cited_refs = _extract_inline_refs(answer_text)

        return AgentAnswer(
            question_id=question_id,
            answer_text=answer_text,
            cited_refs=cited_refs,
            turns=turns,
            tool_calls_made=tool_calls_made,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            wall_time_ms=wall_ms,
        )
