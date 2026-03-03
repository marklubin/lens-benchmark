"""Anthropic LLM client with tool-use support.

Implements the same BaseLLMClient interface as OpenAIClient but uses the
Anthropic Python SDK (anthropic.Anthropic). Translates between LENS
tool definitions and Anthropic's tool_use content blocks.

Environment variables:
    ANTHROPIC_API_KEY       API key (or pass directly)

Usage:
    from lens.agent.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="sk-ant-...", model="claude-sonnet-4-6")
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

from lens.agent.llm_client import AgentTurn, BaseLLMClient, ToolCall, ToolDefinition, ToolResult

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """LLM client using the Anthropic Python SDK with tool-use support."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_dir: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        if cache_dir:
            log.info("AnthropicClient cache_dir specified but not yet implemented — ignoring")

    def _create_with_retry(self, max_retries: int = 5, **kwargs):
        """Call messages.create with exponential backoff on transient errors."""
        import anthropic as _anthropic

        for attempt in range(max_retries + 1):
            try:
                return self._client.messages.create(**kwargs)
            except (
                _anthropic.InternalServerError,
                _anthropic.RateLimitError,
                _anthropic.APIConnectionError,
                _anthropic.APITimeoutError,
            ) as e:
                if attempt == max_retries:
                    raise
                wait = min(2 ** attempt * 2, 60)
                log.warning(
                    "Anthropic API error (attempt %d/%d, retrying in %ds): %s",
                    attempt + 1,
                    max_retries + 1,
                    wait,
                    e,
                )
                time.sleep(wait)
            except _anthropic.BadRequestError as e:
                # Non-transient — don't retry
                log.error("Anthropic bad request: %s", e)
                raise

    def run_agent_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
        tool_executor: Callable[[ToolCall], ToolResult],
        max_turns: int = 10,
        turn_callback: Callable[[AgentTurn], None] | None = None,
    ) -> list[AgentTurn]:
        messages: list[dict] = [
            {"role": "user", "content": user_message},
        ]
        anthropic_tools = [_to_anthropic_tool(td) for td in tools] if tools else []
        turns: list[AgentTurn] = []

        for _ in range(max_turns):
            kwargs: dict = {
                "model": self._model,
                "system": system_prompt,
                "messages": messages,
                "max_tokens": self._max_tokens,
            }
            if self._temperature > 0:
                kwargs["temperature"] = self._temperature
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            response = self._create_with_retry(**kwargs)
            tokens = (
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage else 0
            )

            # Parse response content blocks
            text_parts: list[str] = []
            tool_use_blocks: list[dict] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            combined_text = "\n".join(text_parts) if text_parts else None

            if tool_use_blocks:
                # Assistant turn with tool calls
                parsed_calls = [_from_anthropic_tool_use(tb) for tb in tool_use_blocks]
                assistant_turn = AgentTurn(
                    role="assistant",
                    content=combined_text,
                    tool_calls=parsed_calls,
                    tokens_used=tokens,
                )
                turns.append(assistant_turn)
                if turn_callback:
                    turn_callback(assistant_turn)

                # Build assistant message for conversation history
                assistant_content = []
                for part in text_parts:
                    assistant_content.append({"type": "text", "text": part})
                for tb in tool_use_blocks:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tb["id"],
                        "name": tb["name"],
                        "input": tb["input"],
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute tool calls
                tool_result_blocks: list[dict] = []
                tool_results: list[ToolResult] = []

                if len(parsed_calls) > 1:
                    log.debug("Executing %d tool calls in parallel", len(parsed_calls))
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=len(parsed_calls)
                    ) as pool:
                        future_to_idx = {
                            pool.submit(tool_executor, pc): i
                            for i, pc in enumerate(parsed_calls)
                        }
                        indexed_results: list[tuple[int, ToolResult]] = []
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            indexed_results.append((idx, future.result()))
                        indexed_results.sort(key=lambda x: x[0])
                        for i, tr in indexed_results:
                            tool_results.append(tr)
                            tool_result_blocks.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_blocks[i]["id"],
                                "content": tr.content,
                                "is_error": tr.is_error,
                            })
                else:
                    for tb, parsed_call in zip(tool_use_blocks, parsed_calls, strict=True):
                        result = tool_executor(parsed_call)
                        tool_results.append(result)
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tb["id"],
                            "content": result.content,
                            "is_error": result.is_error,
                        })

                # Anthropic requires tool results in a "user" message
                messages.append({"role": "user", "content": tool_result_blocks})

                turns.append(AgentTurn(
                    role="tool",
                    tool_results=tool_results,
                    tokens_used=0,
                ))

            else:
                # Final text response — no tool calls
                final_turn = AgentTurn(
                    role="assistant",
                    content=combined_text or "",
                    tokens_used=tokens,
                )
                turns.append(final_turn)
                if turn_callback:
                    turn_callback(final_turn)
                break

        return turns


def _to_anthropic_tool(td: ToolDefinition) -> dict:
    """Convert a ToolDefinition to Anthropic's tool format."""
    return {
        "name": td.name,
        "description": td.description,
        "input_schema": td.parameters,
    }


def _from_anthropic_tool_use(tb: dict) -> ToolCall:
    """Convert an Anthropic tool_use block to our ToolCall dataclass."""
    return ToolCall(
        id=tb.get("id") or uuid.uuid4().hex[:12],
        name=tb["name"],
        arguments=tb.get("input", {}),
    )
