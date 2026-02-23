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


_THINK_RE = __import__("re").compile(r"<think>[\s\S]*?</think>\s*", __import__("re").DOTALL)


def _strip_think(text: str | None) -> str:
    """Remove <think>...</think> blocks produced by Qwen3 reasoning mode."""
    if not text:
        return text or ""
    return _THINK_RE.sub("", text).strip()


class OpenAIClient(BaseLLMClient):
    """LLM client using the OpenAI Python SDK with tool-use support.

    Compatible with any OpenAI-compatible API (OpenAI, DeepSeek, OpenRouter, vLLM)
    via the ``base_url`` parameter.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 4096,
        cache_dir: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install 'lens-bench[openai]'"
            ) from exc

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

        # Wrap with disk cache if cache_dir is set
        if cache_dir:
            from lens.agent.llm_cache import CachingOpenAIClient
            self._cache = CachingOpenAIClient(self._client, cache_dir)
            self._client = self._cache
            log.info("LLM response caching enabled: %s", cache_dir)
        else:
            self._cache = None

        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        # Detect Qwen3 models — need /no_think suffix and <think> stripping
        self._is_qwen3 = "qwen3" in model.lower()

    def _completions_with_retry(self, max_retries: int = 5, **kwargs):
        """Call chat.completions.create with exponential backoff on transient errors."""
        import openai as _openai  # noqa: PLC0415

        for attempt in range(max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
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
                    "LLM API error (attempt %d/%d, retrying in %ds): %s",
                    attempt + 1,
                    max_retries + 1,
                    wait,
                    e,
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
        # Suppress Qwen3 thinking mode to avoid massive <think> blocks
        sys_content = system_prompt
        if self._is_qwen3:
            sys_content = system_prompt + "\n/no_think"

        messages: list[dict] = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_message},
        ]
        openai_tools = [_to_openai_tool(td) for td in tools] if tools else None
        turns: list[AgentTurn] = []

        for _ in range(max_turns):
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools
            if self._seed is not None:
                kwargs["seed"] = self._seed

            response = self._completions_with_retry(**kwargs)
            choice = response.choices[0]
            message = choice.message
            usage = response.usage
            tokens = (usage.total_tokens if usage else 0) or 0

            # Strip <think> blocks from Qwen3 responses
            if self._is_qwen3 and message.content:
                message.content = _strip_think(message.content)

            tool_calls_in_msg = message.tool_calls or []

            if tool_calls_in_msg:
                # Assistant turn with tool calls
                parsed_calls = [_from_openai_tool_call(tc) for tc in tool_calls_in_msg]
                assistant_turn = AgentTurn(
                    role="assistant",
                    content=message.content,
                    tool_calls=parsed_calls,
                    tokens_used=tokens,
                )
                turns.append(assistant_turn)
                if turn_callback:
                    turn_callback(assistant_turn)

                # Append the assistant message to conversation
                messages.append(message.model_dump())

                # Execute tool calls — parallelize when multiple calls in a single turn
                tool_results: list[ToolResult] = []
                if len(parsed_calls) > 1:
                    log.debug(
                        "Executing %d tool calls in parallel", len(parsed_calls)
                    )
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=len(parsed_calls)
                    ) as tool_pool:
                        future_to_idx = {
                            tool_pool.submit(tool_executor, pc): i
                            for i, pc in enumerate(parsed_calls)
                        }
                        indexed_results: list[tuple[int, ToolResult]] = []
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            indexed_results.append((idx, future.result()))
                        # Preserve original order for correct tool_call_id pairing
                        indexed_results.sort(key=lambda x: x[0])
                        for i, tr in indexed_results:
                            tool_results.append(tr)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_calls_in_msg[i].id,
                                "content": tr.content,
                            })
                else:
                    for tc_def, parsed_call in zip(tool_calls_in_msg, parsed_calls, strict=True):
                        result = tool_executor(parsed_call)
                        tool_results.append(result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_def.id,
                            "content": result.content,
                        })

                turns.append(AgentTurn(
                    role="tool",
                    tool_results=tool_results,
                    tokens_used=0,
                ))

            else:
                # Final text response — no tool calls
                final_turn = AgentTurn(
                    role="assistant",
                    content=message.content or "",
                    tokens_used=tokens,
                )
                turns.append(final_turn)
                if turn_callback:
                    turn_callback(final_turn)
                break

        return turns


def _to_openai_tool(td: ToolDefinition) -> dict:
    """Convert a ToolDefinition to the OpenAI tool format."""
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }


def _from_openai_tool_call(tc) -> ToolCall:
    """Convert an OpenAI tool call object to our ToolCall dataclass."""
    try:
        arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
    except json.JSONDecodeError:
        arguments = {}

    return ToolCall(
        id=tc.id or uuid.uuid4().hex[:12],
        name=tc.function.name,
        arguments=arguments,
    )
