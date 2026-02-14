from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent."""

    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class ToolCall:
    """A tool invocation requested by the agent."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """Result returned from executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentTurn:
    """A single turn in the agent conversation."""

    role: str  # "assistant" or "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    tokens_used: int = 0


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients that support tool-use agent loops."""

    @abstractmethod
    def run_agent_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
        tool_executor: Callable[[ToolCall], ToolResult],
        max_turns: int = 10,
    ) -> list[AgentTurn]:
        """Run the full agent loop.

        Sends the user message, receives tool calls, dispatches them via
        tool_executor, sends results back, and repeats until the agent
        gives a final text answer or max_turns is reached.
        """


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client for testing. No API keys needed.

    On turn 1: makes a memory_search tool call with the user message as query.
    On turn 2: makes a memory_capabilities tool call.
    On turn 3: produces a final text answer summarizing tool results.
    """

    def __init__(self, search_responses: dict[str, str] | None = None) -> None:
        self.search_responses = search_responses or {}

    def run_agent_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[ToolDefinition],
        tool_executor: Callable[[ToolCall], ToolResult],
        max_turns: int = 10,
    ) -> list[AgentTurn]:
        turns: list[AgentTurn] = []
        tool_results_summary: list[str] = []

        # Turn 1: memory_search
        search_call = ToolCall(
            id="mock-tc-1",
            name="memory_search",
            arguments={"query": user_message},
        )
        turns.append(AgentTurn(
            role="assistant",
            tool_calls=[search_call],
            tokens_used=100,
        ))

        search_result = tool_executor(search_call)
        tool_results_summary.append(search_result.content)
        turns.append(AgentTurn(
            role="tool",
            tool_results=[search_result],
            tokens_used=100,
        ))

        # Turn 2: memory_capabilities
        caps_call = ToolCall(
            id="mock-tc-2",
            name="memory_capabilities",
            arguments={},
        )
        turns.append(AgentTurn(
            role="assistant",
            tool_calls=[caps_call],
            tokens_used=100,
        ))

        caps_result = tool_executor(caps_call)
        tool_results_summary.append(caps_result.content)
        turns.append(AgentTurn(
            role="tool",
            tool_results=[caps_result],
            tokens_used=100,
        ))

        # Turn 3: final text answer
        summary = "; ".join(tool_results_summary)
        answer_text = (
            f"Based on my search, I found: [{summary}]. "
            "The answer is: I could not determine a specific answer from the available data."
        )
        turns.append(AgentTurn(
            role="assistant",
            content=answer_text,
            tokens_used=100,
        ))

        return turns
