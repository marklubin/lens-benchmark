from __future__ import annotations

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
        turn_callback: Callable[[AgentTurn], None] | None = None,
    ) -> list[AgentTurn]:
        """Run the full agent loop.

        Sends the user message, receives tool calls, dispatches them via
        tool_executor, sends results back, and repeats until the agent
        gives a final text answer or max_turns is reached.

        Args:
            system_prompt: System prompt for the agent.
            user_message: The user's question.
            tools: Available tool definitions.
            tool_executor: Callback to execute tool calls.
            max_turns: Maximum number of assistant turns before forcing termination.
            turn_callback: Optional callback invoked after each assistant turn,
                used by the harness for budget enforcement. May raise to stop the loop.
        """


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client for testing. No API keys needed.

    Respects max_turns:
    - max_turns >= 3: search -> capabilities -> final answer (3 assistant turns)
    - max_turns == 2: search -> final answer (2 assistant turns)
    - max_turns == 1: final answer only (1 assistant turn)
    - max_turns == 0: no turns (empty)
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
        turn_callback: Callable[[AgentTurn], None] | None = None,
    ) -> list[AgentTurn]:
        turns: list[AgentTurn] = []
        tool_results_summary: list[str] = []
        turns_remaining = max_turns

        if turns_remaining <= 0:
            return turns

        # Turn 1: memory_search (if we have budget for at least 2 turns)
        if turns_remaining >= 2:
            search_call = ToolCall(
                id="mock-tc-1",
                name="memory_search",
                arguments={"query": user_message},
            )
            assistant_turn = AgentTurn(
                role="assistant",
                tool_calls=[search_call],
                tokens_used=100,
            )
            turns.append(assistant_turn)
            if turn_callback:
                turn_callback(assistant_turn)

            search_result = tool_executor(search_call)
            tool_results_summary.append(search_result.content)
            turns.append(AgentTurn(
                role="tool",
                tool_results=[search_result],
                tokens_used=100,
            ))
            turns_remaining -= 1

        # Turn 2: memory_capabilities (if we have budget for at least 2 more turns)
        if turns_remaining >= 2:
            caps_call = ToolCall(
                id="mock-tc-2",
                name="memory_capabilities",
                arguments={},
            )
            assistant_turn = AgentTurn(
                role="assistant",
                tool_calls=[caps_call],
                tokens_used=100,
            )
            turns.append(assistant_turn)
            if turn_callback:
                turn_callback(assistant_turn)

            caps_result = tool_executor(caps_call)
            tool_results_summary.append(caps_result.content)
            turns.append(AgentTurn(
                role="tool",
                tool_results=[caps_result],
                tokens_used=100,
            ))
            turns_remaining -= 1

        # Final turn: text answer
        summary = "; ".join(tool_results_summary) if tool_results_summary else "no data"
        answer_text = (
            f"Based on my search, I found: [{summary}]. "
            "The answer is: I could not determine a specific answer from the available data."
        )
        final_turn = AgentTurn(
            role="assistant",
            content=answer_text,
            tokens_used=100,
        )
        turns.append(final_turn)
        if turn_callback:
            turn_callback(final_turn)

        return turns
