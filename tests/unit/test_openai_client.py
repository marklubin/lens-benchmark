from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from lens.agent.llm_client import ToolCall, ToolDefinition, ToolResult


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeUsage:
    total_tokens: int


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list | None

    def model_dump(self):
        d: dict = {"role": "assistant"}
        if self.content:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return d


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


def _get_openai_client_cls():
    """Import OpenAIClient with a fake openai SDK module."""
    fake = types.ModuleType("openai")
    fake.OpenAI = MagicMock()
    sys.modules["openai"] = fake
    # Force re-import
    sys.modules.pop("lens.agent.openai_client", None)
    from lens.agent.openai_client import OpenAIClient, _from_openai_tool_call, _to_openai_tool
    return OpenAIClient, _to_openai_tool, _from_openai_tool_call, fake


class TestOpenAIToolConversion:
    def test_to_openai_tool(self):
        _, _to_openai_tool, _, _ = _get_openai_client_cls()

        td = ToolDefinition(
            name="memory_search",
            description="Search memory",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        result = _to_openai_tool(td)
        assert result["type"] == "function"
        assert result["function"]["name"] == "memory_search"
        assert result["function"]["description"] == "Search memory"
        assert result["function"]["parameters"]["required"] == ["query"]

    def test_from_openai_tool_call(self):
        _, _, _from_openai_tool_call, _ = _get_openai_client_cls()

        tc = _FakeToolCall(id="call_123", function=_FakeFunction(name="memory_search", arguments='{"query": "test"}'))
        result = _from_openai_tool_call(tc)
        assert isinstance(result, ToolCall)
        assert result.id == "call_123"
        assert result.name == "memory_search"
        assert result.arguments == {"query": "test"}

    def test_from_openai_tool_call_bad_json(self):
        _, _, _from_openai_tool_call, _ = _get_openai_client_cls()

        tc = _FakeToolCall(id="call_bad", function=_FakeFunction(name="foo", arguments="not json"))
        result = _from_openai_tool_call(tc)
        assert result.arguments == {}


class TestOpenAIClientLoop:
    def _make_client(self, responses):
        """Create an OpenAIClient with a mocked OpenAI SDK."""
        OpenAIClient, _, _, fake = _get_openai_client_cls()

        mock_sdk_client = MagicMock()
        mock_sdk_client.chat.completions.create.side_effect = responses
        fake.OpenAI.return_value = mock_sdk_client

        client = OpenAIClient(api_key="test-key", model="test-model")
        client._client = mock_sdk_client
        return client

    def test_simple_text_response(self):
        response = _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content="Hello!", tool_calls=None))],
            usage=_FakeUsage(total_tokens=50),
        )
        client = self._make_client([response])

        tools = [ToolDefinition(name="test", description="test", parameters={"type": "object", "properties": {}})]
        executor = MagicMock()

        turns = client.run_agent_loop(
            system_prompt="system",
            user_message="user",
            tools=tools,
            tool_executor=executor,
        )

        assert len(turns) == 1
        assert turns[0].role == "assistant"
        assert turns[0].content == "Hello!"
        assert turns[0].tokens_used == 50
        executor.assert_not_called()

    def test_tool_call_then_response(self):
        # First response: tool call
        tool_call_response = _FakeResponse(
            choices=[_FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[_FakeToolCall(id="tc1", function=_FakeFunction(name="memory_search", arguments='{"query": "test"}'))],
                )
            )],
            usage=_FakeUsage(total_tokens=100),
        )
        # Second response: text
        text_response = _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content="Found results.", tool_calls=None))],
            usage=_FakeUsage(total_tokens=80),
        )

        client = self._make_client([tool_call_response, text_response])

        executor = MagicMock(return_value=ToolResult(tool_call_id="tc1", content='[{"ref_id": "ep1"}]'))
        callback = MagicMock()

        turns = client.run_agent_loop(
            system_prompt="system",
            user_message="user",
            tools=[ToolDefinition(name="memory_search", description="search", parameters={"type": "object", "properties": {}})],
            tool_executor=executor,
            turn_callback=callback,
        )

        # 3 turns: assistant(tool_call) + tool(results) + assistant(text)
        assert len(turns) == 3
        assert turns[0].role == "assistant"
        assert turns[0].tool_calls is not None
        assert turns[1].role == "tool"
        assert turns[2].role == "assistant"
        assert turns[2].content == "Found results."

        executor.assert_called_once()
        assert callback.call_count == 2  # One for tool call turn, one for final

    def test_max_turns_respected(self):
        # All responses are tool calls — should stop after max_turns
        tool_response = _FakeResponse(
            choices=[_FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[_FakeToolCall(id="tc", function=_FakeFunction(name="search", arguments='{}'))],
                )
            )],
            usage=_FakeUsage(total_tokens=50),
        )

        client = self._make_client([tool_response, tool_response, tool_response])
        executor = MagicMock(return_value=ToolResult(tool_call_id="tc", content="ok"))

        turns = client.run_agent_loop(
            system_prompt="sys",
            user_message="msg",
            tools=[ToolDefinition(name="search", description="s", parameters={"type": "object", "properties": {}})],
            tool_executor=executor,
            max_turns=2,
        )

        # 2 iterations * (assistant + tool) = 4 turns
        assert len(turns) == 4
