from __future__ import annotations

import pytest

from lens.adapters.null import NullAdapter
from lens.agent.budget_enforcer import QuestionBudget
from lens.agent.harness import AgentHarness
from lens.agent.llm_client import (
    AgentTurn,
    MockLLMClient,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from lens.agent.tool_bridge import build_tool_definitions, dispatch_tool_call


class TestMockLLMClient:
    def test_produces_three_turns(self):
        client = MockLLMClient()

        def executor(tc: ToolCall) -> ToolResult:
            return ToolResult(tool_call_id=tc.id, content="mock result")

        tools = [ToolDefinition(name="memory_search", description="search", parameters={})]
        turns = client.run_agent_loop(
            system_prompt="test",
            user_message="What happened?",
            tools=tools,
            tool_executor=executor,
        )
        # 5 turns: assistant(search), tool(result), assistant(caps), tool(result), assistant(answer)
        assert len(turns) == 5
        assert turns[0].role == "assistant"
        assert turns[0].tool_calls is not None
        assert turns[-1].role == "assistant"
        assert turns[-1].content is not None

    def test_final_answer_contains_text(self):
        client = MockLLMClient()

        def executor(tc: ToolCall) -> ToolResult:
            return ToolResult(tool_call_id=tc.id, content="data")

        tools = []
        turns = client.run_agent_loop("sys", "user q", tools, executor)
        final = turns[-1]
        assert "Based on my search" in final.content
        assert "The answer is" in final.content


class TestToolBridge:
    def test_build_tool_definitions_null_adapter(self):
        adapter = NullAdapter()
        tools = build_tool_definitions(adapter)
        names = {t.name for t in tools}
        assert "memory_search" in names
        assert "memory_retrieve" in names
        assert "memory_capabilities" in names

    def test_dispatch_search(self):
        adapter = NullAdapter()
        call = ToolCall(id="tc1", name="memory_search", arguments={"query": "test"})
        result = dispatch_tool_call(adapter, call)
        assert not result.is_error
        assert "[]" in result.content

    def test_dispatch_retrieve(self):
        adapter = NullAdapter()
        call = ToolCall(id="tc2", name="memory_retrieve", arguments={"ref_id": "xxx"})
        result = dispatch_tool_call(adapter, call)
        assert not result.is_error

    def test_dispatch_capabilities(self):
        adapter = NullAdapter()
        call = ToolCall(id="tc3", name="memory_capabilities", arguments={})
        result = dispatch_tool_call(adapter, call)
        assert not result.is_error
        assert "semantic" in result.content

    def test_dispatch_unknown_tool(self):
        adapter = NullAdapter()
        call = ToolCall(id="tc4", name="unknown_tool", arguments={})
        result = dispatch_tool_call(adapter, call)
        assert result.is_error

    def test_payload_truncation(self):
        adapter = NullAdapter()
        call = ToolCall(id="tc5", name="memory_capabilities", arguments={})
        result = dispatch_tool_call(adapter, call, max_payload_bytes=10)
        assert "[truncated]" in result.content


class TestAgentHarness:
    def test_answer_returns_agent_answer(self):
        client = MockLLMClient()
        budget = QuestionBudget(max_turns=10, max_total_tool_calls=20)
        harness = AgentHarness(client, budget)
        adapter = NullAdapter()

        answer = harness.answer("What patterns emerged?", adapter, "q01")
        assert answer.question_id == "q01"
        assert answer.answer_text  # Should have some text
        assert answer.tool_calls_made >= 0
        assert answer.wall_time_ms > 0

    def test_answer_tracks_tool_calls(self):
        client = MockLLMClient()
        budget = QuestionBudget(max_turns=10, max_total_tool_calls=20)
        harness = AgentHarness(client, budget)
        adapter = NullAdapter()

        answer = harness.answer("test", adapter, "q02")
        # MockLLMClient makes 2 tool calls (search + capabilities)
        assert answer.tool_calls_made == 2

    def test_answer_no_budget_violations(self):
        client = MockLLMClient()
        budget = QuestionBudget(max_turns=10, max_total_tool_calls=20)
        harness = AgentHarness(client, budget)
        adapter = NullAdapter()

        answer = harness.answer("test", adapter)
        assert answer.budget_violations == []
