from __future__ import annotations

import json
import time
from dataclasses import asdict

from lens.agent.budget_enforcer import BudgetEnforcement, BudgetViolation, QuestionBudget
from lens.agent.llm_client import AgentTurn, BaseLLMClient, ToolCall, ToolResult
from lens.agent.tool_bridge import build_tool_definitions, dispatch_tool_call
from lens.core.models import AgentAnswer

SYSTEM_PROMPT = """\
You are a research assistant with access to a memory system. Your task is to \
answer the user's question by searching and retrieving information from memory.

Instructions:
- Use memory_search to find relevant information for the question.
- Use memory_retrieve to get full document details when you need specifics. \
Reference retrieved documents by their ref_id when citing evidence.
- Use memory_capabilities to understand what the memory system offers, \
including available search modes and filter fields.
- Synthesize your findings into a clear, concise answer.
- Cite evidence by referencing the ref_ids of retrieved documents.
- If you cannot find sufficient information, say so clearly.
"""


class AgentHarness:
    """Runs the agent loop for a single question against a memory adapter."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        budget: QuestionBudget | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.budget = budget or QuestionBudget()

    def answer(
        self,
        question_prompt: str,
        adapter,
        question_id: str = "",
    ) -> AgentAnswer:
        """Run the agent to answer a single question.

        Args:
            question_prompt: The question text to answer.
            adapter: A MemoryAdapter instance to use as tools.
            question_id: Optional identifier for the question.

        Returns:
            An AgentAnswer with the result, tool usage stats, and budget info.
        """
        tools = build_tool_definitions(adapter)
        enforcer = BudgetEnforcement(self.budget)
        refs_cited: list[str] = []
        turn_counter = 0

        def tool_executor(tool_call: ToolCall) -> ToolResult:
            nonlocal turn_counter
            try:
                enforcer.check_tool_call()
            except BudgetViolation:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content="Budget exceeded: too many tool calls.",
                    is_error=True,
                )

            result = dispatch_tool_call(adapter, tool_call, self.budget.max_payload_bytes)
            enforcer.record_tool_call()
            enforcer.check_payload(len(result.content.encode("utf-8")))

            # Track ref_ids from memory_retrieve calls
            if tool_call.name == "memory_retrieve" and not result.is_error:
                ref_id = tool_call.arguments.get("ref_id", "")
                if ref_id:
                    refs_cited.append(ref_id)

            return result

        wall_start = time.monotonic()
        try:
            turns = self.llm_client.run_agent_loop(
                system_prompt=SYSTEM_PROMPT,
                user_message=question_prompt,
                tools=tools,
                tool_executor=tool_executor,
                max_turns=self.budget.max_turns,
            )
        except BudgetViolation:
            turns = []
        wall_ms = (time.monotonic() - wall_start) * 1000

        # Record turn and token usage
        total_tokens = 0
        tool_calls_made = 0
        for turn in turns:
            if turn.role == "assistant":
                enforcer.record_turn()
                total_tokens += turn.tokens_used
                if turn.tool_calls:
                    tool_calls_made += len(turn.tool_calls)
            elif turn.role == "tool":
                total_tokens += turn.tokens_used

        # Extract final answer text from the last assistant turn
        answer_text = ""
        for turn in reversed(turns):
            if turn.role == "assistant" and turn.content:
                answer_text = turn.content
                break

        # Serialize turns
        serialized_turns = []
        for turn in turns:
            turn_dict: dict = {"role": turn.role}
            if turn.content is not None:
                turn_dict["content"] = turn.content
            if turn.tool_calls:
                turn_dict["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in turn.tool_calls
                ]
            if turn.tool_results:
                turn_dict["tool_results"] = [
                    {"tool_call_id": tr.tool_call_id, "content": tr.content, "is_error": tr.is_error}
                    for tr in turn.tool_results
                ]
            turn_dict["tokens_used"] = turn.tokens_used
            serialized_turns.append(turn_dict)

        return AgentAnswer(
            question_id=question_id,
            answer_text=answer_text,
            turns=serialized_turns,
            tool_calls_made=tool_calls_made,
            total_tokens=total_tokens,
            wall_time_ms=wall_ms,
            budget_violations=list(enforcer.violations),
            refs_cited=list(refs_cited),
        )
