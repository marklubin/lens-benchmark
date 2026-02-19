from __future__ import annotations

import re
import time

from lens.agent.budget_enforcer import BudgetEnforcement, BudgetViolation, QuestionBudget
from lens.agent.llm_client import AgentTurn, BaseLLMClient, ToolCall, ToolResult
from lens.agent.tool_bridge import build_tool_definitions, dispatch_tool_call
from lens.core.models import AgentAnswer

SYSTEM_PROMPT = """\
You are a research assistant with access to a memory system. Your task is to \
answer the user's question by searching and retrieving information from memory.

Instructions:
- Use memory_search to find relevant information for the question.
- Use memory_retrieve to get full document details when you need specifics.
- Use memory_capabilities to understand what the memory system offers, \
including available search modes, filter fields, and any extra tools.
- If extra tools are available (e.g. batch_retrieve), prefer them for efficiency \
— they can fetch multiple documents in one call instead of one call per document.
- Synthesize your findings into a clear, concise answer.
- IMPORTANT: For each claim in your answer, cite the supporting episode using \
the format [ref_id]. Every factual statement must have at least one citation.
- If you cannot find sufficient information, say so clearly.
- You have a limited number of turns and tool calls. Use them efficiently.
"""


def _extract_inline_refs(text: str) -> list[str]:
    """Extract [ref_id] citations from answer text.

    Matches patterns like [scope_name_ep_NNN] or (ref_id: scope_name_ep_NNN).
    """
    patterns = [
        r'\[([a-z][a-z0-9_]*_ep_\d+)\]',              # [insider_threat_05_ep_007]
        r'\(ref_id:\s*([a-z][a-z0-9_]*_ep_\d+)\)',     # (ref_id: insider_threat_05_ep_007)
    ]
    refs: list[str] = []
    for pat in patterns:
        refs.extend(re.findall(pat, text))
    return list(dict.fromkeys(refs))  # deduplicate preserving order


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

        def tool_executor(tool_call: ToolCall) -> ToolResult:
            # Pre-flight: check tool call budget
            try:
                enforcer.check_tool_call()
            except BudgetViolation:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content="Budget exceeded: too many tool calls.",
                    is_error=True,
                )

            # Execute with latency measurement
            t0 = time.monotonic()
            result = dispatch_tool_call(adapter, tool_call, self.budget.max_payload_bytes)
            latency_ms = (time.monotonic() - t0) * 1000

            # Post-flight: record and check
            enforcer.record_tool_call()
            enforcer.check_latency(latency_ms)
            enforcer.check_payload(len(result.content.encode("utf-8")))

            # Track ref_ids from memory_retrieve and any extended tool with ref_ids
            if not result.is_error:
                if tool_call.name == "memory_retrieve":
                    ref_id = tool_call.arguments.get("ref_id", "")
                    if ref_id:
                        refs_cited.append(ref_id)
                # Extended tools that accept a list of ref_ids (e.g. batch_retrieve)
                ref_ids = tool_call.arguments.get("ref_ids")
                if isinstance(ref_ids, list):
                    refs_cited.extend(r for r in ref_ids if isinstance(r, str) and r)

            return result

        def turn_callback(turn: AgentTurn) -> None:
            """Called by the LLM client after each assistant turn for budget enforcement."""
            enforcer.record_turn()
            enforcer.check_tokens(turn.tokens_used)
            enforcer.check_turn()

        all_turns: list[AgentTurn] = []

        def tracking_turn_callback(turn: AgentTurn) -> None:
            """Track every assistant turn so we preserve work on budget violation."""
            all_turns.append(turn)
            turn_callback(turn)

        wall_start = time.monotonic()
        try:
            turns = self.llm_client.run_agent_loop(
                system_prompt=SYSTEM_PROMPT,
                user_message=question_prompt,
                tools=tools,
                tool_executor=tool_executor,
                max_turns=self.budget.max_turns,
                turn_callback=tracking_turn_callback,
            )
        except BudgetViolation:
            # Preserve the assistant turns we collected before the violation
            turns = all_turns
        wall_ms = (time.monotonic() - wall_start) * 1000

        # Count tool calls from turns (for reporting)
        total_tokens = enforcer.total_tokens
        tool_calls_made = enforcer.tool_calls_used

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

        # Merge inline [ref_id] citations with tool-call refs
        inline_refs = _extract_inline_refs(answer_text)
        all_refs = list(dict.fromkeys(refs_cited + inline_refs))

        return AgentAnswer(
            question_id=question_id,
            answer_text=answer_text,
            turns=serialized_turns,
            tool_calls_made=tool_calls_made,
            total_tokens=total_tokens,
            wall_time_ms=wall_ms,
            budget_violations=list(enforcer.violations),
            refs_cited=all_refs,
        )
