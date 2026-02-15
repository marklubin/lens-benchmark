from __future__ import annotations

from dataclasses import dataclass, field

from lens.core.errors import LensError


class BudgetViolation(LensError):
    """Raised when a hard budget limit is exceeded."""


@dataclass
class QuestionBudget:
    """Per-question budget limits for the agent."""

    max_turns: int = 10
    max_payload_bytes: int = 65536
    max_latency_per_call_ms: float = 5000
    max_total_tool_calls: int = 20
    max_agent_tokens: int = 8192


class BudgetEnforcement:
    """Tracks and enforces per-question budget limits."""

    def __init__(self, budget: QuestionBudget) -> None:
        self.budget = budget
        self.turns_used: int = 0
        self.tool_calls_used: int = 0
        self.total_payload_bytes: int = 0
        self.total_tokens: int = 0
        self.violations: list[str] = []
        self.warnings: list[str] = []

    def check_turn(self) -> None:
        """Raise BudgetViolation if turns exceed max_turns."""
        if self.turns_used >= self.budget.max_turns:
            msg = f"Turn limit exceeded: {self.turns_used} >= {self.budget.max_turns}"
            self.violations.append(msg)
            raise BudgetViolation(msg)

    def check_tool_call(self) -> None:
        """Raise BudgetViolation if tool calls would exceed limit."""
        if self.tool_calls_used >= self.budget.max_total_tool_calls:
            msg = f"Tool call limit exceeded: {self.tool_calls_used} >= {self.budget.max_total_tool_calls}"
            self.violations.append(msg)
            raise BudgetViolation(msg)

    def check_payload(self, payload_bytes: int) -> None:
        """Record payload size. Warns but does not raise or count as violation."""
        self.total_payload_bytes += payload_bytes
        if payload_bytes > self.budget.max_payload_bytes:
            msg = f"Payload size warning: {payload_bytes} > {self.budget.max_payload_bytes}"
            self.warnings.append(msg)

    def check_latency(self, latency_ms: float) -> None:
        """Record a hard violation if a tool call exceeds the latency cap."""
        if latency_ms > self.budget.max_latency_per_call_ms:
            msg = (
                f"Tool call latency exceeded: {latency_ms:.0f}ms "
                f"> {self.budget.max_latency_per_call_ms:.0f}ms"
            )
            self.violations.append(msg)

    def check_tokens(self, tokens: int) -> None:
        """Add tokens and record a violation if total exceeds limit.

        Records the violation for scoring but does NOT raise, since the
        response has already been received and paid for. Hard stops are
        only enforced for turns and tool calls.
        """
        self.total_tokens += tokens
        if self.total_tokens > self.budget.max_agent_tokens:
            msg = f"Token limit exceeded: {self.total_tokens} > {self.budget.max_agent_tokens}"
            self.violations.append(msg)

    def record_turn(self) -> None:
        """Increment turns used."""
        self.turns_used += 1

    def record_tool_call(self) -> None:
        """Increment tool calls used."""
        self.tool_calls_used += 1

    def record_tokens(self, n: int) -> None:
        """Add to total tokens (without enforcement check)."""
        self.total_tokens += n

    @property
    def is_exhausted(self) -> bool:
        """True if any hard limit has been reached."""
        return (
            self.turns_used >= self.budget.max_turns
            or self.tool_calls_used >= self.budget.max_total_tool_calls
            or self.total_tokens >= self.budget.max_agent_tokens
        )

    def summary(self) -> dict:
        """Return a dict of all tracked values, violations, and warnings."""
        return {
            "turns_used": self.turns_used,
            "tool_calls_used": self.tool_calls_used,
            "total_payload_bytes": self.total_payload_bytes,
            "total_tokens": self.total_tokens,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "is_exhausted": self.is_exhausted,
            "budget": {
                "max_turns": self.budget.max_turns,
                "max_payload_bytes": self.budget.max_payload_bytes,
                "max_latency_per_call_ms": self.budget.max_latency_per_call_ms,
                "max_total_tool_calls": self.budget.max_total_tool_calls,
                "max_agent_tokens": self.budget.max_agent_tokens,
            },
        }
