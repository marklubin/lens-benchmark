from __future__ import annotations

from dataclasses import dataclass, field

from lens.core.config import BudgetConfig, BudgetPreset
from lens.core.errors import BudgetExceededError


@dataclass
class UsageRecord:
    """Tracks LLM usage for a single method invocation."""

    phase: str = ""
    method: str = ""
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class BudgetTracker:
    """Tracks cumulative LLM usage across a run."""

    records: list[UsageRecord] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return sum(r.llm_calls for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def by_phase(self, phase: str) -> list[UsageRecord]:
        return [r for r in self.records if r.phase == phase]

    def by_method(self, method: str) -> list[UsageRecord]:
        return [r for r in self.records if r.method == method]

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "by_phase": {
                phase: {
                    "calls": sum(r.llm_calls for r in records),
                    "tokens": sum(r.total_tokens for r in records),
                }
                for phase in {"refresh", "core", "search"}
                if (records := self.by_phase(phase))
            },
        }


class BudgetedLLM:
    """Wrapper that enforces budget limits on LLM access.

    Injected into adapters by the runner. The adapter calls methods on this
    object; the runner decides whether the call is permitted based on the
    active budget preset and current phase/method context.
    """

    def __init__(self, budget_config: BudgetConfig, tracker: BudgetTracker | None = None) -> None:
        self.budget_config = budget_config
        self.tracker = tracker or BudgetTracker()
        self._current_phase: str = ""
        self._current_method: str = ""
        self._method_calls: int = 0
        self._method_tokens: int = 0

    def set_context(self, phase: str, method: str) -> None:
        """Set the current execution context. Called by the runner before each adapter method."""
        self._current_phase = phase
        self._current_method = method
        self._method_calls = 0
        self._method_tokens = 0

    @property
    def _active_budget(self) -> BudgetPreset:
        """Get the budget preset for the current method."""
        method_budgets = {
            "ingest": self.budget_config.ingest,
            "refresh": self.budget_config.refresh,
            "core": self.budget_config.core,
            "search": self.budget_config.search,
        }
        return method_budgets.get(self._current_method, BudgetPreset())

    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str:
        """Request an LLM completion. Raises BudgetExceededError if over budget.

        Returns a placeholder string. Real implementation will call the actual LLM.
        """
        budget = self._active_budget

        # Check call count
        if self._method_calls >= budget.max_llm_calls:
            raise BudgetExceededError(
                phase=self._current_phase,
                method=self._current_method,
                limit_kind="llm_calls",
                limit=budget.max_llm_calls,
                actual=self._method_calls + 1,
            )

        # Check token budget
        if self._method_tokens + max_tokens > budget.max_tokens:
            raise BudgetExceededError(
                phase=self._current_phase,
                method=self._current_method,
                limit_kind="tokens",
                limit=budget.max_tokens,
                actual=self._method_tokens + max_tokens,
            )

        self._method_calls += 1
        self._method_tokens += max_tokens

        # Record usage
        record = UsageRecord(
            phase=self._current_phase,
            method=self._current_method,
            llm_calls=1,
            tokens_out=max_tokens,
        )
        self.tracker.records.append(record)

        # Placeholder — real LLM call goes here
        return ""

    @property
    def calls_remaining(self) -> int:
        budget = self._active_budget
        return max(0, budget.max_llm_calls - self._method_calls)

    @property
    def tokens_remaining(self) -> int:
        budget = self._active_budget
        return max(0, budget.max_tokens - self._method_tokens)
