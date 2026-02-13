from __future__ import annotations

from abc import ABC, abstractmethod

from lens.core.models import Hit, Insight


class MemoryAdapter(ABC):
    """Abstract base class for memory system adapters.

    Every memory backend must implement these 5 methods.
    The runner injects a BudgetedLLM handle before calling any method.
    """

    @abstractmethod
    def reset(self, persona_id: str) -> None:
        """Clear all state for a persona. Called once before episode stream begins."""

    @abstractmethod
    def ingest(
        self,
        episode_id: str,
        persona_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Ingest a single episode. Must complete within 200ms, no LLM calls allowed."""

    @abstractmethod
    def refresh(self, persona_id: str, checkpoint: int) -> None:
        """Phase A: offline refresh. Metered but uncapped. Build/update insights."""

    @abstractmethod
    def core(self, persona_id: str, k: int, checkpoint: int) -> list[Insight]:
        """Phase B: return top-k longitudinal insights. Online, budgeted."""

    @abstractmethod
    def search(self, persona_id: str, query: str, k: int, checkpoint: int) -> list[Hit]:
        """Phase B: search memory. Online, budgeted."""

    def set_budgeted_llm(self, llm: object) -> None:
        """Called by runner to inject the BudgetedLLM handle."""
        self._budgeted_llm = llm

    @property
    def llm(self) -> object:
        """Access the runner-provided BudgetedLLM."""
        if not hasattr(self, "_budgeted_llm"):
            msg = "BudgetedLLM not set. Adapter must be run through the RunEngine."
            raise RuntimeError(msg)
        return self._budgeted_llm
