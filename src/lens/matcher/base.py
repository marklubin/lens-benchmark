from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMatcher(ABC):
    """Abstract base class for insight matchers.

    Matchers determine whether two insight texts refer to the "same" insight.
    Used by stability metrics (Survival@k, Churn@k).
    """

    @abstractmethod
    def match(self, text_a: str, text_b: str) -> bool:
        """Return True if the two texts represent the same insight."""

    @abstractmethod
    def similarity(self, text_a: str, text_b: str) -> float:
        """Return a similarity score in [0, 1]."""
