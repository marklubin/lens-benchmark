from __future__ import annotations

from lens.matcher.base import BaseMatcher


class LLMJudgeMatcher(BaseMatcher):
    """LLM-based matcher for determining insight equivalence.

    Uses an LLM to judge whether two insights are semantically the same.
    More robust than embedding similarity but requires LLM calls (offline only).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._llm: object | None = None

    def set_llm(self, llm: object) -> None:
        """Set the LLM client to use for judging."""
        self._llm = llm

    def match(self, text_a: str, text_b: str) -> bool:
        return self.similarity(text_a, text_b) >= self.threshold

    def similarity(self, text_a: str, text_b: str) -> float:
        """Judge similarity using LLM.

        Returns 1.0 if same, 0.0 if different.
        Falls back to simple heuristic if no LLM is set.
        """
        if self._llm is None:
            # Fallback: simple word overlap
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            if not words_a or not words_b:
                return 0.0
            overlap = len(words_a & words_b)
            return overlap / max(len(words_a), len(words_b))

        # LLM judge — placeholder for real implementation
        # Would call self._llm.complete() with a same/different prompt
        return 0.0
