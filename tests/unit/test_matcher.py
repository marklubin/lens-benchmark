from __future__ import annotations

from lens.matcher.embedding import EmbeddingMatcher, _cosine_similarity, _simple_embedding
from lens.matcher.llm_judge import LLMJudgeMatcher


class TestEmbeddingMatcher:
    def test_identical_texts_match(self):
        matcher = EmbeddingMatcher(threshold=0.85)
        assert matcher.match("anxiety pattern evolution", "anxiety pattern evolution")

    def test_very_different_texts_dont_match(self):
        matcher = EmbeddingMatcher(threshold=0.85)
        assert not matcher.match(
            "anxiety pattern in therapy sessions",
            "export feature requirements for dashboard product",
        )

    def test_similarity_range(self):
        matcher = EmbeddingMatcher()
        sim = matcher.similarity("hello world", "hello world")
        assert 0.0 <= sim <= 1.0

    def test_empty_text(self):
        matcher = EmbeddingMatcher()
        sim = matcher.similarity("", "hello")
        assert 0.0 <= sim <= 1.0


class TestSimpleEmbedding:
    def test_normalized(self):
        import math

        emb = _simple_embedding("test text here")
        mag = math.sqrt(sum(v * v for v in emb))
        assert abs(mag - 1.0) < 0.01

    def test_deterministic(self):
        emb1 = _simple_embedding("reproducible")
        emb2 = _simple_embedding("reproducible")
        assert emb1 == emb2


class TestLLMJudgeMatcher:
    def test_fallback_word_overlap(self):
        matcher = LLMJudgeMatcher(threshold=0.3)
        # Same words = high overlap
        assert matcher.match("the cat sat on mat", "the cat sat on mat")

    def test_no_overlap(self):
        matcher = LLMJudgeMatcher(threshold=0.5)
        assert not matcher.match("alpha beta gamma", "delta epsilon zeta")
