from __future__ import annotations

import hashlib
import math

from lens.matcher.base import BaseMatcher


def _simple_embedding(text: str) -> list[float]:
    """Generate a deterministic pseudo-embedding from text.

    This is a simple bag-of-words hash-based embedding for use when
    no external embedding model is available. For production use,
    replace with a real embedding model.
    """
    dim = 64
    vec = [0.0] * dim
    words = text.lower().split()

    for word in words:
        h = hashlib.md5(word.encode()).hexdigest()  # noqa: S324
        for i in range(dim):
            byte_val = int(h[i % len(h)], 16)
            vec[i] += (byte_val - 7.5) / 7.5

    # Normalize
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]

    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class EmbeddingMatcher(BaseMatcher):
    """Cosine similarity matcher using embeddings.

    Uses a simple hash-based embedding by default. Can be configured
    to use a real embedding model for better accuracy.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def match(self, text_a: str, text_b: str) -> bool:
        return self.similarity(text_a, text_b) >= self.threshold

    def similarity(self, text_a: str, text_b: str) -> float:
        emb_a = _simple_embedding(text_a)
        emb_b = _simple_embedding(text_b)
        return _cosine_similarity(emb_a, emb_b)
