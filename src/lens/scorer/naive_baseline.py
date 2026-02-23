"""Naive baseline generator for head-to-head comparison.

Generates context-stuffed answers by concatenating all episodes up to each
question's checkpoint_after, then asking the same LLM model used by the
adapter to answer the question. This establishes the floor: if a memory
system can't beat context stuffing, it adds no value.

The generator caches answers per (question_id, model) to avoid redundant
LLM calls when re-scoring.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
from pathlib import Path
from typing import Callable

from lens.core.models import Episode, Question

logger = logging.getLogger(__name__)

# Standalone prompt — mirrors datagen naive baseline but without synix imports
_NAIVE_SYSTEM = (
    "You are an analyst. You will be given a series of sequential log entries "
    "and a question about patterns across those entries. "
    "Answer the question based ONLY on the provided data. "
    "Be specific and cite entry IDs (e.g. [entry_id]) when referencing data."
)

_NAIVE_USER_TMPL = """\
SEQUENTIAL LOG ENTRIES (chronological order):

{episodes_block}

QUESTION:
{question}

Answer the question based on the log entries above. Be specific and cite \
entry IDs in [brackets] when referencing data points."""


def build_naive_prompt(
    episodes: list[Episode],
    question_prompt: str,
    max_tokens: int = 0,
) -> tuple[str, str]:
    """Build system + user prompts for the naive baseline.

    Args:
        episodes: Episodes in chronological order (already filtered to checkpoint).
        question_prompt: The question text.
        max_tokens: If > 0, truncate the episode block to roughly this many
            tokens (estimated at 4 chars/token). 0 = unlimited.

    Returns:
        (system_prompt, user_prompt)
    """
    lines: list[str] = []
    total_chars = 0
    char_limit = max_tokens * 4 if max_tokens > 0 else 0

    for ep in episodes:
        entry = f"[{ep.episode_id}] {ep.timestamp.isoformat()}: {ep.text}"
        if char_limit > 0 and total_chars + len(entry) > char_limit:
            break
        lines.append(entry)
        total_chars += len(entry)

    episodes_block = "\n\n".join(lines)
    user_prompt = _NAIVE_USER_TMPL.format(
        episodes_block=episodes_block,
        question=question_prompt,
    )
    return _NAIVE_SYSTEM, user_prompt


class NaiveBaselineGenerator:
    """Generates and caches naive baseline answers for comparison.

    Args:
        llm_fn: Callable(system_prompt, user_prompt) -> answer_text.
        episodes: All episodes for the scope, sorted chronologically.
        cache_dir: Directory to store/load cached answers.
        model_id: Model identifier for cache key deduplication.
        max_result_tokens: Cap on episode context (0 = unlimited).
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str],
        episodes: list[Episode],
        cache_dir: Path | None = None,
        model_id: str = "unknown",
        max_result_tokens: int = 0,
        max_workers: int = 1,
    ) -> None:
        self._llm_fn = llm_fn
        self._episodes = sorted(episodes, key=lambda e: e.timestamp)
        self._cache_dir = cache_dir
        self._model_id = model_id
        self._max_result_tokens = max_result_tokens
        self._max_workers = max_workers
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._load_cache()

    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "naive_baseline_cache.json"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                self._cache = data.get("answers", {})
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to load naive baseline cache, starting fresh")
                self._cache = {}

    def _save_cache(self) -> None:
        path = self._cache_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"model": self._model_id, "answers": self._cache}
        path.write_text(json.dumps(data, indent=2))

    def _cache_key(self, question_id: str) -> str:
        # Include token limit so constrained runs get separate baseline answers
        if self._max_result_tokens > 0:
            return f"{question_id}:{self._model_id}:cap{self._max_result_tokens}"
        return f"{question_id}:{self._model_id}"

    def _episodes_up_to(self, checkpoint: int) -> list[Episode]:
        """Return episodes up to the checkpoint (by sequence number in episode_id)."""
        result: list[Episode] = []
        for ep in self._episodes:
            # Extract episode number from id like "scope_01_ep_005"
            parts = ep.episode_id.rsplit("_", 1)
            try:
                ep_num = int(parts[-1])
            except (ValueError, IndexError):
                # Can't parse number — include it (safe default)
                result.append(ep)
                continue
            if ep_num <= checkpoint:
                result.append(ep)
        return result

    def get_answer(self, question: Question) -> str:
        """Get or generate the naive baseline answer for a question."""
        key = self._cache_key(question.question_id)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        episodes = self._episodes_up_to(question.checkpoint_after)
        if not episodes:
            answer = "(No episodes available for naive baseline)"
            with self._cache_lock:
                self._cache[key] = answer
                self._save_cache()
            return answer

        system, user = build_naive_prompt(
            episodes, question.prompt, max_tokens=self._max_result_tokens
        )

        try:
            answer = self._llm_fn(system, user)
        except Exception as e:
            logger.error("Naive baseline LLM call failed for %s: %s", question.question_id, e)
            answer = "(Naive baseline generation failed)"

        with self._cache_lock:
            self._cache[key] = answer
            self._save_cache()
        return answer

    def generate_all(self, questions: list[Question]) -> dict[str, str]:
        """Pre-generate baseline answers for all questions, optionally in parallel.

        Returns a dict mapping question_id to baseline answer text.
        """
        # Filter to only questions that aren't already cached
        uncached = []
        results: dict[str, str] = {}
        for q in questions:
            key = self._cache_key(q.question_id)
            with self._cache_lock:
                if key in self._cache:
                    results[q.question_id] = self._cache[key]
                    continue
            uncached.append(q)

        if not uncached:
            logger.info("All %d baseline answers served from cache", len(questions))
            return results

        workers = min(self._max_workers, len(uncached))
        if workers > 1:
            logger.info(
                "Generating %d naive baseline answers in parallel (workers=%d)",
                len(uncached), workers,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_q = {
                    pool.submit(self.get_answer, q): q for q in uncached
                }
                for future in concurrent.futures.as_completed(future_to_q):
                    q = future_to_q[future]
                    try:
                        results[q.question_id] = future.result()
                    except Exception as e:
                        logger.error(
                            "Baseline generation failed for %s: %s", q.question_id, e
                        )
                        results[q.question_id] = "(Naive baseline generation failed)"
        else:
            for q in uncached:
                results[q.question_id] = self.get_answer(q)

        return results
