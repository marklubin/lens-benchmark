"""Custom synix validators for the LENS datagen pipeline.

Validators:
  - WordCount: checks episode word counts
  - ContaminationCheck: LLM-based single-episode answering test
  - NaiveBaseline: LLM-based full-context baseline
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from synix.build.validators import BaseValidator, Violation
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact

import scoring
import prompt_utils


# ---------------------------------------------------------------------------
# WordCount validator
# ---------------------------------------------------------------------------


class WordCount(BaseValidator):
    """Check that signal and distractor episodes meet minimum word count."""

    def __init__(self, layers, min_words=350, severity="error"):
        super().__init__()
        self.layers = layers
        self.min_words = min_words
        self.severity = severity

    def validate(self, artifacts: list[Artifact], ctx) -> list[Violation]:
        violations: list[Violation] = []
        for a in artifacts:
            if a.artifact_type not in ("signal_episode", "distractor_episode"):
                continue
            wc = scoring.compute_word_count(a.content)
            if wc < self.min_words:
                violations.append(Violation(
                    violation_type="word_count",
                    severity=self.severity,
                    message=f"Episode {a.label} has {wc} words (minimum: {self.min_words})",
                    label=a.label,
                    field="content",
                    metadata={"word_count": wc, "min_words": self.min_words},
                ))

        return violations

    def to_config_dict(self):
        return {
            "layers": [l.name for l in self.layers],
            "min_words": self.min_words,
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# ContaminationCheck validator
# ---------------------------------------------------------------------------


# Question types that require cross-episode reasoning (checked for contamination)
SYNTHESIS_QUESTION_TYPES = {
    "longitudinal", "negative", "temporal", "counterfactual", "paraphrase",
    "distractor_resistance", "severity_assessment", "evidence_sufficiency",
}


class ContaminationCheck(BaseValidator):
    """LLM-based check: can synthesis questions be answered from single episodes?

    For each synthesis question (longitudinal, negative, temporal, counterfactual,
    paraphrase), tests each signal episode individually. If max single-episode
    fact coverage > 50%, the question is contaminated.

    Side effect: writes detailed results to build_dir/contamination_results.json.
    """

    def __init__(self, layers, llm_config=None, max_single_ep_coverage=0.5):
        super().__init__()
        self.layers = layers
        self.llm_config = llm_config or {}
        self.max_single_ep_coverage = max_single_ep_coverage

    def validate(self, artifacts: list[Artifact], ctx) -> list[Violation]:
        merged_config = {"llm_config": self.llm_config}
        client = _get_llm_client(merged_config)

        # Gather artifacts by type
        question_artifacts = [a for a in artifacts if a.artifact_type == "question"]
        signal_artifacts = sorted(
            [a for a in artifacts if a.artifact_type == "signal_episode"],
            key=lambda a: a.label,
        )

        # Filter to synthesis questions (require cross-episode reasoning)
        longitudinal_qs = []
        for qa in question_artifacts:
            q_data = json.loads(qa.content)
            if q_data.get("question_type") in SYNTHESIS_QUESTION_TYPES:
                longitudinal_qs.append((qa, q_data))

        if not longitudinal_qs:
            return []

        violations: list[Violation] = []
        all_results: list[dict] = []
        any_contaminated = False

        for qa, q_data in longitudinal_qs:
            q_id = q_data["question_id"]
            q_prompt = q_data["prompt"]
            key_facts = q_data.get("ground_truth", {}).get("key_facts", [])
            checkpoint = q_data.get("checkpoint_after", len(signal_artifacts))

            # Skip questions with no key facts — can't measure fact coverage
            if not key_facts:
                all_results.append({
                    "question_id": q_id,
                    "max_single_episode_coverage": None,
                    "worst_episode": None,
                    "contaminated": False,
                    "skipped": "no key_facts in ground_truth",
                    "episode_scores": [],
                })
                continue

            # Get episodes up to checkpoint (sorted by label which is the episode ID)
            relevant_episodes = [
                a for a in signal_artifacts
                if _episode_index_from_label(a.label) <= checkpoint
            ]

            max_single_coverage = 0.0
            worst_episode = None
            episode_scores: list[dict] = []

            for ep in relevant_episodes:
                prompt = prompt_utils.build_contamination_prompt(ep.content, q_prompt)
                response = _logged_complete(
                    client, merged_config,
                    messages=[{"role": "user", "content": prompt}],
                    artifact_desc=f"contamination-{q_id}-{ep.label}",
                )

                cov = scoring.compute_fact_coverage(response.content, key_facts)
                episode_scores.append({
                    "episode_id": ep.label,
                    "coverage": round(cov, 3),
                    "answer": response.content,
                })
                if cov > max_single_coverage:
                    max_single_coverage = cov
                    worst_episode = ep.label

            contaminated = max_single_coverage > self.max_single_ep_coverage
            if contaminated:
                any_contaminated = True
                violations.append(Violation(
                    violation_type="contamination",
                    severity="error",
                    message=(
                        f"Question {q_id} is contaminated: max single-episode "
                        f"coverage {max_single_coverage:.1%} (episode {worst_episode})"
                    ),
                    label=qa.label,
                    field="content",
                    metadata={
                        "question_id": q_id,
                        "max_single_episode_coverage": round(max_single_coverage, 3),
                        "worst_episode": worst_episode,
                    },
                ))

            all_results.append({
                "question_id": q_id,
                "max_single_episode_coverage": round(max_single_coverage, 3),
                "worst_episode": worst_episode,
                "contaminated": contaminated,
                "episode_scores": episode_scores,
            })

        # Write detailed results as side effect
        build_dir = _get_build_dir(ctx)
        if build_dir:
            results_path = Path(build_dir) / "contamination_results.json"
            results_path.write_text(json.dumps({
                "summary": "fail" if any_contaminated else "pass",
                "questions": all_results,
            }, indent=2))

        return violations

    def to_config_dict(self):
        return {
            "layers": [l.name for l in self.layers],
            "llm_config": self.llm_config,
        }


# ---------------------------------------------------------------------------
# NaiveBaseline validator
# ---------------------------------------------------------------------------


class NaiveBaseline(BaseValidator):
    """LLM-based naive baseline: all episodes in context, no memory system.

    Uses LLM-as-judge to score whether the baseline answer demonstrates
    knowledge of each key fact (semantic matching, not word-overlap).

    Three-tier thresholds (applied per question-type average):
      - fail_threshold (default 0.50): error — benchmark is too easy
      - warn_threshold (default 0.30): warning — borderline easy
      - floor_threshold (default 0.05): warning — signal may be missing

    Side effect: writes detailed results to build_dir/baseline_results.json.
    """

    def __init__(self, layers, llm_config=None, fail_threshold=0.50,
                 warn_threshold=0.30, floor_threshold=0.05):
        super().__init__()
        self.layers = layers
        self.llm_config = llm_config or {}
        self.fail_threshold = fail_threshold
        self.warn_threshold = warn_threshold
        self.floor_threshold = floor_threshold

    def validate(self, artifacts: list[Artifact], ctx) -> list[Violation]:
        merged_config = {"llm_config": self.llm_config}
        client = _get_llm_client(merged_config)

        # Gather artifacts
        question_artifacts = [a for a in artifacts if a.artifact_type == "question"]
        signal_artifacts = sorted(
            [a for a in artifacts if a.artifact_type == "signal_episode"],
            key=lambda a: a.label,
        )
        distractor_artifacts = sorted(
            [a for a in artifacts if a.artifact_type == "distractor_episode"],
            key=lambda a: a.label,
        )

        # Merge and sort all episodes by timestamp metadata
        all_episodes = signal_artifacts + distractor_artifacts
        all_episodes.sort(key=lambda a: a.metadata.get("timestamp", a.label))

        question_results: list[dict] = []

        for qa in question_artifacts:
            q_data = json.loads(qa.content)
            q_id = q_data["question_id"]
            q_prompt = q_data["prompt"]
            key_facts = q_data.get("ground_truth", {}).get("key_facts", [])
            checkpoint = q_data.get("checkpoint_after", len(signal_artifacts))

            # Skip questions with no key facts — can't measure fact coverage
            if not key_facts:
                question_results.append({
                    "question_id": q_id,
                    "question_type": q_data.get("question_type", ""),
                    "fact_coverage": None,
                    "answer": None,
                    "per_fact_matches": [],
                    "skipped": "no key_facts in ground_truth",
                })
                continue

            # Episodes up to checkpoint
            signal_texts = []
            distractor_texts = []
            for ep in all_episodes:
                idx = _episode_index_from_label(ep.label)
                if ep.artifact_type == "distractor_episode":
                    distractor_texts.append(ep.content)
                elif idx <= checkpoint:
                    signal_texts.append(ep.content)

            # Estimate tokens and sample distractors if needed to fit context
            # ~2.0 tokens per word for structured data with special chars
            max_tokens = 100_000  # conservative headroom below 128k
            signal_tokens = sum(len(t.split()) for t in signal_texts) * 2.0
            available = max_tokens - signal_tokens - 2000  # prompt overhead
            if available > 0 and distractor_texts:
                distractor_tokens = sum(len(t.split()) for t in distractor_texts) * 2.0
                if distractor_tokens > available:
                    # Sample distractors to fit within budget
                    max_distractors = max(1, int(len(distractor_texts) * available / distractor_tokens))
                    rng = random.Random(42)
                    distractor_texts = rng.sample(distractor_texts, min(max_distractors, len(distractor_texts)))

            episode_texts = signal_texts + distractor_texts

            prompt = prompt_utils.build_naive_baseline_prompt(episode_texts, q_prompt)
            response = _logged_complete(
                client, merged_config,
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"baseline-{q_id}",
            )

            cov, per_fact = scoring.compute_fact_coverage_llm(
                response.content, key_facts, q_prompt, client, merged_config,
            )

            question_results.append({
                "question_id": q_id,
                "question_type": q_data.get("question_type", ""),
                "fact_coverage": round(cov, 3),
                "answer": response.content,
                "per_fact_matches": per_fact,
            })

        # Compute averages by question type (skip questions with no key facts)
        by_type: dict[str, list[float]] = {}
        for r in question_results:
            if r["fact_coverage"] is None:
                continue
            qt = r["question_type"]
            by_type.setdefault(qt, []).append(r["fact_coverage"])

        summary_stats: dict[str, float] = {}
        violations: list[Violation] = []

        for qt, coverages in by_type.items():
            avg = sum(coverages) / len(coverages) if coverages else 0.0
            summary_stats[qt] = round(avg, 3)
            if avg > self.fail_threshold:
                violations.append(Violation(
                    violation_type="naive_baseline_too_easy",
                    severity="error",
                    message=(
                        f"Question type '{qt}' averages {avg:.1%} fact coverage "
                        f"on naive baseline (threshold: {self.fail_threshold:.0%}). "
                        f"Benchmark is too easy."
                    ),
                    label="naive_baseline",
                    field="fact_coverage",
                    metadata={"question_type": qt, "avg_coverage": round(avg, 3)},
                ))
            elif avg > self.warn_threshold:
                violations.append(Violation(
                    violation_type="naive_baseline_too_easy",
                    severity="warning",
                    message=(
                        f"Question type '{qt}' averages {avg:.1%} fact coverage "
                        f"on naive baseline (warn threshold: {self.warn_threshold:.0%}). "
                        f"Benchmark may be too easy."
                    ),
                    label="naive_baseline",
                    field="fact_coverage",
                    metadata={"question_type": qt, "avg_coverage": round(avg, 3)},
                ))
            elif avg < self.floor_threshold:
                violations.append(Violation(
                    violation_type="naive_baseline_too_hard",
                    severity="warning",
                    message=(
                        f"Question type '{qt}' averages {avg:.1%} fact coverage "
                        f"on naive baseline (floor: {self.floor_threshold:.0%}). "
                        f"Signal may be missing or key facts poorly calibrated."
                    ),
                    label="naive_baseline",
                    field="fact_coverage",
                    metadata={"question_type": qt, "avg_coverage": round(avg, 3)},
                ))

        # Write detailed results as side effect
        build_dir = _get_build_dir(ctx)
        if build_dir:
            results_path = Path(build_dir) / "baseline_results.json"
            results_path.write_text(json.dumps({
                "summary": summary_stats,
                "questions": question_results,
            }, indent=2))

        return violations

    def to_config_dict(self):
        return {
            "layers": [l.name for l in self.layers],
            "llm_config": self.llm_config,
            "fail_threshold": self.fail_threshold,
            "warn_threshold": self.warn_threshold,
            "floor_threshold": self.floor_threshold,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _episode_index_from_label(label: str) -> int:
    """Extract the numeric episode index from a label like 'scope_ep_009'.

    For distractor labels, returns a very large number so they sort last.
    """
    parts = label.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return 999999


def _get_build_dir(ctx) -> str | None:
    """Extract build_dir from validation context."""
    if ctx is None:
        return None
    pipeline = getattr(ctx, "pipeline", None)
    if pipeline is not None:
        return getattr(pipeline, "build_dir", None)
    return None
