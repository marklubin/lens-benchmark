"""Custom synix validators for the LENS datagen pipeline.

Validators:
  - WordCount: checks episode word counts
  - ContaminationCheck: LLM-based single-episode answering test
  - NaiveBaseline: LLM-based full-context baseline
"""
from __future__ import annotations

import json
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


class ContaminationCheck(BaseValidator):
    """LLM-based check: can longitudinal questions be answered from single episodes?

    For each longitudinal question, tests each signal episode individually.
    If max single-episode fact coverage > 50%, the question is contaminated.

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

        # Filter to longitudinal questions only
        longitudinal_qs = []
        for qa in question_artifacts:
            q_data = json.loads(qa.content)
            if q_data.get("question_type") == "longitudinal":
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

    Warns if any question type averages > threshold (default 95%) fact coverage,
    indicating the benchmark may be too easy.

    Side effect: writes detailed results to build_dir/baseline_results.json.
    """

    def __init__(self, layers, llm_config=None, warn_threshold=0.95):
        super().__init__()
        self.layers = layers
        self.llm_config = llm_config or {}
        self.warn_threshold = warn_threshold

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

            # Episodes up to checkpoint
            episode_texts = []
            for ep in all_episodes:
                idx = _episode_index_from_label(ep.label)
                # Include all episodes whose index is within checkpoint
                # For distractors, include all (they interleave)
                if ep.artifact_type == "distractor_episode" or idx <= checkpoint:
                    episode_texts.append(ep.content)

            prompt = prompt_utils.build_naive_baseline_prompt(episode_texts, q_prompt)
            response = _logged_complete(
                client, merged_config,
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"baseline-{q_id}",
            )

            cov = scoring.compute_fact_coverage(response.content, key_facts)
            per_fact = scoring.compute_per_fact_matches(response.content, key_facts)

            question_results.append({
                "question_id": q_id,
                "question_type": q_data.get("question_type", ""),
                "fact_coverage": round(cov, 3),
                "answer": response.content,
                "per_fact_matches": per_fact,
            })

        # Compute averages by question type
        by_type: dict[str, list[float]] = {}
        for r in question_results:
            qt = r["question_type"]
            by_type.setdefault(qt, []).append(r["fact_coverage"])

        summary_stats: dict[str, float] = {}
        violations: list[Violation] = []

        for qt, coverages in by_type.items():
            avg = sum(coverages) / len(coverages) if coverages else 0.0
            summary_stats[qt] = round(avg, 3)
            if avg > self.warn_threshold:
                violations.append(Violation(
                    violation_type="naive_baseline_too_easy",
                    severity="warning",
                    message=(
                        f"Question type '{qt}' averages {avg:.1%} fact coverage "
                        f"on naive baseline (threshold: {self.warn_threshold:.0%}). "
                        f"Benchmark may be too easy."
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
            "warn_threshold": self.warn_threshold,
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
