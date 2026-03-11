"""Dataset loader for V1 LENS benchmark scopes.

Reads spec.yaml and generated/ artifacts from the V1 dataset layout
and produces structured data for the V2 bank builder.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EpisodeData:
    """A single episode (signal or distractor) from a scope."""

    episode_id: str
    filename: str
    content: str
    ordinal: int  # 1-based position within its category
    is_distractor: bool


@dataclass
class QuestionData:
    """A benchmark question with ground truth."""

    question_id: str
    scope_id: str
    checkpoint_after: int
    question_type: str
    prompt: str
    ground_truth: dict  # canonical_answer, key_facts, required_evidence_refs


@dataclass
class ScopeData:
    """All data for a single benchmark scope."""

    scope_id: str
    scope_dir: Path
    episodes: list[EpisodeData]
    questions: list[QuestionData]
    checkpoints: list[int]  # sorted unique checkpoint_after values
    spec: dict
    dataset_hash: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_hash:
            self.dataset_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Content-address the dataset for reproducibility checks."""
        h = hashlib.sha256()
        for ep in sorted(self.episodes, key=lambda e: e.episode_id):
            h.update(ep.episode_id.encode())
            h.update(ep.content.encode())
        return h.hexdigest()[:16]

    def episodes_up_to(self, checkpoint: int) -> list[EpisodeData]:
        """Return all episodes valid at a given checkpoint.

        Signal episodes: ordinal <= checkpoint.
        Distractors: proportional — include first N distractors where N = checkpoint
        (keeps signal-to-noise ratio roughly constant).
        """
        signal = [e for e in self.episodes if not e.is_distractor and e.ordinal <= checkpoint]
        distractors = sorted(
            [e for e in self.episodes if e.is_distractor],
            key=lambda e: (e.ordinal, e.episode_id),
        )
        # Include proportional distractors
        n_distractors = min(checkpoint, len(distractors))
        return sorted(signal + distractors[:n_distractors], key=lambda e: (e.ordinal, e.episode_id))

    def questions_at(self, checkpoint: int) -> list[QuestionData]:
        """Return questions whose checkpoint_after matches exactly."""
        return [q for q in self.questions if q.checkpoint_after == checkpoint]


def load_scope(scope_dir: str | Path) -> ScopeData:
    """Load a scope from the V1 dataset layout.

    Expected layout:
        scope_dir/
            spec.yaml
            generated/
                episodes/
                    signal_001.txt
                    signal_002.txt
                    distractor_theme_001.txt
                    ...
                questions.json
    """
    scope_dir = Path(scope_dir)
    spec = _load_spec(scope_dir / "spec.yaml")
    scope_id = spec.get("scope_id", scope_dir.name)
    episodes = _load_episodes(scope_dir / "generated" / "episodes", scope_id)
    questions = _load_questions(scope_dir / "generated" / "questions.json")
    checkpoints = sorted(set(q.checkpoint_after for q in questions))

    return ScopeData(
        scope_id=scope_id,
        scope_dir=scope_dir,
        episodes=episodes,
        questions=questions,
        checkpoints=checkpoints,
        spec=spec,
    )


def _load_spec(spec_path: Path) -> dict:
    """Load and return spec.yaml as a dict."""
    with open(spec_path) as f:
        return yaml.safe_load(f)


def _load_episodes(episodes_dir: Path, scope_id: str) -> list[EpisodeData]:
    """Load all episode text files from the generated/episodes/ directory."""
    episodes: list[EpisodeData] = []

    for path in sorted(episodes_dir.iterdir()):
        if not path.suffix == ".txt":
            continue

        content = path.read_text()
        filename = path.name
        is_distractor = filename.startswith("distractor_")

        # Extract ordinal from filename
        m = re.search(r"(\d+)\.txt$", filename)
        ordinal = int(m.group(1)) if m else 0

        # Build episode_id
        if is_distractor:
            episode_id = f"{scope_id}_{filename[:-4]}"  # strip .txt
        else:
            episode_id = f"{scope_id}_ep_{ordinal:03d}"

        episodes.append(
            EpisodeData(
                episode_id=episode_id,
                filename=filename,
                content=content,
                ordinal=ordinal,
                is_distractor=is_distractor,
            )
        )

    return episodes


def _load_questions(questions_path: Path) -> list[QuestionData]:
    """Load questions.json and return structured QuestionData objects."""
    with open(questions_path) as f:
        raw = json.load(f)

    questions: list[QuestionData] = []
    for item in raw:
        questions.append(
            QuestionData(
                question_id=item["question_id"],
                scope_id=item["scope_id"],
                checkpoint_after=item["checkpoint_after"],
                question_type=item["question_type"],
                prompt=item["prompt"],
                ground_truth=item["ground_truth"],
            )
        )

    return questions
