from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from lens.core.errors import DatasetError


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TimelineConfig:
    start: str  # ISO date string e.g. "2024-01-15"
    interval: str = "1d"  # e.g. "1d", "7d", "14d"

    def start_date(self) -> date:
        return date.fromisoformat(self.start)

    def interval_days(self) -> int:
        m = re.match(r"^(\d+)d$", self.interval)
        if not m:
            raise DatasetError(f"Invalid interval format: {self.interval!r}, expected e.g. '1d', '7d'")
        return int(m.group(1))


@dataclass
class EpisodeConfig:
    count: int
    timeline: TimelineConfig
    format: str = ""
    target_words: int = 500


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    seed: int = 42


@dataclass
class ScenarioConfig:
    setting: str = ""
    voice: str = ""


@dataclass
class PhaseArc:
    id: str
    episodes: str  # e.g. "1-8"
    description: str = ""
    signal_density: str = "none"  # none, low, medium, high

    def episode_range(self) -> tuple[int, int]:
        """Return (start, end) 1-based inclusive indices."""
        parts = self.episodes.split("-")
        if len(parts) != 2:
            raise DatasetError(f"Invalid episode range: {self.episodes!r}, expected 'N-M'")
        return int(parts[0]), int(parts[1])


@dataclass
class NoiseConfig:
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class DistractorTheme:
    id: str
    scenario: str
    excluded_terms: list[str] = field(default_factory=list)


@dataclass
class DistractorConfig:
    count: int = 0
    target_words: int = 0  # 0 = use episodes.target_words
    themes: list[DistractorTheme] = field(default_factory=list)
    seed: int = 99
    max_similarity: float = 0.3


@dataclass
class KeyFact:
    id: str
    fact: str
    first_appears: str  # phase_relative ref e.g. "early_signal:1"
    reinforced_in: list[str] = field(default_factory=list)


@dataclass
class QuestionGroundTruth:
    canonical_answer: str
    key_facts: list[str] = field(default_factory=list)  # key_fact IDs
    evidence: list[str] = field(default_factory=list)  # phase-relative refs


@dataclass
class QuestionSpec:
    id: str
    checkpoint_after: int
    type: str  # "longitudinal", "null_hypothesis", "action_recommendation"
    prompt: str
    ground_truth: QuestionGroundTruth


@dataclass
class ScopeSpec:
    scope_id: str
    domain: str = ""
    description: str = ""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    episodes: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(
        count=30, timeline=TimelineConfig(start="2024-01-01")
    ))
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    arc: list[PhaseArc] = field(default_factory=list)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    distractors: DistractorConfig | None = None
    key_facts: list[KeyFact] = field(default_factory=list)
    questions: list[QuestionSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_spec(path: str | Path) -> ScopeSpec:
    """Load a ScopeSpec from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise DatasetError(
            "PyYAML is required for spec loading. Install with: pip install 'lens-bench[datagen]'"
        )

    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Spec file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise DatasetError(f"Spec must be a YAML mapping, got {type(raw).__name__}")

    return _parse_spec(raw)


def _parse_spec(raw: dict) -> ScopeSpec:
    """Parse a raw YAML dict into a ScopeSpec."""
    if "scope_id" not in raw:
        raise DatasetError("Spec must have a 'scope_id' field")

    # Generation config
    gen_raw = raw.get("generation", {})
    generation = GenerationConfig(
        temperature=gen_raw.get("temperature", 0.7),
        seed=gen_raw.get("seed", 42),
    )

    # Timeline & episodes
    ep_raw = raw.get("episodes", {})
    tl_raw = ep_raw.get("timeline", {})
    timeline = TimelineConfig(
        start=str(tl_raw.get("start", "2024-01-01")),
        interval=tl_raw.get("interval", "1d"),
    )
    episodes_cfg = EpisodeConfig(
        count=ep_raw.get("count", 30),
        timeline=timeline,
        format=ep_raw.get("format", ""),
        target_words=ep_raw.get("target_words", 150),
    )

    # Scenario
    scn_raw = raw.get("scenario", {})
    scenario = ScenarioConfig(
        setting=scn_raw.get("setting", ""),
        voice=scn_raw.get("voice", ""),
    )

    # Arc phases
    arc = []
    for phase_raw in raw.get("arc", []):
        arc.append(PhaseArc(
            id=phase_raw["id"],
            episodes=str(phase_raw["episodes"]),
            description=phase_raw.get("description", ""),
            signal_density=phase_raw.get("signal_density", "none"),
        ))

    # Noise
    noise_raw = raw.get("noise", {})
    noise = NoiseConfig(
        description=noise_raw.get("description", ""),
        examples=noise_raw.get("examples", []),
    )

    # Distractors (optional)
    distractors = None
    dist_raw = raw.get("distractors")
    if dist_raw and isinstance(dist_raw, dict):
        themes = []
        for t_raw in dist_raw.get("themes", []):
            themes.append(DistractorTheme(
                id=t_raw["id"],
                scenario=t_raw.get("scenario", ""),
                excluded_terms=t_raw.get("excluded_terms", []),
            ))
        distractors = DistractorConfig(
            count=dist_raw.get("count", 0),
            target_words=dist_raw.get("target_words", 0),
            themes=themes,
            seed=dist_raw.get("seed", 99),
            max_similarity=dist_raw.get("max_similarity", 0.3),
        )

    # Key facts
    key_facts = []
    for kf_raw in raw.get("key_facts", []):
        key_facts.append(KeyFact(
            id=kf_raw["id"],
            fact=kf_raw["fact"],
            first_appears=kf_raw.get("first_appears", ""),
            reinforced_in=kf_raw.get("reinforced_in", []),
        ))

    # Questions
    questions = []
    for q_raw in raw.get("questions", []):
        gt_raw = q_raw.get("ground_truth", {})
        gt = QuestionGroundTruth(
            canonical_answer=gt_raw.get("canonical_answer", ""),
            key_facts=gt_raw.get("key_facts", []),
            evidence=gt_raw.get("evidence", []),
        )
        questions.append(QuestionSpec(
            id=q_raw["id"],
            checkpoint_after=q_raw["checkpoint_after"],
            type=q_raw["type"],
            prompt=q_raw["prompt"],
            ground_truth=gt,
        ))

    return ScopeSpec(
        scope_id=raw["scope_id"],
        domain=raw.get("domain", ""),
        description=raw.get("description", ""),
        generation=generation,
        episodes=episodes_cfg,
        scenario=scenario,
        arc=arc,
        noise=noise,
        distractors=distractors,
        key_facts=key_facts,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


VALID_QUESTION_TYPES = {"longitudinal", "null_hypothesis", "action_recommendation"}
VALID_SIGNAL_DENSITIES = {"none", "low", "medium", "high"}


def validate_spec(spec: ScopeSpec) -> list[str]:
    """Validate a ScopeSpec. Returns list of error messages (empty = valid)."""
    errors: list[str] = []

    if not spec.scope_id:
        errors.append("scope_id is required")

    if spec.episodes.count < 1:
        errors.append("episodes.count must be >= 1")

    # Validate timeline
    try:
        spec.episodes.timeline.start_date()
    except ValueError:
        errors.append(f"Invalid timeline start date: {spec.episodes.timeline.start!r}")

    try:
        spec.episodes.timeline.interval_days()
    except DatasetError as e:
        errors.append(str(e))

    # Validate arc phases cover all episodes
    if spec.arc:
        covered = set()
        phase_ids = set()
        for phase in spec.arc:
            if phase.id in phase_ids:
                errors.append(f"Duplicate phase id: {phase.id!r}")
            phase_ids.add(phase.id)

            if phase.signal_density not in VALID_SIGNAL_DENSITIES:
                errors.append(
                    f"Phase {phase.id!r}: invalid signal_density {phase.signal_density!r}"
                )

            try:
                start, end = phase.episode_range()
                if start < 1 or end > spec.episodes.count:
                    errors.append(
                        f"Phase {phase.id!r}: episode range {phase.episodes} "
                        f"exceeds episode count {spec.episodes.count}"
                    )
                for i in range(start, end + 1):
                    covered.add(i)
            except DatasetError as e:
                errors.append(f"Phase {phase.id!r}: {e}")

        # Check all episodes are covered (warn, not error — phases can overlap)
        expected = set(range(1, spec.episodes.count + 1))
        uncovered = expected - covered
        if uncovered:
            errors.append(
                f"Episodes not covered by any arc phase: {sorted(uncovered)}"
            )

    # Validate key facts
    kf_ids = {kf.id for kf in spec.key_facts}
    phase_ids = {p.id for p in spec.arc}
    for kf in spec.key_facts:
        if kf.first_appears:
            _validate_phase_ref(kf.first_appears, phase_ids, spec, f"key_fact {kf.id!r}.first_appears", errors)
        for ref in kf.reinforced_in:
            _validate_phase_ref(ref, phase_ids, spec, f"key_fact {kf.id!r}.reinforced_in", errors)

    # Validate questions
    for q in spec.questions:
        if q.type not in VALID_QUESTION_TYPES:
            errors.append(f"Question {q.id!r}: invalid type {q.type!r}")
        if q.checkpoint_after < 1 or q.checkpoint_after > spec.episodes.count:
            errors.append(
                f"Question {q.id!r}: checkpoint_after {q.checkpoint_after} "
                f"out of range [1, {spec.episodes.count}]"
            )
        for fact_id in q.ground_truth.key_facts:
            if fact_id not in kf_ids:
                errors.append(f"Question {q.id!r}: references unknown key_fact {fact_id!r}")
        for ref in q.ground_truth.evidence:
            _validate_phase_ref(ref, phase_ids, spec, f"question {q.id!r}.evidence", errors)

    # Validate distractors
    if spec.distractors is not None:
        dc = spec.distractors
        if dc.count < 0:
            errors.append("distractors.count must be >= 0")
        if dc.max_similarity < 0.0 or dc.max_similarity > 1.0:
            errors.append("distractors.max_similarity must be between 0.0 and 1.0")
        if dc.count > 0 and not dc.themes:
            errors.append("distractors.themes required when distractors.count > 0")
        theme_ids = set()
        for theme in dc.themes:
            if not theme.id:
                errors.append("distractor theme must have an 'id'")
            if theme.id in theme_ids:
                errors.append(f"Duplicate distractor theme id: {theme.id!r}")
            theme_ids.add(theme.id)
            if not theme.scenario.strip():
                errors.append(f"Distractor theme {theme.id!r}: scenario is required")

    return errors


def _validate_phase_ref(
    ref: str, phase_ids: set[str], spec: ScopeSpec, context: str, errors: list[str]
) -> None:
    """Validate a phase-relative reference like 'early_signal:1'."""
    parts = ref.split(":")
    if len(parts) != 2:
        errors.append(f"{context}: invalid phase ref {ref!r}, expected 'phase_id:local_index'")
        return
    phase_id, local_idx_str = parts
    if phase_id not in phase_ids:
        errors.append(f"{context}: unknown phase {phase_id!r} in ref {ref!r}")
        return
    try:
        local_idx = int(local_idx_str)
    except ValueError:
        errors.append(f"{context}: non-integer index in ref {ref!r}")
        return
    # Check local index is within phase bounds
    for phase in spec.arc:
        if phase.id == phase_id:
            start, end = phase.episode_range()
            phase_size = end - start + 1
            if local_idx < 1 or local_idx > phase_size:
                errors.append(
                    f"{context}: local index {local_idx} out of range "
                    f"[1, {phase_size}] for phase {phase_id!r}"
                )
            break


def validate_spec_or_raise(spec: ScopeSpec) -> None:
    """Validate spec and raise DatasetError on any issues."""
    errors = validate_spec(spec)
    if errors:
        msg = f"Spec validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {e}" for e in errors[:20]
        )
        raise DatasetError(msg)


# ---------------------------------------------------------------------------
# Phase-relative reference resolution
# ---------------------------------------------------------------------------


def resolve_phase_ref(ref: str, spec: ScopeSpec) -> str:
    """Resolve a phase-relative ref like 'early_signal:1' to a global episode ID.

    Returns episode ID like '{scope_id}_ep_009'.
    """
    parts = ref.split(":")
    if len(parts) != 2:
        raise DatasetError(f"Invalid phase ref: {ref!r}")

    phase_id, local_idx_str = parts
    local_idx = int(local_idx_str)

    for phase in spec.arc:
        if phase.id == phase_id:
            start, _end = phase.episode_range()
            global_idx = start + local_idx - 1
            return make_episode_id(spec.scope_id, global_idx)

    raise DatasetError(f"Unknown phase: {phase_id!r}")


def make_episode_id(scope_id: str, global_index: int) -> str:
    """Build deterministic episode ID: '{scope_id}_ep_{N:03d}'."""
    return f"{scope_id}_ep_{global_index:03d}"


def make_episode_timestamp(spec: ScopeSpec, global_index: int) -> str:
    """Build episode timestamp from spec timeline. Index is 1-based."""
    start = spec.episodes.timeline.start_date()
    days = spec.episodes.timeline.interval_days()
    dt = start + timedelta(days=days * (global_index - 1))
    return f"{dt.isoformat()}T10:00:00"


def get_phase_for_episode(spec: ScopeSpec, global_index: int) -> PhaseArc | None:
    """Find which phase a global episode index belongs to."""
    for phase in spec.arc:
        start, end = phase.episode_range()
        if start <= global_index <= end:
            return phase
    return None


def get_key_facts_for_phase(spec: ScopeSpec, phase: PhaseArc) -> list[tuple[int, KeyFact]]:
    """Get key facts that should appear in a phase, with their local episode index.

    Returns list of (local_index, KeyFact) tuples.
    """
    results: list[tuple[int, KeyFact]] = []
    for kf in spec.key_facts:
        # Check first_appears
        if kf.first_appears:
            pid, local_str = kf.first_appears.split(":")
            if pid == phase.id:
                results.append((int(local_str), kf))
                continue
        # Check reinforced_in
        for ref in kf.reinforced_in:
            pid, local_str = ref.split(":")
            if pid == phase.id:
                results.append((int(local_str), kf))
                break  # one entry per key_fact per phase
    return results


def compute_spec_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of the spec file for reproducibility tracking."""
    content = Path(path).read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def get_checkpoints(spec: ScopeSpec) -> list[int]:
    """Extract unique sorted checkpoint values from questions."""
    return sorted({q.checkpoint_after for q in spec.questions})
