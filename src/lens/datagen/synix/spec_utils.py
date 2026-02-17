"""Standalone spec YAML parsing for the synix pipeline.

This module duplicates the spec parsing logic from lens.datagen.spec so that
the synix pipeline files remain self-contained (no lens.* imports).  It works
with raw dicts and YAML, producing plain dict / list structures rather than
dataclasses.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading & parsing
# ---------------------------------------------------------------------------


def load_spec(path: str | Path) -> dict:
    """Load a scope spec from a YAML file and return a parsed dict."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise ValueError(f"Spec file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Spec must be a YAML mapping, got {type(raw).__name__}")

    return parse_spec(raw)


def parse_spec(raw: dict) -> dict:
    """Parse a raw YAML dict into a normalised spec dict."""
    if "scope_id" not in raw:
        raise ValueError("Spec must have a 'scope_id' field")

    gen_raw = raw.get("generation", {})
    ep_raw = raw.get("episodes", {})
    tl_raw = ep_raw.get("timeline", {})
    scn_raw = raw.get("scenario", {})
    noise_raw = raw.get("noise", {})

    spec: dict = {
        "scope_id": raw["scope_id"],
        "domain": raw.get("domain", ""),
        "description": raw.get("description", ""),
        "generation": {
            "temperature": gen_raw.get("temperature", 0.7),
            "seed": gen_raw.get("seed", 42),
        },
        "episodes": {
            "count": ep_raw.get("count", 30),
            "timeline": {
                "start": str(tl_raw.get("start", "2024-01-01")),
                "interval": tl_raw.get("interval", "1d"),
            },
            "format": ep_raw.get("format", ""),
            "target_words": ep_raw.get("target_words", 150),
        },
        "scenario": {
            "setting": scn_raw.get("setting", ""),
            "voice": scn_raw.get("voice", ""),
        },
        "arc": [],
        "noise": {
            "description": noise_raw.get("description", ""),
            "examples": noise_raw.get("examples", []),
        },
        "distractors": None,
        "key_facts": [],
        "questions": [],
    }

    # Arc phases
    for p in raw.get("arc", []):
        spec["arc"].append({
            "id": p["id"],
            "episodes": str(p["episodes"]),
            "description": p.get("description", ""),
            "signal_density": p.get("signal_density", "none"),
        })

    # Distractors
    dist_raw = raw.get("distractors")
    if dist_raw and isinstance(dist_raw, dict):
        themes = []
        for t in dist_raw.get("themes", []):
            themes.append({
                "id": t["id"],
                "scenario": t.get("scenario", ""),
                "excluded_terms": t.get("excluded_terms", []),
            })
        spec["distractors"] = {
            "count": dist_raw.get("count", 0),
            "target_words": dist_raw.get("target_words", 0),
            "themes": themes,
            "seed": dist_raw.get("seed", 99),
            "max_similarity": dist_raw.get("max_similarity", 0.3),
        }

    # Key facts
    for kf in raw.get("key_facts", []):
        spec["key_facts"].append({
            "id": kf["id"],
            "fact": kf["fact"],
            "first_appears": kf.get("first_appears", ""),
            "reinforced_in": kf.get("reinforced_in", []),
        })

    # Questions
    for q in raw.get("questions", []):
        gt = q.get("ground_truth", {})
        q_entry: dict = {
            "id": q["id"],
            "checkpoint_after": q["checkpoint_after"],
            "type": q["type"],
            "prompt": q["prompt"],
            "ground_truth": {
                "canonical_answer": gt.get("canonical_answer", ""),
                "key_facts": gt.get("key_facts", []),
                "evidence": gt.get("evidence", []),
            },
        }
        if "variant_of" in q:
            q_entry["variant_of"] = q["variant_of"]
        if "expected_answer_polarity" in q:
            q_entry["expected_answer_polarity"] = q["expected_answer_polarity"]
        spec["questions"].append(q_entry)

    return spec


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALID_QUESTION_TYPES = {
    "longitudinal",
    "null_hypothesis",
    "action_recommendation",
    "negative",
    "paraphrase",
    "temporal",
    "counterfactual",
    "distractor_resistance",
    "severity_assessment",
    "evidence_sufficiency",
}
VALID_SIGNAL_DENSITIES = {"none", "low", "medium", "high"}


def validate_spec(spec: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []

    if not spec.get("scope_id"):
        errors.append("scope_id is required")

    ep_count = spec["episodes"]["count"]
    if ep_count < 1:
        errors.append("episodes.count must be >= 1")

    # Timeline
    try:
        _start_date(spec)
    except ValueError:
        errors.append(f"Invalid timeline start date: {spec['episodes']['timeline']['start']!r}")

    try:
        _interval_days(spec)
    except ValueError as e:
        errors.append(str(e))

    # Arc phases
    if spec["arc"]:
        covered: set[int] = set()
        phase_ids: set[str] = set()
        for phase in spec["arc"]:
            if phase["id"] in phase_ids:
                errors.append(f"Duplicate phase id: {phase['id']!r}")
            phase_ids.add(phase["id"])

            if phase["signal_density"] not in VALID_SIGNAL_DENSITIES:
                errors.append(f"Phase {phase['id']!r}: invalid signal_density {phase['signal_density']!r}")

            try:
                start, end = episode_range(phase)
                if start < 1 or end > ep_count:
                    errors.append(
                        f"Phase {phase['id']!r}: episode range {phase['episodes']} "
                        f"exceeds episode count {ep_count}"
                    )
                for i in range(start, end + 1):
                    covered.add(i)
            except ValueError as e:
                errors.append(f"Phase {phase['id']!r}: {e}")

        expected = set(range(1, ep_count + 1))
        uncovered = expected - covered
        if uncovered:
            errors.append(f"Episodes not covered by any arc phase: {sorted(uncovered)}")

    # Key facts
    kf_ids = {kf["id"] for kf in spec["key_facts"]}
    phase_ids = {p["id"] for p in spec["arc"]}
    for kf in spec["key_facts"]:
        if kf["first_appears"]:
            _validate_phase_ref(kf["first_appears"], phase_ids, spec, f"key_fact {kf['id']!r}.first_appears", errors)
        for ref in kf["reinforced_in"]:
            _validate_phase_ref(ref, phase_ids, spec, f"key_fact {kf['id']!r}.reinforced_in", errors)

    # Questions
    for q in spec["questions"]:
        if q["type"] not in VALID_QUESTION_TYPES:
            errors.append(f"Question {q['id']!r}: invalid type {q['type']!r}")
        if q["checkpoint_after"] < 1 or q["checkpoint_after"] > ep_count:
            errors.append(
                f"Question {q['id']!r}: checkpoint_after {q['checkpoint_after']} "
                f"out of range [1, {ep_count}]"
            )
        for fid in q["ground_truth"]["key_facts"]:
            if fid not in kf_ids:
                errors.append(f"Question {q['id']!r}: references unknown key_fact {fid!r}")
        for ref in q["ground_truth"]["evidence"]:
            _validate_phase_ref(ref, phase_ids, spec, f"question {q['id']!r}.evidence", errors)

    # Distractors
    dc = spec.get("distractors")
    if dc is not None:
        if dc["count"] < 0:
            errors.append("distractors.count must be >= 0")
        if dc["max_similarity"] < 0.0 or dc["max_similarity"] > 1.0:
            errors.append("distractors.max_similarity must be between 0.0 and 1.0")
        if dc["count"] > 0 and not dc["themes"]:
            errors.append("distractors.themes required when distractors.count > 0")
        theme_ids: set[str] = set()
        for theme in dc["themes"]:
            if not theme["id"]:
                errors.append("distractor theme must have an 'id'")
            if theme["id"] in theme_ids:
                errors.append(f"Duplicate distractor theme id: {theme['id']!r}")
            theme_ids.add(theme["id"])
            if not theme["scenario"].strip():
                errors.append(f"Distractor theme {theme['id']!r}: scenario is required")

    return errors


def validate_spec_or_raise(spec: dict) -> None:
    errors = validate_spec(spec)
    if errors:
        msg = f"Spec validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {e}" for e in errors[:20]
        )
        raise ValueError(msg)


def _validate_phase_ref(
    ref: str, phase_ids: set[str], spec: dict, context: str, errors: list[str]
) -> None:
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
    for phase in spec["arc"]:
        if phase["id"] == phase_id:
            start, end = episode_range(phase)
            phase_size = end - start + 1
            if local_idx < 1 or local_idx > phase_size:
                errors.append(
                    f"{context}: local index {local_idx} out of range "
                    f"[1, {phase_size}] for phase {phase_id!r}"
                )
            break


# ---------------------------------------------------------------------------
# Phase-relative reference helpers
# ---------------------------------------------------------------------------


def episode_range(phase: dict) -> tuple[int, int]:
    """Return (start, end) 1-based inclusive indices for a phase."""
    parts = phase["episodes"].split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid episode range: {phase['episodes']!r}, expected 'N-M'")
    return int(parts[0]), int(parts[1])


def resolve_phase_ref(ref: str, spec: dict) -> str:
    """Resolve 'phase_id:local_index' to a global episode ID."""
    parts = ref.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid phase ref: {ref!r}")
    phase_id, local_idx_str = parts
    local_idx = int(local_idx_str)
    for phase in spec["arc"]:
        if phase["id"] == phase_id:
            start, _end = episode_range(phase)
            global_idx = start + local_idx - 1
            return make_episode_id(spec["scope_id"], global_idx)
    raise ValueError(f"Unknown phase: {phase_id!r}")


def make_episode_id(scope_id: str, global_index: int) -> str:
    return f"{scope_id}_ep_{global_index:03d}"


def make_episode_timestamp(spec: dict, global_index: int) -> str:
    start = _start_date(spec)
    days = _interval_days(spec)
    dt = start + timedelta(days=days * (global_index - 1))
    return f"{dt.isoformat()}T10:00:00"


def get_phase_for_episode(spec: dict, global_index: int) -> dict | None:
    for phase in spec["arc"]:
        start, end = episode_range(phase)
        if start <= global_index <= end:
            return phase
    return None


def get_key_facts_for_phase(spec: dict, phase: dict) -> list[tuple[int, dict]]:
    """Return (local_index, key_fact_dict) pairs for facts in this phase."""
    results: list[tuple[int, dict]] = []
    for kf in spec["key_facts"]:
        if kf["first_appears"]:
            pid, local_str = kf["first_appears"].split(":")
            if pid == phase["id"]:
                results.append((int(local_str), kf))
                continue
        for ref in kf["reinforced_in"]:
            pid, local_str = ref.split(":")
            if pid == phase["id"]:
                results.append((int(local_str), kf))
                break
    return results


def compute_spec_hash(path: str | Path) -> str:
    content = Path(path).read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def get_checkpoints(spec: dict) -> list[int]:
    return sorted({q["checkpoint_after"] for q in spec["questions"]})


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------


def _start_date(spec: dict) -> date:
    return date.fromisoformat(spec["episodes"]["timeline"]["start"])


def _interval_days(spec: dict) -> int:
    interval = spec["episodes"]["timeline"]["interval"]
    m = re.match(r"^(\d+)d$", interval)
    if not m:
        raise ValueError(f"Invalid interval format: {interval!r}, expected e.g. '1d', '7d'")
    return int(m.group(1))
