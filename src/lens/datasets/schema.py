from __future__ import annotations

from lens.core.errors import DatasetError


# Required top-level keys in a dataset JSON file
REQUIRED_KEYS = {"version", "personas"}

# Required keys per persona
PERSONA_REQUIRED_KEYS = {"persona_id", "episodes"}

# Required keys per episode
EPISODE_REQUIRED_KEYS = {"episode_id", "persona_id", "timestamp", "text"}


def validate_dataset(data: dict) -> list[str]:
    """Validate a dataset dictionary against the expected schema.

    Returns a list of error messages. Empty list means valid.
    """
    errors: list[str] = []

    # Top-level
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing required top-level key: {key!r}")

    if "personas" not in data:
        return errors

    if not isinstance(data["personas"], list):
        errors.append("'personas' must be a list")
        return errors

    seen_episode_ids: set[str] = set()

    for i, persona in enumerate(data["personas"]):
        prefix = f"personas[{i}]"

        if not isinstance(persona, dict):
            errors.append(f"{prefix}: must be a dict")
            continue

        for key in PERSONA_REQUIRED_KEYS:
            if key not in persona:
                errors.append(f"{prefix}: missing required key {key!r}")

        if "episodes" not in persona:
            continue

        if not isinstance(persona["episodes"], list):
            errors.append(f"{prefix}.episodes: must be a list")
            continue

        for j, episode in enumerate(persona["episodes"]):
            ep_prefix = f"{prefix}.episodes[{j}]"

            if not isinstance(episode, dict):
                errors.append(f"{ep_prefix}: must be a dict")
                continue

            for key in EPISODE_REQUIRED_KEYS:
                if key not in episode:
                    errors.append(f"{ep_prefix}: missing required key {key!r}")

            eid = episode.get("episode_id")
            if eid and eid in seen_episode_ids:
                errors.append(f"{ep_prefix}: duplicate episode_id {eid!r}")
            if eid:
                seen_episode_ids.add(eid)

    # Validate truth_patterns if present
    if "truth_patterns" in data:
        if not isinstance(data["truth_patterns"], list):
            errors.append("'truth_patterns' must be a list")
        else:
            for i, tp in enumerate(data["truth_patterns"]):
                tp_prefix = f"truth_patterns[{i}]"
                for key in ("pattern_id", "persona_id", "canonical_insight"):
                    if key not in tp:
                        errors.append(f"{tp_prefix}: missing required key {key!r}")

    return errors


def validate_or_raise(data: dict) -> None:
    """Validate dataset and raise DatasetError on any issues."""
    errors = validate_dataset(data)
    if errors:
        msg = f"Dataset validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {e}" for e in errors[:20]
        )
        raise DatasetError(msg)
