from __future__ import annotations

from lens.core.errors import DatasetError


# Required top-level keys in a dataset JSON file
REQUIRED_KEYS = {"version", "scopes"}

# Required keys per scope
SCOPE_REQUIRED_KEYS = {"scope_id", "episodes"}

# Required keys per episode
EPISODE_REQUIRED_KEYS = {"episode_id", "scope_id", "timestamp", "text"}

# Required keys per question
QUESTION_REQUIRED_KEYS = {
    "question_id", "scope_id", "checkpoint_after", "question_type", "prompt", "ground_truth",
}

# Required keys in ground_truth
GROUND_TRUTH_REQUIRED_KEYS = {"canonical_answer", "required_evidence_refs", "key_facts"}


def validate_dataset(data: dict) -> list[str]:
    """Validate a dataset dictionary against the expected schema.

    Returns a list of error messages. Empty list means valid.
    """
    errors: list[str] = []

    # Top-level
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing required top-level key: {key!r}")

    if "scopes" not in data:
        return errors

    if not isinstance(data["scopes"], list):
        errors.append("'scopes' must be a list")
        return errors

    seen_episode_ids: set[str] = set()

    for i, scope in enumerate(data["scopes"]):
        prefix = f"scopes[{i}]"

        if not isinstance(scope, dict):
            errors.append(f"{prefix}: must be a dict")
            continue

        for key in SCOPE_REQUIRED_KEYS:
            if key not in scope:
                errors.append(f"{prefix}: missing required key {key!r}")

        if "episodes" not in scope:
            continue

        if not isinstance(scope["episodes"], list):
            errors.append(f"{prefix}.episodes: must be a list")
            continue

        for j, episode in enumerate(scope["episodes"]):
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

    # Validate questions if present
    if "questions" in data:
        if not isinstance(data["questions"], list):
            errors.append("'questions' must be a list")
        else:
            valid_question_types = {"longitudinal", "null_hypothesis", "action_recommendation"}
            seen_question_ids: set[str] = set()
            for i, q in enumerate(data["questions"]):
                q_prefix = f"questions[{i}]"
                if not isinstance(q, dict):
                    errors.append(f"{q_prefix}: must be a dict")
                    continue

                for key in QUESTION_REQUIRED_KEYS:
                    if key not in q:
                        errors.append(f"{q_prefix}: missing required key {key!r}")

                qid = q.get("question_id")
                if qid and qid in seen_question_ids:
                    errors.append(f"{q_prefix}: duplicate question_id {qid!r}")
                if qid:
                    seen_question_ids.add(qid)

                qtype = q.get("question_type")
                if qtype and qtype not in valid_question_types:
                    errors.append(
                        f"{q_prefix}: invalid question_type {qtype!r}, "
                        f"must be one of {valid_question_types}"
                    )

                gt = q.get("ground_truth")
                if gt is not None:
                    if not isinstance(gt, dict):
                        errors.append(f"{q_prefix}.ground_truth: must be a dict")
                    else:
                        for key in GROUND_TRUTH_REQUIRED_KEYS:
                            if key not in gt:
                                errors.append(
                                    f"{q_prefix}.ground_truth: missing required key {key!r}"
                                )

    return errors


def validate_or_raise(data: dict) -> None:
    """Validate dataset and raise DatasetError on any issues."""
    errors = validate_dataset(data)
    if errors:
        msg = f"Dataset validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {e}" for e in errors[:20]
        )
        raise DatasetError(msg)
