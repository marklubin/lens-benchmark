from __future__ import annotations

import json
from pathlib import Path

from lens.core.errors import DatasetError, atomic_write
from lens.datasets.schema import validate_or_raise


def compile_dataset(
    scope_dirs: list[str | Path],
    version: str,
    output_path: str | Path,
) -> Path:
    """Compile multiple generated scopes into a single dataset JSON.

    Args:
        scope_dirs: Paths to scope directories (each containing generated/).
        version: Dataset version string.
        output_path: Where to write the compiled dataset JSON.

    Returns:
        Path to the written dataset file.
    """
    output_path = Path(output_path)
    scopes: list[dict] = []
    all_questions: list[dict] = []
    seen_episode_ids: set[str] = set()
    seen_question_ids: set[str] = set()

    for scope_dir in scope_dirs:
        scope_dir = Path(scope_dir)
        gen_dir = scope_dir / "generated"

        episodes_path = gen_dir / "episodes.json"
        questions_path = gen_dir / "questions.json"

        if not episodes_path.exists():
            raise DatasetError(f"No generated episodes found in {scope_dir}")

        # Load episodes
        episodes = json.loads(episodes_path.read_text())
        if not isinstance(episodes, list):
            raise DatasetError(f"episodes.json must be a list in {scope_dir}")

        # Load and merge distractor episodes if present
        distractors_path = gen_dir / "distractors.json"
        if distractors_path.exists():
            distractors = json.loads(distractors_path.read_text())
            if isinstance(distractors, list):
                episodes = episodes + distractors
                # Sort merged episodes by timestamp for interleaved ordering
                episodes.sort(key=lambda ep: ep.get("timestamp", ""))

        # Check for duplicate episode IDs
        for ep in episodes:
            eid = ep.get("episode_id", "")
            if eid in seen_episode_ids:
                raise DatasetError(f"Duplicate episode_id across scopes: {eid!r}")
            seen_episode_ids.add(eid)

        # Determine scope_id from episodes
        scope_ids = {ep.get("scope_id", "") for ep in episodes}
        if len(scope_ids) != 1:
            raise DatasetError(
                f"Expected exactly one scope_id in {scope_dir}, got {scope_ids}"
            )
        scope_id = scope_ids.pop()

        # Load manifest for domain metadata
        # Prefer release_manifest.json (synix pipeline) over manifest.json (legacy)
        manifest_path = gen_dir / "release_manifest.json"
        if not manifest_path.exists():
            manifest_path = gen_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        # Build scope entry
        scope_entry: dict = {
            "scope_id": scope_id,
            "episodes": episodes,
        }

        # Try to get domain from spec.yaml
        spec_path = scope_dir / "spec.yaml"
        if spec_path.exists():
            try:
                from lens.datagen.spec import load_spec
                spec = load_spec(spec_path)
                if spec.domain:
                    scope_entry["domain"] = spec.domain
            except Exception:
                pass  # Non-critical metadata

        scopes.append(scope_entry)

        # Load questions if available
        if questions_path.exists():
            questions = json.loads(questions_path.read_text())
            if isinstance(questions, list):
                for q in questions:
                    qid = q.get("question_id", "")
                    if qid in seen_question_ids:
                        raise DatasetError(
                            f"Duplicate question_id across scopes: {qid!r}"
                        )
                    seen_question_ids.add(qid)
                all_questions.extend(questions)

    if not scopes:
        raise DatasetError("No scopes to compile")

    # Assemble dataset
    dataset: dict = {
        "version": version,
        "scopes": scopes,
        "questions": all_questions,
    }

    # Validate against existing schema
    validate_or_raise(dataset)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(output_path) as tmp:
        tmp.write_text(json.dumps(dataset, indent=2))

    return output_path
