from __future__ import annotations

import json
from pathlib import Path

from lens.core.errors import LensError
from lens.core.models import RunResult, ScoreCard


def load_run_manifest(run_dir: str | Path) -> dict:
    """Load the run manifest from a run output directory."""
    path = Path(run_dir) / "run_manifest.json"
    if not path.exists():
        raise LensError(f"Run manifest not found: {path}")
    return json.loads(path.read_text())


def load_run_result(run_dir: str | Path) -> RunResult:
    """Reconstruct a RunResult from saved artifacts."""
    manifest = load_run_manifest(run_dir)
    run_dir = Path(run_dir)

    from lens.core.models import CheckpointResult, PersonaResult, QuestionResult

    personas_dir = run_dir / "personas"
    persona_results: list[PersonaResult] = []

    if personas_dir.exists():
        for persona_path in sorted(personas_dir.iterdir()):
            if not persona_path.is_dir():
                continue

            persona_id = persona_path.name
            checkpoints: list[CheckpointResult] = []

            for cp_path in sorted(persona_path.iterdir()):
                if not cp_path.is_dir() or not cp_path.name.startswith("checkpoint_"):
                    continue

                checkpoint_num = int(cp_path.name.split("_", 1)[1])

                # Load question results
                qr_file = cp_path / "question_results.json"
                question_results = []
                if qr_file.exists():
                    question_results = [
                        QuestionResult.from_dict(qr)
                        for qr in json.loads(qr_file.read_text())
                    ]

                # Load validation errors
                validation_errors: list[str] = []
                val_file = cp_path / "validation.json"
                if val_file.exists():
                    validation_errors = json.loads(val_file.read_text())

                checkpoints.append(CheckpointResult(
                    persona_id=persona_id,
                    checkpoint=checkpoint_num,
                    question_results=question_results,
                    validation_errors=validation_errors,
                ))

            persona_results.append(PersonaResult(
                persona_id=persona_id,
                checkpoints=checkpoints,
            ))

    return RunResult(
        run_id=manifest["run_id"],
        adapter=manifest["adapter"],
        dataset_version=manifest.get("dataset_version", "unknown"),
        budget_preset=manifest.get("budget_preset", "standard"),
        personas=persona_results,
    )


def load_scorecard(run_dir: str | Path) -> ScoreCard | None:
    """Load scorecard from a run directory, if it exists."""
    path = Path(run_dir) / "scores" / "scorecard.json"
    if not path.exists():
        return None
    return ScoreCard.from_dict(json.loads(path.read_text()))
