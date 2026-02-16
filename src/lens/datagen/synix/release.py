"""Release step: reads synix build artifacts + validator results,
produces final output files for backward compatibility with the compiler.

Outputs written to generated/:
  - episodes.json — signal episodes
  - distractors.json — distractor episodes
  - questions.json — resolved questions
  - release_manifest.json — generation manifest with validation results
  - verification.json — full verification results
  - verification_report.html — HTML report
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# When invoked as `lens release`, we're inside the lens package,
# so we can use lens imports here.
from lens.datagen.verify_report import generate_verification_report
from lens.datagen.spec import load_spec


def run_release(scope_dir: str | Path, verbose: bool = False) -> Path:
    """Execute the release step.

    Reads synix build artifacts from generated/ and produces legacy JSON
    files + manifest + HTML report.
    """
    scope_dir = Path(scope_dir)
    gen_dir = scope_dir / "generated"
    spec_path = scope_dir / "spec.yaml"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # Load spec (using the canonical lens spec loader)
    spec = load_spec(spec_path)

    _log(f"Releasing scope: {spec.scope_id}")

    # --- Read synix build artifacts ---
    store = _ArtifactReader(gen_dir)

    signal_episodes = store.list_artifacts("signal_episodes")
    distractor_episodes = store.list_artifacts("distractor_episodes")
    question_artifacts = store.list_artifacts("questions")
    audit_artifacts = store.list_artifacts("key_fact_audit")

    _log(f"  Signal episodes: {len(signal_episodes)}")
    _log(f"  Distractor episodes: {len(distractor_episodes)}")
    _log(f"  Questions: {len(question_artifacts)}")

    # --- Build episodes.json ---
    episodes = []
    for a in sorted(signal_episodes, key=lambda x: x["label"]):
        episodes.append({
            "episode_id": a["metadata"].get("episode_id", a["label"]),
            "scope_id": a["metadata"].get("scope_id", spec.scope_id),
            "timestamp": a["metadata"].get("timestamp", ""),
            "text": a["content"],
            "meta": {
                "episode_type": "signal",
                "phase": a["metadata"].get("phase", ""),
                "signal_density": a["metadata"].get("signal_density", ""),
            },
        })

    # --- Build distractors.json ---
    distractors = []
    # Assign sequential dx_ IDs across all distractor artifacts
    sorted_distractors = sorted(distractor_episodes, key=lambda x: x["label"])
    for i, a in enumerate(sorted_distractors):
        # Compute interleaved timestamp
        start_date = spec.episodes.timeline.start_date()
        interval_days = spec.episodes.timeline.interval_days()
        total_days = interval_days * (spec.episodes.count - 1)
        if len(sorted_distractors) > 1:
            offset_days = (total_days * i) / (len(sorted_distractors) - 1)
        else:
            offset_days = total_days / 2
        dt = start_date + timedelta(days=offset_days)
        timestamp = f"{dt.isoformat()}T10:30:00"

        distractors.append({
            "episode_id": f"{spec.scope_id}_dx_{i + 1:03d}",
            "scope_id": spec.scope_id,
            "timestamp": timestamp,
            "text": a["content"],
            "meta": {
                "episode_type": "distractor",
                "theme": a["metadata"].get("theme", ""),
            },
        })

    # --- Build questions.json ---
    questions = []
    for a in sorted(question_artifacts, key=lambda x: x["label"]):
        q_data = json.loads(a["content"])
        questions.append(q_data)

    # --- Build manifest.json ---
    audit_data = {}
    if audit_artifacts:
        audit_data = json.loads(audit_artifacts[0]["content"])

    # Read validator side-effect files
    contamination_results = _read_json(gen_dir / "contamination_results.json")
    baseline_results = _read_json(gen_dir / "baseline_results.json")

    # Run synix validate --json to get violations (best effort)
    validation_violations = _run_synix_validate_json(scope_dir)

    manifest = {
        "scope_id": spec.scope_id,
        "spec_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "pipeline": "synix/lens-datagen",
        },
        "spec_hash": _read_spec_hash(gen_dir),
        "phases": {},
        "total_tokens": {"prompt": 0, "completion": 0},
        "key_fact_coverage": audit_data.get("key_fact_coverage", {}),
        "validation": {
            "key_fact_hit_rate": audit_data.get("hit_rate", 0.0),
            "contamination_check": (
                contamination_results.get("summary", "pending")
                if contamination_results else "pending"
            ),
            "naive_baseline": (
                baseline_results.get("summary", "pending")
                if baseline_results else "pending"
            ),
            "violations": validation_violations,
        },
        "distractors": {
            "count": len(distractors),
            "themes": list({a["metadata"].get("theme", "") for a in sorted_distractors}),
        },
    }

    # --- Build verification.json ---
    verification = {}
    if contamination_results:
        verification["contamination"] = contamination_results
    if baseline_results:
        verification["naive_baseline"] = baseline_results
    verification["_meta"] = {
        "scope_id": spec.scope_id,
        "domain": spec.domain,
        "episode_count": len(episodes),
        "question_count": len(questions),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }

    # --- Write outputs ---
    gen_dir.mkdir(parents=True, exist_ok=True)

    (gen_dir / "episodes.json").write_text(json.dumps(episodes, indent=2))
    _log(f"  Wrote episodes.json ({len(episodes)} episodes)")

    if distractors:
        (gen_dir / "distractors.json").write_text(json.dumps(distractors, indent=2))
        _log(f"  Wrote distractors.json ({len(distractors)} distractors)")

    (gen_dir / "questions.json").write_text(json.dumps(questions, indent=2))
    _log(f"  Wrote questions.json ({len(questions)} questions)")

    (gen_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    _log("  Wrote release_manifest.json")

    (gen_dir / "verification.json").write_text(json.dumps(verification, indent=2))
    _log("  Wrote verification.json")

    # Generate HTML report
    html = generate_verification_report(verification, spec, manifest)
    (gen_dir / "verification_report.html").write_text(html)
    _log("  Wrote verification_report.html")

    _log(f"Release complete: {gen_dir}")
    return gen_dir


# ---------------------------------------------------------------------------
# Artifact reading
# ---------------------------------------------------------------------------


class _ArtifactReader:
    """Read synix build artifacts from the build directory.

    Synix stores artifacts as JSON files under:
        build_dir/layer{level}-{name}/{label}.json
    """

    def __init__(self, build_dir: Path):
        self.build_dir = build_dir
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load the synix manifest.json if it exists."""
        manifest_path = self.build_dir / "manifest.json"
        if manifest_path.exists():
            try:
                return json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def list_artifacts(self, layer_name: str) -> list[dict]:
        """List all artifacts for a given layer."""
        artifacts = []

        # Search for layer directories matching the pattern
        for d in sorted(self.build_dir.iterdir()):
            if d.is_dir() and d.name.endswith(f"-{layer_name}"):
                for f in sorted(d.iterdir()):
                    if f.suffix == ".json":
                        try:
                            data = json.loads(f.read_text())
                            artifacts.append(data)
                        except (json.JSONDecodeError, OSError):
                            continue

        return artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it doesn't exist or is invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_spec_hash(gen_dir: Path) -> str:
    """Try to extract spec hash from the spec artifact."""
    for d in sorted(gen_dir.iterdir()):
        if d.is_dir() and d.name.endswith("-spec"):
            spec_file = d / "spec.json"
            if spec_file.exists():
                try:
                    data = json.loads(spec_file.read_text())
                    content = json.loads(data.get("content", "{}"))
                    return content.get("_spec_hash", "")
                except (json.JSONDecodeError, OSError):
                    pass
    return ""


def _run_synix_validate_json(scope_dir: Path) -> list[dict]:
    """Run synix validate --json and capture violations (best effort)."""
    try:
        pipeline_path = Path(__file__).parent / "pipeline.py"
        result = subprocess.run(
            [
                sys.executable, "-m", "synix", "validate",
                str(pipeline_path), "--json",
            ],
            env={
                **dict(__import__("os").environ),
                "LENS_SCOPE_DIR": str(scope_dir),
                "LENS_BUILD_DIR": str(scope_dir / "generated"),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            return data.get("violations", [])
    except Exception:
        pass
    return []
