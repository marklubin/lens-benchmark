#!/usr/bin/env python3
"""Compile per-scope synix build outputs into the unified benchmark dataset JSON.

Reads episodes.json and questions.json from each scope's generated/ directory
and produces a single datasets/benchmark_dataset.json that passes validate_or_raise().

Usage:
    uv run python scripts/compile_dataset.py
    uv run python scripts/compile_dataset.py --scopes-dir datasets/scopes --output datasets/benchmark_dataset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lens.datasets.schema import validate_or_raise


def _load_layer3_questions(layer3_dir: Path) -> list[dict]:
    """Load questions from individual layer3-questions/*.json files.

    Each file is a synix artifact with a 'content' field containing the
    question JSON as a string.
    """
    questions = []
    for qf in sorted(layer3_dir.glob("*.json")):
        artifact = json.loads(qf.read_text())
        content = artifact.get("content", "")
        if isinstance(content, str):
            try:
                q = json.loads(content)
            except json.JSONDecodeError:
                print(f"  WARNING: could not parse content in {qf.name}", file=sys.stderr)
                continue
        else:
            q = content
        questions.append(q)
    return questions


def compile_dataset(scopes_dir: Path, output: Path) -> dict:
    """Compile all scope artifacts into a unified dataset."""
    scope_dirs = sorted(scopes_dir.glob("*/generated"))
    if not scope_dirs:
        print(f"ERROR: No generated/ directories found under {scopes_dir}", file=sys.stderr)
        sys.exit(1)

    scopes: list[dict] = []
    all_questions: list[dict] = []

    for gen_dir in scope_dirs:
        scope_name = gen_dir.parent.name
        episodes_path = gen_dir / "episodes.json"
        questions_path = gen_dir / "questions.json"

        if not episodes_path.exists():
            print(f"WARNING: skipping {scope_name} — no episodes.json", file=sys.stderr)
            continue

        episodes = json.loads(episodes_path.read_text())
        if not episodes:
            print(f"WARNING: skipping {scope_name} — episodes.json is empty", file=sys.stderr)
            continue

        # Derive scope_id from the first episode
        scope_id = episodes[0]["scope_id"]

        scopes.append({
            "scope_id": scope_id,
            "episodes": episodes,
        })
        print(f"  {scope_name}: {len(episodes)} episodes (scope_id={scope_id})")

        # Prefer layer3-questions/ (has all questions) over questions.json (may be subset)
        layer3_dir = gen_dir / "layer3-questions"
        if layer3_dir.is_dir() and list(layer3_dir.glob("*.json")):
            questions = _load_layer3_questions(layer3_dir)
            all_questions.extend(questions)
            print(f"  {scope_name}: {len(questions)} questions (from layer3-questions/)")
        elif questions_path.exists():
            questions = json.loads(questions_path.read_text())
            all_questions.extend(questions)
            print(f"  {scope_name}: {len(questions)} questions (from questions.json)")

    dataset = {
        "version": "0.1.0",
        "scopes": scopes,
        "questions": all_questions,
    }

    # Validate before writing
    validate_or_raise(dataset)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2))

    total_eps = sum(len(s["episodes"]) for s in scopes)
    print(f"\nCompiled {len(scopes)} scopes, {total_eps} episodes, {len(all_questions)} questions")
    print(f"Written to {output}")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile benchmark dataset from scope artifacts")
    parser.add_argument(
        "--scopes-dir", type=Path, default=Path("datasets/scopes"),
        help="Directory containing scope subdirectories (default: datasets/scopes)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/benchmark_dataset.json"),
        help="Output path (default: datasets/benchmark_dataset.json)",
    )
    args = parser.parse_args()
    compile_dataset(args.scopes_dir, args.output)


if __name__ == "__main__":
    main()
