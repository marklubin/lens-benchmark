#!/usr/bin/env python3
"""Inject pre-generated outlines into synix layer1-outline envelope format.

Reads signal_outline.json and distractor_outlines/*.json from the scope root
and wraps them in the synix artifact envelope, writing to generated/layer1-outline/.

Usage:
    python scripts/inject_outlines.py datasets/scopes/02_financial_irregularity
    python scripts/inject_outlines.py --all   # process scopes 02-06
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(data: str) -> str:
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()


def _wrap_signal(content_dict: dict, spec_path: Path) -> dict:
    content_str = json.dumps(content_dict, separators=(",", ":"))
    spec_str = spec_path.read_text()
    return {
        "label": "signal_outline",
        "artifact_type": "signal_outline",
        "artifact_id": _sha256(content_str),
        "input_ids": [_sha256(spec_str)],
        "prompt_id": None,
        "model_config": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "content": content_str,
        "metadata": {
            "episode_count": len(content_dict.get("episodes", [])),
            "source": "pre-generated",
            "layer_name": "outline",
            "layer_level": 1,
        },
    }


def _wrap_distractor(
    content_dict: dict, theme: str, theme_index: int, spec_path: Path
) -> dict:
    content_str = json.dumps(content_dict, separators=(",", ":"))
    spec_str = spec_path.read_text()
    return {
        "label": f"distractor_outline_{theme}",
        "artifact_type": "distractor_outline",
        "artifact_id": _sha256(content_str),
        "input_ids": [_sha256(spec_str)],
        "prompt_id": None,
        "model_config": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "content": content_str,
        "metadata": {
            "theme": theme,
            "theme_index": theme_index,
            "episode_count": len(content_dict.get("episodes", [])),
            "source": "pre-generated",
            "layer_name": "outline",
            "layer_level": 1,
        },
    }


def inject_scope(scope_dir: Path) -> None:
    spec_path = scope_dir / "spec.yaml"
    outline_dir = scope_dir / "generated" / "layer1-outline"
    outline_dir.mkdir(parents=True, exist_ok=True)

    # Signal outline
    signal_src = scope_dir / "signal_outline.json"
    if not signal_src.exists():
        print(f"  SKIP signal_outline.json (not found)")
        return
    with open(signal_src) as f:
        signal_data = json.load(f)
    envelope = _wrap_signal(signal_data, spec_path)
    dest = outline_dir / "signal_outline.json"
    with open(dest, "w") as f:
        json.dump(envelope, f, indent=2)
    ep_count = len(signal_data.get("episodes", []))
    print(f"  signal_outline.json -> {ep_count} episodes")

    # Distractor outlines
    distractor_dir = scope_dir / "distractor_outlines"
    if not distractor_dir.exists():
        print(f"  SKIP distractor_outlines/ (not found)")
        return
    for idx, src_file in enumerate(sorted(distractor_dir.glob("*.json"))):
        theme = src_file.stem
        with open(src_file) as f:
            dist_data = json.load(f)
        envelope = _wrap_distractor(dist_data, theme, idx, spec_path)
        dest = outline_dir / f"distractor_outline_{theme}.json"
        with open(dest, "w") as f:
            json.dump(envelope, f, indent=2)
        ep_count = len(dist_data.get("episodes", []))
        print(f"  distractor_outline_{theme}.json -> {ep_count} episodes")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inject_outlines.py <scope_dir> | --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        base = Path("datasets/scopes")
        scopes = sorted(
            d
            for d in base.iterdir()
            if d.is_dir() and d.name.startswith(("02", "03", "04", "05", "06"))
        )
    else:
        scopes = [Path(sys.argv[1])]

    for scope_dir in scopes:
        print(f"\n=== {scope_dir.name} ===")
        inject_scope(scope_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
