"""Extract a single fact sheet from an outline and build its renderer prompt.

Usage:
    python scripts/render_episode.py <outline_json> <episode_index> [--spec <spec_yaml>]

Prints the renderer prompt to stdout. The caller (a Task agent) can use this
to get the prompt, then write the rendered episode to the appropriate file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lens" / "datagen" / "synix"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import spec_utils
from narrative_prompts import build_narrative_renderer_prompt


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/render_episode.py <outline_json> <episode_index> [--spec <spec_yaml>]")
        sys.exit(1)

    outline_path = sys.argv[1]
    episode_index = int(sys.argv[2])

    # Find spec by walking up from outline
    outline = Path(outline_path)
    if "--spec" in sys.argv:
        spec_path = sys.argv[sys.argv.index("--spec") + 1]
    else:
        # Assume outline is in generated/, spec is one level up
        spec_path = outline.parent.parent / "spec.yaml"

    spec = spec_utils.load_spec(spec_path)
    data = json.loads(outline.read_text())

    # Find the episode by index
    fact_sheet = None
    for ep in data["episodes"]:
        if ep["index"] == episode_index:
            fact_sheet = ep
            break

    if fact_sheet is None:
        print(f"ERROR: Episode index {episode_index} not found in {outline_path}", file=sys.stderr)
        sys.exit(1)

    prompt = build_narrative_renderer_prompt(spec, fact_sheet)
    print(prompt)


if __name__ == "__main__":
    main()
