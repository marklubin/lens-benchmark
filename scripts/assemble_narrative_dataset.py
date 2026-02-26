"""Assemble narrative episodes into LENS dataset JSON files.

After generating episodes via Claude Code agents (or any other method),
this script reads the raw episode text files from the generated/ directory
and produces:
  - datasets/scope_XX_only.json (signal episodes only)
  - datasets/scope_XX_with_distractors.json (signal + distractor episodes interleaved)

Expected directory structure:
  datasets/scopes/07_tutoring_jailbreak/
    spec.yaml
    generated/
      signal_outline.json          # Planner output (fact sheets)
      distractor_outline_{theme}.json
      episodes/
        signal_001.txt ... signal_020.txt
        distractor_{theme}_001.txt ... distractor_{theme}_007.txt

Usage:
    python scripts/assemble_narrative_dataset.py datasets/scopes/07_tutoring_jailbreak
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lens" / "datagen" / "synix"))
import spec_utils


def assemble_dataset(scope_dir: str) -> None:
    scope_path = Path(scope_dir)
    spec = spec_utils.load_spec(scope_path / "spec.yaml")
    scope_id = spec["scope_id"]
    ep_count = spec["episodes"]["count"]

    gen_dir = scope_path / "generated"
    ep_dir = gen_dir / "episodes"

    if not ep_dir.exists():
        print(f"ERROR: {ep_dir} does not exist. Generate episodes first.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Signal episodes
    # -----------------------------------------------------------------
    signal_episodes: list[dict] = []
    for i in range(1, ep_count + 1):
        txt_path = ep_dir / f"signal_{i:03d}.txt"
        if not txt_path.exists():
            print(f"WARNING: Missing signal episode {txt_path}")
            continue

        text = txt_path.read_text().strip()
        eid = spec_utils.make_episode_id(scope_id, i)
        ts = spec_utils.make_episode_timestamp(spec, i)
        phase = spec_utils.get_phase_for_episode(spec, i)
        phase_id = phase["id"] if phase else "unknown"
        density = phase["signal_density"] if phase else "none"

        signal_episodes.append({
            "episode_id": eid,
            "scope_id": scope_id,
            "timestamp": ts,
            "text": text,
            "meta": {
                "episode_type": "signal",
                "phase": phase_id,
                "signal_density": density,
            },
        })

    print(f"Loaded {len(signal_episodes)} signal episodes")

    # -----------------------------------------------------------------
    # Distractor episodes
    # -----------------------------------------------------------------
    distractor_episodes: list[dict] = []
    dc = spec.get("distractors")
    if dc and dc["count"] > 0:
        themes = dc["themes"]
        per_theme = [dc["count"] // len(themes)] * len(themes)
        for i in range(dc["count"] % len(themes)):
            per_theme[i] += 1

        distractor_idx = 0
        for tidx, theme in enumerate(themes):
            for j in range(1, per_theme[tidx] + 1):
                distractor_idx += 1
                txt_path = ep_dir / f"distractor_{theme['id']}_{j:03d}.txt"
                if not txt_path.exists():
                    print(f"WARNING: Missing distractor {txt_path}")
                    continue

                text = txt_path.read_text().strip()
                # Distractor timestamps interleave with signal
                # Spread them evenly across the timeline
                frac = distractor_idx / max(dc["count"], 1)
                fake_global_idx = max(1, int(frac * ep_count))
                ts = spec_utils.make_episode_timestamp(spec, fake_global_idx)
                # Offset by 6 hours so distractors sort after same-day signals
                ts = ts.replace("T10:00:00", "T16:00:00")

                label = f"{scope_id}_dx_{theme['id']}_{j:03d}"
                distractor_episodes.append({
                    "episode_id": label,
                    "scope_id": scope_id,
                    "timestamp": ts,
                    "text": text,
                    "meta": {
                        "episode_type": "distractor",
                        "theme": theme["id"],
                    },
                })

    print(f"Loaded {len(distractor_episodes)} distractor episodes")

    # -----------------------------------------------------------------
    # Write _only.json
    # -----------------------------------------------------------------
    datasets_dir = Path("datasets")
    scope_num = scope_id.split("_")[-1]  # e.g. "07" from "tutoring_jailbreak_07"

    only_data = {
        "version": "0.1.0",
        "scopes": [{
            "scope_id": scope_id,
            "episodes": sorted(signal_episodes, key=lambda e: e["timestamp"]),
        }],
    }
    only_path = datasets_dir / f"scope_{scope_num}_only.json"
    only_path.write_text(json.dumps(only_data, indent=2))
    print(f"Wrote {only_path} ({len(signal_episodes)} episodes)")

    # -----------------------------------------------------------------
    # Write _with_distractors.json
    # -----------------------------------------------------------------
    all_episodes = signal_episodes + distractor_episodes
    all_episodes.sort(key=lambda e: e["timestamp"])

    with_dist_data = {
        "version": "0.1.0",
        "scopes": [{
            "scope_id": scope_id,
            "episodes": all_episodes,
        }],
    }
    dist_path = datasets_dir / f"scope_{scope_num}_with_distractors.json"
    dist_path.write_text(json.dumps(with_dist_data, indent=2))
    print(f"Wrote {dist_path} ({len(all_episodes)} episodes)")

    # -----------------------------------------------------------------
    # Resolve questions
    # -----------------------------------------------------------------
    questions: list[dict] = []
    kf_map = {kf["id"]: kf["fact"] for kf in spec["key_facts"]}

    for q in spec["questions"]:
        evidence_refs = []
        for ref in q["ground_truth"]["evidence"]:
            try:
                evidence_refs.append(spec_utils.resolve_phase_ref(ref, spec))
            except ValueError:
                pass

        key_fact_texts = [
            kf_map[fid] for fid in q["ground_truth"]["key_facts"] if fid in kf_map
        ]

        questions.append({
            "question_id": q["id"],
            "scope_id": scope_id,
            "checkpoint_after": q["checkpoint_after"],
            "question_type": q["type"],
            "prompt": q["prompt"],
            "ground_truth": {
                "canonical_answer": q["ground_truth"]["canonical_answer"],
                "required_evidence_refs": evidence_refs,
                "key_facts": key_fact_texts,
            },
        })

    questions_path = gen_dir / "questions.json"
    questions_path.write_text(json.dumps(questions, indent=2))
    print(f"Wrote {questions_path} ({len(questions)} questions)")

    print(f"\nDone. Checkpoints: {spec_utils.get_checkpoints(spec)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/assemble_narrative_dataset.py <scope_dir>")
        sys.exit(1)

    for scope_dir in sys.argv[1:]:
        print(f"\n=== Assembling {scope_dir} ===")
        assemble_dataset(scope_dir)
