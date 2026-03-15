#!/usr/bin/env python3
"""Extract answers from new scope state DBs, format grading tasks, and merge results.

Usage:
  # Extract grading tasks from new scopes
  uv run python studies/grid/extract_and_merge.py extract \
    --scopes clinical_trial_10 implicit_decision_13 epoch_classification_14 \
    --output results/grading_tasks_expansion.jsonl

  # Merge scored results into grid_summary_full.json
  uv run python studies/grid/extract_and_merge.py merge \
    --score-files results/claude_scores_m3.jsonl results/claude_scores_new.jsonl results/claude_scores_expansion.jsonl \
    --output results/grid_summary_full.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

GRID_DIR = Path(__file__).resolve().parent
WORK_DIR = GRID_DIR / "work"
RESULTS_DIR = GRID_DIR / "results"

# Add src to path
sys.path.insert(0, str(GRID_DIR.parent.parent / "src"))


def extract_grading_tasks(scope_ids: list[str], output: Path, scope_dirs: dict[str, Path] | None = None) -> None:
    """Extract answers from state DBs for given scopes and write grading_tasks JSONL."""
    tasks = []

    for scope_id in scope_ids:
        work = WORK_DIR / scope_id
        state_db = work / "state.db"

        if not state_db.exists():
            # Also check work_new prefix (used in Phase 2)
            for prefix in ["work", "work_new"]:
                alt = GRID_DIR / prefix / scope_id / "state.db"
                if alt.exists():
                    state_db = alt
                    break

        if not state_db.exists():
            print(f"WARN: No state.db for {scope_id} at {work}/state.db", file=sys.stderr)
            continue

        # Load questions for this scope
        scope_dir = None
        if scope_dirs and scope_id in scope_dirs:
            scope_dir = scope_dirs[scope_id]
        else:
            # Try to find the scope dir from datasets
            datasets_dir = GRID_DIR.parent.parent.parent / "datasets" / "scopes"
            for d in datasets_dir.iterdir():
                if d.name.endswith(scope_id.rsplit("_", 1)[-1]) or scope_id.replace("_", "") in d.name.replace("_", ""):
                    scope_dir = d
                    break

        questions = {}
        if scope_dir:
            q_file = scope_dir / "generated" / "questions.json"
            if q_file.exists():
                questions = {q["question_id"]: q for q in json.loads(q_file.read_text())}

        conn = sqlite3.connect(str(state_db))

        # Get all runs
        runs = conn.execute(
            "SELECT run_id, data FROM runs ORDER BY run_id"
        ).fetchall()

        for run_id, run_data_str in runs:
            run_data = json.loads(run_data_str)
            policy_id = run_data.get("policy_id", "unknown")
            replicate_id = "r01"  # Extract from run_id if available
            import re
            m = re.search(r"(r\d+)", run_id)
            if m:
                replicate_id = m.group(1)

            # Get all answers for this run
            answers = conn.execute(
                "SELECT question_id, checkpoint_id, data FROM answers WHERE run_id = ? ORDER BY checkpoint_id, question_id",
                (run_id,),
            ).fetchall()

            for question_id, checkpoint_id, answer_data_str in answers:
                answer_data = json.loads(answer_data_str)
                answer_text = answer_data.get("answer_text", answer_data.get("text", ""))
                cited_refs = answer_data.get("cited_refs", answer_data.get("references", []))

                q = questions.get(question_id, {})
                key_facts = q.get("key_facts", q.get("required_evidence_refs", []))
                question_prompt = q.get("question", q.get("question_prompt", question_id))
                canonical_answer = q.get("canonical_answer", q.get("ground_truth", ""))

                grade_id = f"grade-{scope_id}-{policy_id}-{replicate_id}-{question_id}"

                task = {
                    "grade_id": grade_id,
                    "scope_id": scope_id,
                    "policy_id": policy_id,
                    "question_id": question_id,
                    "checkpoint_id": checkpoint_id,
                    "run_id": run_id,
                    "replicate_id": replicate_id,
                    "question_prompt": question_prompt,
                    "key_facts": key_facts,
                    "canonical_answer": canonical_answer,
                    "answer_text": answer_text,
                    "cited_refs": cited_refs,
                }
                tasks.append(task)

        conn.close()
        print(f"Extracted {len([t for t in tasks if t['scope_id'] == scope_id])} tasks from {scope_id}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Total: {len(tasks)} grading tasks written to {output}")


def merge_scores(score_files: list[Path], output: Path) -> None:
    """Merge multiple score JSONL files into grid_summary_full.json."""
    all_scores = []
    for sf in score_files:
        if not sf.exists():
            print(f"WARN: Score file not found: {sf}", file=sys.stderr)
            continue
        with open(sf) as f:
            file_scores = [json.loads(line) for line in f if line.strip()]
        all_scores.extend(file_scores)
        print(f"Loaded {len(file_scores)} scores from {sf.name}")

    # Aggregate by policy and scope_policy
    policy_scores: dict[str, list[float]] = defaultdict(list)
    scope_policy_scores: dict[str, list[float]] = defaultdict(list)

    seen = set()
    for s in all_scores:
        # Deduplicate by grade_id (include replicate to preserve M>1)
        replicate = s.get("replicate_id")
        if not replicate:
            import re as _re
            _m = _re.search(r"(r\d+)", s.get("run_id", ""))
            replicate = _m.group(1) if _m else "r01"
        gid = s.get("grade_id", f"{s['scope_id']}-{s['policy_id']}-{replicate}-{s['question_id']}")
        if gid in seen:
            continue
        seen.add(gid)

        f1 = s.get("fact_f1", 0.0)
        pid = s["policy_id"]
        sid = s["scope_id"]

        policy_scores[pid].append(f1)
        scope_policy_scores[f"{sid}/{pid}"].append(f1)

    # Build summary
    summary = {
        "total_graded": len(seen),
        "grading_sources": [f"{sf.name} ({len([1 for l in open(sf) if l.strip()])})" for sf in score_files if sf.exists()],
        "policies": {},
        "scope_policy": {},
    }

    for pid in sorted(policy_scores.keys()):
        scores = policy_scores[pid]
        summary["policies"][pid] = {
            "n": len(scores),
            "mean": round(sum(scores) / len(scores), 4) if scores else 0,
        }

    for key in sorted(scope_policy_scores.keys()):
        scores = scope_policy_scores[key]
        summary["scope_policy"][key] = {
            "n": len(scores),
            "mean": round(sum(scores) / len(scores), 4) if scores else 0,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGrid summary: {len(seen)} unique graded answers")
    print(f"Policies: {len(summary['policies'])}")
    print(f"Scope×Policy cells: {len(summary['scope_policy'])}")
    print(f"Written to {output}")

    # Print ranking
    print("\nPolicy Ranking:")
    ranked = sorted(summary["policies"].items(), key=lambda x: x[1]["mean"], reverse=True)
    for rank, (pid, data) in enumerate(ranked, 1):
        print(f"  {rank}. {pid}: {data['mean']:.4f} (n={data['n']})")


def main():
    parser = argparse.ArgumentParser(description="Extract answers and merge grid results")
    sub = parser.add_subparsers(dest="command")

    # Extract subcommand
    ext = sub.add_parser("extract", help="Extract grading tasks from state DBs")
    ext.add_argument("--scopes", nargs="+", required=True, help="Scope IDs to extract")
    ext.add_argument("--output", type=Path, default=RESULTS_DIR / "grading_tasks_expansion.jsonl")

    # Merge subcommand
    mrg = sub.add_parser("merge", help="Merge score files into grid_summary")
    mrg.add_argument("--score-files", nargs="+", type=Path, required=True)
    mrg.add_argument("--output", type=Path, default=RESULTS_DIR / "grid_summary_full.json")

    args = parser.parse_args()
    if args.command == "extract":
        extract_grading_tasks(args.scopes, args.output)
    elif args.command == "merge":
        merge_scores(args.score_files, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
