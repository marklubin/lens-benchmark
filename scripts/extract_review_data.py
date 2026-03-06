#!/usr/bin/env python3
"""Extract scoring data across all runs for the review documents.

Outputs JSON with per-run metrics, gated vs ungated composites,
organized by scope category (numeric/narrative/SRS).
"""
import json
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path("output")

# Scope category mapping
NUMERIC = {f"{i:02d}" for i in range(1, 7)}
NARRATIVE = {f"{i:02d}" for i in range(7, 10)}
SRS = {f"{i:02d}" for i in range(10, 13)}

# Composite weights (from scorer)
WEIGHTS = {
    "evidence_grounding": 0.10,
    "fact_recall": 0.10,
    "evidence_coverage": 0.10,
    "budget_compliance": 0.10,
    "answer_quality": 0.15,
    "insight_depth": 0.15,
    "reasoning_quality": 0.10,
    "naive_baseline_advantage": 0.15,
    "action_quality": 0.05,
}

def compute_composite(metrics, apply_gate=True):
    """Compute weighted composite score, optionally without hard gate."""
    eg = metrics.get("evidence_grounding", 0)
    bc = metrics.get("budget_compliance", 0)

    score = sum(metrics.get(k, 0) * w for k, w in WEIGHTS.items())

    if apply_gate and (eg < 0.5 or bc < 0.5):
        return 0.0
    return score

def extract_scope_id(config):
    """Extract scope number from config/dataset path."""
    dataset = config.get("dataset", "")
    # Try various patterns
    for pattern in ["scope_", "scope", "scopes/"]:
        if pattern in dataset:
            idx = dataset.index(pattern) + len(pattern)
            num = ""
            for c in dataset[idx:]:
                if c.isdigit():
                    num += c
                elif num:
                    break
            if num:
                return num.zfill(2)
    return None

def categorize_scope(scope_id):
    if scope_id in NUMERIC:
        return "numeric"
    elif scope_id in NARRATIVE:
        return "narrative"
    elif scope_id in SRS:
        return "srs"
    return "unknown"

def process_run(run_dir):
    """Extract all data from a single run."""
    run_id = run_dir.name

    config_path = run_dir / "config.json"
    scorecard_path = run_dir / "scores" / "scorecard.json"

    if not scorecard_path.exists() or not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
        scorecard = json.loads(scorecard_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    adapter = scorecard.get("adapter") or config.get("adapter", "unknown")
    scope_id = extract_scope_id(config)
    if not scope_id:
        return None

    metrics = {m["name"]: m["value"] for m in scorecard.get("metrics", [])}

    budget_preset = config.get("agent_budget", {}).get("preset", "unknown")
    model = config.get("llm", {}).get("model", "unknown")

    # Compute both gated and ungated composites
    composite_gated = compute_composite(metrics, apply_gate=True)
    composite_ungated = compute_composite(metrics, apply_gate=False)
    gate_fired = composite_gated == 0.0 and composite_ungated > 0.0

    return {
        "run_id": run_id,
        "adapter": adapter,
        "scope_id": scope_id,
        "scope_category": categorize_scope(scope_id),
        "budget_preset": budget_preset,
        "model": model,
        "metrics": metrics,
        "composite_gated": round(composite_gated, 4),
        "composite_ungated": round(composite_ungated, 4),
        "gate_fired": gate_fired,
        "gate_reason": (
            "evidence_grounding" if metrics.get("evidence_grounding", 0) < 0.5
            else "budget_compliance" if metrics.get("budget_compliance", 0) < 0.5
            else None
        ) if gate_fired else None,
    }


def main():
    runs = []
    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        result = process_run(run_dir)
        if result:
            runs.append(result)

    # Group by category
    by_category = {"numeric": [], "narrative": [], "srs": []}
    for r in runs:
        cat = r["scope_category"]
        if cat in by_category:
            by_category[cat].append(r)

    # Summary stats
    print(f"Total scored runs: {len(runs)}")
    for cat, cat_runs in by_category.items():
        adapters = set(r["adapter"] for r in cat_runs)
        gated = sum(1 for r in cat_runs if r["gate_fired"])
        print(f"  {cat}: {len(cat_runs)} runs, {len(adapters)} adapters, {gated} gate-fired")

    # Gate impact analysis
    print("\n=== GATE IMPACT BY ADAPTER ===")
    adapter_gate = {}
    for r in runs:
        a = r["adapter"]
        if a not in adapter_gate:
            adapter_gate[a] = {"gated": [], "ungated": [], "fired": 0, "total": 0}
        adapter_gate[a]["gated"].append(r["composite_gated"])
        adapter_gate[a]["ungated"].append(r["composite_ungated"])
        adapter_gate[a]["total"] += 1
        if r["gate_fired"]:
            adapter_gate[a]["fired"] += 1

    print(f"{'Adapter':<25} {'Runs':>5} {'Gated':>6} {'Mean(gated)':>12} {'Mean(ungated)':>14} {'Delta':>8}")
    for a in sorted(adapter_gate.keys()):
        d = adapter_gate[a]
        mg = sum(d["gated"]) / len(d["gated"]) if d["gated"] else 0
        mu = sum(d["ungated"]) / len(d["ungated"]) if d["ungated"] else 0
        delta = mu - mg
        print(f"{a:<25} {d['total']:>5} {d['fired']:>6} {mg:>12.4f} {mu:>14.4f} {delta:>8.4f}")

    # Per-category adapter rankings
    for cat in ["numeric", "narrative", "srs"]:
        cat_runs = by_category.get(cat, [])
        if not cat_runs:
            continue
        print(f"\n=== {cat.upper()} RANKINGS ===")
        adapter_scores = {}
        for r in cat_runs:
            a = r["adapter"]
            if a not in adapter_scores:
                adapter_scores[a] = {"gated": [], "ungated": []}
            adapter_scores[a]["gated"].append(r["composite_gated"])
            adapter_scores[a]["ungated"].append(r["composite_ungated"])

        ranked = sorted(adapter_scores.items(), key=lambda x: sum(x[1]["gated"])/len(x[1]["gated"]), reverse=True)
        print(f"{'Adapter':<25} {'N':>3} {'Mean(gated)':>12} {'Mean(ungated)':>14} {'Gate Delta':>11}")
        for a, d in ranked:
            mg = sum(d["gated"]) / len(d["gated"])
            mu = sum(d["ungated"]) / len(d["ungated"])
            print(f"{a:<25} {len(d['gated']):>3} {mg:>12.4f} {mu:>14.4f} {mu-mg:>11.4f}")

    # Write full data
    output_path = Path("lens-initial-run-review/scoring_data.json")
    output_path.write_text(json.dumps(runs, indent=2))
    print(f"\nFull data written to {output_path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
