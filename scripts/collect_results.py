#!/usr/bin/env python3
"""Collect scorecard results from all runs into a CSV matrix.

Usage:
    python3 scripts/collect_results.py > results/sweep_results.csv
    python3 scripts/collect_results.py --format table   # pretty-print
    python3 scripts/collect_results.py --format json     # JSON output
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("output")

# Metrics to extract from scorecards
METRICS = [
    "composite",
    "evidence_grounding",
    "fact_recall",
    "evidence_coverage",
    "budget_compliance",
    "citation_coverage",
    "answer_quality",
    "insight_depth",
    "reasoning_quality",
    "naive_baseline_advantage",
    "action_quality",
]


def parse_run_dir_name(name: str) -> dict | None:
    """Extract adapter, scope, budget from run directory metadata."""
    # Try config.json first (new format), then run_result.json (old format)
    config_file = OUTPUT_DIR / name / "config.json"
    result_file = OUTPUT_DIR / name / "run_result.json"

    config = None
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    if config is None and result_file.exists():
        try:
            data = json.loads(result_file.read_text())
            config = data.get("config", {})
        except (json.JSONDecodeError, OSError):
            pass
    if config is None:
        return None

    adapter = config.get("adapter", "unknown")
    dataset = config.get("dataset", "")
    budget_config = config.get("agent_budget", {})
    preset = budget_config.get("preset", "standard")

    # Extract scope from dataset path
    scope = "??"
    if "scope_" in dataset:
        # datasets/scope_01_only.json → 01
        parts = dataset.split("scope_")
        if len(parts) >= 2:
            scope = parts[1][:2]

    # Map preset to budget label
    budget = "standard"
    if "4k" in preset:
        budget = "4k"
    elif "2k" in preset:
        budget = "2k"

    return {
        "run_id": name,
        "adapter": adapter,
        "scope": scope,
        "budget": budget,
    }


def load_scorecard(run_dir: str) -> dict | None:
    """Load scorecard.json from a run directory."""
    # Check both locations: scores/scorecard.json (new) and scorecard.json (old)
    for subpath in ["scores/scorecard.json", "scorecard.json"]:
        path = OUTPUT_DIR / run_dir / subpath
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return None


def extract_metrics(scorecard: dict) -> dict[str, float | None]:
    """Extract metric values from scorecard."""
    metrics_raw = scorecard.get("metrics", [])
    # Handle both list format (new) and dict format (old)
    if isinstance(metrics_raw, list):
        metrics_data = {m["name"]: m for m in metrics_raw if isinstance(m, dict) and "name" in m}
    else:
        metrics_data = metrics_raw

    result = {}
    for m in METRICS:
        if m == "composite":
            result[m] = scorecard.get("composite_score")
        else:
            metric_info = metrics_data.get(m, {})
            if isinstance(metric_info, dict):
                result[m] = metric_info.get("value") or metric_info.get("score")
            else:
                result[m] = None
    return result


def collect_all() -> list[dict]:
    """Scan all run directories and collect results."""
    rows = []

    if not OUTPUT_DIR.exists():
        return rows

    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        info = parse_run_dir_name(run_dir.name)
        if info is None:
            continue

        scorecard = load_scorecard(run_dir.name)
        if scorecard is None:
            continue

        metrics = extract_metrics(scorecard)
        row = {**info, **metrics}
        rows.append(row)

    return rows


def output_csv(rows: list[dict]) -> None:
    """Write CSV to stdout."""
    if not rows:
        print("No results found.", file=sys.stderr)
        return

    fieldnames = ["adapter", "scope", "budget", "run_id"] + METRICS
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r["adapter"], r["scope"], r["budget"])):
        writer.writerow(row)


def output_table(rows: list[dict]) -> None:
    """Pretty-print a summary table."""
    if not rows:
        print("No results found.")
        return

    # Group by adapter × scope, show composite for each budget
    from collections import defaultdict

    matrix: dict[str, dict[str, dict[str, float | None]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        matrix[row["adapter"]][row["scope"]][row["budget"]] = row.get("composite")

    adapters = sorted(matrix.keys())
    scopes = sorted({s for a in matrix.values() for s in a.keys()})
    budgets_seen = sorted({b for a in matrix.values() for s in a.values() for b in s.keys()})

    # Header
    header = f"{'Adapter':<25}"
    for scope in scopes:
        for budget in budgets_seen:
            header += f" {scope}/{budget:>8}"
    print(header)
    print("-" * len(header))

    for adapter in adapters:
        line = f"{adapter:<25}"
        for scope in scopes:
            for budget in budgets_seen:
                val = matrix[adapter].get(scope, {}).get(budget)
                if val is not None:
                    line += f" {val:>12.4f}"
                else:
                    line += f" {'---':>12}"
        print(line)


def output_json(rows: list[dict]) -> None:
    """Write JSON to stdout."""
    print(json.dumps(rows, indent=2))


def main():
    fmt = "csv"
    if "--format" in sys.argv:
        idx = sys.argv.index("--format")
        if idx + 1 < len(sys.argv):
            fmt = sys.argv[idx + 1]

    rows = collect_all()
    print(f"Found {len(rows)} scored runs.", file=sys.stderr)

    if fmt == "csv":
        output_csv(rows)
    elif fmt == "table":
        output_table(rows)
    elif fmt == "json":
        output_json(rows)
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
