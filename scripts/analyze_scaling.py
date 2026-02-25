#!/usr/bin/env python3
"""Scaling analysis: how system performance changes as data grows from episode 1 to 120.

Extracts timing, token, and tool-call data from Phase 5 runs and generates 5 figures:
  A. Ingest latency per episode
  B. Question wall time per checkpoint
  C. Token consumption per checkpoint
  D. Tool calls per checkpoint
  E. Empty answer rate per checkpoint

Usage:
    uv run python scripts/analyze_scaling.py [--budget 16k] [--output-dir results/figures] [--no-plot]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Adapter color map (consistent across all figures) ─────────────────────────
ADAPTER_COLORS = {
    "sqlite-chunked-hybrid": "#2196F3",
    "cognee": "#4CAF50",
    "graphiti": "#FF9800",
    "mem0-raw": "#9C27B0",
    "letta": "#F44336",
    "letta-sleepy": "#E91E63",
    "compaction": "#795548",
    "null": "#607D8B",
}

ADAPTER_ORDER = [
    "sqlite-chunked-hybrid",
    "cognee",
    "graphiti",
    "mem0-raw",
    "letta",
    "letta-sleepy",
    "compaction",
    "null",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_state(state_file: str) -> dict:
    """Load constrained_validation.json."""
    with open(state_file) as f:
        return json.load(f)


def parse_label(label: str) -> tuple[str, str, str] | None:
    """Parse 'adapter/sXX/budget' into (adapter, scope, budget)."""
    parts = label.split("/")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


def discover_scope_id(run_id: str) -> str | None:
    """Find the actual scope directory name for a run."""
    scopes_dir = Path(f"output/{run_id}/scopes")
    if not scopes_dir.exists():
        return None
    dirs = [d.name for d in scopes_dir.iterdir() if d.is_dir()]
    return dirs[0] if len(dirs) == 1 else None


def extract_ingest_timing(run_id: str) -> list[dict]:
    """Parse log.jsonl for ingest step entries. Returns [{episode_index, elapsed_ms}]."""
    log_path = Path(f"output/{run_id}/log.jsonl")
    if not log_path.exists():
        logger.warning("log.jsonl not found for run %s", run_id)
        return []

    entries = []
    idx = 0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("step") == "ingest":
                entries.append({
                    "episode_index": idx,
                    "elapsed_ms": entry.get("elapsed_ms", 0.0),
                })
                idx += 1
    return entries


def extract_checkpoint_metrics(run_id: str, scope_id: str) -> list[dict]:
    """Extract per-checkpoint aggregated metrics from question_results.json files.

    Returns list of {checkpoint_episode, checkpoint_index, mean_wall_time_ms,
    mean_total_tokens, mean_tool_calls, empty_rate, n_questions}.
    """
    scope_dir = Path(f"output/{run_id}/scopes/{scope_id}")
    if not scope_dir.exists():
        logger.warning("Scope dir not found: %s", scope_dir)
        return []

    checkpoint_dirs = sorted(
        [d for d in scope_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    results = []
    for ci, cp_dir in enumerate(checkpoint_dirs):
        qr_path = cp_dir / "question_results.json"
        if not qr_path.exists():
            continue

        try:
            with open(qr_path) as f:
                questions = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", qr_path, exc)
            continue

        if not questions:
            continue

        wall_times = []
        total_tokens = []
        tool_calls = []
        empty_count = 0

        for q in questions:
            ans = q.get("answer", {})
            wt = ans.get("wall_time_ms")
            if wt is not None:
                wall_times.append(wt)
            tt = ans.get("total_tokens")
            if tt is not None:
                total_tokens.append(tt)
            tc = ans.get("tool_calls_made")
            if tc is not None:
                tool_calls.append(tc)
            answer_text = ans.get("answer_text", "")
            if not answer_text or not answer_text.strip():
                empty_count += 1

        n = len(questions)
        cp_episode = int(cp_dir.name.split("_")[1])

        results.append({
            "checkpoint_episode": cp_episode,
            "checkpoint_index": ci,
            "mean_wall_time_ms": sum(wall_times) / len(wall_times) if wall_times else 0,
            "mean_total_tokens": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
            "mean_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else 0,
            "empty_rate": empty_count / n if n > 0 else 0,
            "n_questions": n,
        })

    return results


def collect_all_data(
    state: dict, budget_filter: str
) -> dict[str, dict[str, Any]]:
    """Collect ingest timing + checkpoint metrics for all runs matching budget.

    Returns {adapter: {scopes: [{scope_code, ingest_timing, checkpoints}], ...}}.
    """
    runs = state.get("runs", {})
    adapter_data: dict[str, list[dict]] = {}

    for label, info in runs.items():
        if info.get("status") != "scored":
            continue
        parsed = parse_label(label)
        if parsed is None:
            continue
        adapter, scope_code, budget = parsed
        if budget != budget_filter:
            continue

        run_id = info["run_id"]
        scope_id = discover_scope_id(run_id)
        if scope_id is None:
            logger.warning("No scope found for run %s (%s)", run_id, label)
            continue

        ingest = extract_ingest_timing(run_id)
        checkpoints = extract_checkpoint_metrics(run_id, scope_id)

        entry = {
            "scope_code": scope_code,
            "run_id": run_id,
            "scope_id": scope_id,
            "ingest_timing": ingest,
            "checkpoints": checkpoints,
        }
        adapter_data.setdefault(adapter, []).append(entry)

    return adapter_data


def aggregate_across_scopes(
    adapter_data: dict[str, list[dict]],
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics across scopes for each adapter.

    Returns {adapter: {
        ingest_timing_avg: [{episode_index, mean_elapsed_ms}],
        checkpoints_avg: [{checkpoint_index, mean_wall_time_ms, mean_total_tokens,
                          mean_tool_calls, empty_rate}],
        n_scopes: int,
    }}.
    """
    aggregated = {}

    for adapter, scope_entries in adapter_data.items():
        # ── Ingest timing: average across scopes at each episode index ──
        ingest_by_idx: dict[int, list[float]] = {}
        for entry in scope_entries:
            for pt in entry["ingest_timing"]:
                ingest_by_idx.setdefault(pt["episode_index"], []).append(pt["elapsed_ms"])

        ingest_avg = sorted(
            [
                {"episode_index": idx, "mean_elapsed_ms": sum(vals) / len(vals)}
                for idx, vals in ingest_by_idx.items()
            ],
            key=lambda x: x["episode_index"],
        )

        # ── Checkpoint metrics: average across scopes at each checkpoint index ──
        cp_by_idx: dict[int, list[dict]] = {}
        for entry in scope_entries:
            for cp in entry["checkpoints"]:
                cp_by_idx.setdefault(cp["checkpoint_index"], []).append(cp)

        checkpoints_avg = []
        for ci in sorted(cp_by_idx.keys()):
            cps = cp_by_idx[ci]
            n = len(cps)
            checkpoints_avg.append({
                "checkpoint_index": ci,
                "mean_wall_time_ms": sum(c["mean_wall_time_ms"] for c in cps) / n,
                "mean_total_tokens": sum(c["mean_total_tokens"] for c in cps) / n,
                "mean_tool_calls": sum(c["mean_tool_calls"] for c in cps) / n,
                "empty_rate": sum(c["empty_rate"] for c in cps) / n,
                "n_scopes": n,
            })

        aggregated[adapter] = {
            "ingest_timing_avg": ingest_avg,
            "checkpoints_avg": checkpoints_avg,
            "n_scopes": len(scope_entries),
        }

    return aggregated


# ── Plotting ──────────────────────────────────────────────────────────────────

def _setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def _adapter_label(adapter: str) -> str:
    """Shorter display names for plot legends."""
    return {
        "sqlite-chunked-hybrid": "chunked-hybrid",
        "letta-sleepy": "letta-sleepy",
    }.get(adapter, adapter)


def plot_ingest_latency(aggregated: dict, output_dir: str):
    """Figure A: Ingest latency per episode (log scale)."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))

    for adapter in ADAPTER_ORDER:
        if adapter not in aggregated or adapter == "null":
            continue
        data = aggregated[adapter]["ingest_timing_avg"]
        if not data:
            continue
        xs = [d["episode_index"] for d in data]
        ys = [max(d["mean_elapsed_ms"], 0.001) for d in data]  # floor for log scale
        color = ADAPTER_COLORS.get(adapter, "#333333")
        ax.plot(xs, ys, label=_adapter_label(adapter), color=color, linewidth=1.5, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Ingest Latency (ms, log scale)")
    ax.set_title("A. Per-Episode Ingest Latency as Data Grows")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Annotation for buffered adapters
    ax.annotate(
        "cognee/graphiti/compaction buffer\nduring ingest → near-zero here",
        xy=(60, 0.01), fontsize=8, fontstyle="italic", color="#666",
    )

    out = Path(output_dir) / "scaling_ingest_latency.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_checkpoint_metric(
    aggregated: dict,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: str,
    log_scale: bool = False,
):
    """Generic checkpoint-indexed line plot."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))

    for adapter in ADAPTER_ORDER:
        if adapter not in aggregated:
            continue
        cps = aggregated[adapter]["checkpoints_avg"]
        if not cps:
            continue
        xs = [c["checkpoint_index"] for c in cps]
        ys = [c[metric_key] for c in cps]
        color = ADAPTER_COLORS.get(adapter, "#333333")
        ax.plot(
            xs, ys,
            label=_adapter_label(adapter),
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4,
            alpha=0.85,
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Checkpoint Index (earlier → later)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add checkpoint episode labels on x-axis
    if aggregated:
        sample = next(iter(aggregated.values()))
        cps = sample.get("checkpoints_avg", [])
        # We don't have episode numbers in aggregated, so just use index
        ax.set_xticks(range(len(cps)))

    out = Path(output_dir) / filename
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def generate_all_plots(aggregated: dict, output_dir: str):
    """Generate all 5 scaling figures."""
    os.makedirs(output_dir, exist_ok=True)

    plot_ingest_latency(aggregated, output_dir)

    plot_checkpoint_metric(
        aggregated,
        metric_key="mean_wall_time_ms",
        ylabel="Mean Wall Time (ms)",
        title="B. Question Answering Latency per Checkpoint",
        filename="scaling_wall_time.png",
        output_dir=output_dir,
    )

    plot_checkpoint_metric(
        aggregated,
        metric_key="mean_total_tokens",
        ylabel="Mean Total Tokens",
        title="C. Token Consumption per Checkpoint",
        filename="scaling_tokens.png",
        output_dir=output_dir,
    )

    plot_checkpoint_metric(
        aggregated,
        metric_key="mean_tool_calls",
        ylabel="Mean Tool Calls per Question",
        title="D. Tool Calls per Checkpoint",
        filename="scaling_tool_calls.png",
        output_dir=output_dir,
    )

    plot_checkpoint_metric(
        aggregated,
        metric_key="empty_rate",
        ylabel="Empty Answer Rate",
        title="E. Empty Answer Rate per Checkpoint",
        filename="scaling_empty_rate.png",
        output_dir=output_dir,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scaling analysis across Phase 5 runs")
    parser.add_argument(
        "--state-file",
        default="results/constrained_validation.json",
        help="Path to constrained_validation.json",
    )
    parser.add_argument("--budget", default="16k", choices=["8k", "16k"])
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading state from %s (budget=%s)", args.state_file, args.budget)
    state = load_state(args.state_file)

    logger.info("Collecting data from scored runs...")
    adapter_data = collect_all_data(state, args.budget)

    scored_count = sum(len(v) for v in adapter_data.values())
    logger.info("Found %d scored runs across %d adapters", scored_count, len(adapter_data))

    logger.info("Aggregating across scopes...")
    aggregated = aggregate_across_scopes(adapter_data)

    for adapter, agg in sorted(aggregated.items()):
        n_ingest = len(agg["ingest_timing_avg"])
        n_cp = len(agg["checkpoints_avg"])
        logger.info(
            "  %s: %d scopes, %d ingest points, %d checkpoints",
            adapter, agg["n_scopes"], n_ingest, n_cp,
        )

    # Export raw data
    json_out = Path("results/scaling_analysis.json")
    os.makedirs(json_out.parent, exist_ok=True)

    # Make JSON-serializable
    export = {}
    for adapter, agg in aggregated.items():
        export[adapter] = {
            "n_scopes": agg["n_scopes"],
            "ingest_timing_avg": agg["ingest_timing_avg"],
            "checkpoints_avg": agg["checkpoints_avg"],
        }

    with open(json_out, "w") as f:
        json.dump({"budget": args.budget, "adapters": export}, f, indent=2)
    logger.info("Exported %s", json_out)

    if not args.no_plot:
        logger.info("Generating plots...")
        generate_all_plots(aggregated, args.output_dir)
        logger.info("Done. Figures saved to %s/scaling_*.png", args.output_dir)
    else:
        logger.info("Skipping plots (--no-plot)")


if __name__ == "__main__":
    main()
