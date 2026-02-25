#!/usr/bin/env python3
"""Canonical scenario trace: extract full agent interaction for one question across all adapters.

Default scenario: scope 04 (environmental_drift), Q3 (chromium source), checkpoint 99, 16k budget.
Generates a swimlane figure, LaTeX comparison table, and structured trace JSON.

Usage:
    uv run python scripts/analyze_scenario_trace.py [--budget 16k] [--scope s04]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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

# Correctness markers for ed04_q03
CORRECTNESS_MARKERS = {
    "discharge_pipe": re.compile(r"discharge\s+pipe", re.IGNORECASE),
    "rm_18_6": re.compile(r"RM\s*18\.6|river.?mile\s*18\.6|18\.6", re.IGNORECASE),
    "wq03_peak": re.compile(r"WQ.?03.*(?:peak|highest|132|105|52)", re.IGNORECASE),
    "spatial_gradient": re.compile(
        r"(?:gradient|upstream|downstream|dilut|decay|spatial)", re.IGNORECASE
    ),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_state(state_file: str) -> dict:
    with open(state_file) as f:
        return json.load(f)


def find_runs(state: dict, scope: str, budget: str) -> dict[str, str]:
    """Return {adapter: run_id} for all scored runs matching scope+budget."""
    runs = state.get("runs", {})
    result = {}
    for label, info in runs.items():
        if info.get("status") != "scored":
            continue
        parts = label.split("/")
        if len(parts) != 3:
            continue
        adapter, sc, bgt = parts
        if sc == scope and bgt == budget:
            result[adapter] = info["run_id"]
    return result


def discover_scope_id(run_id: str) -> str | None:
    scopes_dir = Path(f"output/{run_id}/scopes")
    if not scopes_dir.exists():
        return None
    dirs = [d.name for d in scopes_dir.iterdir() if d.is_dir()]
    return dirs[0] if len(dirs) == 1 else None


def find_question_in_checkpoint(
    run_id: str, scope_id: str, checkpoint: int, question_pattern: str
) -> dict | None:
    """Find a question matching pattern in a specific checkpoint."""
    qr_path = Path(f"output/{run_id}/scopes/{scope_id}/checkpoint_{checkpoint}/question_results.json")
    if not qr_path.exists():
        # Try nearby checkpoints
        scope_dir = Path(f"output/{run_id}/scopes/{scope_id}")
        cp_dirs = sorted(
            [d for d in scope_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        # Find closest checkpoint
        best = None
        best_dist = float("inf")
        for d in cp_dirs:
            ep = int(d.name.split("_")[1])
            dist = abs(ep - checkpoint)
            if dist < best_dist:
                best_dist = dist
                best = d
        if best is None:
            return None
        qr_path = best / "question_results.json"
        logger.info("Checkpoint %d not found, using %s", checkpoint, best.name)

    if not qr_path.exists():
        return None

    with open(qr_path) as f:
        questions = json.load(f)

    pat = re.compile(question_pattern, re.IGNORECASE)
    for q in questions:
        qid = q.get("question", {}).get("question_id", "")
        if pat.search(qid):
            return q
    return None


def extract_question_trace(question_result: dict) -> dict[str, Any]:
    """Extract structured trace from a question_results.json entry."""
    question = question_result.get("question", {})
    answer = question_result.get("answer", {})

    # Parse turns for tool calls
    turns = answer.get("turns", [])
    search_queries: list[str] = []
    batch_retrieves: list[list[str]] = []
    tool_sequence: list[dict] = []  # [{type, detail, tokens}]

    cumulative_tokens = 0
    for turn in turns:
        turn_tokens = turn.get("tokens_used", 0)
        cumulative_tokens += turn_tokens

        if turn.get("role") == "assistant":
            for tc in turn.get("tool_calls", []):
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                if name == "memory_search":
                    query = args.get("query", "")
                    search_queries.append(query)
                    tool_sequence.append({
                        "type": "memory_search",
                        "query": query,
                        "cumulative_tokens": cumulative_tokens,
                    })
                elif name == "batch_retrieve":
                    refs = args.get("ref_ids", [])
                    batch_retrieves.append(refs)
                    tool_sequence.append({
                        "type": "batch_retrieve",
                        "ref_ids": refs,
                        "cumulative_tokens": cumulative_tokens,
                    })
                else:
                    tool_sequence.append({
                        "type": name,
                        "cumulative_tokens": cumulative_tokens,
                    })

        elif turn.get("role") == "tool":
            for tr in turn.get("tool_results", []):
                content = tr.get("content", "")
                # Count results returned
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        if tool_sequence and tool_sequence[-1]["type"] == "memory_search":
                            tool_sequence[-1]["results_count"] = len(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass

    answer_text = answer.get("answer_text", "")
    refs_cited = answer.get("refs_cited", [])

    # Correctness checks
    checks = {}
    for name, pattern in CORRECTNESS_MARKERS.items():
        checks[name] = bool(pattern.search(answer_text))

    return {
        "question_id": question.get("question_id", ""),
        "question_prompt": question.get("prompt", ""),
        "ground_truth": question.get("ground_truth", {}),
        "answer_text": answer_text,
        "answer_length": len(answer_text),
        "tool_calls_made": answer.get("tool_calls_made", 0),
        "total_tokens": answer.get("total_tokens", 0),
        "wall_time_ms": answer.get("wall_time_ms", 0),
        "budget_violations": answer.get("budget_violations", []),
        "refs_cited": refs_cited,
        "search_queries": search_queries,
        "batch_retrieves": batch_retrieves,
        "tool_sequence": tool_sequence,
        "correctness": checks,
        "is_empty": not answer_text.strip(),
        "is_correct": checks.get("discharge_pipe", False),
    }


# ── Output generation ─────────────────────────────────────────────────────────

def generate_comparison_table(traces: dict[str, dict], output_dir: str):
    """Generate markdown + LaTeX comparison table."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Markdown ──
    md_lines = [
        "| Adapter | Tool Calls | Tokens | Wall Time | Answer? | Discharge Pipe? | RM 18.6? | Refs Cited |",
        "|---------|-----------|--------|-----------|---------|-----------------|----------|------------|",
    ]
    for adapter in ADAPTER_ORDER:
        if adapter not in traces:
            continue
        t = traces[adapter]
        answer_status = "Empty" if t["is_empty"] else f'{t["answer_length"]} chars'
        pipe = "Yes" if t["correctness"].get("discharge_pipe") else "No"
        rm = "Yes" if t["correctness"].get("rm_18_6") else "No"
        refs = ", ".join(t["refs_cited"][:3]) if t["refs_cited"] else "none"
        md_lines.append(
            f'| {adapter} | {t["tool_calls_made"]} | {t["total_tokens"]:,} | '
            f'{t["wall_time_ms"]:.0f}ms | {answer_status} | {pipe} | {rm} | {refs} |'
        )

    md_path = Path(output_dir) / "scenario_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    logger.info("Saved %s", md_path)

    # ── LaTeX ──
    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Canonical Scenario: Chromium Source Question (ed04\_q03)}",
        r"\label{tab:scenario}",
        r"\begin{tabular}{lrrrcccl}",
        r"\toprule",
        r"System & Calls & Tokens & Time (ms) & Answer & Pipe? & RM 18.6? & Refs \\",
        r"\midrule",
    ]
    for adapter in ADAPTER_ORDER:
        if adapter not in traces:
            continue
        t = traces[adapter]
        safe_name = adapter.replace("-", "{-}").replace("_", r"\_")
        answer_status = "Empty" if t["is_empty"] else f'{t["answer_length"]}ch'
        pipe = r"\cmark" if t["correctness"].get("discharge_pipe") else r"\xmark"
        rm = r"\cmark" if t["correctness"].get("rm_18_6") else r"\xmark"
        n_refs = len(t["refs_cited"])
        tex_lines.append(
            f"  {safe_name} & {t['tool_calls_made']} & {t['total_tokens']:,} & "
            f"{t['wall_time_ms']:.0f} & {answer_status} & {pipe} & {rm} & {n_refs} refs \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_path = Path("results/tables") / "scenario_comparison.tex"
    os.makedirs(tex_path.parent, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines) + "\n")
    logger.info("Saved %s", tex_path)


def generate_swimlane(traces: dict[str, dict], output_dir: str):
    """Horizontal swimlane: one row per adapter, x = cumulative tokens, colored blocks for tool calls."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # Filter to adapters with data, maintain order
    adapters_with_data = [a for a in ADAPTER_ORDER if a in traces]
    n_rows = len(adapters_with_data)

    fig, ax = plt.subplots(figsize=(14, max(3, n_rows * 0.8 + 1)))

    tool_colors = {
        "memory_search": "#2196F3",
        "batch_retrieve": "#4CAF50",
        "memory_retrieve": "#8BC34A",
    }
    bar_height = 0.5

    max_tokens = 0
    for i, adapter in enumerate(adapters_with_data):
        t = traces[adapter]
        y = n_rows - 1 - i

        for step in t["tool_sequence"]:
            x = step["cumulative_tokens"]
            max_tokens = max(max_tokens, x)
            tool_type = step["type"]
            color = tool_colors.get(tool_type, "#999999")

            # Draw a small block at this token position
            block_width = max(max_tokens * 0.008, 200)
            ax.barh(y, block_width, left=x - block_width / 2, height=bar_height,
                    color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

            # Label: result count or ref count
            label = ""
            if "results_count" in step:
                label = str(step["results_count"])
            elif "ref_ids" in step:
                label = str(len(step["ref_ids"]))
            if label:
                ax.text(x, y + bar_height / 2 + 0.05, label,
                        ha="center", va="bottom", fontsize=7, color="#333")

        # Mark answer at the end
        total_tok = t["total_tokens"]
        max_tokens = max(max_tokens, total_tok)
        if t["is_empty"]:
            ax.plot(total_tok, y, "x", color="red", markersize=10, markeredgewidth=2)
        elif t["is_correct"]:
            ax.plot(total_tok, y, "D", color="green", markersize=8)
        else:
            ax.plot(total_tok, y, "s", color="orange", markersize=8)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [_short_name(a) for a in reversed(adapters_with_data)]
    )
    ax.set_xlabel("Cumulative Tokens")
    ax.set_title("Scenario Trace: Tool Call Sequence per Adapter (ed04_q03)")
    ax.set_xlim(-500, max_tokens * 1.05)
    ax.grid(True, axis="x", alpha=0.2)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2196F3", label="memory_search"),
        mpatches.Patch(color="#4CAF50", label="batch_retrieve"),
        plt.Line2D([], [], marker="D", color="green", linestyle="None",
                   markersize=6, label="Correct answer"),
        plt.Line2D([], [], marker="s", color="orange", linestyle="None",
                   markersize=6, label="Partial/wrong"),
        plt.Line2D([], [], marker="x", color="red", linestyle="None",
                   markersize=8, markeredgewidth=2, label="Empty answer"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

    out = Path(output_dir) / "scenario_trace_swimlane.png"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


def _short_name(adapter: str) -> str:
    return {
        "sqlite-chunked-hybrid": "chunked-hybrid",
        "letta-sleepy": "letta-sleepy",
    }.get(adapter, adapter)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Canonical scenario trace analysis")
    parser.add_argument(
        "--state-file",
        default="results/constrained_validation.json",
        help="Path to constrained_validation.json",
    )
    parser.add_argument("--scope", default="s04")
    parser.add_argument("--budget", default="16k", choices=["8k", "16k"])
    parser.add_argument("--question", default="q03_longitudinal")
    parser.add_argument("--checkpoint", type=int, default=99)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    state = load_state(args.state_file)
    adapter_runs = find_runs(state, args.scope, args.budget)

    logger.info(
        "Found %d adapters for %s/%s: %s",
        len(adapter_runs), args.scope, args.budget,
        ", ".join(f"{a}={r}" for a, r in sorted(adapter_runs.items())),
    )

    traces: dict[str, dict] = {}

    for adapter, run_id in sorted(adapter_runs.items()):
        scope_id = discover_scope_id(run_id)
        if scope_id is None:
            logger.warning("No scope found for %s (%s)", adapter, run_id)
            continue

        qr = find_question_in_checkpoint(run_id, scope_id, args.checkpoint, args.question)
        if qr is None:
            logger.warning("Question %s not found for %s at checkpoint %d", args.question, adapter, args.checkpoint)
            continue

        trace = extract_question_trace(qr)
        trace["adapter"] = adapter
        trace["run_id"] = run_id
        trace["scope_id"] = scope_id
        traces[adapter] = trace

        status = "EMPTY" if trace["is_empty"] else ("CORRECT" if trace["is_correct"] else "PARTIAL/WRONG")
        logger.info(
            "  %s: %d calls, %d tokens, %dms → %s",
            adapter, trace["tool_calls_made"], trace["total_tokens"],
            trace["wall_time_ms"], status,
        )

    if not traces:
        logger.error("No traces found — exiting")
        return

    # Export structured data
    json_out = Path("results/scenario_trace_data.json")
    os.makedirs(json_out.parent, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(
            {
                "scope": args.scope,
                "budget": args.budget,
                "checkpoint": args.checkpoint,
                "question_pattern": args.question,
                "traces": traces,
            },
            f, indent=2,
        )
    logger.info("Exported %s", json_out)

    # Generate outputs
    generate_comparison_table(traces, args.output_dir)
    generate_swimlane(traces, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
