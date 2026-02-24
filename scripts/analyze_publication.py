#!/usr/bin/env python3
"""Publication-ready analysis of LENS benchmark results.

Generates statistical analyses, figures, and LaTeX-ready tables for the paper.

Analyses:
  D1. Bootstrap CIs on answer_quality per adapter across 6 scopes
  D2. Cross-scope consistency: heatmap + Kendall's W
  D3. Distractor effect: Phase 1-2 (30 eps) vs Phase 5 (120 eps)
  D4. Question-type analysis: per-type adapter rankings + discrimination

Usage:
    python3 scripts/analyze_publication.py
    python3 scripts/analyze_publication.py --state-file constrained_validation_state.json
    python3 scripts/analyze_publication.py --no-plot
    python3 scripts/analyze_publication.py --output-dir results/publication

Reads from:
  - constrained_validation_state.json (Phase 5 results)
  - output/*/scores/scorecard.json (detailed per-question data)

Outputs to results/figures/ and results/tables/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze_pub")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
STATE_FILE = PROJECT_DIR / "constrained_validation_state.json"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def parse_label(label: str) -> tuple[str, str, str] | None:
    """Parse 'adapter/sXX/budget' into (adapter, scope, budget)."""
    if "/" in label:
        parts = label.split("/")
        if len(parts) == 3:
            return parts[0], parts[1].lstrip("s"), parts[2]
    # Old format
    for suffix in ("_4k", "_2k", "_8k", "_16k"):
        if label.endswith(suffix):
            return label[: -len(suffix)], "01", suffix.lstrip("_")
    return None


def load_scorecard(run_id: str) -> dict | None:
    """Load scorecard.json from a run directory."""
    for subpath in ["scores/scorecard.json", "scorecard.json"]:
        path = OUTPUT_DIR / run_id / subpath
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


def load_results_json(run_id: str) -> dict | None:
    """Load results.json from a run directory."""
    path = OUTPUT_DIR / run_id / "results.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


def load_state(state_file: Path) -> dict:
    """Load validation state file."""
    if not state_file.exists():
        log.error("State file not found: %s", state_file)
        sys.exit(1)
    return json.loads(state_file.read_text())


def collect_metric_data(state: dict) -> dict[str, dict]:
    """Collect per-run metric data.

    Returns {label: {metric_name: value, ...}} for all scored runs.
    """
    data = {}
    for label, info in state.items():
        if not isinstance(info, dict) or info.get("status") != "scored":
            continue
        run_id = info.get("run_id")
        if not run_id:
            continue
        scorecard = load_scorecard(run_id)
        if not scorecard:
            continue

        metrics = {}
        for m in scorecard.get("metrics", []):
            if isinstance(m, dict):
                metrics[m["name"]] = m["value"]
        metrics["composite"] = scorecard.get("composite_score", 0)
        data[label] = metrics
    return data


def collect_per_question_data(state: dict) -> dict[str, list[dict]]:
    """Collect per-question results for each scored run.

    Returns {label: [{"question_id", "question_type", "answer_text", ...}]}.
    """
    data = {}
    for label, info in state.items():
        if not isinstance(info, dict) or info.get("status") != "scored":
            continue
        run_id = info.get("run_id")
        if not run_id:
            continue
        results = load_results_json(run_id)
        if not results:
            continue

        questions = []
        for scope in results.get("scopes", []):
            for cp in scope.get("checkpoints", []):
                for qr in cp.get("question_results", []):
                    q = qr.get("question", {})
                    a = qr.get("answer", {})
                    questions.append({
                        "question_id": q.get("question_id", ""),
                        "question_type": q.get("question_type", ""),
                        "checkpoint_after": q.get("checkpoint_after", 0),
                        "answer_text": a.get("answer_text", ""),
                        "key_facts": q.get("ground_truth", {}).get("key_facts", []),
                    })
        data[label] = questions
    return data


def collect_nba_per_question(state: dict) -> dict[str, list[dict]]:
    """Collect per-question NBA data from scorecards.

    Returns {label: [{question_id, win_rate, ...}]}.
    """
    data = {}
    for label, info in state.items():
        if not isinstance(info, dict) or info.get("status") != "scored":
            continue
        run_id = info.get("run_id")
        if not run_id:
            continue
        scorecard = load_scorecard(run_id)
        if not scorecard:
            continue
        for m in scorecard.get("metrics", []):
            if isinstance(m, dict) and m.get("name") == "naive_baseline_advantage":
                pq = m.get("details", {}).get("per_question", [])
                if pq:
                    data[label] = pq
                break
    return data


# ---------------------------------------------------------------------------
# D1: Bootstrap CIs
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean and CI. Returns (mean, ci_low, ci_high)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    return float(data.mean()), float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    diff = x - y
    try:
        from scipy.stats import wilcoxon
        nonzero = diff[diff != 0]
        if len(nonzero) < 3:
            return 0.0, 1.0
        stat, p = wilcoxon(nonzero)
        return float(stat), float(p)
    except ImportError:
        log.warning("scipy not available — using sign test fallback")
        n_pos = int(np.sum(diff > 0))
        n_neg = int(np.sum(diff < 0))
        n_total = n_pos + n_neg
        if n_total == 0:
            return 0.0, 1.0
        from math import comb
        k = min(n_pos, n_neg)
        p = 2 * sum(comb(n_total, i) * 0.5 ** n_total for i in range(k + 1))
        return float(n_pos - n_neg), min(float(p), 1.0)


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation (effect size for Wilcoxon)."""
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    n = len(nonzero)
    # r = (n_positive_ranks - n_negative_ranks) / n
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    pos_rank_sum = ranks[nonzero > 0].sum()
    neg_rank_sum = ranks[nonzero < 0].sum()
    return float((pos_rank_sum - neg_rank_sum) / (n * (n + 1) / 2))


def d1_bootstrap_analysis(
    metric_data: dict[str, dict],
    metric_name: str = "answer_quality",
    budget_filter: str | None = None,
) -> dict[str, dict]:
    """D1: Bootstrap CIs per adapter across scopes.

    Returns {adapter: {mean, ci_low, ci_high, n_scopes, scope_values}}.
    """
    # Group by adapter
    adapter_values: dict[str, list[float]] = defaultdict(list)
    adapter_scope_detail: dict[str, dict[str, float]] = defaultdict(dict)

    for label, metrics in metric_data.items():
        parsed = parse_label(label)
        if not parsed:
            continue
        adapter, scope, budget = parsed
        if budget_filter and budget != budget_filter:
            continue
        val = metrics.get(metric_name)
        if val is not None:
            adapter_values[adapter].append(val)
            adapter_scope_detail[adapter][scope] = val

    results = {}
    rng = np.random.default_rng(42)
    for adapter in sorted(adapter_values):
        vals = np.array(adapter_values[adapter])
        mean, lo, hi = bootstrap_ci(vals, rng=rng)
        results[adapter] = {
            "mean": round(mean, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "n_scopes": len(vals),
            "scope_values": adapter_scope_detail[adapter],
        }
    return results


def d1_pairwise_tests(
    metric_data: dict[str, dict],
    metric_name: str = "answer_quality",
    budget_filter: str | None = None,
) -> list[dict]:
    """Pairwise Wilcoxon signed-rank tests between all adapter pairs.

    Returns list of {adapter_a, adapter_b, statistic, p_value, effect_size, sig}.
    """
    # Group by adapter, matching scopes
    adapter_scope: dict[str, dict[str, float]] = defaultdict(dict)
    for label, metrics in metric_data.items():
        parsed = parse_label(label)
        if not parsed:
            continue
        adapter, scope, budget = parsed
        if budget_filter and budget != budget_filter:
            continue
        val = metrics.get(metric_name)
        if val is not None:
            adapter_scope[adapter][scope] = val

    adapters = sorted(adapter_scope.keys())
    results = []
    for a, b in itertools.combinations(adapters, 2):
        common = sorted(set(adapter_scope[a]) & set(adapter_scope[b]))
        if len(common) < 3:
            continue
        x = np.array([adapter_scope[a][s] for s in common])
        y = np.array([adapter_scope[b][s] for s in common])
        stat, p = wilcoxon_signed_rank(x, y)
        effect = rank_biserial_correlation(x, y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        results.append({
            "adapter_a": a,
            "adapter_b": b,
            "mean_a": round(float(x.mean()), 4),
            "mean_b": round(float(y.mean()), 4),
            "statistic": round(stat, 4),
            "p_value": round(p, 4),
            "effect_size": round(effect, 4),
            "n_scopes": len(common),
            "sig": sig,
        })
    return results


# ---------------------------------------------------------------------------
# D2: Cross-Scope Consistency
# ---------------------------------------------------------------------------


def d2_heatmap_matrix(
    metric_data: dict[str, dict],
    metric_name: str = "answer_quality",
    budget_filter: str | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """Build adapter × scope matrix for heatmap.

    Returns (adapters, scopes, matrix) where matrix[i][j] is
    adapter i's score on scope j.
    """
    adapter_scope: dict[str, dict[str, float]] = defaultdict(dict)
    for label, metrics in metric_data.items():
        parsed = parse_label(label)
        if not parsed:
            continue
        adapter, scope, budget = parsed
        if budget_filter and budget != budget_filter:
            continue
        val = metrics.get(metric_name)
        if val is not None:
            adapter_scope[adapter][scope] = val

    adapters = sorted(adapter_scope.keys())
    all_scopes = sorted(set(s for d in adapter_scope.values() for s in d))
    matrix = np.full((len(adapters), len(all_scopes)), np.nan)
    for i, adapter in enumerate(adapters):
        for j, scope in enumerate(all_scopes):
            if scope in adapter_scope[adapter]:
                matrix[i][j] = adapter_scope[adapter][scope]

    return adapters, all_scopes, matrix


def d2_kendalls_w(matrix: np.ndarray) -> float:
    """Compute Kendall's W (coefficient of concordance).

    Measures how consistently adapters are ranked across scopes.
    """
    # matrix is adapters × scopes
    n_adapters, n_scopes = matrix.shape
    if n_scopes < 2 or n_adapters < 2:
        return 0.0

    # Rank adapters within each scope (column)
    try:
        from scipy.stats import rankdata
        ranks = np.zeros_like(matrix)
        for j in range(n_scopes):
            col = matrix[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                ranks[valid, j] = rankdata(-col[valid])  # Higher score = lower rank
            elif valid.sum() == 1:
                ranks[valid, j] = 1.0
    except ImportError:
        # Simple ranking fallback
        ranks = np.zeros_like(matrix)
        for j in range(n_scopes):
            col = matrix[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                order = np.argsort(-col[valid])
                r = np.empty_like(order, dtype=float)
                r[order] = np.arange(1, valid.sum() + 1, dtype=float)
                ranks[valid, j] = r

    # W = 12 * S / (k^2 * (n^3 - n))
    # where S = sum of squared deviations of rank sums from mean rank sum
    k = n_scopes  # number of raters (scopes)
    n = n_adapters  # number of items (adapters)
    rank_sums = ranks.sum(axis=1)
    mean_rank_sum = rank_sums.mean()
    s = np.sum((rank_sums - mean_rank_sum) ** 2)
    w = 12 * s / (k ** 2 * (n ** 3 - n))
    return float(min(w, 1.0))


# ---------------------------------------------------------------------------
# D3: Distractor Effect Analysis
# ---------------------------------------------------------------------------


def d3_distractor_effect(
    phase5_data: dict[str, dict],
    phase12_state: dict | None = None,
    metric_name: str = "answer_quality",
    budget_filter: str | None = None,
) -> list[dict]:
    """Compare Phase 5 (with distractors) vs earlier phases (without).

    If phase12_state is None, attempts to load from older state files.
    Returns [{adapter, scope, phase5_score, phase12_score, delta}].
    """
    results = []

    # Phase 5 scores
    p5_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for label, metrics in phase5_data.items():
        parsed = parse_label(label)
        if not parsed:
            continue
        adapter, scope, budget = parsed
        if budget_filter and budget != budget_filter:
            continue
        val = metrics.get(metric_name)
        if val is not None:
            p5_scores[adapter][scope] = val

    # Phase 1-2 scores (from older state or separate data)
    p12_scores: dict[str, dict[str, float]] = defaultdict(dict)
    if phase12_state:
        for label, info in phase12_state.items():
            if not isinstance(info, dict) or info.get("status") != "scored":
                continue
            parsed = parse_label(label)
            if not parsed:
                continue
            adapter, scope, budget = parsed
            if budget_filter and budget != budget_filter:
                continue
            run_id = info.get("run_id")
            if not run_id:
                continue
            scorecard = load_scorecard(run_id)
            if scorecard:
                for m in scorecard.get("metrics", []):
                    if isinstance(m, dict) and m["name"] == metric_name:
                        p12_scores[adapter][scope] = m["value"]
                        break

    for adapter in sorted(set(p5_scores) & set(p12_scores)):
        for scope in sorted(set(p5_scores[adapter]) & set(p12_scores[adapter])):
            p5 = p5_scores[adapter][scope]
            p12 = p12_scores[adapter][scope]
            results.append({
                "adapter": adapter,
                "scope": scope,
                "phase5_score": round(p5, 4),
                "phase12_score": round(p12, 4),
                "delta": round(p5 - p12, 4),
            })

    return results


# ---------------------------------------------------------------------------
# D4: Question-Type Analysis
# ---------------------------------------------------------------------------


def d4_question_type_analysis(
    nba_per_question: dict[str, list[dict]],
    per_question_data: dict[str, list[dict]],
    budget_filter: str | None = None,
) -> dict[str, dict]:
    """Analyze per-question-type performance across adapters.

    Returns {question_type: {adapter: {mean_nba, n_questions}}}.
    """
    # Build question_id -> question_type mapping from per_question_data
    qid_to_type: dict[str, str] = {}
    for label, questions in per_question_data.items():
        for q in questions:
            qid = q.get("question_id", "")
            qtype = q.get("question_type", "")
            if qid and qtype:
                qid_to_type[qid] = qtype

    # Group NBA by adapter × question_type
    type_adapter_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for label, pq_data in nba_per_question.items():
        parsed = parse_label(label)
        if not parsed:
            continue
        adapter, scope, budget = parsed
        if budget_filter and budget != budget_filter:
            continue

        for pq in pq_data:
            qid = pq.get("question_id", "")
            win_rate = pq.get("win_rate")
            if win_rate is None:
                continue
            qtype = qid_to_type.get(qid, "")
            if not qtype:
                # Infer from question_id suffix
                parts = qid.rsplit("_", 1)
                qtype = parts[-1] if len(parts) > 1 else "unknown"
            type_adapter_values[qtype][adapter].append(win_rate)

    results = {}
    for qtype in sorted(type_adapter_values):
        results[qtype] = {}
        for adapter in sorted(type_adapter_values[qtype]):
            vals = type_adapter_values[qtype][adapter]
            results[qtype][adapter] = {
                "mean_nba": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "n_questions": len(vals),
            }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

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
        return plt, mticker
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return None, None


def plot_d1_ci_bars(d1_results: dict, output_path: Path, metric_name: str = "answer_quality") -> None:
    """Horizontal bar chart with bootstrap CIs."""
    plt, _ = _setup_matplotlib()
    if plt is None:
        return

    adapters = sorted(d1_results.keys(), key=lambda a: d1_results[a]["mean"], reverse=True)
    means = [d1_results[a]["mean"] for a in adapters]
    ci_low = [d1_results[a]["mean"] - d1_results[a]["ci_low"] for a in adapters]
    ci_high = [d1_results[a]["ci_high"] - d1_results[a]["mean"] for a in adapters]

    fig, ax = plt.subplots(figsize=(8, max(4, len(adapters) * 0.5)))
    y = range(len(adapters))
    colors = ["#607D8B" if a == "null" else "#2196F3" for a in adapters]
    ax.barh(y, means, xerr=[ci_low, ci_high], color=colors, capsize=4, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(adapters)
    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Adapter {metric_name.replace('_', ' ').title()} (95% Bootstrap CI)")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.4, label="parity")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    log.info("D1 CI bars saved to %s", output_path)


def plot_d2_heatmap(adapters: list, scopes: list, matrix: np.ndarray, output_path: Path) -> None:
    """Adapter × scope heatmap."""
    plt, _ = _setup_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(scopes) * 1.2), max(4, len(adapters) * 0.6)))

    try:
        import seaborn as sns
        sns.heatmap(
            matrix, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=[f"Scope {s}" for s in scopes],
            yticklabels=adapters,
            ax=ax, vmin=0, vmax=1, linewidths=0.5,
        )
    except ImportError:
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(scopes)))
        ax.set_xticklabels([f"Scope {s}" for s in scopes])
        ax.set_yticks(range(len(adapters)))
        ax.set_yticklabels(adapters)
        for i in range(len(adapters)):
            for j in range(len(scopes)):
                val = matrix[i][j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)

    ax.set_title("Answer Quality: Adapter × Scope")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    log.info("D2 heatmap saved to %s", output_path)


def plot_d4_question_types(d4_results: dict, output_path: Path) -> None:
    """Grouped bar chart: adapters per question type."""
    plt, _ = _setup_matplotlib()
    if plt is None:
        return

    qtypes = sorted(d4_results.keys())
    all_adapters = sorted(set(
        a for qt in d4_results.values() for a in qt.keys()
    ))

    x = np.arange(len(qtypes))
    width = 0.8 / max(len(all_adapters), 1)
    colors = [
        "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5",
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(qtypes) * 1.5), 6))
    for i, adapter in enumerate(all_adapters):
        vals = [d4_results[qt].get(adapter, {}).get("mean_nba", 0) for qt in qtypes]
        ax.bar(x + i * width, vals, width, label=adapter,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (len(all_adapters) - 1) / 2)
    ax.set_xticklabels(qtypes, rotation=45, ha="right")
    ax.set_ylabel("Mean NBA (win rate)")
    ax.set_title("Naive Baseline Advantage by Question Type")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    log.info("D4 question-type chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# LaTeX Table Generation
# ---------------------------------------------------------------------------


def latex_d1_table(d1_results: dict, pairwise: list[dict], metric_name: str) -> str:
    """Generate LaTeX table for D1 results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + metric_name.replace("_", " ").title() + r" across 6 scopes (95\% Bootstrap CI)}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Adapter & Mean & 95\% CI & $N$ & vs.\ null \\",
        r"\midrule",
    ]
    adapters = sorted(d1_results.keys(), key=lambda a: d1_results[a]["mean"], reverse=True)
    null_tests = {t["adapter_a"] if t["adapter_b"] == "null" else t["adapter_b"]: t
                  for t in pairwise if "null" in (t["adapter_a"], t["adapter_b"])}

    for adapter in adapters:
        r = d1_results[adapter]
        ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        sig = ""
        if adapter != "null" and adapter in null_tests:
            sig = null_tests[adapter].get("sig", "")
        line = f"  {adapter} & {r['mean']:.3f} & {ci_str} & {r['n_scopes']} & {sig} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def latex_d2_heatmap_table(adapters: list, scopes: list, matrix: np.ndarray) -> str:
    """Generate LaTeX table for D2 heatmap data."""
    cols = "l" + "c" * len(scopes) + "c"
    header = "Adapter & " + " & ".join(f"S{s}" for s in scopes) + r" & Mean \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Answer Quality: Adapter $\times$ Scope}",
        r"\label{tab:heatmap}",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for i, adapter in enumerate(adapters):
        vals = []
        for j in range(len(scopes)):
            v = matrix[i][j]
            vals.append(f"{v:.3f}" if not np.isnan(v) else "---")
        row_mean = np.nanmean(matrix[i])
        line = f"  {adapter} & " + " & ".join(vals) + f" & {row_mean:.3f} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Publication-ready LENS benchmark analysis")
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--budget", default="8k", help="Budget to analyze (default: 8k)")
    parser.add_argument("--metric", default="answer_quality", help="Primary metric (default: answer_quality)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    state = load_state(args.state_file)
    metric_data = collect_metric_data(state)

    if not metric_data:
        log.error("No scored runs found in state file")
        sys.exit(1)

    # Output directories
    out_base = args.output_dir or (PROJECT_DIR / "results")
    fig_dir = out_base / "figures"
    tab_dir = out_base / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loaded %d scored runs", len(metric_data))
    log.info("Analyzing metric=%s, budget=%s", args.metric, args.budget)

    # ===================================================================
    # D1: Bootstrap CIs
    # ===================================================================
    print(f"\n{'=' * 75}")
    print("D1: Bootstrap CIs on %s (per adapter, across scopes)" % args.metric)
    print(f"{'=' * 75}")

    d1 = d1_bootstrap_analysis(metric_data, args.metric, budget_filter=args.budget)
    print(f"{'Adapter':<25} {'Mean':>8} {'CI Low':>8} {'CI High':>8} {'N':>4}")
    print("-" * 60)
    for adapter in sorted(d1, key=lambda a: d1[a]["mean"], reverse=True):
        r = d1[adapter]
        print(f"{adapter:<25} {r['mean']:>8.4f} {r['ci_low']:>8.4f} {r['ci_high']:>8.4f} {r['n_scopes']:>4}")

    # Pairwise tests
    pairwise = d1_pairwise_tests(metric_data, args.metric, budget_filter=args.budget)
    if pairwise:
        print(f"\n{'Comparison':<45} {'p':>8} {'r':>8} {'Sig':>5}")
        print("-" * 70)
        for t in sorted(pairwise, key=lambda t: t["p_value"]):
            comp = f"{t['adapter_a']} vs {t['adapter_b']}"
            print(f"{comp:<45} {t['p_value']:>8.4f} {t['effect_size']:>8.4f} {t['sig']:>5}")

    # ===================================================================
    # D2: Cross-Scope Consistency
    # ===================================================================
    print(f"\n{'=' * 75}")
    print("D2: Cross-Scope Consistency")
    print(f"{'=' * 75}")

    adapters, scopes, matrix = d2_heatmap_matrix(metric_data, args.metric, budget_filter=args.budget)
    if len(adapters) > 0 and len(scopes) > 1:
        w = d2_kendalls_w(matrix)
        print(f"Kendall's W: {w:.4f}")
        if w > 0.7:
            print("  -> Strong concordance: rankings are consistent across scopes")
        elif w > 0.5:
            print("  -> Moderate concordance: rankings mostly consistent")
        else:
            print("  -> Weak concordance: rankings vary substantially across scopes")

        print(f"\n{'Adapter':<25}", end="")
        for s in scopes:
            print(f"  S{s:>3}", end="")
        print(f"  {'Mean':>6}")
        print("-" * (25 + 6 * len(scopes) + 8))
        for i, adapter in enumerate(adapters):
            print(f"{adapter:<25}", end="")
            for j in range(len(scopes)):
                v = matrix[i][j]
                print(f"  {v:>5.3f}" if not np.isnan(v) else "    ---", end="")
            print(f"  {np.nanmean(matrix[i]):>6.3f}")

    # ===================================================================
    # D4: Question-Type Analysis
    # ===================================================================
    print(f"\n{'=' * 75}")
    print("D4: Question-Type Analysis")
    print(f"{'=' * 75}")

    nba_pq = collect_nba_per_question(state)
    pq_data = collect_per_question_data(state)
    d4 = d4_question_type_analysis(nba_pq, pq_data, budget_filter=args.budget)

    if d4:
        all_adapters = sorted(set(a for qt in d4.values() for a in qt.keys()))
        # Header
        print(f"{'Type':<20}", end="")
        for a in all_adapters:
            print(f"  {a[:12]:>12}", end="")
        print()
        print("-" * (20 + 14 * len(all_adapters)))
        for qtype in sorted(d4):
            print(f"{qtype:<20}", end="")
            for a in all_adapters:
                v = d4[qtype].get(a, {}).get("mean_nba", 0)
                print(f"  {v:>12.3f}", end="")
            print()

        # Discrimination analysis: which question types have highest variance across adapters
        print("\nQuestion type discrimination (std across adapters):")
        for qtype in sorted(d4):
            vals = [d4[qtype][a]["mean_nba"] for a in d4[qtype]]
            if len(vals) >= 2:
                std = float(np.std(vals))
                print(f"  {qtype:<20} std={std:.4f}  range=[{min(vals):.3f}, {max(vals):.3f}]")

    # ===================================================================
    # Plots
    # ===================================================================
    if not args.no_plot:
        plot_d1_ci_bars(d1, fig_dir / "d1_answer_quality_ci.png", args.metric)
        if len(adapters) > 0 and len(scopes) > 1:
            plot_d2_heatmap(adapters, scopes, matrix, fig_dir / "d2_heatmap.png")
        if d4:
            plot_d4_question_types(d4, fig_dir / "d4_question_types.png")

    # ===================================================================
    # LaTeX Tables
    # ===================================================================
    latex_d1 = latex_d1_table(d1, pairwise, args.metric)
    (tab_dir / "d1_main_results.tex").write_text(latex_d1)
    log.info("LaTeX table saved to %s", tab_dir / "d1_main_results.tex")

    if len(adapters) > 0 and len(scopes) > 1:
        latex_d2 = latex_d2_heatmap_table(adapters, scopes, matrix)
        (tab_dir / "d2_heatmap.tex").write_text(latex_d2)
        log.info("LaTeX table saved to %s", tab_dir / "d2_heatmap.tex")

    # ===================================================================
    # Export JSON
    # ===================================================================
    export = {
        "d1_bootstrap_cis": d1,
        "d1_pairwise_tests": pairwise,
        "d2_kendalls_w": w if len(scopes) > 1 else None,
        "d4_question_types": d4,
        "config": {
            "metric": args.metric,
            "budget_filter": args.budget,
            "state_file": str(args.state_file),
        },
    }
    export_path = out_base / "publication_analysis.json"
    export_path.write_text(json.dumps(export, indent=2) + "\n")
    log.info("Analysis exported to %s", export_path)

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    if d1:
        best = max(d1, key=lambda a: d1[a]["mean"])
        worst = min(d1, key=lambda a: d1[a]["mean"])
        print(f"Best adapter: {best} ({d1[best]['mean']:.4f} [{d1[best]['ci_low']:.4f}, {d1[best]['ci_high']:.4f}])")
        print(f"Worst adapter: {worst} ({d1[worst]['mean']:.4f})")
        null_score = d1.get("null", {}).get("mean", 0)
        print(f"Null baseline: {null_score:.4f}")
        above_null = [a for a in d1 if a != "null" and d1[a]["mean"] > null_score]
        print(f"Adapters above null: {', '.join(above_null) if above_null else 'none'}")

    sig_tests = [t for t in pairwise if t["sig"]]
    print(f"Significant pairwise differences (p<0.05): {len(sig_tests)}/{len(pairwise)}")
    print(f"\nOutputs: {fig_dir}/, {tab_dir}/, {export_path}")


if __name__ == "__main__":
    main()
