#!/usr/bin/env python3
"""
LENS V2 Grid Ablation — Analysis & Visualization

Produces 8 publication-quality figures and statistical analysis from the
10-scope x 7-policy x M=3 grid study.

Usage:
    python analyze_grid.py [--results-dir results/] [--output-dir brief/] [--scopes N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Color palette — cyberpunk / dark theme
# ---------------------------------------------------------------------------
BG_DARK = "#0a0a0f"
BG_PANEL = "#1a1a2e"
PRIMARY = "#06b6d4"      # cyan
ACCENT = "#ff00ff"        # magenta
TEXT_LIGHT = "#e0e0e8"
TEXT_DIM = "#8888aa"
GRID_COLOR = "#2a2a3e"

# Cyan-to-magenta colormap for heatmaps
CMAP_CYBER = mcolors.LinearSegmentedColormap.from_list(
    "cyber", [BG_PANEL, PRIMARY, "#a855f7", ACCENT], N=256
)

# Policy display names and canonical order
POLICY_ORDER = [
    "null",
    "policy_base",
    "policy_summary",
    "policy_core_maintained",
    "policy_core_structured",
    "policy_core",
    "policy_core_faceted",
]
POLICY_LABELS = {
    "null": "Null",
    "policy_base": "Base",
    "policy_summary": "Summary",
    "policy_core": "Core",
    "policy_core_structured": "Core-Structured",
    "policy_core_maintained": "Core-Maintained",
    "policy_core_faceted": "Core-Faceted",
}


# ---------------------------------------------------------------------------
# Dark theme setup
# ---------------------------------------------------------------------------
def apply_dark_theme():
    """Apply cyberpunk-inspired dark matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_LIGHT,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "text.color": TEXT_LIGHT,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "legend.facecolor": BG_PANEL,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_LIGHT,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "savefig.facecolor": BG_DARK,
        "savefig.edgecolor": BG_DARK,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_scores(results_dir: Path) -> list[dict]:
    """Load all claude_scores_*.jsonl files from results_dir."""
    records: list[dict] = []
    score_files = sorted(results_dir.glob("claude_scores_*.jsonl"))
    if not score_files:
        print(f"ERROR: No claude_scores_*.jsonl files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    for fpath in score_files:
        print(f"  Loading {fpath.name} ...", end=" ")
        count = 0
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Extract replicate_id from run_id: run-...-r01-grid-...
                m = re.search(r"-(r\d+)-", rec.get("run_id", ""))
                rec["replicate_id"] = m.group(1) if m else "r01"
                records.append(rec)
                count += 1
        print(f"{count} records")

    print(f"  Total: {len(records)} score records")
    return records


def load_summary(results_dir: Path) -> dict:
    """Load grid_summary_full.json if present."""
    fpath = results_dir / "grid_summary_full.json"
    if fpath.exists():
        with open(fpath) as f:
            return json.load(f)
    return {}


def build_arrays(records: list[dict]):
    """Build structured numpy arrays from score records.

    Returns:
        scopes: sorted list of scope_ids
        policies: list of policy_ids in canonical order
        data: dict  (scope, policy) -> list[float] of fact_f1 values
        per_replicate: dict (scope, policy, replicate) -> list[float]
    """
    data: dict[tuple[str, str], list[float]] = {}
    per_replicate: dict[tuple[str, str, str], list[float]] = {}

    all_scopes = set()
    all_policies = set()

    for rec in records:
        scope = rec["scope_id"]
        policy = rec["policy_id"]
        rep = rec["replicate_id"]
        f1 = rec.get("fact_f1", 0.0)
        if f1 is None:
            f1 = 0.0

        all_scopes.add(scope)
        all_policies.add(policy)

        data.setdefault((scope, policy), []).append(f1)
        per_replicate.setdefault((scope, policy, rep), []).append(f1)

    # Sort scopes by mean F1 across non-null policies (ascending = hardest first)
    scope_means = {}
    for scope in all_scopes:
        vals = []
        for policy in all_policies:
            if policy == "null":
                continue
            vals.extend(data.get((scope, policy), []))
        scope_means[scope] = np.mean(vals) if vals else 0.0

    scopes = sorted(all_scopes, key=lambda s: scope_means[s])

    # Canonical policy order, filtered to what's in data
    policies = [p for p in POLICY_ORDER if p in all_policies]
    # Add any policies not in the canonical list
    for p in sorted(all_policies):
        if p not in policies:
            policies.append(p)

    return scopes, policies, data, per_replicate


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, ci: float = 0.95,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(np.mean(values)), float(lo), float(hi)


def policy_f1_array(data: dict, scopes: list[str], policy: str) -> np.ndarray:
    """Flat array of all fact_f1 values for a given policy across scopes."""
    vals = []
    for scope in scopes:
        vals.extend(data.get((scope, policy), []))
    return np.asarray(vals, dtype=float)


def paired_policy_scores(data: dict, scopes: list[str],
                         p1: str, p2: str) -> tuple[np.ndarray, np.ndarray]:
    """Build paired score arrays (per question, matched by scope).

    For unpaired tests we aggregate per-scope means and pair those.
    """
    a, b = [], []
    for scope in scopes:
        v1 = data.get((scope, p1), [])
        v2 = data.get((scope, p2), [])
        if v1 and v2:
            a.append(np.mean(v1))
            b.append(np.mean(v2))
    return np.asarray(a), np.asarray(b)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------
def gradient_colors(n: int) -> list[str]:
    """Return n colors along the cyan→magenta gradient."""
    cmap = mcolors.LinearSegmentedColormap.from_list("cg", [PRIMARY, ACCENT], N=n)
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def fig1_heatmap(scopes, policies, data, fig_dir):
    """Scopes (rows) x Policies (cols) heatmap of mean Fact F1."""
    nonnull = [p for p in policies if p != "null"]
    matrix = np.full((len(scopes), len(nonnull)), np.nan)
    for i, scope in enumerate(scopes):
        for j, policy in enumerate(nonnull):
            vals = data.get((scope, policy), [])
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=CMAP_CYBER, aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(scopes)):
        for j in range(len(nonnull)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = TEXT_LIGHT if val < 0.5 else BG_DARK
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(nonnull)))
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in nonnull], rotation=35, ha="right")
    ax.set_yticks(range(len(scopes)))
    ax.set_yticklabels(scopes)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Scope (hardest → easiest)")
    ax.set_title("Fact F1 — Scopes × Policies", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Fact F1")
    cbar.ax.yaxis.label.set_color(TEXT_LIGHT)
    cbar.ax.tick_params(colors=TEXT_DIM)

    fig.tight_layout()
    _save(fig, fig_dir / "fig1_heatmap")
    plt.close(fig)


def fig2_policy_ranking(scopes, policies, data, fig_dir):
    """Horizontal bar chart of mean F1 per policy with bootstrap 95% CI."""
    nonnull = [p for p in policies if p != "null"]
    means, ci_lo, ci_hi = [], [], []
    rng = np.random.default_rng(42)
    for p in nonnull:
        arr = policy_f1_array(data, scopes, p)
        m, lo, hi = bootstrap_ci(arr, rng=rng)
        means.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)

    # Sort by mean descending
    order = np.argsort(means)[::-1]
    labels = [POLICY_LABELS.get(nonnull[i], nonnull[i]) for i in order]
    vals = [means[i] for i in order]
    errs_lo = [ci_lo[i] for i in order]
    errs_hi = [ci_hi[i] for i in order]
    colors = gradient_colors(len(order))

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(order))
    bars = ax.barh(y_pos, vals, xerr=[errs_lo, errs_hi],
                   color=colors, edgecolor=TEXT_DIM, linewidth=0.5,
                   capsize=4, error_kw={"ecolor": TEXT_LIGHT, "linewidth": 1.2})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Fact F1")
    ax.set_title("Policy Ranking — Mean F1 with 95% Bootstrap CI", fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, min(max(vals) * 1.3, 1.0))

    # Value labels
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", color=TEXT_LIGHT, fontsize=10)

    fig.tight_layout()
    _save(fig, fig_dir / "fig2_policy_ranking")
    plt.close(fig)


def fig3_scope_difficulty(scopes, policies, data, fig_dir):
    """Vertical bar chart of scope difficulty (mean F1 across non-null policies)."""
    nonnull = [p for p in policies if p != "null"]
    means, stds = [], []
    for scope in scopes:
        policy_means = []
        for p in nonnull:
            vals = data.get((scope, p), [])
            if vals:
                policy_means.append(np.mean(vals))
        means.append(np.mean(policy_means) if policy_means else 0.0)
        stds.append(np.std(policy_means, ddof=1) if len(policy_means) > 1 else 0.0)

    colors = gradient_colors(len(scopes))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scopes))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor=TEXT_DIM,
           linewidth=0.5, capsize=4, error_kw={"ecolor": TEXT_LIGHT, "linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(scopes, rotation=35, ha="right")
    ax.set_ylabel("Mean Fact F1 (non-null policies)")
    ax.set_title("Scope Difficulty — Hardest to Easiest", fontweight="bold")
    ax.set_ylim(0, 1.0)

    for i, v in enumerate(means):
        ax.text(i, v + stds[i] + 0.015, f"{v:.3f}", ha="center", color=TEXT_LIGHT, fontsize=9)

    fig.tight_layout()
    _save(fig, fig_dir / "fig3_scope_difficulty")
    plt.close(fig)


def fig4_ablation_waterfall(scopes, policies, data, fig_dir):
    """Waterfall chart: null → base → core → faceted incremental gains."""
    chain = ["null", "policy_base", "policy_core", "policy_core_faceted"]
    chain = [p for p in chain if p in policies]

    means = []
    for p in chain:
        arr = policy_f1_array(data, scopes, p)
        means.append(np.mean(arr) if len(arr) else 0.0)

    deltas = [means[0]] + [means[i] - means[i - 1] for i in range(1, len(means))]
    bottoms = [0.0]
    for i in range(1, len(deltas)):
        bottoms.append(bottoms[-1] + deltas[i - 1])

    labels = [POLICY_LABELS.get(p, p) for p in chain]
    colors_bar = []
    for d in deltas:
        if d >= 0:
            colors_bar.append(PRIMARY)
        else:
            colors_bar.append("#ff4444")

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(chain))
    bars = ax.bar(x, deltas, bottom=bottoms, color=colors_bar, edgecolor=TEXT_DIM,
                  linewidth=0.8, width=0.6)

    # Connector lines
    for i in range(len(chain) - 1):
        top = bottoms[i] + deltas[i]
        ax.plot([i + 0.3, i + 0.7], [top, top], color=TEXT_DIM, linewidth=1, linestyle="--")

    # Delta labels
    for i, (d, b) in enumerate(zip(deltas, bottoms)):
        y_text = b + d + (0.015 if d >= 0 else -0.03)
        prefix = "+" if d > 0 and i > 0 else ""
        ax.text(i, y_text, f"{prefix}{d:.3f}", ha="center", va="bottom" if d >= 0 else "top",
                color=TEXT_LIGHT, fontsize=11, fontweight="bold")

    # Total line
    total = means[-1]
    ax.axhline(total, color=ACCENT, linewidth=1, linestyle=":", alpha=0.7)
    ax.text(len(chain) - 0.5, total + 0.01, f"Total: {total:.3f}",
            color=ACCENT, fontsize=10, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fact F1")
    ax.set_title("Ablation Waterfall — Null → Base → Core → Faceted", fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)

    fig.tight_layout()
    _save(fig, fig_dir / "fig4_ablation_waterfall")
    plt.close(fig)


def fig5_consolidation_effect(scopes, policies, data, fig_dir):
    """Grouped bar chart comparing consolidation policies vs baseline per scope."""
    compare_policies = ["policy_base", "policy_summary", "policy_core_maintained", "policy_core_structured"]
    compare_policies = [p for p in compare_policies if p in policies]
    n_policies = len(compare_policies)
    n_scopes = len(scopes)

    colors = gradient_colors(n_policies)
    width = 0.8 / n_policies

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_scopes)

    for j, policy in enumerate(compare_policies):
        vals = []
        for scope in scopes:
            v = data.get((scope, policy), [])
            vals.append(np.mean(v) if v else 0.0)
        offset = (j - n_policies / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=POLICY_LABELS.get(policy, policy),
               color=colors[j], edgecolor=TEXT_DIM, linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(scopes, rotation=35, ha="right")
    ax.set_ylabel("Mean Fact F1")
    ax.set_title("Consolidation Strategies vs Base — Per Scope", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    fig.tight_layout()
    _save(fig, fig_dir / "fig5_consolidation_effect")
    plt.close(fig)


def fig6_variance(scopes, policies, data, fig_dir):
    """Box + jitter plots of F1 distribution per policy."""
    nonnull = [p for p in policies if p != "null"]
    plot_data = []
    labels = []
    for p in nonnull:
        arr = policy_f1_array(data, scopes, p)
        plot_data.append(arr)
        labels.append(POLICY_LABELS.get(p, p))

    # Sort by median descending
    medians = [np.median(a) for a in plot_data]
    order = np.argsort(medians)[::-1]
    plot_data = [plot_data[i] for i in order]
    labels = [labels[i] for i in order]

    colors = gradient_colors(len(nonnull))

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(plot_data, vert=True, patch_artist=True, widths=0.5,
                    showfliers=False,
                    medianprops=dict(color=TEXT_LIGHT, linewidth=2),
                    whiskerprops=dict(color=TEXT_DIM),
                    capprops=dict(color=TEXT_DIM))

    for i, (patch, c) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
        patch.set_edgecolor(TEXT_LIGHT)

    # Jittered scatter
    rng = np.random.default_rng(42)
    for i, arr in enumerate(plot_data):
        jitter = rng.uniform(-0.15, 0.15, size=len(arr))
        ax.scatter(np.full_like(arr, i + 1) + jitter, arr,
                   color=colors[i], alpha=0.5, s=12, edgecolors="none", zorder=5)

    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Fact F1")
    ax.set_title("Score Distribution per Policy — M=3 × All Scopes", fontweight="bold")
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    _save(fig, fig_dir / "fig6_variance")
    plt.close(fig)


def fig7_scope_clusters(scopes, policies, data, fig_dir):
    """Radar/spider chart of per-scope performance across non-null policies."""
    nonnull = [p for p in policies if p != "null"]
    n_policies = len(nonnull)

    # Build matrix: scopes x policies  (normalized 0-1 within each policy)
    matrix = np.zeros((len(scopes), n_policies))
    for j, p in enumerate(nonnull):
        col = []
        for scope in scopes:
            v = data.get((scope, p), [])
            col.append(np.mean(v) if v else 0.0)
        col = np.array(col)
        # Normalize to 0-1 (min-max within policy)
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            matrix[:, j] = (col - cmin) / (cmax - cmin)
        else:
            matrix[:, j] = 0.5

    angles = np.linspace(0, 2 * np.pi, n_policies, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    scope_colors = gradient_colors(len(scopes))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor(BG_PANEL)

    for i, scope in enumerate(scopes):
        values = matrix[i].tolist() + [matrix[i, 0]]
        ax.plot(angles, values, "-o", markersize=4, label=scope,
                color=scope_colors[i], linewidth=1.5, alpha=0.85)
        ax.fill(angles, values, color=scope_colors[i], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in nonnull], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=TEXT_DIM)
    ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.5)
    ax.xaxis.grid(True, color=GRID_COLOR, alpha=0.5)
    ax.spines["polar"].set_color(GRID_COLOR)

    ax.set_title("Scope Performance Profiles (normalized 0-1)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8, framealpha=0.8)

    fig.tight_layout()
    _save(fig, fig_dir / "fig7_scope_clusters")
    plt.close(fig)


def fig8_statistical_tests(scopes, policies, data, fig_dir):
    """Heatmap of pairwise Wilcoxon signed-rank p-values (upper triangle)."""
    nonnull = [p for p in policies if p != "null"]
    n = len(nonnull)
    pval_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = paired_policy_scores(data, scopes, nonnull[i], nonnull[j])
            if len(a) >= 5:
                try:
                    stat, pval = stats.wilcoxon(a, b)
                    pval_matrix[i, j] = pval
                except ValueError:
                    # All differences are zero
                    pval_matrix[i, j] = 1.0
            else:
                pval_matrix[i, j] = np.nan

    labels = [POLICY_LABELS.get(p, p) for p in nonnull]

    fig, ax = plt.subplots(figsize=(9, 8))

    # Mask lower triangle
    mask = np.tril(np.ones_like(pval_matrix, dtype=bool))
    np.fill_diagonal(mask, True)

    # Use a diverging colormap anchored at 0.05
    cmap_sig = mcolors.LinearSegmentedColormap.from_list(
        "sig", [ACCENT, "#a855f7", PRIMARY, BG_PANEL], N=256
    )

    sns.heatmap(pval_matrix, mask=mask, ax=ax, cmap=cmap_sig,
                vmin=0, vmax=0.2,
                annot=True, fmt=".3f", annot_kws={"size": 9, "color": TEXT_LIGHT},
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "p-value (Wilcoxon)", "shrink": 0.7},
                linewidths=0.5, linecolor=GRID_COLOR)

    # Significance markers
    for i in range(n):
        for j in range(i + 1, n):
            pv = pval_matrix[i, j]
            if not np.isnan(pv):
                bonf = pv * (n * (n - 1) / 2)  # Bonferroni
                if bonf < 0.001:
                    marker = "***"
                elif bonf < 0.01:
                    marker = "**"
                elif bonf < 0.05:
                    marker = "*"
                else:
                    marker = ""
                if marker:
                    ax.text(j + 0.5, i + 0.82, marker, ha="center", va="center",
                            color=ACCENT, fontsize=10, fontweight="bold")

    ax.set_title("Pairwise Wilcoxon Signed-Rank p-values\n(* p<.05, ** p<.01, *** p<.001 Bonferroni-corrected)",
                 fontweight="bold", fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_LIGHT)
    cbar.ax.tick_params(colors=TEXT_DIM)

    fig.tight_layout()
    _save(fig, fig_dir / "fig8_statistical_tests")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
def run_statistics(scopes, policies, data) -> dict[str, Any]:
    """Run all statistical tests, return structured results."""
    nonnull = [p for p in policies if p != "null"]
    rng = np.random.default_rng(42)
    results: dict[str, Any] = {}

    # --- Bootstrap CIs ---
    boot_cis = {}
    for p in policies:
        arr = policy_f1_array(data, scopes, p)
        mean, lo, hi = bootstrap_ci(arr, rng=rng)
        boot_cis[p] = {"mean": round(mean, 4), "ci_95_low": round(lo, 4),
                        "ci_95_high": round(hi, 4), "n": len(arr)}
    results["bootstrap_ci"] = boot_cis

    # --- Friedman test ---
    # Build per-scope mean matrix (scopes x policies)
    if len(nonnull) >= 3 and len(scopes) >= 3:
        scope_policy_means = []
        for p in nonnull:
            col = []
            for scope in scopes:
                v = data.get((scope, p), [])
                col.append(np.mean(v) if v else 0.0)
            scope_policy_means.append(col)
        # Friedman expects: each row = a block (scope), each col = a treatment (policy)
        matrix = np.array(scope_policy_means).T  # scopes x policies
        try:
            stat, pval = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
            results["friedman"] = {"statistic": round(float(stat), 4),
                                   "p_value": round(float(pval), 6),
                                   "n_scopes": len(scopes), "n_policies": len(nonnull)}
        except Exception as e:
            results["friedman"] = {"error": str(e)}
    else:
        results["friedman"] = {"error": "insufficient groups"}

    # --- Pairwise Wilcoxon with Bonferroni ---
    n_comp = len(nonnull) * (len(nonnull) - 1) // 2
    pairwise = {}
    for i in range(len(nonnull)):
        for j in range(i + 1, len(nonnull)):
            a, b = paired_policy_scores(data, scopes, nonnull[i], nonnull[j])
            key = f"{nonnull[i]} vs {nonnull[j]}"
            if len(a) >= 5:
                try:
                    stat, pval = stats.wilcoxon(a, b)
                    bonf_p = min(pval * n_comp, 1.0)
                    pairwise[key] = {
                        "statistic": round(float(stat), 4),
                        "p_value_raw": round(float(pval), 6),
                        "p_value_bonferroni": round(float(bonf_p), 6),
                        "significant_05": bonf_p < 0.05,
                        "n_pairs": len(a)
                    }
                except ValueError:
                    pairwise[key] = {"error": "zero differences", "n_pairs": len(a)}
            else:
                pairwise[key] = {"error": "insufficient pairs", "n_pairs": len(a)}
    results["pairwise_wilcoxon"] = pairwise

    # --- Kendall's W ---
    if len(nonnull) >= 3 and len(scopes) >= 3:
        # Rankings per scope
        matrix_raw = np.zeros((len(scopes), len(nonnull)))
        for j, p in enumerate(nonnull):
            for i, scope in enumerate(scopes):
                v = data.get((scope, p), [])
                matrix_raw[i, j] = np.mean(v) if v else 0.0

        # Rank within each scope (row)
        from scipy.stats import rankdata
        rank_matrix = np.zeros_like(matrix_raw)
        for i in range(len(scopes)):
            rank_matrix[i] = rankdata(matrix_raw[i])

        k = len(nonnull)  # number of judges (policies)
        n = len(scopes)   # number of objects (scopes)
        rank_sums = rank_matrix.sum(axis=0)
        mean_rank_sum = n * (k + 1) / 2
        ss = np.sum((rank_sums - mean_rank_sum) ** 2)
        # But Kendall's W: judges=scopes, objects=policies
        # Actually: W = 12 * S / (k^2 * (n^3 - n))
        # where k=judges (scopes), n=objects (policies)
        # S = sum of squared deviations of rank sums from their mean
        # Re-orient: each scope ranks the policies
        rank_matrix2 = np.zeros_like(matrix_raw)
        for i in range(len(scopes)):
            rank_matrix2[i] = rankdata(matrix_raw[i])
        R = rank_matrix2.sum(axis=0)  # sum of ranks per policy
        R_mean = R.mean()
        S = np.sum((R - R_mean) ** 2)
        k_judges = len(scopes)
        n_objects = len(nonnull)
        W = (12 * S) / (k_judges**2 * (n_objects**3 - n_objects))

        results["kendalls_w"] = {
            "W": round(float(W), 4),
            "n_scopes": len(scopes),
            "n_policies": len(nonnull),
            "interpretation": (
                "strong" if W > 0.7 else "moderate" if W > 0.4 else "weak"
            )
        }
    else:
        results["kendalls_w"] = {"error": "insufficient data"}

    # --- Cohen's d for key comparisons ---
    key_comparisons = [
        ("policy_core_faceted", "policy_core", "faceted vs core"),
        ("policy_core_maintained", "policy_core", "maintained vs core"),
        ("policy_core", "policy_base", "core vs base"),
        ("policy_core_faceted", "policy_base", "faceted vs base"),
    ]
    cohens = {}
    for p1, p2, label in key_comparisons:
        if p1 in policies and p2 in policies:
            a = policy_f1_array(data, scopes, p1)
            b = policy_f1_array(data, scopes, p2)
            d = cohens_d(a, b)
            cohens[label] = {
                "d": round(d, 4),
                "interpretation": (
                    "large" if abs(d) > 0.8 else
                    "medium" if abs(d) > 0.5 else
                    "small" if abs(d) > 0.2 else "negligible"
                ),
                "n1": len(a), "n2": len(b)
            }
    results["cohens_d"] = cohens

    return results


def print_statistics(stats_results: dict):
    """Pretty-print statistical results."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    print("\n--- Bootstrap 95% CI on Mean Fact F1 (10,000 resamples) ---")
    for policy, ci in stats_results["bootstrap_ci"].items():
        label = POLICY_LABELS.get(policy, policy)
        print(f"  {label:20s}  {ci['mean']:.4f}  [{ci['ci_95_low']:.4f}, {ci['ci_95_high']:.4f}]  (n={ci['n']})")

    print("\n--- Friedman Test (non-parametric repeated measures) ---")
    fr = stats_results["friedman"]
    if "error" not in fr:
        sig = "SIGNIFICANT" if fr["p_value"] < 0.05 else "not significant"
        print(f"  chi2 = {fr['statistic']:.4f}, p = {fr['p_value']:.6f}  ({sig})")
        print(f"  k={fr['n_policies']} policies, n={fr['n_scopes']} scopes")
    else:
        print(f"  Error: {fr['error']}")

    print("\n--- Pairwise Wilcoxon Signed-Rank (Bonferroni corrected) ---")
    for pair, res in stats_results["pairwise_wilcoxon"].items():
        if "error" in res:
            print(f"  {pair}: {res['error']}")
        else:
            sig = "*" if res["significant_05"] else ""
            print(f"  {pair}: p_raw={res['p_value_raw']:.4f}, p_bonf={res['p_value_bonferroni']:.4f} {sig}")

    print("\n--- Kendall's W (concordance across scopes) ---")
    kw = stats_results["kendalls_w"]
    if "error" not in kw:
        print(f"  W = {kw['W']:.4f} ({kw['interpretation']})")
        print(f"  n_scopes={kw['n_scopes']}, n_policies={kw['n_policies']}")
    else:
        print(f"  Error: {kw['error']}")

    print("\n--- Cohen's d (effect sizes) ---")
    for label, res in stats_results["cohens_d"].items():
        print(f"  {label}: d = {res['d']:.4f} ({res['interpretation']})")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _save(fig, stem: Path):
    """Save figure as PNG (300 dpi) and PDF."""
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    print(f"  Saved: {stem.name}.png + .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LENS V2 Grid Ablation — Analysis & Visualization"
    )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Directory containing claude_scores_*.jsonl (default: results/ relative to script)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures and stats (default: brief/ relative to script)")
    parser.add_argument("--scopes", type=int, default=None,
                        help="Limit to N scopes (sorted by difficulty ascending)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    results_dir = args.results_dir or (script_dir / "results")
    output_dir = args.output_dir or (script_dir / "brief")
    fig_dir = output_dir / "figures"

    results_dir = results_dir.resolve()
    output_dir = output_dir.resolve()
    fig_dir = fig_dir.resolve()

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Apply theme
    apply_dark_theme()

    # Load data
    print("\nLoading score data:")
    records = load_scores(results_dir)
    summary = load_summary(results_dir)

    # Build arrays
    scopes, policies, data, per_replicate = build_arrays(records)

    if args.scopes and args.scopes < len(scopes):
        scopes = scopes[:args.scopes]
        print(f"\nLimited to {args.scopes} hardest scopes: {scopes}")

    n_nonnull = len([p for p in policies if p != "null"])
    print(f"\nGrid: {len(scopes)} scopes x {len(policies)} policies ({n_nonnull} non-null)")
    print(f"Scopes (hardest first): {scopes}")
    print(f"Policies: {policies}\n")

    # Generate figures
    print("Generating figures:")
    fig1_heatmap(scopes, policies, data, fig_dir)
    fig2_policy_ranking(scopes, policies, data, fig_dir)
    fig3_scope_difficulty(scopes, policies, data, fig_dir)
    fig4_ablation_waterfall(scopes, policies, data, fig_dir)
    fig5_consolidation_effect(scopes, policies, data, fig_dir)
    fig6_variance(scopes, policies, data, fig_dir)
    fig7_scope_clusters(scopes, policies, data, fig_dir)
    fig8_statistical_tests(scopes, policies, data, fig_dir)

    # Statistical analysis
    stats_results = run_statistics(scopes, policies, data)
    print_statistics(stats_results)

    # Save statistics
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2, default=str)
    print(f"Statistics saved: {stats_path}")

    # Summary counts
    print(f"\nDone. {len(records)} records analyzed, 8 figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
