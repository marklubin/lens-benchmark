#!/usr/bin/env python3
"""Generate visualization plots for narrative scope benchmark brief."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = "/home/mark/lens-benchmark/results/narrative_brief"

# --- Data ---
narrative_data = [
    ("sqlite-chunked-hybrid", "S07", 1, 0.2865),
    ("sqlite-chunked-hybrid", "S07", 2, 0.2865),
    ("sqlite-chunked-hybrid", "S07", 3, 0.2865),
    ("sqlite-chunked-hybrid", "S08", 1, 0.3949),
    ("sqlite-chunked-hybrid", "S08", 2, 0.3992),
    ("sqlite-chunked-hybrid", "S08", 3, 0.3992),
    ("sqlite-chunked-hybrid", "S09", 1, 0.3585),
    ("sqlite-chunked-hybrid", "S09", 2, 0.3585),
    ("sqlite-chunked-hybrid", "S09", 3, 0.3585),
    ("cognee", "S07", 1, 0.2980),
    ("cognee", "S07", 2, 0.2980),
    ("cognee", "S07", 3, 0.2980),
    ("cognee", "S08", 1, 0.3637),
    ("cognee", "S08", 2, 0.3637),
    ("cognee", "S08", 3, 0.3637),
    ("cognee", "S09", 1, 0.3190),
    ("cognee", "S09", 2, 0.3190),
    ("cognee", "S09", 3, 0.3190),
    ("letta", "S07", 1, 0.3340),
    ("letta", "S07", 2, 0.3340),
    ("letta", "S07", 3, 0.3340),
    ("letta", "S08", 1, 0.3458),
    ("letta", "S08", 2, 0.3458),
    ("letta", "S08", 3, 0.3388),
    ("letta", "S09", 1, 0.3490),
    ("letta", "S09", 2, 0.3490),
    ("letta", "S09", 3, 0.3490),
    ("letta-sleepy", "S07", 1, 0.3285),
    ("letta-sleepy", "S07", 2, 0.3040),
    ("letta-sleepy", "S07", 3, 0.2905),
    ("letta-sleepy", "S08", 1, 0.2932),
    ("letta-sleepy", "S08", 2, 0.2831),
    ("letta-sleepy", "S08", 3, 0.3016),
    ("letta-sleepy", "S09", 1, 0.3055),
    ("letta-sleepy", "S09", 2, 0.3055),
    ("letta-sleepy", "S09", 3, 0.2975),
    ("compaction", "S07", 1, 0.2675),
    ("compaction", "S07", 2, 0.2675),
    ("compaction", "S07", 3, 0.2675),
    ("compaction", "S08", 1, 0.2872),
    ("compaction", "S08", 2, 0.2872),
    ("compaction", "S08", 3, 0.2872),
    ("compaction", "S09", 1, 0.2735),
    ("compaction", "S09", 2, 0.2735),
    ("compaction", "S09", 3, 0.2735),
    ("mem0-raw", "S07", 1, 0.2468),
    ("mem0-raw", "S07", 2, 0.2398),
    ("mem0-raw", "S07", 3, 0.2606),
    ("mem0-raw", "S08", 1, 0.2552),
    ("mem0-raw", "S08", 2, 0.2552),
    ("mem0-raw", "S08", 3, 0.2552),
    ("mem0-raw", "S09", 1, 0.2585),
    ("mem0-raw", "S09", 2, 0.2585),
    ("mem0-raw", "S09", 3, 0.2585),
    ("null", "S07", 1, 0.1785),
    ("null", "S07", 2, 0.1785),
    ("null", "S07", 3, 0.1785),
    ("null", "S08", 1, 0.1785),
    ("null", "S08", 2, 0.1785),
    ("null", "S08", 3, 0.1785),
    ("null", "S09", 1, 0.1785),
    ("null", "S09", 2, 0.1785),
    ("null", "S09", 3, 0.1785),
]

numeric_data = {
    "sqlite-chunked-hybrid": 0.454,
    "cognee": 0.421,
    "graphiti": 0.393,
    "mem0-raw": 0.330,
    "letta": 0.327,
    "letta-sleepy": 0.322,
    "compaction": 0.245,
}

df = pd.DataFrame(narrative_data, columns=["adapter", "scope", "rep", "composite"])

# Sort order: by overall narrative mean descending
adapter_means = df.groupby("adapter")["composite"].mean().sort_values(ascending=False)
adapter_order = adapter_means.index.tolist()

# Consistent color palette
palette_adapters = {
    "sqlite-chunked-hybrid": "#2196F3",
    "cognee": "#4CAF50",
    "letta": "#FF9800",
    "letta-sleepy": "#FFC107",
    "compaction": "#9C27B0",
    "mem0-raw": "#F44336",
    "graphiti": "#00BCD4",
    "null": "#9E9E9E",
}

scope_colors = {"S07": "#5C6BC0", "S08": "#26A69A", "S09": "#EF5350"}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ============================================================
# Plot 1: Grouped bar chart — adapter rankings by scope
# ============================================================
def plot_adapter_rankings():
    fig, ax = plt.subplots(figsize=(11, 6))

    # Non-null adapters sorted by mean
    order_no_null = [a for a in adapter_order if a != "null"]
    scopes = ["S07", "S08", "S09"]
    n_adapters = len(order_no_null)
    n_scopes = len(scopes)
    bar_width = 0.22
    x = np.arange(n_adapters)

    for i, scope in enumerate(scopes):
        means = []
        stds = []
        for adapter in order_no_null:
            sub = df[(df["adapter"] == adapter) & (df["scope"] == scope)]
            means.append(sub["composite"].mean())
            stds.append(sub["composite"].std())
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=3, label=scope,
            color=scope_colors[scope], edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 1, "capthick": 1},
        )

    # Null baseline line
    null_mean = df[df["adapter"] == "null"]["composite"].mean()
    ax.axhline(y=null_mean, color="#9E9E9E", linestyle="--", linewidth=1.5,
               label=f"null baseline ({null_mean:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(order_no_null, rotation=25, ha="right")
    ax.set_ylabel("Composite Score")
    ax.set_title("Narrative Scope Performance by Adapter and Scope")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0.1, 0.45)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/adapter_rankings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  adapter_rankings.png")


# ============================================================
# Plot 2: Heatmap — adapters x scopes
# ============================================================
def plot_heatmap():
    fig, ax = plt.subplots(figsize=(7, 6))

    scopes = ["S07", "S08", "S09"]
    # Sort adapters by overall mean (exclude null last)
    order_no_null = [a for a in adapter_order if a != "null"]
    order_with_null = order_no_null + ["null"]

    pivot = df.groupby(["adapter", "scope"])["composite"].mean().unstack()
    pivot = pivot.reindex(index=order_with_null, columns=scopes)

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Composite Score", "shrink": 0.8},
        ax=ax, vmin=0.15, vmax=0.42,
    )

    ax.set_title("Mean Composite Score: Adapter x Scope")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  heatmap.png")


# ============================================================
# Plot 3: Numeric vs Narrative comparison
# ============================================================
def plot_numeric_vs_narrative():
    fig, ax = plt.subplots(figsize=(11, 6))

    # Adapters present in both datasets (plus graphiti for numeric only)
    common_adapters = ["sqlite-chunked-hybrid", "cognee", "letta", "letta-sleepy",
                       "compaction", "mem0-raw"]
    all_adapters = ["sqlite-chunked-hybrid", "cognee", "graphiti", "letta",
                    "letta-sleepy", "compaction", "mem0-raw"]

    # Sort by numeric score descending
    all_adapters_sorted = sorted(all_adapters, key=lambda a: numeric_data.get(a, 0), reverse=True)

    n = len(all_adapters_sorted)
    x = np.arange(n)
    bar_width = 0.35

    numeric_vals = [numeric_data.get(a, 0) for a in all_adapters_sorted]

    narrative_vals = []
    for a in all_adapters_sorted:
        if a in common_adapters:
            narrative_vals.append(df[df["adapter"] == a]["composite"].mean())
        else:
            narrative_vals.append(0)  # graphiti has no narrative data

    bars1 = ax.bar(x - bar_width/2, numeric_vals, bar_width,
                   label="Numeric (Phase 5, 8k)", color="#5C6BC0",
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + bar_width/2, narrative_vals, bar_width,
                   label="Narrative (S07-S09)", color="#26A69A",
                   edgecolor="white", linewidth=0.5)

    # Mark graphiti's missing narrative bar
    graphiti_idx = all_adapters_sorted.index("graphiti")
    ax.annotate("N/A\n(context\noverflow)", xy=(graphiti_idx + bar_width/2, 0.01),
                ha="center", va="bottom", fontsize=8, color="#666666", style="italic")

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_adapters_sorted, rotation=25, ha="right")
    ax.set_ylabel("Composite Score")
    ax.set_title("Numeric vs Narrative Scope Performance")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 0.52)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/numeric_vs_narrative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  numeric_vs_narrative.png")


# ============================================================
# Plot 4: Score distribution — violin + strip
# ============================================================
def plot_score_distribution():
    fig, ax = plt.subplots(figsize=(11, 6))

    order_no_null = [a for a in adapter_order if a != "null"]
    df_no_null = df[df["adapter"] != "null"].copy()

    colors = [palette_adapters[a] for a in order_no_null]

    parts = ax.violinplot(
        [df_no_null[df_no_null["adapter"] == a]["composite"].values for a in order_no_null],
        positions=range(len(order_no_null)),
        showmeans=True, showmedians=False, showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.4)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    # Overlay strip plot for individual points
    for i, adapter in enumerate(order_no_null):
        vals = df_no_null[df_no_null["adapter"] == adapter]["composite"].values
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(i + jitter, vals, color=colors[i], s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.5, zorder=5)

    # Null baseline line
    null_mean = df[df["adapter"] == "null"]["composite"].mean()
    ax.axhline(y=null_mean, color="#9E9E9E", linestyle="--", linewidth=1.5,
               label=f"null baseline ({null_mean:.3f})")

    ax.set_xticks(range(len(order_no_null)))
    ax.set_xticklabels(order_no_null, rotation=25, ha="right")
    ax.set_ylabel("Composite Score")
    ax.set_title("Score Distribution Across Narrative Scopes")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0.15, 0.45)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  score_distribution.png")


if __name__ == "__main__":
    print("Generating plots...")
    plot_adapter_rankings()
    plot_heatmap()
    plot_numeric_vs_narrative()
    plot_score_distribution()
    print("Done. All plots saved to:", OUT_DIR)
