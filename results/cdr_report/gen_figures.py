#!/usr/bin/env python3
"""Generate figures for the CDR LENS report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Agent-driven comparison bar chart ──────────────────────────────

adapters = [
    "letta-v4\n(full CDR)",
    "sqlite-chunked\n-hybrid",
    "cognee",
    "letta\n(no CDR)",
    "letta-sleepy\n(partial CDR)",
]

s07 = [0.586, 0.480, 0.409, 0.334, 0.329]
s08 = [0.489, 0.445, 0.403, 0.379, 0.382]
s09 = [0.612, 0.425, 0.438, 0.405, 0.305]
means = [0.562, 0.450, 0.417, 0.373, 0.339]

x = np.arange(len(adapters))
width = 0.18

fig, ax = plt.subplots(figsize=(9, 4.5))

colors = ["#90CAF9", "#FFB74D", "#A5D6A7", "#37474F"]
bars1 = ax.bar(x - 1.5 * width, s07, width, label="S07 (Tutoring)", color=colors[0], edgecolor="white")
bars2 = ax.bar(x - 0.5 * width, s08, width, label="S08 (Corporate)", color=colors[1], edgecolor="white")
bars3 = ax.bar(x + 0.5 * width, s09, width, label="S09 (Shadow API)", color=colors[2], edgecolor="white")
bars4 = ax.bar(x + 1.5 * width, means, width, label="Mean", color=colors[3], edgecolor="white")

ax.set_ylabel("Composite Score", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(adapters, fontsize=9)
ax.set_ylim(0, 0.75)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Agent-Driven Runs: Composite Scores on Narrative Scopes", fontsize=12, fontweight="bold")

# Add mean value labels on the mean bars
for bar, val in zip(bars4, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/agent_comparison.pdf", bbox_inches="tight")
plt.savefig("figures/agent_comparison.png", dpi=200, bbox_inches="tight")
print("Generated figures/agent_comparison.pdf")


# ── CDR progression chart ──────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 4))

progression = {
    "letta\n(no CDR)": 0.373,
    "letta-sleepy\n(undirected renewal)": 0.339,
    "letta-v4\n(full CDR)": 0.562,
}
labels = list(progression.keys())
vals = list(progression.values())
bar_colors = ["#BDBDBD", "#FFAB91", "#4CAF50"]

bars = ax2.bar(labels, vals, color=bar_colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax2.set_ylabel("Mean Composite Score", fontsize=11)
ax2.set_ylim(0, 0.7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("CDR Implementation Progression", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/cdr_progression.pdf", bbox_inches="tight")
plt.savefig("figures/cdr_progression.png", dpi=200, bbox_inches="tight")
print("Generated figures/cdr_progression.pdf")


# ── V4 per-metric breakdown ───────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(8, 4))

metrics = [
    "evidence\ngrounding", "budget\ncompliance", "insight\ndepth",
    "action\nquality", "NBA", "answer\nquality",
    "evidence\ncoverage", "fact\nrecall"
]
v4_vals = [1.0, 1.0, 0.767, 0.794, 0.673, 0.523, 0.169, 0.05]

bar_colors3 = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.2 else "#F44336" for v in v4_vals]
bars3 = ax3.barh(metrics, v4_vals, color=bar_colors3, edgecolor="white", height=0.6)

for bar, val in zip(bars3, v4_vals):
    ax3.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", ha="left", va="center", fontsize=9)

ax3.set_xlim(0, 1.15)
ax3.set_xlabel("Score", fontsize=11)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_title("Letta V4: Per-Metric Breakdown (Mean across S07–S09)", fontsize=12, fontweight="bold")
ax3.invert_yaxis()

plt.tight_layout()
plt.savefig("figures/v4_metrics.pdf", bbox_inches="tight")
plt.savefig("figures/v4_metrics.png", dpi=200, bbox_inches="tight")
print("Generated figures/v4_metrics.pdf")
