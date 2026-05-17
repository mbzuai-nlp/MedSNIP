"""Plot inter-annotator agreement on the IAA test batch.

Reads data/3-annotated/iaa_stats.json and writes
data/3-annotated/iaa_reasoning.png.

Three panels:
  (a) ARI per entry (grouping agreement) + overall mean
  (b) Fleiss' κ per labeled field, with Landis & Koch interpretation bands
  (c) Pairwise Cohen's κ heatmap for `label_in_general` (the headline label)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "3-annotated" / "iaa_stats.json"
OUT_PATH = ROOT / "data" / "3-annotated" / "iaa_reasoning.png"

# Landis & Koch (1977) bands for κ interpretation
KAPPA_BANDS = [
    (0.00, 0.20, "slight",       "#F5C6C6"),
    (0.20, 0.40, "fair",         "#FAD7B7"),
    (0.40, 0.60, "moderate",     "#FCEDB4"),
    (0.60, 0.80, "substantial",  "#CFE6BB"),
    (0.80, 1.00, "almost perfect", "#A9D3A4"),
]


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    stats = json.loads(IN_PATH.read_text())
    ids = stats["annotator_ids"]
    grouping = stats["grouping_ari"]
    metrics = stats["metrics"]

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.32)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) ARI per entry + overall mean
    per_entry = grouping["per_entry"]
    entries = sorted(per_entry.keys(), key=int)
    aris = [per_entry[e]["mean_ari"] for e in entries]
    bar = ax1.bar(entries, aris, color="#3274A1", alpha=0.9,
                  edgecolor="white", linewidth=0.6)
    for b, e in zip(bar, entries):
        info = per_entry[e]
        ax1.annotate(f"{info['mean_ari']:.2f}",
                     xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9.5, fontweight="bold")
        ax1.annotate(f"n={info['atoms']}",
                     xy=(b.get_x() + b.get_width() / 2, 0.02),
                     ha="center", fontsize=8, color="#555")
    overall = grouping["overall_mean"]
    ax1.axhline(overall, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax1.annotate(f"Overall: {overall:.2f}", xy=(-0.45, overall + 0.02),
                 fontsize=8, color="gray", fontstyle="italic", ha="left")
    ax1.set_xlabel("Entry ID")
    ax1.set_ylabel("Mean Pairwise ARI")
    ax1.set_title("(a) Grouping Agreement\n(Adjusted Rand Index per entry)", fontweight="bold")
    ax1.set_ylim(0, 1.08)

    # (b) Fleiss' κ per field with Landis & Koch bands
    fields = list(metrics.keys())
    kappas = [metrics[f]["fleiss_kappa"] for f in fields]
    field_labels = [f.replace("label_", "") for f in fields]
    for lo, hi, _, color in KAPPA_BANDS:
        ax2.axhspan(lo, hi, color=color, alpha=0.55, zorder=0)
    bars = ax2.bar(field_labels, kappas, color="#2C3E50", alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=2)
    for b, k in zip(bars, kappas):
        ax2.annotate(f"κ = {k:.2f}",
                     xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold", zorder=3)
    band_handles = [Patch(facecolor=c, alpha=0.55, label=name)
                    for _, _, name, c in KAPPA_BANDS]
    ax2.legend(handles=band_handles, title="Landis & Koch", fontsize=7.5,
               loc="upper right", framealpha=0.95, ncol=1, title_fontsize=8.5)
    ax2.set_ylabel("Fleiss' κ")
    ax2.set_title("(b) Label Agreement\n(Fleiss' κ per field, 6 annotators)", fontweight="bold")
    ax2.set_ylim(0, 1.0)
    plt.setp(ax2.get_xticklabels(), rotation=14, ha="right", fontsize=9)

    # (c) Pairwise Cohen's κ heatmap for label_in_general
    n = len(ids)
    mat = np.full((n, n), np.nan)
    pw = metrics["label_in_general"]["pairwise_cohen_kappa"]
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j:
                mat[i, j] = 1.0
            elif i < j:
                k = pw.get(f"{a}_{b}")
                if k is not None:
                    mat[i, j] = k
                    mat[j, i] = k
    tick_labels = [f"a{i}" for i in ids]
    sns.heatmap(mat, ax=ax3, vmin=0, vmax=1, cmap="RdYlGn",
                annot=True, fmt=".2f", annot_kws={"fontsize": 9},
                xticklabels=tick_labels, yticklabels=tick_labels,
                cbar_kws={"label": "Cohen's κ"}, square=True,
                linewidths=0.6, linecolor="white")
    ax3.set_title("(c) Pairwise Cohen's κ\n(label_in_general)", fontweight="bold")
    ax3.tick_params(axis="x", rotation=0)
    ax3.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
