"""Plot the train/dev/test split: entry counts and false-rate per cell.

(a) Stacked entry counts per subset, broken down by split.
(b) Heatmap of false-rates across (label type) × (subset × split).
    Heatmap keeps 18 numbers readable when grouped bars would overlap.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "4-split" / "stats.json"
OUT_PATH = ROOT / "data" / "4-split" / "split_reasoning.png"

PALETTE = {"consumer": "#3274A1", "vignette": "#C44E52"}
SUBSET_LABEL = {"consumer": "Consumer Health", "vignette": "Clinical Vignettes"}
SPLITS = ["train", "dev", "test"]
SPLIT_ALPHA = {"train": 1.0, "dev": 0.62, "test": 0.32}
LABEL_PRETTY = {
    "atomic_false_rate":           "atomic",
    "human_general_false_rate":    "general",
    "human_contextual_false_rate": "contextual",
}


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    stats = json.loads(IN_PATH.read_text())
    by = stats["by_subset"]
    subsets = ["consumer", "vignette"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 4.8),
        gridspec_kw={"wspace": 0.28, "width_ratios": [0.85, 1.15]}
    )

    # (a) Stacked entry counts per subset
    x = np.arange(len(subsets))
    width = 0.55
    bottoms = np.zeros(len(subsets))
    for sp in SPLITS:
        vals = np.array([by[sub][sp]["entries"] for sub in subsets])
        bars = ax1.bar(x, vals, width, bottom=bottoms,
                       color=[PALETTE[sub] for sub in subsets],
                       alpha=SPLIT_ALPHA[sp],
                       edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            y = bar.get_y() + bar.get_height() / 2
            ax1.annotate(f"{int(v)}", xy=(bar.get_x() + bar.get_width() / 2, y),
                         ha="center", va="center", fontsize=10,
                         fontweight="bold", color="white")
        bottoms += vals
    for i, sub in enumerate(subsets):
        total = sum(by[sub][sp]["entries"] for sp in SPLITS)
        ax1.annotate(f"n = {total}", xy=(i, total + 3),
                     ha="center", fontsize=9, color="#666666",
                     fontstyle="italic")
    ax1.set_xticks(x)
    ax1.set_xticklabels([SUBSET_LABEL[s] for s in subsets])
    ax1.set_ylabel("Number of Entries")
    ax1.set_title("(a) Train / Dev / Test Entries by Subset", fontweight="bold")

    legend_handles = [
        (Patch(facecolor=PALETTE["consumer"], alpha=SPLIT_ALPHA[sp],
               edgecolor="white", linewidth=0.6),
         Patch(facecolor=PALETTE["vignette"], alpha=SPLIT_ALPHA[sp],
               edgecolor="white", linewidth=0.6))
        for sp in SPLITS
    ]
    ax1.legend(handles=legend_handles, labels=list(SPLITS),
               title="Split", fontsize=9, framealpha=0.95,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
               handlelength=2.6, loc="upper right")
    # Extra headroom so the legend doesn't overlap the n=... annotations
    ax1.set_ylim(0, max(sum(by[sub][sp]["entries"] for sp in SPLITS)
                        for sub in subsets) * 1.45)

    # (b) Heatmap: rows = label types, cols = (subset × split) cells
    label_rows = list(LABEL_PRETTY.values())
    cell_cols = [(sub, sp) for sub in subsets for sp in SPLITS]
    matrix = np.zeros((len(label_rows), len(cell_cols)))
    for j, (sub, sp) in enumerate(cell_cols):
        for i, key in enumerate(LABEL_PRETTY.keys()):
            matrix[i, j] = 100 * by[sub][sp][key]

    col_labels = [f"{sub[:3]}\n{sp}" for sub, sp in cell_cols]
    sns.heatmap(matrix, ax=ax2,
                xticklabels=col_labels, yticklabels=label_rows,
                annot=True, fmt=".1f", annot_kws={"fontsize": 10, "fontweight": "bold"},
                cmap="RdYlGn_r", vmin=0, vmax=max(20, matrix.max() * 1.05),
                cbar_kws={"label": "False Rate (%)", "pad": 0.02},
                linewidths=0.6, linecolor="white", square=False)
    ax2.set_title("(b) Snippet-Level False Rate (%) per (Subset × Split)",
                  fontweight="bold")
    ax2.tick_params(axis="x", rotation=0)
    ax2.tick_params(axis="y", rotation=0, labelsize=10)

    # Vertical separator between consumer and vignette column groups
    ax2.axvline(x=len(SPLITS), color="white", linewidth=4)
    ax2.axvline(x=len(SPLITS), color="#222222", linewidth=1.4)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
