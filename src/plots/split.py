"""Plot the train/dev/test split: entry counts and false-rate per cell.

Color encodes subset (blue=consumer, red=vignette); alpha encodes
split (solid=train, mid=dev, light=test).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "3-split" / "stats.json"
OUT_PATH = ROOT / "data" / "3-split" / "split_reasoning.png"

PALETTE = {"consumer": "#3274A1", "vignette": "#C44E52"}
SUBSET_LABEL = {"consumer": "Consumer Health", "vignette": "Clinical Vignettes"}
SPLITS = ["train", "dev", "test"]
SPLIT_ALPHA = {"train": 1.0, "dev": 0.62, "test": 0.32}


def style_bars(ax, subsets):
    """Recolor a grouped bar plot: base color per x-category, alpha per hue."""
    for sp_idx, sp in enumerate(SPLITS):
        container = ax.containers[sp_idx]
        for bar, subset in zip(container, subsets):  # type: ignore[arg-type]
            bar.set_facecolor(PALETTE[subset])
            bar.set_alpha(SPLIT_ALPHA[sp])
            bar.set_edgecolor("white")
            bar.set_linewidth(0.6)


def split_legend(ax):
    """Each split row shows a paired (consumer-blue + vignette-red) swatch at
    the split's alpha — matches the two-color bars in the chart."""
    handles = [
        (Patch(facecolor=PALETTE["consumer"], alpha=SPLIT_ALPHA[sp],
               edgecolor="white", linewidth=0.6),
         Patch(facecolor=PALETTE["vignette"], alpha=SPLIT_ALPHA[sp],
               edgecolor="white", linewidth=0.6))
        for sp in SPLITS
    ]
    ax.legend(handles=handles, labels=list(SPLITS),
              title="Split", fontsize=9, framealpha=0.9,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
              handlelength=2.6)


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)

    stats = json.loads(IN_PATH.read_text())
    by = stats["by_subset"]
    subsets = ["consumer", "vignette"]

    rows = []
    for s in subsets:
        for sp in SPLITS:
            c = by[s][sp]
            rate = 100 * c["false_claims"] / c["claims"] if c["claims"] else 0.0
            rows.append({"subset": s, "split": sp, "entries": c["entries"],
                         "claims": c["claims"], "rate": rate})
    df = pd.DataFrame(rows)

    overall = (100 * sum(by[s][sp]["false_claims"] for s in subsets for sp in SPLITS)
                     / sum(by[s][sp]["claims"]       for s in subsets for sp in SPLITS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.28})

    # (a) entries
    sns.barplot(data=df, x="subset", y="entries", hue="split",
                hue_order=SPLITS, order=subsets, ax=ax1)
    style_bars(ax1, subsets)
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%d", fontsize=9.5, fontweight="bold", padding=3)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([SUBSET_LABEL[s] for s in subsets])
    ax1.set_xlabel("")
    ax1.set_ylabel("Number of Entries")
    ax1.set_title("(a) Train / Dev / Test Entries by Subset", fontweight="bold")
    split_legend(ax1)
    ax1.set_ylim(0, df["entries"].max() * 1.18)

    # (b) false-claim rate
    sns.barplot(data=df, x="subset", y="rate", hue="split",
                hue_order=SPLITS, order=subsets, ax=ax2)
    style_bars(ax2, subsets)
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%", fontsize=9.5, fontweight="bold", padding=3)
    ax2.axhline(overall, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.annotate(f"Overall: {overall:.1f}%", xy=(-0.45, overall + 0.15),
                 fontsize=8, color="gray", fontstyle="italic", ha="left")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([SUBSET_LABEL[s] for s in subsets])
    ax2.set_xlabel("")
    ax2.set_ylabel("False Atomic-Claim Rate (%)")
    ax2.set_title("(b) False Atomic-Claim Rate per (Subset × Split)", fontweight="bold")
    split_legend(ax2)
    ax2.set_ylim(0, df["rate"].max() * 1.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
