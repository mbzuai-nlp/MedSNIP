"""Plot the annotation step: snippet counts and false-rate breakdown per subset.

Reads data/3-annotated/stats.json and writes
data/3-annotated/annotation_reasoning.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "3-annotated" / "stats.json"
OUT_PATH = ROOT / "data" / "3-annotated" / "annotation_reasoning.png"

PALETTE = {"consumer": "#3274A1", "vignette": "#C44E52"}
SUBSET_LABEL = {"consumer": "Consumer Health", "vignette": "Clinical Vignettes"}


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    stats = json.loads(IN_PATH.read_text())
    by = stats["by_subset"]
    subsets = ["consumer", "vignette"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.3})

    # (a) Snippet counts (stacked: true vs false on label_human_general)
    true_counts = [round(by[s]["snippets"] * (1 - by[s]["human_general_false_rate"])) for s in subsets]
    false_counts = [by[s]["snippets"] - t for s, t in zip(subsets, true_counts)]
    colors = [PALETTE[s] for s in subsets]
    x = list(range(len(subsets)))
    ax1.bar(x, true_counts, color=colors, alpha=0.35, edgecolor="white", linewidth=0.6, label="True")
    ax1.bar(x, false_counts, bottom=true_counts, color=colors, alpha=1.0, edgecolor="white", linewidth=0.6, label="False")

    for i, s in enumerate(subsets):
        total = by[s]["snippets"]
        n_false = false_counts[i]
        rate = 100 * by[s]["human_general_false_rate"]
        ax1.annotate(f"{total:,} snippets", xy=(i, total),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=9.5, fontweight="bold", color=PALETTE[s])
        ax1.annotate(f"{n_false} false ({rate:.1f}%)",
                     xy=(i, total - n_false / 2),
                     ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels([SUBSET_LABEL[s] for s in subsets])
    ax1.set_ylabel("Number of Snippets")
    ax1.set_title("(a) Snippet Composition by Subset\n(by label_human_general)", fontweight="bold")
    ax1.set_ylim(0, max(by[s]["snippets"] for s in subsets) * 1.18)

    # (b) False rates grouped: per (subset × label type)
    rows = []
    label_pretty = {
        "atomic_false_rate":             "label_atomic",
        "human_general_false_rate":      "label_human_general",
        "human_contextual_false_rate":   "label_human_contextual",
    }
    for s in subsets:
        for key, label in label_pretty.items():
            rows.append({"subset": s, "field": label, "rate": 100 * by[s][key]})
    df = pd.DataFrame(rows)

    sns.barplot(data=df, x="field", y="rate", hue="subset",
                palette=PALETTE, hue_order=subsets,
                order=list(label_pretty.values()),
                edgecolor="white", linewidth=0.6, ax=ax2)
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%", fontsize=9, fontweight="bold", padding=3)
    ax2.set_xlabel("")
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("(b) Snippet-Level Rates per Subset", fontweight="bold")
    leg = ax2.legend(title=None, fontsize=9, framealpha=0.9)
    for t, s in zip(leg.get_texts(), subsets):
        t.set_text(SUBSET_LABEL[s])
    ax2.set_ylim(0, df["rate"].max() * 1.25)
    plt.setp(ax2.get_xticklabels(), rotation=18, ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
