"""Plot the dev/test split: entry counts and false-rate per (subset, split) cell."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "3-split" / "stats.json"
OUT_PATH = ROOT / "data" / "3-split" / "split_reasoning.png"

CONSUMER_COLOR = "#3274A1"
VIGNETTE_COLOR = "#C44E52"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    stats = json.loads(IN_PATH.read_text())
    by = stats["by_subset"]

    cells = {
        ("consumer", "dev"):  by["consumer"]["dev"],
        ("consumer", "test"): by["consumer"]["test"],
        ("vignette", "dev"):  by["vignette"]["dev"],
        ("vignette", "test"): by["vignette"]["test"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.35})

    # (a) Entries: stacked dev+test per subset
    subsets = ["consumer", "vignette"]
    dev_vals = [cells[(s, "dev")]["entries"] for s in subsets]
    test_vals = [cells[(s, "test")]["entries"] for s in subsets]
    x = np.arange(len(subsets))
    width = 0.5
    b1 = ax1.bar(x, dev_vals, width, color=[CONSUMER_COLOR, VIGNETTE_COLOR], alpha=0.9, label="dev")
    b2 = ax1.bar(x, test_vals, width, bottom=dev_vals,
                 color=[CONSUMER_COLOR, VIGNETTE_COLOR], alpha=0.45,
                 edgecolor="white", linewidth=1, label="test")
    for bars, vals, offset in [(b1, dev_vals, 0), (b2, test_vals, dev_vals)]:
        for i, (bar, v) in enumerate(zip(bars, vals)):
            y = bar.get_y() + bar.get_height() / 2
            ax1.annotate(f"{v}", xy=(bar.get_x() + bar.get_width() / 2, y),
                         ha="center", va="center", fontsize=10, fontweight="bold",
                         color="white")
    for i, s in enumerate(subsets):
        total = dev_vals[i] + test_vals[i]
        test_pct = 100 * test_vals[i] / total
        ax1.annotate(f"test = {test_pct:.1f}%", xy=(i, total + 3),
                     ha="center", fontsize=9, color="#666666", fontstyle="italic")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Consumer Health", "Clinical Vignettes"])
    ax1.set_ylabel("Number of Entries")
    ax1.set_title("(a) Dev/Test Entries by Subset", fontweight="bold", pad=10)
    ax1.set_ylim(0, max(d + t for d, t in zip(dev_vals, test_vals)) * 1.18)

    # (b) False-claim rate per cell
    labels = ["consumer\ndev", "consumer\ntest", "vignette\ndev", "vignette\ntest"]
    rates = [
        100 * cells[k]["false_claims"] / cells[k]["claims"]
        for k in [("consumer", "dev"), ("consumer", "test"),
                  ("vignette", "dev"), ("vignette", "test")]
    ]
    colors = [CONSUMER_COLOR, CONSUMER_COLOR, VIGNETTE_COLOR, VIGNETTE_COLOR]
    alphas = [0.9, 0.45, 0.9, 0.45]
    bars = ax2.bar(labels, rates, width=0.6, color=colors)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    for bar, val in zip(bars, rates):
        ax2.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold")
    overall = 100 * sum(c["false_claims"] for c in cells.values()) \
              / sum(c["claims"] for c in cells.values())
    ax2.axhline(y=overall, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax2.annotate(f"Overall: {overall:.1f}%", xy=(3.5, overall + 0.2),
                 fontsize=8, color="gray", fontstyle="italic", ha="right")
    ax2.set_ylabel("False Claim Rate (%)")
    ax2.set_title("(b) False-Claim Rate per (Subset × Split)", fontweight="bold", pad=10)
    ax2.set_ylim(0, max(rates) * 1.4)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
