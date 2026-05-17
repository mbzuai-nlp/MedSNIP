"""Plot the MedQA subset split: word-count histogram + counts + false-rate.

Reads data/2-subset/medqa.json (which already has the `subset` key) and
writes data/2-subset/subset_reasoning.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "2-subset" / "medqa.json"
OUT_PATH = ROOT / "data" / "2-subset" / "subset_reasoning.png"

WORD_THRESHOLD = 80

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    data = json.loads(IN_PATH.read_text())
    consumer = [e for e in data if e["subset"] == "consumer"]
    vignette = [e for e in data if e["subset"] == "vignette"]
    consumer_wc = [len(e["query"].split()) for e in consumer]
    vignette_wc = [len(e["query"].split()) for e in vignette]

    n_c, n_v = len(consumer), len(vignette)
    consumer_false = sum(1 for e in consumer for s in e["model_response"]["statements"] if not s["label"])
    vignette_false = sum(1 for e in vignette for s in e["model_response"]["statements"] if not s["label"])
    consumer_claims = sum(len(e["model_response"]["statements"]) for e in consumer)
    vignette_claims = sum(len(e["model_response"]["statements"]) for e in vignette)
    consumer_rate = 100 * consumer_false / consumer_claims
    vignette_rate = 100 * vignette_false / vignette_claims
    overall_rate = 100 * (consumer_false + vignette_false) / (consumer_claims + vignette_claims)
    consumer_uniq = len({e["query"] for e in consumer})
    vignette_uniq = len({e["query"] for e in vignette})

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.8, 0.8], wspace=0.35)

    # (a) Histogram
    ax1 = fig.add_subplot(gs[0])
    bins_c = np.arange(0, 36, 2)
    bins_v = np.arange(100, 270, 10)
    ax1.hist(consumer_wc, bins=bins_c, color="#3274A1", edgecolor="white", linewidth=0.8, alpha=0.9, label="Consumer Health")
    ax1.hist(vignette_wc, bins=bins_v, color="#C44E52", edgecolor="white", linewidth=0.8, alpha=0.9, label="Clinical Vignettes")
    ax1.axvspan(min(vignette_wc) - 1, min(vignette_wc) - 1, alpha=0)
    gap_lo, gap_hi = max(consumer_wc) + 1, min(vignette_wc) - 1
    ax1.axvspan(gap_lo, gap_hi, alpha=0.08, color="#888888", zorder=0)

    ax1.annotate(f"Consumer Health\n{min(consumer_wc)}–{max(consumer_wc)} words (n={n_c})",
                 xy=(15, 28), fontsize=9.5, ha="center", fontweight="bold", color="#3274A1",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#3274A1", alpha=0.8))
    ax1.annotate(f"Clinical Vignettes\n{min(vignette_wc)}–{max(vignette_wc)} words (n={n_v})",
                 xy=(220, 28), fontsize=9.5, ha="center", fontweight="bold", color="#C44E52",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C44E52", alpha=0.8))
    ax1.annotate("No queries\nin this range", xy=((gap_lo + gap_hi) / 2, 15),
                 fontsize=8.5, ha="center", va="center", color="#666666", fontstyle="italic")

    ax1.set_xlabel("Query Word Count")
    ax1.set_ylabel("Number of Entries")
    ax1.set_title("(a) Query Length Distribution", fontweight="bold", pad=10)
    ax1.set_ylim(0, 34)
    ax1.set_xlim(-5, 265)

    # (b) Counts
    ax2 = fig.add_subplot(gs[1])
    metrics = ["Entries", "Unique\nQueries", "False\nStatements"]
    c_vals = [n_c, consumer_uniq, consumer_false]
    v_vals = [n_v, vignette_uniq, vignette_false]
    x = np.arange(len(metrics))
    width = 0.32
    bars1 = ax2.bar(x - width / 2, c_vals, width, color="#3274A1", alpha=0.9, label="Consumer Health")
    bars2 = ax2.bar(x + width / 2, v_vals, width, color="#C44E52", alpha=0.9, label="Clinical Vignettes")
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Subset Counts", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9.5)
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax2.set_ylim(0, max(max(c_vals), max(v_vals)) * 1.2)

    # (c) False rate
    ax3 = fig.add_subplot(gs[2])
    categories = ["Consumer\nHealth", "Clinical\nVignettes"]
    false_rates = [round(consumer_rate, 1), round(vignette_rate, 1)]
    colors = ["#3274A1", "#C44E52"]
    bars = ax3.bar(categories, false_rates, width=0.5, color=colors, alpha=0.9,
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, false_rates):
        ax3.annotate(f"{val}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=11, fontweight="bold")
    ax3.set_ylabel("False Rate (%)")
    ax3.set_title("(c) False Statement Rate", fontweight="bold", pad=10)
    ax3.set_ylim(0, max(false_rates) * 1.5)
    ax3.axhline(y=overall_rate, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax3.annotate(f"Overall: {overall_rate:.1f}%", xy=(1.35, overall_rate + 0.2),
                 fontsize=8, color="gray", fontstyle="italic")

    plt.tight_layout(w_pad=3)
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
