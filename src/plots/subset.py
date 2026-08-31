"""Plot the Kim subset split as two separate panels.

(a) Query word-count distribution with the natural gap that justifies the
    consumer/vignette partition.
(b) Atomic-claim composition per subset, with the false-rate annotated.

Reads data/2-subset/kim.json (flat) + stats.json and writes
data/2-subset/subset_reasoning_{a,b}.{png,pdf}.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "2-subset"

PALETTE = {"consumer": "#3274A1", "vignette": "#C44E52"}
SUBSET_LABEL = {"consumer": "Consumer Health", "vignette": "Clinical Vignettes"}

PANEL_FIGSIZE = (5.5, 5.5)


def entry_id_of(row_id) -> int:
    return int(str(row_id).split("-")[0])


def save_panel(fig, name: str) -> None:
    out_png = DATA_DIR / f"{name}.png"
    out_pdf = DATA_DIR / f"{name}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")
    plt.close(fig)


def panel_a(entry_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=PANEL_FIGSIZE)
    ax1.set_box_aspect(1)

    sns.histplot(
        data=entry_df, x="word_count", hue="subset", bins=40,
        palette=PALETTE, hue_order=["consumer", "vignette"],
        edgecolor="white", linewidth=0.6, alpha=0.9, ax=ax1, legend=False,
    )

    consumer_max = entry_df.loc[entry_df["subset"] == "consumer", "word_count"].max()
    vignette_min = entry_df.loc[entry_df["subset"] == "vignette", "word_count"].min()
    gap_lo, gap_hi = consumer_max + 1, vignette_min - 1
    ax1.axvspan(gap_lo, gap_hi, alpha=0.12, color="#888888", zorder=0)
    ax1.annotate("No queries\nin this range",
                 xy=((gap_lo + gap_hi) / 2, 13),
                 ha="center", va="center", fontsize=9,
                 color="#555555", fontstyle="italic")

    pill_x_axfrac = {"consumer": 0.20, "vignette": 0.78}
    for s in ("consumer", "vignette"):
        wc = entry_df.loc[entry_df["subset"] == s, "word_count"]
        ax1.annotate(
            f"{SUBSET_LABEL[s]}\n{wc.min()}–{wc.max()} words (n={len(wc)})",
            xy=(pill_x_axfrac[s], 0.93), xycoords="axes fraction",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=PALETTE[s],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=PALETTE[s], linewidth=1.4, alpha=0.95),
        )
    ax1.set_xlabel("Query Word Count")
    ax1.set_ylabel("Number of Entries")
    ymax = entry_df.groupby(pd.cut(entry_df["word_count"], bins=40)).size().max()
    ax1.set_ylim(0, ymax * 1.45)

    save_panel(fig, "subset_reasoning_a")


def panel_b(stats: dict) -> None:
    fig, ax2 = plt.subplots(figsize=PANEL_FIGSIZE)
    ax2.set_box_aspect(1)

    subsets = ["consumer", "vignette"]
    x = list(range(len(subsets)))
    true_counts = [stats[s]["claims"] - stats[s]["false_claims"] for s in subsets]
    false_counts = [stats[s]["false_claims"] for s in subsets]
    colors = [PALETTE[s] for s in subsets]

    ax2.bar(x, true_counts, color=colors, alpha=0.35, edgecolor="white",
            linewidth=0.6, label="True")
    ax2.bar(x, false_counts, bottom=true_counts, color=colors, alpha=1.0,
            edgecolor="white", linewidth=0.6, label="False")

    for i, s in enumerate(subsets):
        total = stats[s]["claims"]
        n_false = stats[s]["false_claims"]
        rate = 100 * stats[s]["false_rate"]
        ax2.annotate(f"{total:,} atomic claims", xy=(i, total),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=9.5, fontweight="bold",
                     color=PALETTE[s])
        ax2.annotate(f"{n_false} false ({rate:.1f}%)",
                     xy=(i, total - n_false / 2),
                     ha="center", va="center", fontsize=10, fontweight="bold",
                     color="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels([SUBSET_LABEL[s] for s in subsets])
    ax2.set_xlabel("")
    ax2.set_ylabel("Number of Atomic Claims")
    ax2.set_ylim(0, max(stats[s]["claims"] for s in subsets) * 1.18)

    save_panel(fig, "subset_reasoning_b")


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)

    rows = json.loads((DATA_DIR / "kim.json").read_text())
    stats = json.loads((DATA_DIR / "stats.json").read_text())

    seen: dict[int, dict] = {}
    for r in rows:
        eid = entry_id_of(r["id"])
        if eid not in seen:
            seen[eid] = {"subset": r["subset"], "word_count": len(r["query"].split())}
    entry_df = pd.DataFrame(seen.values())

    panel_a(entry_df)
    panel_b(stats)


if __name__ == "__main__":
    main()
