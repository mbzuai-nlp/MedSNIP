"""Plot inter-annotator agreement on the IAA test batch as three separate panels.

Reads data/3-annotated/iaa_stats.json and writes
  data/3-annotated/iaa_reasoning_{a,b,c}.{png,pdf}.

  (a) ARI per entry (grouping agreement) + overall mean
  (b) Fleiss' kappa per labeled field, with Landis & Koch interpretation bands
  (c) Pairwise Cohen's kappa heatmap for `label_in_general` (the headline label)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "3-annotated"
IN_PATH = DATA_DIR / "iaa_stats.json"

KAPPA_BANDS = [
    (0.00, 0.20, "slight",       "#F5C6C6"),
    (0.20, 0.40, "fair",         "#FAD7B7"),
    (0.40, 0.60, "moderate",     "#FCEDB4"),
    (0.60, 0.80, "substantial",  "#CFE6BB"),
    (0.80, 1.00, "almost perfect", "#A9D3A4"),
]

PANEL_FIGSIZE = (5.0, 5.0)


def save_panel(fig, name: str) -> None:
    out_png = DATA_DIR / f"{name}.png"
    out_pdf = DATA_DIR / f"{name}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")
    plt.close(fig)


def panel_a(grouping: dict) -> None:
    fig, ax1 = plt.subplots(figsize=PANEL_FIGSIZE)
    ax1.set_box_aspect(1)

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
    ax1.set_ylim(0, 1.08)

    save_panel(fig, "iaa_reasoning_a")


def panel_b(metrics: dict) -> None:
    fig, ax2 = plt.subplots(figsize=PANEL_FIGSIZE)
    ax2.set_box_aspect(1)

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
    ax2.set_ylim(0, 1.0)
    plt.setp(ax2.get_xticklabels(), rotation=14, ha="right", fontsize=9)

    save_panel(fig, "iaa_reasoning_b")


def panel_c(ids: list, metrics: dict) -> None:
    fig, ax3 = plt.subplots(figsize=PANEL_FIGSIZE)

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
    ax3.tick_params(axis="x", rotation=0)
    ax3.tick_params(axis="y", rotation=0)

    save_panel(fig, "iaa_reasoning_c")


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    stats = json.loads(IN_PATH.read_text())
    panel_a(stats["grouping_ari"])
    panel_b(stats["metrics"])
    panel_c(stats["annotator_ids"], stats["metrics"])


if __name__ == "__main__":
    main()
