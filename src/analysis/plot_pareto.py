"""Generate the call-count--F1_F trade-off figure for the paper.

Plots snippet- vs atom-level baselines.
X-axis is total verifier calls per sweep.
Y-axis is false-class F1.
The calibrated retrieval-augmented verifier is reported separately in Table 7.

Output:
    report/figs/pareto-cost-f1.png
    report/figs/pareto-cost-f1.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT_PNG = ROOT / "report" / "figs" / "pareto-cost-f1.png"
OUT_PDF = ROOT / "report" / "figs" / "pareto-cost-f1.pdf"

COLORS = {
    "MedSnip":  "#1f77b4",
    "HealthFC": "#d62728",
    "MedHallu": "#2ca02c",
}

# dataset, split label, n_snip, f_snip, n_atom, f_atom  (all claim-only)
BASELINE_PAIRS = [
    ("MedSnip",  "train", 1599, 0.433, 3735, 0.302),
    ("MedSnip",  "dev",    494, 0.407, 1078, 0.356),
    ("MedSnip",  "test",   431, 0.451,  942, 0.351),
    ("HealthFC", "",       750, 0.670,  967, 0.630),
    ("MedHallu", "",      2000, 0.749, 3798, 0.694),
]

# Label offset in screen points (dx, dy) per (dataset, split).
# Labels sit above-left of each snippet point to stay clear of the arrows
# (which come from the atom point to the lower-right).
LABEL_OFFSETS = {
    ("MedSnip",  "train"): (-72, 12),
    ("MedSnip",  "dev"):   (-72, 12),
    ("MedSnip",  "test"):  (-72, 12),
    ("HealthFC", ""):      (-72, 12),
    ("MedHallu", ""):      (-72, 12),
}


def label_for(ds: str, split: str) -> str:
    if ds == "MedSnip":
        return f"MedSnip {split.upper()}"
    return ds


def add_arrow(ax, n_atom, f_atom, n_snip, f_snip, color):
    ax.annotate(
        "",
        xy=(n_snip, f_snip),
        xytext=(n_atom, f_atom),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.15,
            shrinkA=8,
            shrinkB=8,
        ),
        zorder=1,
    )


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="serif",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.6,
            "axes.axisbelow": True,
            "grid.color": "#cfd6df",
            "grid.linestyle": "-",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 4.45))

    for ds, split, n_snip, f_snip, n_atom, f_atom in BASELINE_PAIRS:
        color = COLORS[ds]

        add_arrow(ax, n_atom, f_atom, n_snip, f_snip, color)

        ax.scatter(
            n_atom,
            f_atom,
            s=105,
            marker="s",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        ax.scatter(
            n_snip,
            f_snip,
            s=105,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )

        dx, dy = LABEL_OFFSETS.get((ds, split), (-72, 12))
        ax.annotate(
            label_for(ds, split),
            xy=(n_snip, f_snip),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor=color,
                linewidth=0.4,
                alpha=0.9,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.5,
                alpha=0.55,
                shrinkA=2,
                shrinkB=4,
            ),
            zorder=5,
        )

    ax.set_xlabel("Verifier calls per sweep, log scale")
    ax.set_ylabel(r"False-class F1, $\mathrm{F1}_{F}$")
    ax.set_xscale("log")
    ax.set_xlim(200, 6500)
    ax.set_ylim(0.25, 0.82)

    ax.grid(True, which="major", color="#cfd6df", linewidth=0.55, alpha=0.75)
    ax.grid(True, which="minor", color="#e6eaef", linewidth=0.4,  alpha=0.6)
    ax.minorticks_on()
    ax.tick_params(which="both", color="#aaaaaa")

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            label="Snippet",
            markersize=7,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="black",
            label="Atom",
            markersize=7,
            linestyle="None",
            markerfacecolor="white",
        ),
        Line2D([0], [0], color=COLORS["MedSnip"],  lw=3, label="MedSnip"),
        Line2D([0], [0], color=COLORS["HealthFC"], lw=3, label="HealthFC"),
        Line2D([0], [0], color=COLORS["MedHallu"], lw=3, label="MedHallu"),
    ]

    leg = ax.legend(
        handles=legend_items,
        loc="center right",
        title=r"$\bf{Better}$ $\nwarrow$ (up and left)",
        title_fontsize=9,
        frameon=True,
        handletextpad=0.5,
        borderpad=0.6,
        labelspacing=0.45,
    )
    leg.get_title().set_color("#555555")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
