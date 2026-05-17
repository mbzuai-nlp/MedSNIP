"""Single bar plot of baseline Macro F1 across all configurations.

22 bars total: 10 atom (5 models × 2 modes) + 12 snippet (2 models × 2
modes × 3 ground-truth labels). Color encodes grain (blue = atom,
orange = snippet). Sorted descending by macro F1 so the strongest
config is at the top.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "5-baselines" / "metrics.json"
OUT_PATH = ROOT / "data" / "5-baselines" / "baselines_reasoning.png"

GRAIN_COLOR = {"atom": "#3274A1", "snippet": "#E69A45"}
# Show only models common to both atom and snippet grains so atom-vs-
# snippet comparisons are fair.
ALLOWED_MODELS = {"gpt-4o-none", "gpt-5.4-high"}


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    metrics = json.loads(IN_PATH.read_text())

    rows = []
    for mode, models in metrics["atom"].items():
        for model, table in models.items():
            if model not in ALLOWED_MODELS:
                continue
            rows.append({
                "grain": "atom",
                "label": f"atom · {model} · {mode}",
                "macro_f1": table["overall"]["all"]["label"]["macro_f1"],
            })
    for mode, models in metrics["snippet"].items():
        for model, table in models.items():
            if model not in ALLOWED_MODELS:
                continue
            for gt in ("label_atomic", "label_human_general", "label_human_contextual"):
                gt_short = gt.replace("label_", "").replace("_", " ")
                rows.append({
                    "grain": "snippet",
                    "label": f"snippet · {model} · {mode} · vs {gt_short}",
                    "macro_f1": table["overall"]["all"][gt]["macro_f1"],
                })

    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    labels = [r["label"] for r in rows]
    values = [r["macro_f1"] for r in rows]
    colors = [GRAIN_COLOR[r["grain"]] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = list(range(len(rows)))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3f}",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # best at top
    ax.set_xlabel("Macro F1")
    ax.set_title("Baseline Macro F1 across all configurations\n(grain: atom = blue, snippet = orange)",
                 fontweight="bold")
    ax.set_xlim(0, max(values) * 1.12)

    legend = [
        Patch(facecolor=GRAIN_COLOR["atom"], edgecolor="white", linewidth=0.6, label="atom"),
        Patch(facecolor=GRAIN_COLOR["snippet"], edgecolor="white", linewidth=0.6, label="snippet"),
    ]
    ax.legend(handles=legend, title="Grain", fontsize=9, framealpha=0.95, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
