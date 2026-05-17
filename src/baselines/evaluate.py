"""Compute baseline metrics from saved predictions — no LLM calls.

Walks data/5-baselines/predictions/{atom,snippet}/{mode}/<model-config>/
predictions.json and produces data/5-baselines/metrics.json with
per-(grain × mode × model × split × subset × ground_truth) metrics.

For each (grain, mode, model, split, subset, ground_truth):
  - n, n_true, n_false (gold)
  - accuracy
  - precision_F, recall_F, f1_F  (false class — main signal)
  - precision_T, recall_T, f1_T
  - macro_f1
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
PRED_BASE = ROOT / "data" / "5-baselines" / "predictions"
SPLIT_PATH = ROOT / "data" / "4-split" / "medqa.json"
OUT_PATH = ROOT / "data" / "5-baselines" / "metrics.json"

ATOM_GTS = ["label"]
SNIPPET_GTS = ["label_atomic", "label_human_general", "label_human_contextual"]


def f1_for_class(rows: list[dict], gt_field: str, positive: bool) -> dict:
    """Compute precision/recall/F1 where `positive` is the class of interest."""
    tp = fp = fn = tn = 0
    for r in rows:
        gold = bool(r[gt_field])
        pred = bool(r["prediction"])
        gold_pos = gold == positive
        pred_pos = pred == positive
        if gold_pos and pred_pos:   tp += 1
        elif gold_pos and not pred_pos: fn += 1
        elif not gold_pos and pred_pos: fp += 1
        else:                            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_block(rows: list[dict], gt_field: str) -> dict:
    """Compute the full metrics block for one (rows, ground_truth) pair."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    n_true_gold  = sum(1 for r in rows if bool(r[gt_field]))
    n_false_gold = n - n_true_gold
    acc = sum(1 for r in rows if bool(r[gt_field]) == bool(r["prediction"])) / n
    F = f1_for_class(rows, gt_field, positive=False)
    T = f1_for_class(rows, gt_field, positive=True)
    macro_f1 = (F["f1"] + T["f1"]) / 2
    return {
        "n":           n,
        "n_true_gold": n_true_gold,
        "n_false_gold": n_false_gold,
        "accuracy":    round(acc, 4),
        "precision_F": F["precision"], "recall_F": F["recall"], "f1_F": F["f1"],
        "precision_T": T["precision"], "recall_T": T["recall"], "f1_T": T["f1"],
        "macro_f1":    round(macro_f1, 4),
    }


def iter_pred_files(grain: str) -> Iterable[tuple[str, str, Path]]:
    """Yield (mode, model_config, predictions_path) for one grain."""
    grain_dir = PRED_BASE / grain
    if not grain_dir.exists():
        return
    for mode_dir in sorted(grain_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        for model_dir in sorted(mode_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            pred = model_dir / "predictions.json"
            if pred.exists():
                yield mode_dir.name, model_dir.name, pred


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Per-snippet split lookup
    snip_rows = json.loads(SPLIT_PATH.read_text())
    snippet_split: dict[str, str] = {r["snippet_id"]: r["split"] for r in snip_rows}
    # Per-atom split lookup via the entry: each atom of an entry shares the same split
    entry_split: dict[int, str] = {r["entry_id"]: r["split"] for r in snip_rows}

    results: dict = {"atom": {}, "snippet": {}}

    # Atom grain
    for mode, model, pred_path in iter_pred_files("atom"):
        preds = json.loads(pred_path.read_text())
        # Augment each row with `split` (atoms inherit their entry's split)
        for r in preds:
            eid = int(str(r["id"]).split("-")[0])
            r["split"] = entry_split.get(eid)
        results["atom"].setdefault(mode, {})[model] = compute_table(preds, ATOM_GTS)

    # Snippet grain
    for mode, model, pred_path in iter_pred_files("snippet"):
        preds = json.loads(pred_path.read_text())
        for r in preds:
            r["split"] = snippet_split.get(r["id"])
        results["snippet"].setdefault(mode, {})[model] = compute_table(preds, SNIPPET_GTS)

    OUT_PATH.write_text(json.dumps(results, indent=2))

    # Print headline table: F1_F at the "overall" cell for each config
    print(f"\nHeadline (F1 on FALSE class, all splits combined, all subsets):")
    print(f"{'grain':>8s} {'mode':>14s} {'model':>22s} {'gt':>22s} {'n':>5s} {'F1_F':>6s} {'F1_T':>6s} {'macro':>6s}")
    for grain in ("atom", "snippet"):
        for mode in sorted(results[grain].keys()):
            for model in sorted(results[grain][mode].keys()):
                for gt in (ATOM_GTS if grain == "atom" else SNIPPET_GTS):
                    cell = results[grain][mode][model]["overall"]["all"][gt]
                    print(f"{grain:>8s} {mode:>14s} {model:>22s} {gt:>22s} "
                          f"{cell['n']:>5d} {cell['f1_F']:>6.3f} {cell['f1_T']:>6.3f} {cell['macro_f1']:>6.3f}")

    print(f"\nwrote {OUT_PATH}")


def compute_table(preds: list[dict], gt_fields: list[str]) -> dict:
    """Compute metrics per split × subset × overall × ground_truth field."""
    splits = ("all", "train", "dev", "test")
    subsets = ("all", "consumer", "vignette")

    def select(rows, split_filter, subset_filter):
        out = []
        for r in rows:
            if split_filter != "all" and r.get("split") != split_filter:
                continue
            if subset_filter != "all" and r.get("subset") != subset_filter:
                continue
            out.append(r)
        return out

    table: dict = {}
    for sp in splits:
        table.setdefault(sp if sp != "all" else "overall", {})
        key = sp if sp != "all" else "overall"
        for sub in subsets:
            sub_key = sub if sub != "all" else "all"
            rows = select(preds, sp, sub)
            table[key][sub_key] = {gt: metrics_block(rows, gt) for gt in gt_fields}
    return table


if __name__ == "__main__":
    main()
