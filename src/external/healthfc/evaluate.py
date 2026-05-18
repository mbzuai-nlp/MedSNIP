"""Evaluate HealthFC predictions at claim level for both grains.

For the snippet grain: 1 prediction per claim, directly compared to gold.
For the atom grain: N predictions per claim, aggregated to claim level via
OR-False (any atom False ⇒ claim False), then compared to gold.

Headline metric: F1_F (false class) on the binary subset (label ∈ {0, 2}).
Also reports: F1_T, macro, accuracy, n; plus secondary stats on the
"insufficient evidence" bucket (label = 1, treated as a separate report).

Output: data/9-healthfc/metrics.json + console headline table.

Usage:
    python -m src.external.healthfc.evaluate
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRED_BASE = ROOT / "data" / "9-healthfc" / "baselines" / "predictions"
OUT_PATH = ROOT / "data" / "9-healthfc" / "metrics.json"


def f1_block(rows: list[dict]) -> dict:
    """rows: each has 'gold_bool' (T/F/None) and 'pred' (T/F).
    None gold_bool rows are excluded from binary F1."""
    binary = [r for r in rows if r["gold_bool"] is not None]
    tp_f = fp_f = fn_f = 0
    tp_t = fp_t = fn_t = 0
    n_correct = 0
    for r in binary:
        gv = r["gold_bool"]
        pv = bool(r["pred"])
        if pv == gv:
            n_correct += 1
        if gv is False and pv is False: tp_f += 1
        elif gv is False and pv is True: fn_f += 1
        elif gv is True and pv is False: fp_f += 1
        if gv is True and pv is True: tp_t += 1
        elif gv is True and pv is False: fn_t += 1
        elif gv is False and pv is True: fp_t += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    p_f, r_f, f1_f = prf(tp_f, fp_f, fn_f)
    p_t, r_t, f1_t = prf(tp_t, fp_t, fn_t)
    return {
        "n":           len(binary),
        "n_true_gold": sum(1 for r in binary if r["gold_bool"] is True),
        "n_false_gold": sum(1 for r in binary if r["gold_bool"] is False),
        "accuracy":    round(n_correct / len(binary), 4) if binary else 0.0,
        "precision_F": p_f, "recall_F": r_f, "f1_F": f1_f,
        "precision_T": p_t, "recall_T": r_t, "f1_T": f1_t,
        "macro_f1":    round((f1_f + f1_t) / 2, 4),
    }


def insufficient_report(rows: list[dict]) -> dict:
    """How does the model behave on label=1 (insufficient evidence)?"""
    insuf = [r for r in rows if r["label"] == 1]
    if not insuf:
        return {"n": 0}
    n_true = sum(1 for r in insuf if bool(r["pred"]) is True)
    n_false = sum(1 for r in insuf if bool(r["pred"]) is False)
    return {
        "n":       len(insuf),
        "n_pred_True":  n_true,
        "n_pred_False": n_false,
        "pct_True":  round(n_true / len(insuf), 4),
        "pct_False": round(n_false / len(insuf), 4),
    }


def aggregate_atoms_to_claims(atom_preds: list[dict]) -> list[dict]:
    """Roll up atom-level predictions to claim level via OR-False rule:
    a claim is True only if ALL its atoms are True; any False atom → False.
    """
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for r in atom_preds:
        by_claim[r["claim_id"]].append(r)
    claim_rows = []
    for cid, group in by_claim.items():
        any_false = any(bool(g["prediction"]) is False for g in group)
        pred = not any_false
        # Take gold/label from any atom (they all share)
        g0 = group[0]
        claim_rows.append({
            "claim_id":   cid,
            "pred":       pred,
            "n_atoms":    len(group),
            "label":      g0["label"],
            "gold_bool":  g0["gold_bool"],
        })
    return claim_rows


def load_snippet_predictions(pred_path: Path) -> list[dict]:
    raw = json.loads(pred_path.read_text())
    return [{
        "claim_id":  r["claim_id"],
        "pred":      bool(r["prediction"]),
        "label":     r["label"],
        "gold_bool": r["gold_bool"],
    } for r in raw]


def find_runs(grain: str) -> list[tuple[str, Path]]:
    """Yield (tag, predictions.json) under the given grain."""
    root = PRED_BASE / grain / "claim-only"
    if not root.exists():
        return []
    out = []
    for model_dir in sorted(root.iterdir()):
        p = model_dir / "predictions.json"
        if p.exists():
            out.append((model_dir.name, p))
    return out


def main():
    results: dict = {"snippet": {}, "atom": {}}
    headline_rows: list[tuple] = []

    # Snippet grain: direct
    for tag, path in find_runs("snippet"):
        rows = load_snippet_predictions(path)
        cell = f1_block(rows)
        results["snippet"][tag] = {
            "binary": cell,
            "insufficient_eval_bucket": insufficient_report([
                {"label": r["label"], "pred": r["pred"]} for r in rows
            ]),
            "n_predictions": len(rows),
        }
        headline_rows.append(("snippet", tag, cell))

    # Atom grain: aggregate first, then F1
    for tag, path in find_runs("atom"):
        atom_preds = json.loads(path.read_text())
        claim_rows = aggregate_atoms_to_claims(atom_preds)
        cell = f1_block(claim_rows)
        results["atom"][tag] = {
            "binary": cell,
            "insufficient_eval_bucket": insufficient_report([
                {"label": r["label"], "pred": r["pred"]} for r in claim_rows
            ]),
            "n_predictions": len(claim_rows),
            "n_atoms": len(atom_preds),
            "avg_atoms_per_claim": round(len(atom_preds) / max(len(claim_rows), 1), 2),
        }
        headline_rows.append(("atom", tag, cell))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print("\nHeadline F1 (HealthFC binary subset, claim-level):")
    print(f"{'grain':<8s} {'model-tag':<24s} {'n':>4s} {'acc':>6s} "
          f"{'P_F':>6s} {'R_F':>6s} {'F1_F':>6s} {'F1_T':>6s} {'macro':>6s}")
    for grain, tag, cell in headline_rows:
        if cell["n"] == 0:
            continue
        print(f"{grain:<8s} {tag:<24s} {cell['n']:>4d} {cell['accuracy']:>6.3f} "
              f"{cell['precision_F']:>6.3f} {cell['recall_F']:>6.3f} "
              f"{cell['f1_F']:>6.3f} {cell['f1_T']:>6.3f} {cell['macro_f1']:>6.3f}")

    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
