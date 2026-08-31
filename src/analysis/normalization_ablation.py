"""Three-way result for the D/F normalization ablation (meta-review #2).

Compares, over the same units:

    raw atom        original expert atom text        (paper's atom run)
    normalized atom same atom, decontextualised      (this ablation)
    human snippet   annotator's rewrite              (paper's snippet run)

Nothing is grouped in any condition - these are the keep-atomic patterns - so
any movement between columns is attributable to text alone.

Reported subgroups:
  - `changed`   the normalizer actually rewrote the atom
  - `unchanged` the normalizer judged the atom already self-contained even
                though a human had reworded it. An internal control: with no
                rewrite applied, a wording account predicts no gain here.

The verifier's noise floor, measured on byte-identical text verified twice
(df_rewrite_split.py), is about 0.013 F1_F. Differences below that are not
meaningful.

Usage:
  python -m src.analysis.normalization_ablation
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / "data" / "14-normalization-ablation"
FINAL = ROOT / "data" / "!-final" / "dataset.json"
PRED = ROOT / "data" / "5-baselines" / "predictions"
NORM_PRED = ABL / "predictions" / "claim-only" / "openai__gpt-5.4-high" / "predictions.json"
NORM_PRED_ABCE = ABL / "predictions_abce" / "claim-only" / "openai__gpt-5.4-high" / "predictions.json"
UNITS_DF = ABL / "normalized_atoms.json"
UNITS_ABCE = ABL / "normalized_atoms_abce.json"
OUT_JSON = ABL / "ablation_result.json"

VERIFIER = "gpt-5.4-high"
NOISE_FLOOR = 0.013


def f1_false(pairs) -> float:
    tp = fp = fn = 0
    for gold, pred in pairs:
        gf, pf = (not gold), (not pred)
        if gf and pf:
            tp += 1
        elif not gf and pf:
            fp += 1
        elif gf and not pf:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> None:
    units = json.loads((ABL / "normalized_atoms.json").read_text())
    norm_pred = {r["snippet_id"]: r["prediction"]
                 for r in json.loads(NORM_PRED.read_text())}
    atom_pred = {r["id"]: r["prediction"] for r in json.loads(
        (PRED / "atom" / "claim-only" / VERIFIER / "predictions.json").read_text())}
    snip_pred = {r["snippet_id"]: r["prediction"] for r in json.loads(
        (PRED / "snippet" / "claim-only" / VERIFIER / "predictions.json").read_text())}

    groups: dict[str, dict[str, list]] = {}
    for u in units:
        if u["id"] not in norm_pred or u["id"] not in atom_pred:
            continue
        if u["snippet_id"] not in snip_pred:
            continue
        gold = u["label_atomic"]
        for key in ("all", "changed" if u.get("changed") else "unchanged",
                    f"pattern {u['pattern']}"):
            g = groups.setdefault(key, {"raw": [], "norm": [], "snip": []})
            g["raw"].append((gold, atom_pred[u["id"]]))
            g["norm"].append((gold, norm_pred[u["id"]]))
            g["snip"].append((gold, snip_pred[u["snippet_id"]]))

    order = ["all", "changed", "unchanged", "pattern D", "pattern F"]
    out = []
    for key in order:
        g = groups.get(key)
        if not g:
            continue
        raw, nrm, snp = f1_false(g["raw"]), f1_false(g["norm"]), f1_false(g["snip"])
        total = snp - raw
        out.append({
            "subgroup": key, "n": len(g["raw"]),
            "raw_atom": round(raw, 4), "normalized_atom": round(nrm, 4),
            "human_snippet": round(snp, 4),
            "norm_minus_raw": round(nrm - raw, 4),
            "snip_minus_raw": round(total, 4),
            "snip_minus_norm": round(snp - nrm, 4),
            "wording_share_pct": (round(100 * (nrm - raw) / total, 1)
                                  if abs(total) > 1e-9 else None),
        })
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print("\n".join(L))


if __name__ == "__main__":
    main()
