"""Per-pattern snippet-vs-atom F1_F gap analysis.

For each structural pattern (A/B/C/D/E/F), compute:
  - n snippets carrying that pattern
  - snippet baseline F1_F
  - atom baseline F1_F (aggregated to snippet via OR-False)
  - delta (snippet - atom)
  - 95% paired bootstrap CI on the delta
  - bootstrap-estimated P(delta > 0)

Uses gpt-5.4-high claim-only baselines and label_human_general as gold.

Usage:
    python -m src.analysis.pattern_f1_gap
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "15-final" / "dataset.json"
SNIPPET_PRED = ROOT / "data" / "5-baselines" / "predictions" / "snippet" / "claim-only" / "gpt-5.4-high" / "predictions.json"
ATOM_PRED = ROOT / "data" / "5-baselines" / "predictions" / "atom" / "claim-only" / "gpt-5.4-high" / "predictions.json"

GOLD_KEY = "label_human_general"
PATTERNS = ["A", "B", "C", "D", "E", "F"]
N_BOOT = 10000
SEED = 0


def f1_f(y_gold: list[bool], y_pred: list[bool]) -> float:
    """F1 on the False class (the harder, less-trivial class)."""
    tp = sum(1 for g, p in zip(y_gold, y_pred) if g is False and p is False)
    fp = sum(1 for g, p in zip(y_gold, y_pred) if g is True and p is False)
    fn = sum(1 for g, p in zip(y_gold, y_pred) if g is False and p is True)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def paired_bootstrap_delta(
    y_gold: list[bool],
    y_snip: list[bool],
    y_atom: list[bool],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    rng = random.Random(seed)
    n = len(y_gold)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [y_gold[i] for i in idx]
        s = [y_snip[i] for i in idx]
        a = [y_atom[i] for i in idx]
        deltas.append(f1_f(g, s) - f1_f(g, a))
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    p_positive = sum(1 for d in deltas if d > 0) / n_boot
    return {"ci_lo": lo, "ci_hi": hi, "P(delta>0)": p_positive}


def main() -> None:
    print("loading dataset + predictions ...")
    data = json.load(open(DATASET))
    snip_preds = {p["snippet_id"]: p["prediction"] for p in json.load(open(SNIPPET_PRED))}
    atom_preds = {p["id"]: p["prediction"] for p in json.load(open(ATOM_PRED))}
    print(f"  snippets in dataset: {len(data)}")
    print(f"  snippet predictions: {len(snip_preds)}")
    print(f"  atom predictions:    {len(atom_preds)}")

    # Per snippet, get snippet prediction and OR-False aggregated atom prediction.
    rows = []
    missing_snip = 0
    missing_atom = 0
    for r in data:
        sid = r["snippet_id"]
        eid = r["entry_id"]
        gold = r.get(GOLD_KEY)
        pattern = r.get("pattern")
        subset = r.get("subset")
        if gold not in (True, False):
            continue
        if sid not in snip_preds:
            missing_snip += 1
            continue
        # OR-False over atoms: True iff ALL atoms predicted True; else False
        atom_indices = [a["index"] for a in r.get("atoms", [])]
        atom_ids = [f"{eid}-{i}" for i in atom_indices]
        if not all(aid in atom_preds for aid in atom_ids):
            missing_atom += 1
            continue
        atom_agg = all(atom_preds[aid] is True for aid in atom_ids)
        snip_pred = snip_preds[sid] is True
        rows.append({
            "snippet_id": sid,
            "pattern": pattern,
            "subset": subset,
            "split": r.get("split"),
            "gold": gold is True,
            "snip_pred": snip_pred,
            "atom_pred": atom_agg,
        })

    print(f"\nrows usable: {len(rows)}")
    if missing_snip or missing_atom:
        print(f"  missing snippet pred: {missing_snip}")
        print(f"  missing atom pred:    {missing_atom}")

    # By pattern
    print(f"\n{'Pattern':<10} {'n':>5}  {'snip_F1F':>9}  {'atom_F1F':>9}  {'delta':>7}  {'CI':>20}  {'P(>0)':>6}")
    print("-" * 80)

    overall_g = [r["gold"] for r in rows]
    overall_s = [r["snip_pred"] for r in rows]
    overall_a = [r["atom_pred"] for r in rows]
    snip_overall = f1_f(overall_g, overall_s)
    atom_overall = f1_f(overall_g, overall_a)
    boot = paired_bootstrap_delta(overall_g, overall_s, overall_a)
    print(f"{'overall':<10} {len(overall_g):>5}  {snip_overall:>9.3f}  {atom_overall:>9.3f}  "
          f"{snip_overall-atom_overall:>+7.3f}  "
          f"[{boot['ci_lo']:>+.3f}, {boot['ci_hi']:>+.3f}]  "
          f"{boot['P(delta>0)']:>6.3f}")

    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pattern[r["pattern"]].append(r)

    summary = {"overall": {
        "n": len(rows),
        "snip_F1F": snip_overall,
        "atom_F1F": atom_overall,
        "delta": snip_overall - atom_overall,
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "P(delta>0)": boot["P(delta>0)"],
    }}
    for p in PATTERNS:
        rr = by_pattern.get(p, [])
        if not rr:
            continue
        g = [r["gold"] for r in rr]
        s = [r["snip_pred"] for r in rr]
        a = [r["atom_pred"] for r in rr]
        snip_f1 = f1_f(g, s)
        atom_f1 = f1_f(g, a)
        bootp = paired_bootstrap_delta(g, s, a)
        print(f"{p:<10} {len(rr):>5}  {snip_f1:>9.3f}  {atom_f1:>9.3f}  "
              f"{snip_f1-atom_f1:>+7.3f}  "
              f"[{bootp['ci_lo']:>+.3f}, {bootp['ci_hi']:>+.3f}]  "
              f"{bootp['P(delta>0)']:>6.3f}")
        summary[p] = {
            "n": len(rr),
            "snip_F1F": snip_f1,
            "atom_F1F": atom_f1,
            "delta": snip_f1 - atom_f1,
            "ci_lo": bootp["ci_lo"],
            "ci_hi": bootp["ci_hi"],
            "P(delta>0)": bootp["P(delta>0)"],
        }

    # Merge-only (A/B/C) vs Atomic-only (D/E/F)
    print()
    for group_name, codes in [("MERGE (A+B+C)", ["A", "B", "C"]),
                                ("ATOMIC (D+E+F)", ["D", "E", "F"])]:
        rr = [r for r in rows if r["pattern"] in codes]
        g = [r["gold"] for r in rr]
        s = [r["snip_pred"] for r in rr]
        a = [r["atom_pred"] for r in rr]
        snip_f1 = f1_f(g, s)
        atom_f1 = f1_f(g, a)
        bootp = paired_bootstrap_delta(g, s, a)
        print(f"{group_name:<14} {len(rr):>5}  {snip_f1:>9.3f}  {atom_f1:>9.3f}  "
              f"{snip_f1-atom_f1:>+7.3f}  "
              f"[{bootp['ci_lo']:>+.3f}, {bootp['ci_hi']:>+.3f}]  "
              f"{bootp['P(delta>0)']:>6.3f}")

    out = ROOT / "data" / "11-analysis" / "pattern-analysis" / "pattern_f1_gap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
