"""Per-pattern snippet-vs-atom F1_F gap under entry-clustered resampling.

`pattern_f1_gap.py` resamples individual snippets. Snippets drawn from the same
source answer are not independent: they share a topic, a shared-context block,
and one annotator's segmentation decisions, so a flat bootstrap understates the
variance of the gap. Table 3 of the paper already clusters by entry; this
script applies the same resampling unit to the per-pattern table so the two
report comparable intervals.

Both estimators are computed side by side so the difference is visible rather
than asserted. The point estimates are identical by construction; only the
intervals move.

Intervals are percentile, matching `pattern_f1_gap.py` and the clustered
analyses elsewhere in the repo. A cluster statistic is not of the flat form the
BCa jackknife acceleration term is defined for.

Usage:
    python -m src.analysis.pattern_f1_gap_clustered
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "!-final" / "dataset.json"
SNIPPET_PRED = ROOT / "data" / "5-baselines" / "predictions" / "snippet" / "claim-only" / "gpt-5.4-high" / "predictions.json"
ATOM_PRED = ROOT / "data" / "5-baselines" / "predictions" / "atom" / "claim-only" / "gpt-5.4-high" / "predictions.json"
OUT_JSON = ROOT / "data" / "11-analysis" / "pattern-analysis" / "pattern_f1_gap_clustered.json"

GOLD_KEY = "label_human_general"
PATTERNS = ["A", "B", "C", "D", "E", "F"]
N_BOOT = 10000
SEED = 42


def f1_f(gold: list[bool], pred: list[bool]) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g is False and p is False)
    fp = sum(1 for g, p in zip(gold, pred) if g is True and p is False)
    fn = sum(1 for g, p in zip(gold, pred) if g is False and p is True)
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def _interval(deltas: list[float]) -> dict:
    deltas.sort()
    n = len(deltas)
    return {
        "ci_lo": deltas[int(0.025 * n)],
        "ci_hi": deltas[int(0.975 * n)],
        "P(delta>0)": sum(1 for d in deltas if d > 0) / n,
    }


def boot_flat(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Resample individual snippets, as in pattern_f1_gap.py."""
    rng = random.Random(seed)
    n = len(rows)
    deltas = []
    for _ in range(n_boot):
        draw = [rows[rng.randrange(n)] for _ in range(n)]
        g = [r["gold"] for r in draw]
        deltas.append(f1_f(g, [r["snip"] for r in draw])
                      - f1_f(g, [r["atom"] for r in draw]))
    return _interval(deltas)


def boot_clustered(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Resample source entries, carrying all of an entry's rows together."""
    by_entry: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_entry[r["entry"]].append(r)
    entries = sorted(by_entry)
    rng = random.Random(seed)
    k = len(entries)
    deltas = []
    for _ in range(n_boot):
        draw = [r for _ in range(k) for r in by_entry[entries[rng.randrange(k)]]]
        if not draw:
            continue
        g = [r["gold"] for r in draw]
        deltas.append(f1_f(g, [r["snip"] for r in draw])
                      - f1_f(g, [r["atom"] for r in draw]))
    return _interval(deltas)


def load_rows() -> list[dict]:
    data = json.load(open(DATASET))
    snip = {p["snippet_id"]: p["prediction"] for p in json.load(open(SNIPPET_PRED))}
    atom = {p["id"]: p["prediction"] for p in json.load(open(ATOM_PRED))}
    rows = []
    for r in data:
        gold = r.get(GOLD_KEY)
        if gold not in (True, False) or r["snippet_id"] not in snip:
            continue
        ids = [f"{r['entry_id']}-{a['index']}" for a in r.get("atoms", [])]
        if not all(i in atom for i in ids):
            continue
        rows.append({
            "entry": r["entry_id"],
            "pattern": r.get("pattern"),
            "gold": gold is True,
            "snip": snip[r["snippet_id"]] is True,
            # OR-False: predicted True only if every covered atom is predicted True.
            "atom": all(atom[i] is True for i in ids),
        })
    return rows


def main() -> None:
    rows = load_rows()
    entries = len({r["entry"] for r in rows})
    print(f"rows={len(rows)}  entries={entries}\n")

    out = []
    groups = [("Overall", rows)] + [(p, [r for r in rows if r["pattern"] == p])
                                    for p in PATTERNS]
    hdr = (f"{'Pattern':<8} {'n':>5} {'entries':>7} {'snip':>7} {'atom':>7} {'delta':>8}  "
           f"{'flat CI':>18} {'clustered CI':>18}  {'flat':>5} {'clus':>5}")
    print(hdr)
    print("-" * len(hdr))
    for name, grp in groups:
        if not grp:
            continue
        g = [r["gold"] for r in grp]
        s = f1_f(g, [r["snip"] for r in grp])
        a = f1_f(g, [r["atom"] for r in grp])
        flat = boot_flat(grp, N_BOOT, SEED)
        clus = boot_clustered(grp, N_BOOT, SEED)
        rec = {
            "pattern": name, "n": len(grp),
            "entries": len({r["entry"] for r in grp}),
            "snip_F1F": round(s, 4), "atom_F1F": round(a, 4),
            "delta": round(s - a, 4),
            "flat_ci": [round(flat["ci_lo"], 4), round(flat["ci_hi"], 4)],
            "clustered_ci": [round(clus["ci_lo"], 4), round(clus["ci_hi"], 4)],
            "flat_sig": flat["ci_lo"] > 0 or flat["ci_hi"] < 0,
            "clustered_sig": clus["ci_lo"] > 0 or clus["ci_hi"] < 0,
        }
        out.append(rec)
        print(f"{name:<8} {rec['n']:>5} {rec['entries']:>7} {s:>7.3f} {a:>7.3f} {s-a:>+8.3f}  "
              f"[{flat['ci_lo']:>+.3f},{flat['ci_hi']:>+.3f}] "
              f"[{clus['ci_lo']:>+.3f},{clus['ci_hi']:>+.3f}]  "
              f"{'*' if rec['flat_sig'] else '-':>5} {'*' if rec['clustered_sig'] else '-':>5}")

    flips = [r for r in out if r["flat_sig"] != r["clustered_sig"]]
    print(f"\ncells whose significance changes: {len(flips)}"
          + (" (" + ", ".join(r["pattern"] for r in flips) + ")" if flips else ""))

    OUT_JSON.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
