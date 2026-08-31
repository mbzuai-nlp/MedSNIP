"""Confidence intervals for every \\medsnipbench{} column in Table 3.

The published table marks significance only on the human-snippet column. The
two automatic-snippet columns were never tested, so this computes intervals for
all three against the shared expert-atom baseline.

Units are resampled by entry rather than individually. Snippets drawn from the
same answer are not independent, and the atom and snippet arms hold different
numbers of units per entry, so entry-level resampling is what keeps the two
arms comparable under the same draw.

Percentile intervals, for the same reason as the external script: the statistic
is computed over resampled entries rather than a flat sample.

Usage:
  python -m src.analysis.msb_column_cis
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "5-baselines" / "predictions"
OUT_JSON = ROOT / "data" / "11-analysis" / "significance-analysis" / "msb_column_cis.json"

VERIFIERS = [
    ("GPT-5.4-high", "gpt-5.4-high"),
    ("GPT-4o", "gpt-4o-none"),
    ("gemma-4-31B-it", "google__gemma-4-31B-it-none"),
    ("gpt-oss-20b", "openai__gpt-oss-20b-none"),
    ("Llama-3.3-70B", "meta-llama__Llama-3.3-70B-Instruct-none"),
    ("Llama-3.1-8B", "meta-llama__Llama-3.1-8B-Instruct-none"),
]

TP, FN, FP, TN = 0, 1, 2, 3


def code(gold: bool, pred: bool) -> int:
    gf, pf = (not gold), (not pred)
    if gf and pf:
        return TP
    if gf and not pf:
        return FN
    if not gf and pf:
        return FP
    return TN


def by_entry_atom(path: Path):
    out = defaultdict(list)
    for r in json.loads(path.read_text()):
        eid = int(str(r["id"]).split("-")[0])
        out[eid].append(code(bool(r["label"]), bool(r["prediction"])))
    return out


def by_entry_snip(path: Path, mask: Path | None = None):
    """Entry -> outcome codes.

    The automatic-snippet columns are scored under the drop-mixed projection,
    which discards snippets whose source atoms straddle sentences carrying
    conflicting human labels. That mask lives with the decomposition rather
    than the predictions, so it has to be joined back in. Skipping it scores
    112 extra Mode 1 snippets and inflates the gap from +0.108 to +0.176.
    """
    if not path.exists():
        return None
    keep = None
    if mask is not None and mask.exists():
        keep = {r["snippet_id"]: r for r in json.loads(mask.read_text())
                if not r.get("_touches_mixed", False)}
    out = defaultdict(list)
    for r in json.loads(path.read_text()):
        if keep is not None:
            lab = keep.get(r["id"])
            if lab is None:
                continue
            gold = bool(lab["label_atomic_and"])
        else:
            gold = bool(r["label_atomic"])
        out[int(r["entry_id"])].append(code(gold, bool(r["prediction"])))
    return out


def f1(codes) -> float:
    a = np.asarray(codes)
    tp = int((a == TP).sum()); fn = int((a == FN).sum()); fp = int((a == FP).sum())
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def gap(atom, snip, entries) -> float:
    A = [c for e in entries for c in atom.get(e, ())]
    S = [c for e in entries for c in snip.get(e, ())]
    if not A or not S:
        return 0.0
    return f1(S) - f1(A)


def boot(atom, snip, n_boot, seed):
    entries = sorted(set(atom) & set(snip))
    point = gap(atom, snip, entries)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(entries))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = [entries[i] for i in rng.choice(idx, size=len(entries), replace=True)]
        draws[b] = gap(atom, snip, pick)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for disp, slug in VERIFIERS:
        atom = by_entry_atom(PRED / "atom" / "claim-only" / slug / "predictions.json")
        SP = ROOT / "data" / "6-snippet-processor"
        cols = [
            ("snippet", PRED / "snippet" / "claim-only" / slug / "predictions.json", None),
            ("Mode 1", PRED / "snippet" / "claim-only" / "auto-a2s" / slug / "predictions.json",
             SP / "atom-to-snippet" / "gpt-5.4-high" / "labeled_snippets.json"),
            ("Mode 2", PRED / "snippet" / "claim-only" / "auto-direct" / slug / "predictions.json",
             SP / "snippet-direct" / "gpt-5.4-high" / "labeled_snippets.json"),
        ]
        for name, path, mask in cols:
            snip = by_entry_snip(path, mask)
            if snip is None:
                continue
            d, lo, hi, n = boot(atom, snip, args.n_boot, args.seed)
            sig = lo > 0 or hi < 0
            rows.append({"verifier": disp, "column": name, "delta": round(d, 4),
                         "ci": [round(lo, 4), round(hi, 4)],
                         "significant": bool(sig), "n_entries": n})
            print(f"{disp:15s} {name:8s} d={d:+.4f} [{lo:+.3f},{hi:+.3f}]"
                  f"{'*' if sig else ''}", flush=True)

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    n_sig = sum(1 for r in rows if r["significant"])
    print(f"\n{n_sig}/{len(rows)} significant")


if __name__ == "__main__":
    main()
