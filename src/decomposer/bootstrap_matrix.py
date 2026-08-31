"""95% BCa confidence intervals on every snippet-vs-atom delta in the matrix.

Reuses the encoding, F1 and bootstrap machinery from
`src.analysis.bootstrap_baseline_gaps` so these intervals are computed exactly
the way the paper's published intervals are, and remain comparable to them.

Two resampling schemes are reported:

  - **row**    — resample individual units, matching the paper's published
                 method. Comparable to Tables 3 and 4.
  - **entry**  — resample whole entries (answers), carrying all their units
                 together. Snippets within an answer are not independent, so
                 row-level intervals are too narrow; the existing baseline
                 script flags this as the better choice for final reporting.
                 Where the two disagree, trust `entry`.

A delta is called significant only when the interval excludes zero.

Usage:
  python -m src.decomposer.bootstrap_matrix
  python -m src.decomposer.bootstrap_matrix --n-boot 20000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

from ..analysis.bootstrap_baseline_gaps import encode, f1_F_axis, gap_stat
from .build_matrix import EXPERT_SLUG, cell_score  # noqa: F401  (cell_score kept for parity)
from .models import DISPLAY, MATRIX_MODELS
from .run_verify import cell_path, labeled_path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "12-decomposer"
OUT_JSON = OUT_DIR / "matrix_cis_bca.json"

REFERENCE_MODEL = "openai/gpt-5.4"


def load_arm(decomposer: str, unit: str, verifier: str,
             decomp_mode: str = "atom-to-snippet"):
    """Return (codes, entry_ids) for a complete cell, else None."""
    lp = labeled_path(decomposer, unit, decomp_mode)
    pp = cell_path(decomposer, unit, verifier, decomp_mode)
    if not (lp.exists() and pp.exists()):
        return None
    labels = {r["snippet_id"]: r for r in json.loads(lp.read_text())}
    preds = json.loads(pp.read_text())
    if len(preds) < len(labels):
        return None
    rows, eids = [], []
    for p in preds:
        lab = labels.get(p["snippet_id"])
        if lab is None or lab.get("_touches_mixed"):
            continue  # drop-mixed projection, as in the main table
        if lab.get("label_atomic_and") is None:
            continue
        rows.append({"prediction": p["prediction"],
                     "label_atomic_and": lab["label_atomic_and"]})
        eids.append(lab["entry_id"])
    if not rows:
        return None
    return encode(rows, "label_atomic_and"), np.array(eids)


def bca_row(atom_codes, snip_codes, n_boot, seed):
    point = float(gap_stat(atom_codes, snip_codes))
    res = bootstrap(
        data=(atom_codes, snip_codes), statistic=gap_stat, method="BCa",
        n_resamples=n_boot, paired=False, vectorized=True,
        confidence_level=0.95, random_state=np.random.default_rng(seed),
    )
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def bca_entry(atom_codes, atom_eids, snip_codes, snip_eids, n_boot, seed):
    """Cluster bootstrap: resample entries, keeping each entry's units together.

    Percentile interval rather than BCa — the jackknife acceleration term BCa
    needs is defined over the resampling unit, and scipy cannot express a
    cluster statistic in the form its BCa implementation requires. Percentile
    intervals are slightly wider here, which is the conservative direction.
    """
    by_atom, by_snip = defaultdict(list), defaultdict(list)
    for c, e in zip(atom_codes, atom_eids):
        by_atom[e].append(c)
    for c, e in zip(snip_codes, snip_eids):
        by_snip[e].append(c)
    entries = sorted(set(by_atom) | set(by_snip))
    a_arr = {e: np.array(v, dtype=np.int8) for e, v in by_atom.items()}
    s_arr = {e: np.array(v, dtype=np.int8) for e, v in by_snip.items()}
    empty = np.array([], dtype=np.int8)

    point = float(f1_F_axis(snip_codes) - f1_F_axis(atom_codes))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(entries))
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(entries), replace=True)
        a = np.concatenate([a_arr.get(entries[i], empty) for i in pick])
        s = np.concatenate([s_arr.get(entries[i], empty) for i in pick])
        draws[b] = f1_F_axis(s) - f1_F_axis(a)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decomp-mode", default="atom-to-snippet",
                    choices=["atom-to-snippet", "snippet-direct"],
                    help="which snippet arm to test. Mode 2 emits no atoms, so "
                         "its baseline is Mode 1's atoms from the same decomposer")
    ap.add_argument("--out-stem", default=None,
                    help="output filename stem (default derived from mode)")
    args = ap.parse_args()

    global OUT_JSON
    stem = args.out_stem or ("matrix_cis_bca" if args.decomp_mode == "atom-to-snippet"
                             else "matrix_cis_bca_mode2")
    OUT_JSON = OUT_DIR / f"{stem}.json"

    decomposers = [m for m in MATRIX_MODELS if m != REFERENCE_MODEL]
    results = []

    for v in MATRIX_MODELS:
        for d in decomposers:
            # Mode 2 has no atoms of its own; its baseline is Mode 1's atoms
            # from the same decomposer, which keeps the comparison within-model.
            a = load_arm(d, "atom", v, "atom-to-snippet")
            s = load_arm(d, "snippet", v, args.decomp_mode)
            if a is None or s is None:
                continue
            a_codes, a_eids = a
            s_codes, s_eids = s
            pt, lo, hi = bca_row(a_codes, s_codes, args.n_boot, args.seed)
            ept, elo, ehi = bca_entry(a_codes, a_eids, s_codes, s_eids,
                                      args.n_boot, args.seed)
            results.append({
                "decomposer": d, "verifier": v, "decomp_mode": args.decomp_mode,
                "delta": round(pt, 4),
                "row_ci": [round(lo, 4), round(hi, 4)],
                "row_sig": bool(lo > 0 or hi < 0),
                "entry_ci": [round(elo, 4), round(ehi, 4)],
                "entry_sig": bool(elo > 0 or ehi < 0),
                "n_atom": int(len(a_codes)), "n_snippet": int(len(s_codes)),
            })
            print(f"{DISPLAY[v]:16s} {DISPLAY[d]:16s} "
                  f"Δ={pt:+.4f} row[{lo:+.3f},{hi:+.3f}]"
                  f"{'*' if (lo > 0 or hi < 0) else ' '} "
                  f"entry[{elo:+.3f},{ehi:+.3f}]"
                  f"{'*' if (elo > 0 or ehi < 0) else ' '}", flush=True)

    OUT_JSON.write_text(json.dumps(results, indent=2))

    lines = ["# Matrix deltas with 95% bootstrap CIs", "",
             f"Snippet minus atom false-class F1, {args.n_boot:,} resamples, "
             f"seed {args.seed}. `row` resamples units (the paper's published "
             f"method, BCa). `entry` resamples whole answers (cluster "
             f"percentile) and is the more honest interval, since units within "
             f"an answer are dependent. `*` marks an interval excluding zero.", ""]
    for v in MATRIX_MODELS:
        col = [r for r in results if r["verifier"] == v]
        if not col:
            continue
        lines += [f"## Verifier: {DISPLAY[v]}", "",
                  "| Decomposer | Δ | row 95% CI | entry 95% CI |",
                  "|---|---:|---|---|"]
        for r in col:
            lines.append(
                f"| {DISPLAY[r['decomposer']]} | {r['delta']:+.3f} | "
                f"[{r['row_ci'][0]:+.3f}, {r['row_ci'][1]:+.3f}]"
                f"{' *' if r['row_sig'] else ''} | "
                f"[{r['entry_ci'][0]:+.3f}, {r['entry_ci'][1]:+.3f}]"
                f"{' *' if r['entry_sig'] else ''} |")
        lines.append("")
    n_row = sum(1 for r in results if r["row_sig"])
    n_ent = sum(1 for r in results if r["entry_sig"])
    lines += ["## Summary", "",
              f"- pairs: **{len(results)}**",
              f"- significant under row resampling: **{n_row}/{len(results)}**",
              f"- significant under entry clustering: **{n_ent}/{len(results)}**", ""]
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
