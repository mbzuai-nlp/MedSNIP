"""Bootstrap confidence intervals on the external snippet-vs-atom deltas.

Table 3 marks significance on \\medsnipbench{} but the external columns were
only shaded where a mode beats its atom baseline, which is not the same claim.
This computes the missing intervals.

Both arms aggregate unit predictions to the claim with OR-FALSE and are scored
over the same claim set, so the comparison is naturally paired at claim level.
Resampling claims keeps that pairing and keeps units belonging to one claim
together, which unit-level resampling would break.

Percentile intervals rather than BCa: the statistic is computed over resampled
claims rather than over a flat sample, which is not the form scipy's BCa
acceleration term is defined for. Percentile intervals run slightly wider here,
which is the conservative direction.

Usage:
  python -m src.analysis.external_cis
  python -m src.analysis.external_cis --n-boot 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "data" / "11-analysis" / "external-analysis" / "external_cis.json"

VERIFIERS = [
    ("GPT-5.4-high", "gpt-5.4-high", "openai__gpt-5.4-high"),
    ("GPT-4o", "gpt-4o-none", "openai__gpt-4o-none"),
    ("gemma-4-31B-it", "google__gemma-4-31B-it-none", "google__gemma-4-31b-it-none"),
    ("gpt-oss-20b", "openai__gpt-oss-20b-none", "openai__gpt-oss-20b-none"),
    ("Llama-3.3-70B", "meta-llama__Llama-3.3-70B-Instruct-none",
     "meta-llama__llama-3.3-70b-instruct-none"),
    ("Llama-3.1-8B", "meta-llama__Llama-3.1-8B-Instruct-none",
     "meta-llama__llama-3.1-8b-instruct-none"),
]


def healthfc_gold():
    out = {}
    for i, r in enumerate(json.loads((ROOT / "data/1-raw/healthfc.json").read_text())):
        l = r["label"]
        out[f"hfc-{i}"] = True if l == 0 else (False if l == 2 else None)
    return out


def medhallu_gold(cid):
    return cid.rsplit("-", 1)[1] == "T"


def claim_preds(path: Path):
    """claim id -> OR-FALSE aggregated prediction."""
    if not path.exists():
        return None
    by = {}
    for r in json.loads(path.read_text()):
        by.setdefault(r.get("claim_id") or r["id"], []).append(bool(r["prediction"]))
    return {c: all(v) for c, v in by.items()}


def f1_false(gold, pred, ids):
    tp = fp = fn = 0
    for c in ids:
        gf, pf = (not gold[c]), (not pred[c])
        if gf and pf:
            tp += 1
        elif not gf and pf:
            fp += 1
        elif gf and not pf:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def boot(gold, atom, snip, n_boot, seed):
    ids = [c for c in atom if c in snip and gold.get(c) is not None]
    point = f1_false(gold, snip, ids) - f1_false(gold, atom, ids)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = [ids[i] for i in rng.choice(idx, size=len(ids), replace=True)]
        draws[b] = f1_false(gold, snip, pick) - f1_false(gold, atom, pick)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hg = healthfc_gold()
    corpora = [
        ("HealthFC", ROOT / "data/9-healthfc/baselines/predictions", hg, "decl-"),
        ("MedHallu", ROOT / "data/10-medhallu/baselines/predictions",
         {}, ""),  # gold computed from id
    ]
    rows = []
    for corpus, base, gold_map, pfx in corpora:
        for disp, pub_slug, new_slug in VERIFIERS:
            atom_p = base / "atom" / "claim-only" / (f"{pfx}{new_slug}" if pfx else pub_slug) / "predictions.json"
            atom = claim_preds(atom_p)
            if atom is None:
                continue
            gold = gold_map if gold_map else {c: medhallu_gold(c) for c in atom}
            for mode, tag in (("Mode 1", f"{pfx}{new_slug}"),
                              ("Mode 2", f"{pfx}mode2-{new_slug}")):
                snip = claim_preds(base / "pipeline-snippet" / "claim-only" / tag / "predictions.json")
                if snip is None:
                    continue
                d, lo, hi, n = boot(gold, atom, snip, args.n_boot, args.seed)
                sig = lo > 0 or hi < 0
                rows.append({"corpus": corpus, "verifier": disp, "mode": mode,
                             "delta": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
                             "significant": bool(sig), "n_claims": n})
                print(f"{corpus:9s} {disp:15s} {mode}  d={d:+.4f}  "
                      f"[{lo:+.3f},{hi:+.3f}]{'*' if sig else ''}", flush=True)

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    n_sig = sum(1 for r in rows if r["significant"])
    print(f"\n{n_sig}/{len(rows)} significant")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
