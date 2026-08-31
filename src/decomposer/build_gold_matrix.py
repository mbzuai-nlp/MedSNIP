"""Decomposer-by-verifier matrix against the gold atom baseline.

\\medsnipbench{} inherits expert atomic claims from Kim et al., so the atom
baseline is given rather than generated. Every decomposer row is therefore
scored against the same human atom set, which is also the atom column of
Table 3. That keeps rows comparable to each other and to the main table, and
means no decomposer needs its own atoms verified.

(The external corpora are different: they ship no atoms, so there the pipeline
has to produce both arms.)

Snippet scores use each decomposer's own drop-mixed projection, matching the
convention used for the automatic-snippet columns in the main table.

Usage:
  python -m src.decomposer.build_gold_matrix
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SP = ROOT / "data" / "6-snippet-processor"
PRED = ROOT / "data" / "5-baselines" / "predictions"
MATRIX = ROOT / "data" / "12-decomposer" / "predictions" / "claim-only"
OUT_JSON = ROOT / "data" / "12-decomposer" / "gold_matrix.json"

VERIFIERS = [
    ("GPT-5.4", "gpt-5.4-high", "openai__gpt-5.4-high"),
    ("GPT-4o", "gpt-4o-none", "openai__gpt-4o-none"),
    ("gemma", "google__gemma-4-31B-it-none", "google__gemma-4-31b-it-none"),
    ("oss-20b", "openai__gpt-oss-20b-none", "openai__gpt-oss-20b-none"),
    ("L-70B", "meta-llama__Llama-3.3-70B-Instruct-none", "meta-llama__llama-3.3-70b-instruct-none"),
    ("L-8B", "meta-llama__Llama-3.1-8B-Instruct-none", "meta-llama__llama-3.1-8b-instruct-none"),
]
# decomposer -> (slug in data dirs, is the reference row)
DECOMPOSERS = [
    ("GPT-5.4", "gpt-5.4-high", True),
    ("GPT-4o", "openai__gpt-4o-none", False),
    ("gemma", "google__gemma-4-31b-it-none", False),
    ("oss-20b", "openai__gpt-oss-20b-none", False),
    ("L-70B", "meta-llama__llama-3.3-70b-instruct-none", False),
    ("L-8B", "meta-llama__llama-3.1-8b-instruct-none", False),
]
MODES = [("Mode 1", "atom-to-snippet", "auto-a2s"),
         ("Mode 2", "snippet-direct", "auto-direct")]


def f1_false(pairs):
    tp = fp = fn = 0
    for g, p in pairs:
        gf, pf = (not g), (not p)
        if gf and pf:
            tp += 1
        elif not gf and pf:
            fp += 1
        elif gf and not pf:
            fn += 1
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def gold_atom(verifier_slug):
    """Kim et al. expert atoms, entry -> outcome pairs."""
    rows = json.loads((PRED / "atom" / "claim-only" / verifier_slug / "predictions.json").read_text())
    out = {}
    for r in rows:
        eid = int(str(r["id"]).split("-")[0])
        out.setdefault(eid, []).append((bool(r["label"]), bool(r["prediction"])))
    return out


def snippet_arm(dec_slug, is_ref, mode_dir, ref_tag, verifier_slug, verifier_new):
    """Entry -> outcome pairs under each decomposer's drop-mixed projection."""
    mask_p = SP / mode_dir / dec_slug / "labeled_snippets.json"
    if not mask_p.exists():
        return None
    keep = {r["snippet_id"]: r for r in json.loads(mask_p.read_text())
            if not r.get("_touches_mixed", False)}
    if is_ref:
        pred_p = PRED / "snippet" / "claim-only" / ref_tag / verifier_slug / "predictions.json"
    else:
        unit = "snippet" if mode_dir == "atom-to-snippet" else "mode2-snippet"
        pred_p = MATRIX / dec_slug / unit / verifier_new / "predictions.json"
    if not pred_p.exists():
        return None
    out = {}
    for r in json.loads(pred_p.read_text()):
        lab = keep.get(r["id"] if is_ref else r["snippet_id"])
        if lab is None:
            continue
        out.setdefault(lab["entry_id"], []).append(
            (bool(lab["label_atomic_and"]), bool(r["prediction"])))
    return out or None


def boot(atom, snip, n_boot=5000, seed=42):
    entries = sorted(set(atom) & set(snip))
    flat = lambda D, ids: [x for e in ids for x in D.get(e, ())]
    point = f1_false(flat(snip, entries)) - f1_false(flat(atom, entries))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(entries))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = [entries[i] for i in rng.choice(idx, size=len(entries), replace=True)]
        draws[b] = f1_false(flat(snip, pick)) - f1_false(flat(atom, pick))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    cells = []
    for mode_name, mode_dir, ref_tag in MODES:
        for dec_name, dec_slug, is_ref in DECOMPOSERS:
            for v_name, v_slug, v_new in VERIFIERS:
                atom = gold_atom(v_slug)
                snip = snippet_arm(dec_slug, is_ref, mode_dir, ref_tag, v_slug, v_new)
                if snip is None:
                    continue
                d, lo, hi = boot(atom, snip)
                cells.append({"mode": mode_name, "decomposer": dec_name,
                              "verifier": v_name, "delta": round(d, 4),
                              "ci": [round(lo, 4), round(hi, 4)],
                              "significant": bool(lo > 0 or hi < 0)})
                print(f"{mode_name} {dec_name:8s} {v_name:8s} "
                      f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]"
                      f"{'*' if (lo > 0 or hi < 0) else ''}", flush=True)
    OUT_JSON.write_text(json.dumps(cells, indent=2))

    L = ["# Decomposer x verifier matrix against the gold atom baseline", "",
         "Every cell is snippet \\fF{} minus the Kim et al. expert-atom \\fF{} for "
         "the same verifier. Rows are decomposers, columns verifiers. "
         "`*` marks an interval excluding zero under an entry-clustered "
         "bootstrap.", ""]
    for mode_name, _, _ in MODES:
        L += [f"## {mode_name}", "",
              "| Decomposer | " + " | ".join(v for v, _, _ in VERIFIERS) + " |",
              "|---" * (len(VERIFIERS) + 1) + "|"]
        for dec_name, _, _ in DECOMPOSERS:
            row = []
            for v_name, _, _ in VERIFIERS:
                c = next((x for x in cells if x["mode"] == mode_name
                          and x["decomposer"] == dec_name and x["verifier"] == v_name), None)
                row.append("—" if not c else
                           f"{c['delta']:+.3f}{'*' if c['significant'] else ''}")
            L.append(f"| {dec_name} | " + " | ".join(row) + " |")
        L.append("")
    n = sum(1 for c in cells if c["significant"])
    L += [f"**{n}/{len(cells)} significant.**", ""]
    print("\n".join(L))
    print(f"\n{n}/{len(cells)} significant")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
