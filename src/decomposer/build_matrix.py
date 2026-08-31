"""Assemble the decomposer x verifier matrix from whatever cells are on disk.

Reports, per (decomposer, verifier) cell, false-class F1 for the snippet arm
and the atom arm, and their delta. Both arms come from the same decomposition
call, so the delta isolates the verification unit rather than decomposer
quality.

Two atom baselines are reported because they answer different questions:

  - `own`    — the decomposer's own generated atoms. Symmetric: the only thing
               differing between arms is the unit.
  - `expert` — the expert atom set from Kim et al. that the paper's Table 3
               uses. Not symmetric (a different decomposer produced it), but it
               is the published baseline, so it keeps this table commensurable
               with the paper.

Cells are read straight from disk rather than from verify_status.json, since
that file undercounts cells completed before its path bookkeeping was fixed.
Partial cells are skipped, never averaged over survivors.

Usage:
  python -m src.decomposer.build_matrix
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

from .models import DISPLAY, MATRIX_MODELS
from .run_verify import cell_path, labeled_path, row_slug

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "12-decomposer"
EXPERT_GLOB = str(ROOT / "data/5-baselines/predictions/atom/claim-only/*/predictions.json")
MATRIX_JSON = OUT_DIR / "matrix.json"

# Verifier OpenRouter id -> directory slug of the paper's expert-atom run.
EXPERT_SLUG = {
    "openai/gpt-5.4":                    "gpt-5.4-high",
    "openai/gpt-4o":                     "gpt-4o-none",
    "google/gemma-4-31b-it":             "google__gemma-4-31B-it-none",
    "openai/gpt-oss-20b":                "openai__gpt-oss-20b-none",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama__Llama-3.3-70B-Instruct-none",
    "meta-llama/llama-3.1-8b-instruct":  "meta-llama__Llama-3.1-8B-Instruct-none",
}


def f1_false(rows, gt_field: str, drop_mixed: bool) -> tuple[float, int]:
    """False-class F1: the positive class is a FALSE label."""
    tp = fp = fn = n = 0
    for r in rows:
        if drop_mixed and r.get("_touches_mixed"):
            continue
        gt = r.get(gt_field)
        if gt is None:
            continue
        n += 1
        pred = bool(r["prediction"])
        if not pred and not gt:
            tp += 1
        elif not pred and gt:
            fp += 1
        elif pred and not gt:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * rc / (p + rc) if p + rc else 0.0), n


def cell_score(decomposer: str, unit: str, verifier: str):
    """None unless the cell is complete — partial cells are never scored."""
    lp, pp = labeled_path(decomposer, unit), cell_path(decomposer, unit, verifier)
    if not (lp.exists() and pp.exists()):
        return None
    labels = {r["snippet_id"]: r for r in json.loads(lp.read_text())}
    preds = json.loads(pp.read_text())
    if len(preds) < len(labels):
        return None
    joined = [{**labels[p["snippet_id"]], "prediction": p["prediction"]}
              for p in preds if p.get("snippet_id") in labels]
    return f1_false(joined, "label_atomic_and", drop_mixed=True)


def expert_baseline() -> dict[str, tuple[float, int]]:
    out = {}
    for f in glob.glob(EXPERT_GLOB):
        out[Path(f).parent.name] = f1_false(json.loads(Path(f).read_text()),
                                            "label", drop_mixed=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-model", default="openai/gpt-5.4",
                    help="decomposer excluded from rows (reused paper row)")
    args = ap.parse_args()

    expert = expert_baseline()
    decomposers = [m for m in MATRIX_MODELS if m != args.reference_model]

    cells, by_verifier = [], {}
    for v in MATRIX_MODELS:
        exp = expert.get(EXPERT_SLUG.get(v, ""), (None, 0))
        col = []
        for d in decomposers:
            s = cell_score(d, "snippet", v)
            a = cell_score(d, "atom", v)
            if s is None or a is None:
                continue
            row = {
                "decomposer": d, "verifier": v,
                "snippet_f1": round(s[0], 4), "snippet_n": s[1],
                "atom_f1": round(a[0], 4), "atom_n": a[1],
                "delta_own": round(s[0] - a[0], 4),
                "expert_atom_f1": round(exp[0], 4) if exp[0] is not None else None,
                "delta_expert": round(s[0] - exp[0], 4) if exp[0] is not None else None,
            }
            cells.append(row)
            col.append(row)
        if col:
            by_verifier[v] = col

    MATRIX_JSON.write_text(json.dumps(
        {"cells": cells,
         "expert_baseline": {k: {"f1_F": round(v[0], 4), "n": v[1]}
                             for k, v in sorted(expert.items())}},
        indent=2))

    lines = [
        "# Decomposer x verifier matrix",
        "",
        f"False-class F1 on MedSNIP-Bench, claim-only verification, "
        f"drop-mixed label projection. {len(cells)}/{len(decomposers)*len(MATRIX_MODELS)} "
        f"decomposer-by-verifier pairs complete (each pair = a snippet arm and an atom arm).",
        "",
        "`own` is the decomposer's own atoms (symmetric — both arms from one "
        "decomposition call). `expert` is the Kim et al. atom set the paper's "
        "Table 3 uses. Positive delta favours snippets.",
        "",
    ]
    for v in MATRIX_MODELS:
        col = by_verifier.get(v)
        if not col:
            continue
        lines += [f"## Verifier: {DISPLAY[v]}", "",
                  "| Decomposer | snippet | own atom | Δ own | expert atom | Δ expert |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in col:
            lines.append(
                f"| {DISPLAY[r['decomposer']]} | {r['snippet_f1']:.3f} | "
                f"{r['atom_f1']:.3f} | {r['delta_own']:+.3f} | "
                f"{r['expert_atom_f1']:.3f} | {r['delta_expert']:+.3f} |")
        d_own = [r["delta_own"] for r in col]
        d_exp = [r["delta_expert"] for r in col if r["delta_expert"] is not None]
        lines += ["",
                  f"mean Δ own **{st.mean(d_own):+.4f}** "
                  f"({sum(1 for x in d_own if x > 0)}/{len(d_own)} positive)"
                  + (f", mean Δ expert **{st.mean(d_exp):+.4f}** "
                     f"({sum(1 for x in d_exp if x > 0)}/{len(d_exp)} positive)"
                     if d_exp else ""),
                  ""]

    if cells:
        allo = [r["delta_own"] for r in cells]
        lines += ["## Overall", "",
                  f"- pairs complete: **{len(cells)}/{len(decomposers)*len(MATRIX_MODELS)}**",
                  f"- mean Δ own: **{st.mean(allo):+.4f}** "
                  f"({sum(1 for x in allo if x > 0)}/{len(allo)} positive)",
                  f"- range: {min(allo):+.3f} to {max(allo):+.3f}", ""]

    print("\n".join(lines))
    print(f"wrote {MATRIX_JSON}")


if __name__ == "__main__":
    main()
