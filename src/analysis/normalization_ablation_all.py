"""Wording-vs-granularity ablation across all six patterns.

Extends the D/F ablation to the merge patterns A/B/C (and the small E), which
is where the paper's headline granularity claim lives.

Three conditions, scored the way the paper's per-pattern table scores them:

    raw atom        original expert atoms          (paper's atom run)
    normalized atom same atoms, decontextualised   (this ablation)
    human snippet   annotator's snippets           (paper's snippet run)

For merge patterns the atom and snippet conditions have different unit counts
(several atoms per snippet). That asymmetry is inherent to the comparison the
paper makes, and is preserved here rather than papered over.

`wording share` is (normalized - raw) / (snippet - raw): how much of the gain
decontextualisation alone recovers. A high share means the gain is text, not
grouping; a low share means grouping is doing the work.

Usage:
  python -m src.analysis.normalization_ablation_all
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / "data" / "14-normalization-ablation"
FINAL = ROOT / "data" / "!-final" / "dataset.json"
PRED = ROOT / "data" / "5-baselines" / "predictions"
VERIFIER = "gpt-5.4-high"
NOISE_FLOOR = 0.013

SOURCES = [
    (ABL / "normalized_atoms.json",
     ABL / "predictions" / "claim-only" / "openai__gpt-5.4-high" / "predictions.json"),
    (ABL / "normalized_atoms_abce.json",
     ABL / "predictions_abce" / "claim-only" / "openai__gpt-5.4-high" / "predictions.json"),
]
OUT_JSON = ABL / "ablation_all_patterns.json"

MERGE = {"A", "B", "C"}

# Normalization leak guard.
#
# The normalizer is told to inline relevant context from `shared_context`. For
# Pattern C (conclusion + premises) on clinical vignettes, those premises ARE
# the patient's presenting findings, which live in shared_context - so on some
# units the normalizer reconstructs the merge instead of ablating it. The
# signature is unmistakable: on those units the "ablated" atom outscores the
# snippet it is supposed to be a stripped-down version of (0.621 vs 0.493),
# which is only possible if normalization added information the snippet lacked.
#
# Units whose text grew by more than this many characters are excluded from the
# headline numbers and reported separately. 382 of 4,868 merge atoms (7.8%) are
# affected. The threshold is coarse, but the effect is stark enough that the
# conclusion does not hinge on where exactly it is drawn.
LEAK_GROWTH_CHARS = 100


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
    atom_pred = {r["id"]: r["prediction"] for r in json.loads(
        (PRED / "atom" / "claim-only" / VERIFIER / "predictions.json").read_text())}
    snip_pred = {r["snippet_id"]: r["prediction"] for r in json.loads(
        (PRED / "snippet" / "claim-only" / VERIFIER / "predictions.json").read_text())}
    snip_gold = {r["snippet_id"]: bool(r["label_atomic"])
                 for r in json.loads(FINAL.read_text())}

    # atom-level conditions, keyed by pattern
    raw: dict[str, list] = {}
    nrm: dict[str, list] = {}
    seen_snips: dict[str, set] = {}
    leaked_raw: dict[str, list] = {}
    leaked_nrm: dict[str, list] = {}
    leaked_snips: dict[str, set] = {}

    for units_path, pred_path in SOURCES:
        if not (units_path.exists() and pred_path.exists()):
            continue
        norm_pred = {r["snippet_id"]: r["prediction"]
                     for r in json.loads(pred_path.read_text())}
        for u in json.loads(units_path.read_text()):
            pat = u["pattern"]
            if u["id"] not in atom_pred or u["id"] not in norm_pred:
                continue
            # The leak is reconstruction of a MERGE, so it can only occur where
            # a merge exists. For keep-atomic patterns the snippet is a single
            # rewritten atom, so heavy inlining there is the wording effect
            # under measurement, not contamination - and indeed those units show
            # no overshoot (normalized 0.602 vs snippet 0.610), unlike merge
            # units (0.507 vs 0.450). Applying the filter to them would discard
            # exactly the signal we are trying to quantify.
            leaked = (pat in MERGE and
                      (len(u["normalized_text"]) - len(u["atom_text"])) >= LEAK_GROWTH_CHARS)
            bucket = leaked_raw if leaked else raw
            bucket_n = leaked_nrm if leaked else nrm
            bucket_s = leaked_snips if leaked else seen_snips
            gold_ = u["label_atomic"]
            bucket.setdefault(pat, []).append((gold_, atom_pred[u["id"]]))
            bucket_n.setdefault(pat, []).append((gold_, norm_pred[u["id"]]))
            bucket_s.setdefault(pat, set()).add(u["snippet_id"])
            continue
            gold = u["label_atomic"]
            raw.setdefault(pat, []).append((gold, atom_pred[u["id"]]))
            nrm.setdefault(pat, []).append((gold, norm_pred[u["id"]]))
            seen_snips.setdefault(pat, set()).add(u["snippet_id"])

    rows = []
    for pat in sorted(raw):
        snips = [(snip_gold[s], snip_pred[s]) for s in sorted(seen_snips[pat])
                 if s in snip_pred and s in snip_gold]
        r_f1, n_f1, s_f1 = f1_false(raw[pat]), f1_false(nrm[pat]), f1_false(snips)
        total = s_f1 - r_f1
        rows.append({
            "pattern": pat, "kind": "merge" if pat in MERGE else "keep-atomic",
            "n_atoms": len(raw[pat]), "n_snippets": len(snips),
            "raw_atom": round(r_f1, 4), "normalized_atom": round(n_f1, 4),
            "human_snippet": round(s_f1, 4),
            "norm_minus_raw": round(n_f1 - r_f1, 4),
            "snip_minus_raw": round(total, 4),
            "wording_share_pct": round(100 * (n_f1 - r_f1) / total, 1) if abs(total) > 1e-9 else None,
        })

    for label, keep in (("merge A+B+C", MERGE), ("keep-atomic D+E+F", {"D", "E", "F"})):
        R = [x for p in keep for x in raw.get(p, [])]
        N = [x for p in keep for x in nrm.get(p, [])]
        S = [(snip_gold[s], snip_pred[s]) for p in keep
             for s in sorted(seen_snips.get(p, ())) if s in snip_pred and s in snip_gold]
        if not R:
            continue
        r_f1, n_f1, s_f1 = f1_false(R), f1_false(N), f1_false(S)
        total = s_f1 - r_f1
        rows.append({
            "pattern": label, "kind": "pooled",
            "n_atoms": len(R), "n_snippets": len(S),
            "raw_atom": round(r_f1, 4), "normalized_atom": round(n_f1, 4),
            "human_snippet": round(s_f1, 4),
            "norm_minus_raw": round(n_f1 - r_f1, 4),
            "snip_minus_raw": round(total, 4),
            "wording_share_pct": round(100 * (n_f1 - r_f1) / total, 1) if abs(total) > 1e-9 else None,
        })

    # Leaked units, reported separately rather than silently dropped.
    for label, keep in (("LEAKED merge A+B+C", MERGE),):
        R = [x for p in keep for x in leaked_raw.get(p, [])]
        N = [x for p in keep for x in leaked_nrm.get(p, [])]
        S = [(snip_gold[s], snip_pred[s]) for p in keep
             for s in sorted(leaked_snips.get(p, ())) if s in snip_pred and s in snip_gold]
        if not R:
            continue
        r_f1, n_f1, s_f1 = f1_false(R), f1_false(N), f1_false(S)
        total = s_f1 - r_f1
        rows.append({
            "pattern": label, "kind": "excluded (leak)",
            "n_atoms": len(R), "n_snippets": len(S),
            "raw_atom": round(r_f1, 4), "normalized_atom": round(n_f1, 4),
            "human_snippet": round(s_f1, 4),
            "norm_minus_raw": round(n_f1 - r_f1, 4),
            "snip_minus_raw": round(total, 4),
            "wording_share_pct": round(100 * (n_f1 - r_f1) / total, 1) if abs(total) > 1e-9 else None,
        })

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
