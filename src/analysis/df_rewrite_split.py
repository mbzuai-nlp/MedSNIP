"""Split the D/F snippet-vs-atom gain by whether the snippet was reworded.

Reviewer FJrT observed that patterns D and F improve despite involving no
grouping at all - one atom in, one snippet out - and suggested the gain may
come from the pipeline writing cleaner, more decontextualised text rather than
from granularity.

Roughly half of D/F snippets are byte-identical to their source atom, so for
those there is no wording difference that could explain anything. That makes a
free partial test possible before running the paid ablation:

    identical subgroup  -> any gain here CANNOT be a wording effect
    reworded subgroup   -> a gain here MAY be a wording effect

If the gain is similar in both subgroups, rewording is not the driver. If it
concentrates in the reworded subgroup, the reviewer is right and the paid
ablation should quantify how much.

This compares the same units under two verifications that already exist: the
paper's expert-atom run and its human-snippet run, both claim-only.

Usage:
  python -m src.analysis.df_rewrite_split
  python -m src.analysis.df_rewrite_split --patterns D F E
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "data" / "!-final" / "dataset.json"
PRED = ROOT / "data" / "5-baselines" / "predictions"
OUT_DIR = ROOT / "data" / "13-cost-accounting"  # analysis outputs live alongside
OUT_JSON = ROOT / "data" / "11-analysis" / "rewrite-analysis" / "df_rewrite_split.json"

VERIFIER = "gpt-5.4-high"


def norm(s: str | None) -> str:
    """Whitespace/punctuation-insensitive comparison, so trivial formatting
    differences are not counted as rewording."""
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def f1_false(rows: list[dict]) -> tuple[float, int]:
    tp = fp = fn = 0
    for r in rows:
        gold_false = not bool(r["gold"])
        pred_false = not bool(r["pred"])
        if gold_false and pred_false:
            tp += 1
        elif not gold_false and pred_false:
            fp += 1
        elif gold_false and not pred_false:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * rc / (p + rc) if p + rc else 0.0), len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="*", default=["D", "F"])
    ap.add_argument("--verifier", default=VERIFIER)
    args = ap.parse_args()

    rows = json.loads(FINAL.read_text())
    atom_pred = {r["id"]: r for r in json.loads(
        (PRED / "atom" / "claim-only" / args.verifier / "predictions.json").read_text())}
    snip_pred = {r["snippet_id"]: r for r in json.loads(
        (PRED / "snippet" / "claim-only" / args.verifier / "predictions.json").read_text())}

    groups: dict[str, dict[str, list]] = {}
    for r in rows:
        pat = (r.get("pattern") or "").strip().upper()[:1]
        if pat not in args.patterns:
            continue
        atoms = r.get("atoms") or []
        if len(atoms) != 1:
            continue  # keep-atomic patterns only; guard against stray multi-atom
        sp = snip_pred.get(r["snippet_id"])
        # Atom prediction ids are "<entry>-<atom index>".
        ap_row = atom_pred.get(f"{r['entry_id']}-{atoms[0].get('index')}")
        if sp is None or ap_row is None:
            continue
        reworded = norm(atoms[0].get("text")) != norm(r["snippet_text"])
        key = f"{pat}:{'reworded' if reworded else 'identical'}"
        g = groups.setdefault(key, {"atom": [], "snippet": []})
        gold = bool(r["label_atomic"])
        g["atom"].append({"gold": gold, "pred": ap_row["prediction"]})
        g["snippet"].append({"gold": gold, "pred": sp["prediction"]})

    out = []
    for key in sorted(groups):
        a_f1, n = f1_false(groups[key]["atom"])
        s_f1, _ = f1_false(groups[key]["snippet"])
        pat, sub = key.split(":")
        out.append({"pattern": pat, "subgroup": sub, "n": n,
                    "atom_f1_F": round(a_f1, 4), "snippet_f1_F": round(s_f1, 4),
                    "delta": round(s_f1 - a_f1, 4)})

    # Pooled across the requested patterns.
    for sub in ("identical", "reworded"):
        A = [x for k, g in groups.items() if k.endswith(sub) for x in g["atom"]]
        S = [x for k, g in groups.items() if k.endswith(sub) for x in g["snippet"]]
        if not A:
            continue
        a_f1, n = f1_false(A)
        s_f1, _ = f1_false(S)
        out.append({"pattern": "+".join(args.patterns), "subgroup": sub, "n": n,
                    "atom_f1_F": round(a_f1, 4), "snippet_f1_F": round(s_f1, 4),
                    "delta": round(s_f1 - a_f1, 4)})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print("\n".join(L))


if __name__ == "__main__":
    main()
