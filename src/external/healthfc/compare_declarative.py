"""HealthFC: published condition vs the corrected declarative conditions.

The published HealthFC rows compare an *undecomposed question* (the snippet
arm) against pipeline atoms. Because 99% of HealthFC `en_claim` values are
questions and the extract prompt assumes declarative input, 26% of those atoms
came out as meta-statements ("The user asks whether X") - vacuously true, with
no medical content to verify. 80 of 327 scorable claims had an atom baseline
made entirely of such statements.

Converting each question to the proposition it asserts removes them entirely
(0/1076). This reports three conditions:

  published   raw question           vs atoms from questions   (what is in the paper)
  decl M1     pipeline snippets      vs atoms, same call       (the symmetric comparison)
  decl M2     pipeline snippets (M2) vs M1 atoms               (mode generalisation)

These are NOT a before/after of one quantity. The published row compares an
undecomposed question against damaged atoms; the new rows compare clean
snippets against clean atoms. A smaller gain in the new rows means the old
comparison measured something else, not that the effect shrank.

Unit-level predictions are aggregated to the claim with OR-FALSE (a claim is
predicted false if any of its units is), matching the paper.

Usage:
  python -m src.external.healthfc.compare_declarative
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "1-raw" / "healthfc.json"
PRED = ROOT / "data" / "9-healthfc" / "baselines" / "predictions"
OUT_JSON = ROOT / "data" / "9-healthfc" / "declarative_comparison.json"

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


def gold_map() -> dict[str, bool | None]:
    out = {}
    for i, r in enumerate(json.loads(RAW.read_text())):
        l = r["label"]
        out[f"hfc-{i}"] = True if l == 0 else (False if l == 2 else None)
    return out


def f1_false(pairs) -> tuple[float, int]:
    tp = fp = fn = 0
    for g, p in pairs:
        gf, pf = (not g), (not p)
        if gf and pf:
            tp += 1
        elif not gf and pf:
            fp += 1
        elif gf and not pf:
            fn += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return (2 * P * R / (P + R) if P + R else 0.0), len(pairs)


def score(path: Path, gold: dict) -> tuple[float, int] | None:
    """Aggregate unit predictions to the claim with OR-FALSE, then score."""
    if not path.exists():
        return None
    by: dict[str, list[bool]] = {}
    for r in json.loads(path.read_text()):
        cid = r.get("claim_id") or r["id"]
        by.setdefault(cid, []).append(bool(r["prediction"]))
    pairs = [(gold[c], all(p)) for c, p in by.items()
             if gold.get(c) is not None]
    return f1_false(pairs)


def main() -> None:
    gold = gold_map()
    rows = []
    for disp, pub_slug, new_slug in VERIFIERS:
        cells = {
            "pub_atom":  PRED / "atom" / "claim-only" / pub_slug / "predictions.json",
            "pub_snip":  PRED / "snippet" / "claim-only" / pub_slug / "predictions.json",
            "m1_atom":   PRED / "atom" / "claim-only" / f"decl-{new_slug}" / "predictions.json",
            "m1_snip":   PRED / "pipeline-snippet" / "claim-only" / f"decl-{new_slug}" / "predictions.json",
            "m2_snip":   PRED / "pipeline-snippet" / "claim-only" / f"decl-mode2-{new_slug}" / "predictions.json",
        }
        s = {k: score(p, gold) for k, p in cells.items()}
        row = {"verifier": disp}
        for k, v in s.items():
            row[k] = round(v[0], 4) if v else None
            row[k + "_n"] = v[1] if v else 0
        if s["pub_atom"] and s["pub_snip"]:
            row["pub_delta"] = round(s["pub_snip"][0] - s["pub_atom"][0], 4)
        if s["m1_atom"] and s["m1_snip"]:
            row["m1_delta"] = round(s["m1_snip"][0] - s["m1_atom"][0], 4)
        if s["m1_atom"] and s["m2_snip"]:
            row["m2_delta"] = round(s["m2_snip"][0] - s["m1_atom"][0], 4)
        rows.append(row)

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else "—"
    d = lambda v: f"{v:+.3f}" if isinstance(v, float) else "—"
    print("\n".join(L))


if __name__ == "__main__":
    main()
