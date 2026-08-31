"""External results: pipeline snippets vs pipeline atoms.

Replaces the published external rows, which compared the *undecomposed* claim
or answer against pipeline atoms. The paper describes those units as
"generated automatically with the MedSNIP pipeline" (§3, §4.1), but the
verification code read the raw dataset text and never the pipeline's snippet
output - 787 HealthFC and 2,636 MedHallu snippets were produced and discarded.

This reports the comparison the paper describes: both arms from the same
decomposition call, so only the verification unit differs.

  atom          pipeline atoms
  snippet M1    pipeline snippets, atomize-then-group
  snippet M2    pipeline snippets, snippet-direct

HealthFC additionally uses declarative input. 99% of its `en_claim` values are
questions, and the extract prompt assumes declarative text, so 26% of the
published atoms were meta-statements ("The user asks whether X") that cannot be
verified. Converting each question to the proposition it asserts removes them
(0/1076). MedHallu needs no such fix (0.03% meta rate).

The published column is retained for reference. It answers a different question
- undecomposed text vs atoms - and is not a before/after of the same quantity.

Usage:
  python -m src.analysis.external_results
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "data" / "11-analysis" / "external-analysis" / "external_results.json"

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


def f1_false(pairs) -> float:
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
    return 2 * P * R / (P + R) if P + R else 0.0


def healthfc_gold() -> dict:
    out = {}
    for i, r in enumerate(json.loads((ROOT / "data/1-raw/healthfc.json").read_text())):
        l = r["label"]
        out[f"hfc-{i}"] = True if l == 0 else (False if l == 2 else None)
    return out


def medhallu_gold(cid: str) -> bool:
    # ids are mh-<id>-T (ground truth) or mh-<id>-H (hallucinated)
    return cid.rsplit("-", 1)[1] == "T"


def score(path: Path, gold) -> float | None:
    """OR-FALSE aggregation to the claim, then false-class F1."""
    if not path.exists():
        return None
    by: dict[str, list[bool]] = {}
    for r in json.loads(path.read_text()):
        by.setdefault(r.get("claim_id") or r["id"], []).append(bool(r["prediction"]))
    pairs = []
    for cid, preds in by.items():
        g = gold(cid) if callable(gold) else gold.get(cid)
        if g is None:
            continue
        pairs.append((g, all(preds)))
    return f1_false(pairs) if pairs else None


def main() -> None:
    hg = healthfc_gold()
    corpora = [
        ("HealthFC", ROOT / "data/9-healthfc/baselines/predictions", hg, "decl-"),
        ("MedHallu", ROOT / "data/10-medhallu/baselines/predictions", medhallu_gold, ""),
    ]
    out = []
    for corpus, base, gold, pfx in corpora:
        for disp, pub_slug, new_slug in VERIFIERS:
            row = {"corpus": corpus, "verifier": disp}
            pub_a = score(base / "atom" / "claim-only" / pub_slug / "predictions.json", gold)
            pub_s = score(base / "snippet" / "claim-only" / pub_slug / "predictions.json", gold)
            # corrected atom arm: HealthFC uses declarative input, MedHallu reuses published
            new_a = score(base / "atom" / "claim-only" / f"{pfx}{new_slug}" / "predictions.json",
                          gold) if pfx else pub_a
            m1 = score(base / "pipeline-snippet" / "claim-only" / f"{pfx}{new_slug}"
                       / "predictions.json", gold)
            m2 = score(base / "pipeline-snippet" / "claim-only" / f"{pfx}mode2-{new_slug}"
                       / "predictions.json", gold)
            row.update({"pub_atom": pub_a, "pub_snip": pub_s, "atom": new_a,
                        "m1": m1, "m2": m2})
            row["pub_delta"] = (pub_s - pub_a) if (pub_a is not None and pub_s is not None) else None
            row["m1_delta"] = (m1 - new_a) if (m1 is not None and new_a is not None) else None
            row["m2_delta"] = (m2 - new_a) if (m2 is not None and new_a is not None) else None
            out.append(row)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    f = lambda v: f"{v:.3f}" if isinstance(v, float) else "—"
    d = lambda v: f"{v:+.3f}" if isinstance(v, float) else "—"
    L = ["# External results: pipeline snippets vs pipeline atoms", "",
         "Both arms come from the same decomposition call, so only the "
         "verification unit differs. HealthFC uses declarative input (its "
         "claims are questions, which the extract prompt mishandled). "
         "`published` is the old comparison - undecomposed text vs atoms - kept "
         "for reference; it answers a different question.", ""]
    for corpus, _, _, _ in corpora:
        rows = [r for r in out if r["corpus"] == corpus]
        if not rows:
            continue
        L += [f"## {corpus}", "",
              "| Verifier | atom | snippet M1 | Δ M1 | snippet M2 | Δ M2 | "
              "| published Δ |",
              "|---|---:|---:|---:|---:|---:|---|---:|"]
        for r in rows:
            L.append(f"| {r['verifier']} | {f(r['atom'])} | {f(r['m1'])} | "
                     f"{d(r['m1_delta'])} | {f(r['m2'])} | {d(r['m2_delta'])} | | "
                     f"{d(r['pub_delta'])} |")
        for key, lab in (("m1_delta", "Mode 1"), ("m2_delta", "Mode 2"),
                         ("pub_delta", "published")):
            v = [r[key] for r in rows if isinstance(r.get(key), float)]
            if v:
                L.append(f"\n**{lab}**: mean {sum(v)/len(v):+.4f}, "
                         f"positive {sum(1 for x in v if x > 0)}/{len(v)}")
        L.append("")
    print("\n".join(L))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
