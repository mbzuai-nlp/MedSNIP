"""End-to-end cost accounting: decomposition overhead included.

Answers meta-review revision #3 and reviewer W5ZC's objection that the
"22-47% cost reduction" ignores the extra call needed to build snippets.

The accounting charges each pipeline for everything it actually needs:

    atom pipeline    = extract call            + N_atom    verifier calls
    snippet pipeline = extract + cluster calls + N_snippet verifier calls

That split is exact rather than modelled: Mode 1 logs `usage_steps` per step,
and the extract step is precisely the work an atom pipeline would do on its
own, so the atom arm is charged one decomposition call and the snippet arm
two. Mode 2 (snippet-direct) needs only one call and is reported separately.

Costs are measured where available. Every run routed through OpenRouter
carries a per-call `cost` field. The paper's original runs predate that and
carry token counts only, so those are priced at list rates (PRICING below);
such rows are marked `priced` rather than `measured` in the output.

A separate table reports end-to-end cost by decomposer, because the headline
number is not the decomposer's own price: a weak decomposer merges less,
produces more units, and shifts cost onto the verifier.

Usage:
  python -m src.analysis.cost_accounting
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "13-cost-accounting"
OUT_JSON = OUT_DIR / "cost_accounting.json"

DECOMP_ROOT = ROOT / "data" / "6-snippet-processor" / "atom-to-snippet"
DIRECT_ROOT = ROOT / "data" / "6-snippet-processor" / "snippet-direct"
MATRIX_PRED = ROOT / "data" / "12-decomposer" / "predictions" / "claim-only"
PAPER_PRED = ROOT / "data" / "5-baselines" / "predictions"

# USD per million tokens (input, output). Used only for runs that predate
# per-call cost logging; cached input is billed at 10% on OpenAI models.
PRICING = {
    "gpt-5.4":                           (2.50, 15.00),
    "openai/gpt-5.4":                    (2.50, 15.00),
    "gpt-4o":                            (2.50, 10.00),
    "openai/gpt-4o":                     (2.50, 10.00),
    "google/gemma-4-31b-it":             (0.09, 0.34),
    "openai/gpt-oss-20b":                (0.03, 0.13),
    "meta-llama/llama-3.3-70b-instruct": (0.71, 0.71),
    "meta-llama/llama-3.1-8b-instruct":  (0.05, 0.08),
}
CACHED_DISCOUNT = 0.10


def price(model: str, inp: int, out: int, cached: int = 0) -> float | None:
    rate = PRICING.get(model) or PRICING.get(model.split("/")[-1])
    if not rate:
        return None
    pin, pout = rate
    billed_in = (inp - cached) + cached * CACHED_DISCOUNT
    return billed_in / 1e6 * pin + out / 1e6 * pout


def step_cost(step: dict, model: str) -> tuple[float | None, bool]:
    """(cost, measured?). Falls back to list pricing when not logged."""
    if step.get("cost") is not None:
        return float(step["cost"]), True
    c = price(model, step.get("input_tokens", 0), step.get("output_tokens", 0),
              step.get("cached_input_tokens", 0))
    return c, False


def decomposition_costs(slug_dir: Path) -> dict:
    """Per-slug decomposition cost, split by pipeline step."""
    files = sorted(slug_dir.glob("e*.json"))
    if not files:
        return {}
    extract = cluster = 0.0
    n_atoms = n_snips = 0
    measured = True
    model = None
    for f in files:
        r = json.loads(f.read_text())
        model = r.get("model") or model
        n_atoms += len(r.get("generated_atoms") or [])
        n_snips += len(r.get("snippets") or [])
        for st in (r.get("usage_steps") or []):
            c, m = step_cost(st, model or "")
            if c is None:
                continue
            measured &= m
            if st.get("step") == "extract":
                extract += c
            else:
                cluster += c
    return {"entries": len(files), "model": model,
            "extract_usd": round(extract, 4), "cluster_usd": round(cluster, 4),
            "decomp_total_usd": round(extract + cluster, 4),
            "atoms": n_atoms, "snippets": n_snips,
            "measured": measured}


def verification_cost(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    total = 0.0
    measured = True
    for r in rows:
        u = r.get("usage") or {}
        if u.get("cost") is not None:
            total += float(u["cost"])
            continue
        measured = False
        c = price(r.get("model", ""), u.get("prompt_tokens", 0),
                  u.get("completion_tokens", 0))
        if c:
            total += c
    return {"calls": len(rows), "usd": round(total, 4), "measured": measured}


# Per-call verifier rates measured on the paper's own cells (200 fresh calls
# each, data/11-analysis/baseline_cost_validation.md). Used only for the
# paper's original runs, whose prediction files carry no usage at all.
PAPER_RATES_USD = {"atom": 0.0027, "snippet": 0.0045}

# Mode 1 and Mode 2 decomposition outputs per external corpus. HealthFC uses
# the declarative rewrite, since that is the arm the paper evaluates: its raw
# claims are questions, which the extract prompt mishandled.
EXTERNAL = [
    ("HealthFC",
     ROOT / "data/9-healthfc/snippet-processor/atom-to-snippet/declarative/healthfc.json",
     ROOT / "data/9-healthfc/snippet-processor/snippet-direct/declarative/healthfc.json"),
    ("MedHallu",
     ROOT / "data/10-medhallu/snippet-processor/atom-to-snippet/medhallu.json",
     ROOT / "data/10-medhallu/snippet-processor/snippet-direct/medhallu.json"),
]


def call_accounting() -> list[dict]:
    """Exact call counts per pipeline, decomposition included.

    The reviewer's objection is fundamentally about calls ("you also need a
    query to create the snippets"), and calls are countable without pricing
    anything, so this answers it exactly for all three corpora.
    """
    def _snips(path: Path) -> int:
        return sum(len(r.get("snippets") or [])
                   for r in json.loads(path.read_text()))

    out = []
    # MedSNIP-Bench inherits expert atoms from Kim et al., so the atom arm is
    # the 5,755 gold atoms the paper verifies, not a decomposer's own output.
    # Unit counts come from the scored prediction files, so the reduction is
    # over exactly the units Table 3 evaluates in each arm.
    msb = PAPER_PRED / "atom" / "claim-only" / "gpt-5.4-high" / "predictions.json"
    ref, direct = DECOMP_ROOT / "gpt-5.4-high", DIRECT_ROOT / "gpt-5.4-high"
    if msb.exists() and ref.exists():
        files = sorted(ref.glob("e*.json"))
        out.append({
            "dataset": "MedSNIP-Bench", "items": len(files),
            "atom_units": len(json.loads(msb.read_text())),
            "snippet_units": sum(len(json.loads(f.read_text()).get("snippets") or [])
                                 for f in files),
            "mode2_units": sum(len(json.loads(f.read_text()).get("snippets") or [])
                               for f in sorted(direct.glob("e*.json")))})
    for name, m1_path, m2_path in EXTERNAL:
        if not m1_path.exists():
            continue
        recs = json.loads(m1_path.read_text())
        out.append({"dataset": name, "items": len(recs),
                    "atom_units": sum(len(r.get("generated_atoms") or []) for r in recs),
                    "snippet_units": sum(len(r.get("snippets") or []) for r in recs),
                    "mode2_units": _snips(m2_path) if m2_path.exists() else None})
    for r in out:
        n = r["items"]
        r["atom_calls_total"] = n + r["atom_units"]          # 1 extract + verify
        r["snippet_calls_total"] = 2 * n + r["snippet_units"]  # extract+cluster + verify
        r["verify_only_red_pct"] = round(
            100 * (1 - r["snippet_units"] / r["atom_units"]), 1)
        r["end_to_end_red_pct"] = round(
            100 * (1 - r["snippet_calls_total"] / r["atom_calls_total"]), 1)
        # Mode 2 needs one decomposition call, not two, and merges differently,
        # so it has its own unit count rather than reusing Mode 1's.
        if r.get("mode2_units"):
            r["mode2_verify_only_red_pct"] = round(
                100 * (1 - r["mode2_units"] / r["atom_units"]), 1)
            r["mode2_calls_total"] = n + r["mode2_units"]
            r["mode2_red_pct"] = round(
                100 * (1 - r["mode2_calls_total"] / r["atom_calls_total"]), 1)
    return out


def per_decomposer_calls() -> list[dict]:
    """Mode 1 vs Mode 2 end-to-end calls, per decomposer, on MedSNIP-Bench.

    Which mode is cheaper end-to-end is not fixed: Mode 2 saves a decomposition
    call per answer but merges less, so it produces more units to verify. The
    winner depends on the decomposer's merge ratio, so it has to be measured
    rather than argued.
    """
    out = []
    for d in sorted(DECOMP_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m1 = decomposition_costs(d)
        if not m1:
            continue
        m2 = decomposition_costs(DIRECT_ROOT / d.name)
        n = m1["entries"]
        atom_calls = n + m1["atoms"]
        m1_calls = 2 * n + m1["snippets"]
        row = {"decomposer": d.name, "entries": n,
               "atom_units": m1["atoms"], "atom_calls": atom_calls,
               "m1_snippets": m1["snippets"], "m1_calls": m1_calls,
               "m1_red_pct": round(100 * (1 - m1_calls / atom_calls), 1),
               "m1_decomp_usd": m1["decomp_total_usd"]}
        if m2:
            m2_calls = m2["entries"] + m2["snippets"]
            row.update({"m2_snippets": m2["snippets"], "m2_calls": m2_calls,
                        "m2_red_pct": round(100 * (1 - m2_calls / atom_calls), 1),
                        "m2_decomp_usd": m2["decomp_total_usd"],
                        "m2_entries": m2["entries"]})
        out.append(row)
    return out


# Measured GPT-5.4 verifier cost per snippet call, over the 9,412 snippet-arm
# calls in the decomposer matrix. Used to project Mode 2 end-to-end dollars,
# since Mode 2 verification has not been run.
VERIFY_USD_PER_SNIPPET = 0.004668


def mode_dollar_comparison() -> list[dict]:
    """Mode 1 vs Mode 2 end-to-end dollars under a GPT-5.4 verifier.

    Decomposition is measured. Verification is projected at the measured
    per-call rate, because Mode 2 verification has not been run — so these are
    estimates on the verification side and are labelled as such.

    The saving comes entirely from halving decomposition, so it scales with how
    expensive the decomposer is. Mode 2 also merges differently, which can add
    or remove verification units and sometimes reverses the sign.
    """
    out = []
    for d in sorted(DECOMP_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m1 = decomposition_costs(d)
        m2 = decomposition_costs(DIRECT_ROOT / d.name)
        if not (m1 and m2):
            continue
        m1_total = m1["decomp_total_usd"] + m1["snippets"] * VERIFY_USD_PER_SNIPPET
        m2_total = m2["decomp_total_usd"] + m2["snippets"] * VERIFY_USD_PER_SNIPPET
        out.append({
            "decomposer": d.name,
            "m1_decomp": m1["decomp_total_usd"], "m2_decomp": m2["decomp_total_usd"],
            "m1_snippets": m1["snippets"], "m2_snippets": m2["snippets"],
            "m1_verify_est": round(m1["snippets"] * VERIFY_USD_PER_SNIPPET, 2),
            "m2_verify_est": round(m2["snippets"] * VERIFY_USD_PER_SNIPPET, 2),
            "m1_total_est": round(m1_total, 2), "m2_total_est": round(m2_total, 2),
            "m2_saving_usd": round(m1_total - m2_total, 2),
            "m2_saving_pct": round(100 * (m1_total - m2_total) / m1_total, 1),
        })
    return out


def paper_row() -> dict | None:
    """The paper's own configuration: GPT-5.4 decomposer, GPT-5.4 verifier."""
    dc = decomposition_costs(DECOMP_ROOT / "gpt-5.4-high")
    if not dc:
        return None
    atom_p = PAPER_PRED / "atom" / "claim-only" / "gpt-5.4-high" / "predictions.json"
    snip_p = PAPER_PRED / "snippet" / "claim-only" / "auto-a2s" / "gpt-5.4-high" / "predictions.json"
    if not (atom_p.exists() and snip_p.exists()):
        return None
    n_atom = len(json.loads(atom_p.read_text()))
    # Charge every unit the pipeline emits, not only the ones the drop-mixed
    # projection scores. The atom arm verifies all 5,755 expert atoms, so
    # billing the snippet arm for its scored subset would flatter it.
    n_snip = dc["snippets"]
    atom_total = dc["extract_usd"] + n_atom * PAPER_RATES_USD["atom"]
    snip_total = dc["decomp_total_usd"] + n_snip * PAPER_RATES_USD["snippet"]
    return {"decomposer": "gpt-5.4-high (paper config)", "entries": dc["entries"],
            "atom_calls": n_atom, "snippet_calls": n_snip,
            "call_reduction_pct": round(100 * (1 - n_snip / n_atom), 1),
            "atom_decomp_usd": dc["extract_usd"],
            "snippet_decomp_usd": dc["decomp_total_usd"],
            "atom_verify_usd": round(n_atom * PAPER_RATES_USD["atom"], 4),
            "snippet_verify_usd": round(n_snip * PAPER_RATES_USD["snippet"], 4),
            "atom_total_usd": round(atom_total, 4),
            "snippet_total_usd": round(snip_total, 4),
            "verify_only_reduction_pct": round(
                100 * (1 - (n_snip * PAPER_RATES_USD["snippet"]) /
                       (n_atom * PAPER_RATES_USD["atom"])), 1),
            "end_to_end_reduction_pct": round(100 * (1 - snip_total / atom_total), 1),
            "measured": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="openai__gpt-5.4-high",
                    help="verifier slug for the end-to-end table")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {"decomposition": {}, "end_to_end": [], "modes": {}}

    for d in sorted(DECOMP_ROOT.iterdir()):
        if d.is_dir():
            c = decomposition_costs(d)
            if c:
                report["decomposition"][d.name] = c
    if DIRECT_ROOT.exists():
        for d in sorted(DIRECT_ROOT.iterdir()):
            if d.is_dir():
                c = decomposition_costs(d)
                if c:
                    report["modes"][d.name] = c

    # End-to-end, per decomposer, under one verifier.
    for slug, dc in report["decomposition"].items():
        atom_v = verification_cost(MATRIX_PRED / slug / "atom" / args.verifier / "predictions.json")
        snip_v = verification_cost(MATRIX_PRED / slug / "snippet" / args.verifier / "predictions.json")
        if not (atom_v and snip_v):
            continue
        # Atom pipeline pays for extract only; snippet pipeline pays for both steps.
        atom_total = dc["extract_usd"] + atom_v["usd"]
        snip_total = dc["decomp_total_usd"] + snip_v["usd"]
        report["end_to_end"].append({
            "decomposer": slug,
            "entries": dc["entries"],
            "atom_calls": atom_v["calls"], "snippet_calls": snip_v["calls"],
            "call_reduction_pct": round(100 * (1 - snip_v["calls"] / atom_v["calls"]), 1),
            "atom_decomp_usd": dc["extract_usd"],
            "snippet_decomp_usd": dc["decomp_total_usd"],
            "atom_verify_usd": atom_v["usd"], "snippet_verify_usd": snip_v["usd"],
            "atom_total_usd": round(atom_total, 4),
            "snippet_total_usd": round(snip_total, 4),
            "verify_only_reduction_pct": round(100 * (1 - snip_v["usd"] / atom_v["usd"]), 1),
            "end_to_end_reduction_pct": round(100 * (1 - snip_total / atom_total), 1),
            "measured": dc["measured"] and atom_v["measured"] and snip_v["measured"],
        })

    pr = paper_row()
    if pr:
        report["end_to_end"].insert(0, pr)
    report["calls"] = call_accounting()
    report["per_decomposer_calls"] = per_decomposer_calls()
    report["mode_dollars"] = mode_dollar_comparison()
    OUT_JSON.write_text(json.dumps(report, indent=2))

    print("\n".join(L))


if __name__ == "__main__":
    main()
