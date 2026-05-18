"""Compare atom→claim aggregation rules.

For each (dataset, mode, model), we have atom-level predictions. We aggregate
them to claim level under several rules, then compare to the snippet baseline
F1 on the same claims. Result: does snippet > atom hold under aggregations
other than OR-False?

Aggregation rules implemented:
  - or_false      : claim=False if ANY atom False  (strict; current rule used in paper)
  - majority      : claim = majority vote of atom predictions (ties → True)
  - and_false     : claim=False only if ALL atoms False (very lenient on False class)
  - threshold_k2  : claim=False only if ≥2 atoms False (intermediate)
  - first_atom    : claim = prediction on first atom (sanity check)

Output:
  data/!-analysis/aggregation_summary.json — per-(dataset, mode, model, rule) F1 + bootstrap CIs
  data/!-analysis/aggregation_summary.md   — human-readable table

Usage:
    python -m src.analysis.aggregation
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "!-analysis"
OUT_JSON = OUT_DIR / "aggregation_summary.json"
OUT_MD = OUT_DIR / "aggregation_summary.md"


# ───────────────────────────────────────────────────────── aggregation rules ──

def agg_or_false(preds: list[bool]) -> bool:
    return not any(p is False for p in preds)


def agg_majority(preds: list[bool]) -> bool:
    n_t = sum(1 for p in preds if p is True)
    n_f = len(preds) - n_t
    return n_t >= n_f  # ties → True


def agg_and_false(preds: list[bool]) -> bool:
    return any(p is True for p in preds)  # False only when ALL atoms False


def make_threshold_k(k: int):
    def _f(preds: list[bool]) -> bool:
        n_false = sum(1 for p in preds if p is False)
        return n_false < k  # False only if ≥k atoms False
    _f.__name__ = f"threshold_k{k}"
    return _f


def agg_first(preds: list[bool]) -> bool:
    return preds[0] if preds else True


RULES = {
    "or_false":     agg_or_false,
    "majority":     agg_majority,
    "and_false":    agg_and_false,
    "threshold_k2": make_threshold_k(2),
    "first_atom":   agg_first,
}


# ─────────────────────────────────────────────────────────── F1 utilities ──

def f1f(rows: list[dict]) -> dict:
    """rows: each has 'pred' (bool) and 'gold' (bool)."""
    binary = [r for r in rows if r["gold"] is not None]
    tp_f = fp_f = fn_f = tp_t = fp_t = fn_t = 0
    n_c = 0
    for r in binary:
        gv, pv = r["gold"], bool(r["pred"])
        if pv == gv: n_c += 1
        if gv is False and pv is False: tp_f += 1
        elif gv is False and pv is True: fn_f += 1
        elif gv is True and pv is False: fp_f += 1
        if gv is True and pv is True: tp_t += 1
        elif gv is True and pv is False: fn_t += 1
        elif gv is False and pv is True: fp_t += 1

    def prf(tp, fp, fn):
        p = tp/(tp+fp) if (tp+fp) else 0
        r = tp/(tp+fn) if (tp+fn) else 0
        return 2*p*r/(p+r) if (p+r) else 0

    f1_F = prf(tp_f, fp_f, fn_f)
    f1_T = prf(tp_t, fp_t, fn_t)
    return {
        "n": len(binary),
        "acc": n_c/len(binary) if binary else 0,
        "F1_F": round(f1_F, 4),
        "F1_T": round(f1_T, 4),
        "macro": round((f1_F+f1_T)/2, 4),
    }


def bootstrap_gap(rows_a: list[dict], rows_b: list[dict], B: int = 5000,
                  seed: int = 0) -> dict:
    """Bootstrap CI on F1_F(a) - F1_F(b). Rows must be paired by index."""
    assert len(rows_a) == len(rows_b)
    n = len(rows_a)
    rng = random.Random(seed)
    gaps = []
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        a_s = [rows_a[i] for i in idxs]
        b_s = [rows_b[i] for i in idxs]
        gaps.append(f1f(a_s)["F1_F"] - f1f(b_s)["F1_F"])
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "mean_gap": round(float(np.mean(gaps)), 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "P_pos": round(sum(1 for g in gaps if g > 0)/B, 4),
    }


# ──────────────────────────────────────────── loaders for each dataset ──

def load_medqa_set(split: str, mode: str, model: str):
    """Return (snippet_rows, atom_groups_by_snippet, all_gold_True_count, all_gold_False_count)."""
    gold = json.load(open(ROOT/"data/4-split/medqa.json"))
    snippet_gold = {r["snippet_id"]: bool(r["label_atomic"])
                    for r in gold if r["split"] == split}
    # Build (entry_id, text_lower) → snippet_id mapping
    text_to_sid = {}
    for snip in gold:
        eid = snip["entry_id"]
        for a in snip.get("atoms", []):
            text_to_sid[(eid, a["text"].strip().lower())] = snip["snippet_id"]
    # 2-subset has the atom text under 'claim'
    aid_to_text = {a["id"]: a["claim"].strip().lower()
                   for a in json.load(open(ROOT/"data/2-subset/medqa.json"))}

    snip_path = ROOT/f"data/5-baselines/predictions/snippet/{mode}/{model}/predictions.json"
    atom_path = ROOT/f"data/5-baselines/predictions/atom/{mode}/{model}/predictions.json"
    if not snip_path.exists() or not atom_path.exists():
        return None

    snip_preds = json.load(open(snip_path))
    atom_preds = json.load(open(atom_path))

    # Snippet rows for this split
    snip_rows = []
    snip_map = {(p.get("snippet_id") or p["id"]): bool(p["prediction"]) for p in snip_preds}
    for sid, gv in snippet_gold.items():
        if sid in snip_map:
            snip_rows.append({"sid": sid, "pred": snip_map[sid], "gold": gv})

    # Atom predictions grouped by snippet (via text matching)
    atom_groups: dict[str, list[bool]] = defaultdict(list)
    for p in atom_preds:
        aid = p["id"]
        eid = int(str(aid).split("-")[0])
        text = aid_to_text.get(aid, "").strip().lower()
        if not text: continue
        sid = text_to_sid.get((eid, text))
        if sid is None or sid not in snippet_gold: continue
        atom_groups[sid].append(bool(p["prediction"]))
    return snip_rows, atom_groups, snippet_gold


def load_healthfc(model_tag: str = "gpt-5.4-high"):
    """HealthFC: gold at claim level (label∈{0=T,2=F}); atoms from snippet-processor."""
    atoms_full = json.load(open(ROOT/"data/9-healthfc/snippet-processor/atom-to-snippet/healthfc.json"))
    gold = {r["id"]: r["label"] for r in atoms_full}  # 0/1/2
    snip_path = ROOT/f"data/9-healthfc/baselines/predictions/snippet/claim-only/{model_tag}/predictions.json"
    atom_path = ROOT/f"data/9-healthfc/baselines/predictions/atom/claim-only/{model_tag}/predictions.json"
    if not snip_path.exists() or not atom_path.exists():
        return None
    snip_preds = json.load(open(snip_path))
    atom_preds = json.load(open(atom_path))
    # Only binary gold rows (0 or 2)
    binary_cids = {c: l for c, l in gold.items() if l in (0, 2)}
    def to_bool(label): return True if label == 0 else (False if label == 2 else None)

    snip_rows = []
    for p in snip_preds:
        cid = p["claim_id"]
        if cid not in binary_cids: continue
        snip_rows.append({"sid": cid, "pred": bool(p["prediction"]),
                          "gold": to_bool(binary_cids[cid])})

    atom_groups: dict[str, list[bool]] = defaultdict(list)
    for p in atom_preds:
        cid = p["claim_id"]
        if cid not in binary_cids: continue
        atom_groups[cid].append(bool(p["prediction"]))
    return snip_rows, atom_groups, {c: to_bool(binary_cids[c]) for c in binary_cids}


def load_medhallu(model_tag: str = "gpt-5.4-high"):
    atoms_full = json.load(open(ROOT/"data/10-medhallu/snippet-processor/atom-to-snippet/medhallu.json"))
    gold = {r["id"]: bool(r["gold_bool"]) for r in atoms_full}
    snip_path = ROOT/f"data/10-medhallu/baselines/predictions/snippet/claim-only/{model_tag}/predictions.json"
    atom_path = ROOT/f"data/10-medhallu/baselines/predictions/atom/claim-only/{model_tag}/predictions.json"
    if not snip_path.exists() or not atom_path.exists():
        return None
    snip_preds = json.load(open(snip_path))
    atom_preds = json.load(open(atom_path))

    snip_rows = []
    for p in snip_preds:
        cid = p["claim_id"]
        if cid not in gold: continue
        snip_rows.append({"sid": cid, "pred": bool(p["prediction"]), "gold": gold[cid]})

    atom_groups: dict[str, list[bool]] = defaultdict(list)
    for p in atom_preds:
        cid = p["claim_id"]
        if cid not in gold: continue
        atom_groups[cid].append(bool(p["prediction"]))
    return snip_rows, atom_groups, gold


# ──────────────────────────────────────────────────────── analysis driver ──

def analyze(snip_rows, atom_groups, gold, atom_counts_needed=None):
    """For each aggregation rule, build atom-aggregated rows aligned with snippet rows,
    compute F1, and bootstrap the gap (snippet − atom) on the COMMON subset.
    """
    # Common claims: snippet has it AND all needed atoms are predicted
    snip_map = {r["sid"]: r for r in snip_rows}
    common = []
    for sid, preds in atom_groups.items():
        if sid not in snip_map: continue
        if atom_counts_needed:
            needed = atom_counts_needed.get(sid)
            if needed is not None and len(preds) != needed: continue
        common.append(sid)
    common.sort()

    snip_common = [snip_map[s] for s in common]
    m_snip = f1f(snip_common)

    out = {"n_common": len(common), "snippet": m_snip, "rules": {}}
    for name, rule in RULES.items():
        atom_common = [{"sid": s, "pred": rule(atom_groups[s]), "gold": gold[s]}
                       for s in common]
        m_a = f1f(atom_common)
        bs = bootstrap_gap(snip_common, atom_common, B=5000)
        out["rules"][name] = {**m_a, "gap_vs_snippet": bs}
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"medqa": {}, "healthfc": {}, "medhallu": {}}

    # MedQA: train/dev/test × full-context/claim-only × gpt-5.4-high/gpt-4o-none
    for split in ("train", "dev", "test"):
        for mode in ("full-context", "claim-only"):
            for model in ("gpt-5.4-high", "gpt-4o-none"):
                got = load_medqa_set(split, mode, model)
                if got is None: continue
                snip_rows, atom_groups, gold = got
                # For MedQA, all atoms expected per snippet — count from gold snippet atoms
                gold_snip = {r["snippet_id"]: r for r in json.load(open(ROOT/"data/4-split/medqa.json"))}
                atom_counts = {sid: len(gold_snip[sid].get("atoms", []))
                               for sid in gold_snip}
                analysis = analyze(snip_rows, atom_groups, gold, atom_counts_needed=atom_counts)
                results["medqa"].setdefault(split, {}).setdefault(mode, {})[model] = analysis

    # HealthFC
    hfc = load_healthfc("gpt-5.4-high")
    if hfc:
        snip_rows, atom_groups, gold = hfc
        atoms_full = json.load(open(ROOT/"data/9-healthfc/snippet-processor/atom-to-snippet/healthfc.json"))
        atom_counts = {r["id"]: len(r["generated_atoms"]) for r in atoms_full}
        results["healthfc"]["gpt-5.4-high"] = analyze(snip_rows, atom_groups, gold, atom_counts_needed=atom_counts)

    # MedHallu
    mh = load_medhallu("gpt-5.4-high")
    if mh:
        snip_rows, atom_groups, gold = mh
        atoms_full = json.load(open(ROOT/"data/10-medhallu/snippet-processor/atom-to-snippet/medhallu.json"))
        atom_counts = {r["id"]: len(r["generated_atoms"]) for r in atoms_full}
        results["medhallu"]["gpt-5.4-high"] = analyze(snip_rows, atom_groups, gold, atom_counts_needed=atom_counts)

    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Markdown table
    md = ["# Aggregation rule comparison (snippet vs atom-aggregated)\n"]
    md.append("Tests robustness of the snippet > atom claim under different atom→claim aggregation rules.\n")
    md.append("`gap` = snippet F1_F − atom F1_F under the rule. Bootstrap 95% CI in brackets.\n")
    md.append("`P(>0)` = bootstrap probability that snippet > atom (the granularity claim).\n\n")

    def fmt_run(label, run, indent=""):
        md.append(f"{indent}**{label}**: snippet F1_F={run['snippet']['F1_F']} "
                  f"(n={run['n_common']})\n\n")
        md.append(f"{indent}| rule | atom F1_F | atom F1_T | atom macro | "
                  f"gap | 95% CI | P(snippet>atom) |\n")
        md.append(f"{indent}|---|---:|---:|---:|---:|---|---:|\n")
        for rname, r in run["rules"].items():
            g = r["gap_vs_snippet"]
            md.append(f"{indent}| `{rname}` | {r['F1_F']} | {r['F1_T']} | {r['macro']} | "
                      f"{g['mean_gap']:+.4f} | [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}] | "
                      f"{g['P_pos']:.3f} |\n")
        md.append("\n")

    for split in ("train", "dev", "test"):
        for mode in ("full-context", "claim-only"):
            for model in ("gpt-5.4-high", "gpt-4o-none"):
                run = results["medqa"].get(split, {}).get(mode, {}).get(model)
                if run: fmt_run(f"MedQA {split} · {mode} · {model}", run)

    if results["healthfc"]:
        fmt_run("HealthFC · claim-only · gpt-5.4-high", results["healthfc"]["gpt-5.4-high"])
    if results["medhallu"]:
        fmt_run("MedHallu · claim-only · gpt-5.4-high", results["medhallu"]["gpt-5.4-high"])

    OUT_MD.write_text("".join(md))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")

    # Compact console summary
    print("\nHEADLINE — P(snippet > atom) by rule:")
    print(f"{'setting':<48s} {'or_false':>10s} {'majority':>10s} {'and_false':>10s} {'thresh_k2':>10s} {'first_atom':>10s}")
    rows_to_print = []
    for split in ("train", "dev", "test"):
        for mode in ("full-context", "claim-only"):
            for model in ("gpt-5.4-high", "gpt-4o-none"):
                r = results["medqa"].get(split, {}).get(mode, {}).get(model)
                if r:
                    rows_to_print.append((f"MedQA/{split}/{mode}/{model}", r))
    for tag, r in [("HealthFC/claim-only/gpt-5.4-high", results["healthfc"].get("gpt-5.4-high")),
                   ("MedHallu/claim-only/gpt-5.4-high", results["medhallu"].get("gpt-5.4-high"))]:
        if r: rows_to_print.append((tag, r))
    for tag, r in rows_to_print:
        row = []
        for rname in ("or_false", "majority", "and_false", "threshold_k2", "first_atom"):
            p = r["rules"][rname]["gap_vs_snippet"]["P_pos"]
            row.append(f"{p:.3f}")
        print(f"{tag:<48s} {row[0]:>10s} {row[1]:>10s} {row[2]:>10s} {row[3]:>10s} {row[4]:>10s}")


if __name__ == "__main__":
    main()
