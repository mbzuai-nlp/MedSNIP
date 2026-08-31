"""Normalization-vs-grouping ablation for patterns D and F (meta-review #2).

Reviewer FJrT observed that patterns D and F improve without any grouping -
one atom in, one snippet out - and suggested the gain comes from the pipeline
writing cleaner, decontextualised text rather than from granularity.

`df_rewrite_split.py` established the correlation for free: where annotators
left the text alone the gain is +0.013 (the verifier's own noise floor), and
where they reworded it the gain is +0.098. But the two subgroups are not
matched - reworded units started harder (atom F1_F 0.373 vs 0.495) - so that
split confounds text with difficulty.

This isolates it causally. Units are held fixed and only the text varies:

    raw atom        the original expert atom (already verified)
    normalized atom the same atom, decontextualised but NOT merged   <- new
    human snippet   the annotator's rewrite (already verified)

If the normalized atom reaches the snippet score, wording explains the gain.
If it stays at the raw-atom score, something other than wording is at work.

The normalizer applies the snippet pipeline's own text rules with every
grouping instruction removed, so the ablation tests the operation rather than
introducing a new one. It rewrites one claim at a time and is explicitly
forbidden from adding facts or changing the claim's stance - a normalizer that
silently corrected wrong claims would manufacture the effect it is measuring.

Usage:
  python -m src.analysis.normalize_atoms --limit 5      # smoke test
  python -m src.analysis.normalize_atoms
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from ..baselines._openrouter_runner import build_kwargs, make_client, usage_row

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "data" / "!-final" / "dataset.json"
OUT_DIR = ROOT / "data" / "14-normalization"
OUT_PATH = OUT_DIR / "normalized_atoms.json"

MODEL = "openai/gpt-5.4"
REASONING = "high"

SYSTEM = """\
You rewrite a single medical claim so it can be judged on its own, without \
any surrounding text.

Apply exactly these edits:
  - Resolve pronouns and vague references to the entity they denote \
(e.g. "it" -> "Benadryl", "both medications" -> "NyQuil and Benadryl").
  - Inline the relevant patient demographics or query framing from the \
provided context when the claim depends on them.
  - Keep the claim's meaning, scope and stance unchanged.

Hard constraints:
  - Do NOT merge in any other claim. The output states the same single \
proposition as the input.
  - Do NOT introduce facts absent from the source, and do NOT add hedges.
  - Do NOT correct a claim that looks medically wrong. Preserve the stance \
exactly; a rewrite that fixes errors would invalidate this measurement.
  - If the claim is already self-contained, return it unchanged.

Return JSON: {"output": "<rewritten claim>"}\
"""


def user_message(atom_text: str, query: str, shared_context: dict) -> str:
    ctx = json.dumps(shared_context or {}, ensure_ascii=False)[:1200]
    return (f"Question the response answered:\n{query}\n\n"
            f"Shared context:\n{ctx}\n\n"
            f"Claim to rewrite:\n{atom_text}")


def norm(s: str | None) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def select_units(patterns: list[str], subgroup: str) -> list[dict]:
    """One row per ATOM, not per snippet.

    Merge patterns (A/B/C) group several atoms into one snippet, so the
    ablation normalizes each atom separately and scores them as the atom
    condition does. The `reworded` subgroup filter only has meaning for
    keep-atomic patterns, where atom and snippet are 1:1; for multi-atom
    snippets there is no single text to compare against, so it is left None
    and the filter does not apply.
    """
    rows = json.loads(FINAL.read_text())
    out = []
    for r in rows:
        pat = (r.get("pattern") or "").strip().upper()[:1]
        if pat not in patterns:
            continue
        atoms = r.get("atoms") or []
        if not atoms:
            continue
        single = len(atoms) == 1
        for a in atoms:
            reworded = (norm(a.get("text")) != norm(r["snippet_text"])
                        if single else None)
            if single and subgroup == "reworded" and not reworded:
                continue
            if single and subgroup == "identical" and reworded:
                continue
            out.append({
                "id": f"{r['entry_id']}-{a.get('index')}",
                "snippet_id": r["snippet_id"], "entry_id": r["entry_id"],
                "pattern": pat, "subset": r["subset"],
                "atoms_in_snippet": len(atoms),
                "atom_text": a.get("text"), "snippet_text": r["snippet_text"],
                "query": r.get("query", ""), "shared_context": r.get("shared_context") or {},
                # Per-atom gold, which is what the atom condition is scored on.
                "label_atomic": bool(a.get("label", r["label_atomic"])),
                "reworded_by_human": reworded,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="*", default=["D", "F"])
    ap.add_argument("--subgroup", default="reworded",
                    choices=["reworded", "identical", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-name", default="normalized_atoms.json",
                    help="output filename inside data/14-normalization/")
    args = ap.parse_args()

    global OUT_PATH
    OUT_PATH = OUT_DIR / args.out_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    units = select_units(args.patterns, args.subgroup)
    if args.limit:
        units = units[: args.limit]

    existing = {}
    if OUT_PATH.exists():
        existing = {r["id"]: r for r in json.loads(OUT_PATH.read_text())}
    todo = [u for u in units if u["id"] not in existing]
    print(f"units={len(units)} done={len(existing)} todo={len(todo)} "
          f"(patterns={args.patterns}, subgroup={args.subgroup})", flush=True)
    if not todo:
        return

    client = make_client()
    lock = Lock()
    results = list(existing.values())
    stats = {"n": 0, "unchanged": 0, "cost": 0.0, "errors": 0}

    def work(u: dict) -> dict:
        kw = build_kwargs(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user_message(
                          u["atom_text"], u["query"], u["shared_context"])}],
            reasoning=REASONING, temperature=0.0, max_tokens=1200)
        kw["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kw)
        raw = (resp.choices[0].message.content or "").strip()
        text = json.loads(raw).get("output") or u["atom_text"]
        return {**u, "normalized_text": text,
                "changed": norm(text) != norm(u["atom_text"]),
                "usage": usage_row(resp)}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, u): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {**u, "normalized_text": u["atom_text"], "changed": False,
                       "error": f"{type(e).__name__}: {str(e)[:140]}"}
                stats["errors"] += 1
            with lock:
                results.append(row)
                stats["n"] += 1
                if not row.get("changed"):
                    stats["unchanged"] += 1
                stats["cost"] += ((row.get("usage") or {}).get("cost") or 0.0)
                if stats["n"] % 25 == 0 or stats["n"] == len(todo):
                    print(f"  {stats['n']}/{len(todo)}  unchanged={stats['unchanged']}  "
                          f"errors={stats['errors']}  ${stats['cost']:.3f}", flush=True)
                    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(results)} rows -> {OUT_PATH}")
    print(f"changed={stats['n'] - stats['unchanged']}/{stats['n']}  "
          f"errors={stats['errors']}  cost=${stats['cost']:.2f}")


if __name__ == "__main__":
    main()
