"""Project gold labels from human snippets onto auto-generated (MedSNIP)
snippets via sentence-level mapping.

For each auto snippet, we resolve its covered original sentences (via
`source_claims` → `generated_atoms[*].source_sentences` for atom-to-snippet,
or `source_sentences` directly for snippet-direct).

For each covered sentence, we look up the human-snippet label_atomic of each
human snippet whose atoms map to that sentence. A sentence may have multiple
contributors (different human snippets touching the same sentence). We
classify each sentence as:
  - unanimous_T: all human contributors labeled True
  - unanimous_F: all human contributors labeled False
  - mixed     : at least one True and at least one False contributor

Each kept auto snippet is annotated with three label-projection variants so
downstream evaluation can choose the policy without re-running the prep:
  - label_atomic_and       (Option 1): AND of all covered sentences, where
                            mixed sentences are treated as False.
  - label_atomic_majority  (Option 2): AND of all covered sentences, where
                            mixed sentences use majority vote (ties = True).
  - _touches_mixed         flag → True iff any covered sentence is mixed.
                            Option 3 (drop-mixed) filters these out at eval time.

`label_atomic` is set to Option 3 semantics (= label_atomic_and on non-mixed
snippets, None on mixed) where possible — but since run_snippet.py expects a
boolean, we default it to label_atomic_and for compatibility. Downstream
evaluator should ignore label_atomic and use the explicit *_and / *_majority
fields with the `_touches_mixed` mask.

Outputs a flat snippet file matching the schema of `data/3-annotated/medsnip-bench.json`.

Usage:
  python -m src.baselines.prep_auto_labels --mode atom-to-snippet --model-slug gpt-5.4-high
  python -m src.baselines.prep_auto_labels --mode snippet-direct  --model-slug gpt-5.4-high
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HUMAN_PATH = ROOT / "data" / "3-annotated" / "medsnip-bench.json"
ALIGN_PATH = ROOT / "data" / "6-snippet-processor" / "sentence_alignment.json"
SP_DIR = ROOT / "data" / "6-snippet-processor"
SPLIT_PATH = ROOT / "data" / "4-split" / "medsnip-bench.json"


def build_sentence_label_map(human_rows, alignment):
    """Return (sentence_label_and, sentence_label_majority, mixed_sentences).

    Each entry maps {entry_id: {sentence_idx: bool}}, plus a set of
    "mixed" sentences (multiple human contributors with disagreement).
    """
    contributors: dict[int, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in human_rows:
        eid = r["entry_id"]
        a2s = alignment.get(str(eid), {}).get("atom_to_sentence", {})
        lbl = bool(r["label_atomic"])
        for atom in r.get("atoms") or []:
            sent = a2s.get(str(atom["index"]), -1)
            if sent is None or sent < 0:
                continue
            contributors[eid][int(sent)].append(lbl)

    label_and: dict[int, dict[int, bool]] = {}
    label_majority: dict[int, dict[int, bool]] = {}
    mixed: dict[int, set[int]] = defaultdict(set)
    for eid, sents in contributors.items():
        label_and[eid] = {}
        label_majority[eid] = {}
        for s, ls in sents.items():
            n_true = sum(1 for x in ls if x)
            n_false = len(ls) - n_true
            label_and[eid][s] = all(ls)
            # majority vote; ties → True (give benefit of the doubt)
            label_majority[eid][s] = n_true >= n_false
            if n_true > 0 and n_false > 0:
                mixed[eid].add(s)
    return label_and, label_majority, dict(mixed)


def project_one_entry(entry: dict, unit: str = "snippet"):
    """Yield (record, covered_sentence_set, source_index_list) per unit.

    `unit="atom"` projects onto the decomposer's own generated atoms instead of
    its snippets. Atoms carry `source_sentences` directly, so the same
    sentence-level label map applies. This is what makes the atom-vs-snippet
    comparison symmetric *within* a decomposer: both arms come from the same
    decomposition call, so a difference cannot be attributed to atom quality.
    """
    atoms = entry.get("generated_atoms") or []

    if unit == "atom":
        for a in atoms:
            covered = sorted(int(i) for i in (a.get("source_sentences") or []))
            yield a, covered, [int(a["index"])]
        return

    atom_sents: dict[int, set[int]] = {}
    for a in atoms:
        atom_sents[int(a["index"])] = set(a.get("source_sentences") or [])

    for s in entry.get("snippets") or []:
        covered: set[int] = set()
        src: list[int] = []
        if atoms:
            # atom-to-snippet: source_claims → generated_atoms[idx-1].source_sentences
            src = [int(i) for i in (s.get("source_claims") or [])]
            for idx in src:
                covered.update(atom_sents.get(idx, set()))
        else:
            # snippet-direct: source_sentences on the snippet directly
            covered.update(int(i) for i in (s.get("source_sentences") or []))
            src = sorted(covered)
        yield s, sorted(covered), src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("atom-to-snippet", "snippet-direct"),
                    required=True)
    ap.add_argument("--model-slug", default="gpt-5.4-high")
    ap.add_argument("--unit", choices=("snippet", "atom"), default="snippet",
                    help="project onto the decomposer's snippets (default) or "
                         "its own generated atoms")
    args = ap.parse_args()

    human_rows = json.loads(HUMAN_PATH.read_text())
    alignment = json.loads(ALIGN_PATH.read_text())
    split_rows = json.loads(SPLIT_PATH.read_text())
    entry_meta = {r["entry_id"]: r for r in split_rows}

    lbl_and, lbl_maj, mixed = build_sentence_label_map(human_rows, alignment)
    n_mixed_total = sum(len(v) for v in mixed.values())
    print(f"sentence-label map: {len(lbl_and)} entries, "
          f"{n_mixed_total} mixed-label sentences")

    auto_dir = SP_DIR / args.mode / args.model_slug
    out_path = auto_dir / f"labeled_{args.unit}s.json"
    if not auto_dir.exists():
        raise SystemExit(f"missing: {auto_dir}")

    out_rows: list[dict] = []
    stats = Counter()

    for f in sorted(auto_dir.glob("e*.json"), key=lambda p: int(p.stem[1:])):
        entry = json.loads(f.read_text())
        eid = entry["entry_id"]
        ent_and = lbl_and.get(eid, {})
        ent_maj = lbl_maj.get(eid, {})
        ent_mixed = mixed.get(eid, set())
        em = entry_meta.get(eid, {})

        for i, (s, covered, src) in enumerate(project_one_entry(entry, args.unit), 1):
            stats["total"] += 1
            labels_and = [ent_and[c] for c in covered if c in ent_and]
            labels_maj = [ent_maj[c] for c in covered if c in ent_maj]
            touches_mixed = any(c in ent_mixed for c in covered)

            if not labels_and:
                stats["dropped_no_coverage"] += 1
                continue

            label_and  = all(labels_and)
            label_maj  = all(labels_maj)
            stats["kept"] += 1
            if touches_mixed: stats["touches_mixed"] += 1
            if label_and:  stats["and_true"]  += 1
            else:          stats["and_false"] += 1
            if label_maj:  stats["maj_true"]  += 1
            else:          stats["maj_false"] += 1

            out_rows.append({
                "snippet_id":             f"auto{'atom' if args.unit == 'atom' else ''}-{eid}-{i}",
                "entry_id":               eid,
                "query":                  entry.get("full_query", ""),
                "full_text":              em.get("full_text", ""),
                "subset":                 entry["subset"],
                "batch_id":               entry["batch_id"],
                "shared_context":         entry.get("shared_context") or {},
                "snippet_text":           s.get("output") or s.get("text") or "",
                "unit":                   args.unit,
                "atoms":                  [
                    {"index": int(a["index"]), "text": a["text"]}
                    for a in (entry.get("generated_atoms") or [])
                    if int(a["index"]) in src
                ] or [{"index": idx, "text": ""} for idx in src],
                "is_ambiguous":           False,
                "notes":                  s.get("notes", ""),
                # default label_atomic = Option 1 (AND), for run_snippet.py compatibility
                "label_atomic":           bool(label_and),
                # three projection variants for downstream policy choice
                "label_atomic_and":       bool(label_and),
                "label_atomic_majority":  bool(label_maj),
                "_touches_mixed":         touches_mixed,
                # human-judgement fields don't transfer — placeholder = AND
                "label_human_general":    bool(label_and),
                "label_human_contextual": bool(label_and),
                "_covered_sentences":     covered,
                "_source_claims":         src,
                "_label_source":          "projected_from_human_label_atomic",
            })

    out_path.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False))
    print()
    print(f"wrote {len(out_rows)} labeled snippet rows → {out_path}")
    print(f"  total auto snippets seen: {stats['total']}")
    print(f"  dropped (no coverage):    {stats['dropped_no_coverage']}")
    print(f"  kept:                     {stats['kept']}")
    print(f"  └─ touches mixed sentence: {stats['touches_mixed']} "
          f"({100*stats['touches_mixed']/max(stats['kept'],1):.1f}% of kept)")
    print()
    print(f"  Option 1 (AND, mixed=False):     "
          f"true={stats['and_true']}, false={stats['and_false']}")
    print(f"  Option 2 (majority on mixed):    "
          f"true={stats['maj_true']}, false={stats['maj_false']}")
    n_drop = stats["kept"] - stats["touches_mixed"]
    print(f"  Option 3 (drop mixed):           "
          f"effective set size {n_drop} (drops {stats['touches_mixed']})")


if __name__ == "__main__":
    main()
