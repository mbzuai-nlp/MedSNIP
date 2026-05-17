"""Merge human annotations onto a snippet-level flat MedQA schema.

After this step, medqa.json carries one row per *snippet* (not per
atom). Snippets are the meaningful unit downstream — the annotator's
chunks of the response — and annotators sometimes produce overlapping
snippet groupings (the same atom appearing in multiple snippets, e.g.,
a fine-grained per-atom snippet plus a coarse meta-snippet covering
the same atoms). One-row-per-atom can't represent that many-to-many;
one-row-per-snippet does.

`split` is *not* assigned here — the split step runs after annotation
now so it can stratify on human labels rather than LLM atom labels.

Each snippet row carries:

  - snippet_id            ("<entry_id>-S<idx>")
  - entry_id              (parent entry)
  - query, full_text      (denormalized from entry)
  - subset                (consumer | vignette, from step 2)
  - batch_id              (which annotation batch the entry was in)
  - shared_context        (annotator-supplied context dict)
  - snippet_text          (human-edited snippet text)
  - atoms                 (list of {index, text, llm_label} for the
                           atoms this snippet covers; llm_label is the
                           per-atom LLM self-label from raw MedQA)
  - is_ambiguous          ("yes"/"no")
  - notes                 (annotator's free-text)
  - label_atomic          (AND over atoms' llm_label; bool)
  - label_human_general   (annotator's label_in_general; bool)
  - label_human_contextual (annotator's label_with_patient_context; bool)
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "2-subset" / "medqa.json"
BATCHES_DIR = ROOT / "data" / "3-annotated" / "batches"
OUT_DIR = ROOT / "data" / "3-annotated"

BATCH_RE = re.compile(r"batch_(\d+)_annotations\.json$")


def to_bool(s: str) -> bool:
    s = (s or "").strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    raise ValueError(f"unexpected label value: {s!r}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atom_rows = json.loads(IN_PATH.read_text())

    # Index atoms by (entry_id, atom_index)
    atom_by_key: dict[tuple[int, int], dict] = {}
    entry_attrs: dict[int, dict] = {}
    for r in atom_rows:
        eid, aid = map(int, str(r["id"]).split("-"))
        atom_by_key[(eid, aid)] = {
            "index": aid,
            "text": r["claim"],
            "llm_label": bool(r["label"]),
        }
        if eid not in entry_attrs:
            entry_attrs[eid] = {
                "query": r["query"],
                "full_text": r["full_text"],
                "subset": r["subset"],
            }

    # Walk every annotated snippet across all batches; one output row each
    out_rows = []
    n_atoms_missing = 0
    for f in sorted(BATCHES_DIR.glob("batch_*_annotations.json")):
        m = BATCH_RE.search(f.name)
        if not m:
            continue
        batch_id = int(m.group(1))
        for entry in json.loads(f.read_text()):
            eid = entry["entry_id"]
            attrs = entry_attrs[eid]
            shared_context = entry.get("shared_context") or {}

            for s_idx, sn in enumerate(entry["snippets"], 1):
                atoms = []
                for atom_idx in sn["source_claims"]:
                    a = atom_by_key.get((eid, atom_idx))
                    if a is None:
                        n_atoms_missing += 1
                        continue
                    atoms.append(a)
                label_atomic = all(a["llm_label"] for a in atoms) if atoms else True

                out_rows.append({
                    "snippet_id":    f"{eid}-S{s_idx}",
                    "entry_id":      eid,
                    "query":         attrs["query"],
                    "full_text":     attrs["full_text"],
                    "subset":        attrs["subset"],
                    "batch_id":      batch_id,
                    "shared_context": shared_context,
                    "snippet_text":  sn["output"],
                    "atoms":         atoms,
                    "is_ambiguous":  sn.get("is_ambiguous", "no"),
                    "notes":         sn.get("notes", ""),
                    "label_atomic":           label_atomic,
                    "label_human_general":    to_bool(sn["label_in_general"]),
                    "label_human_contextual": to_bool(sn["label_with_patient_context"]),
                })

    # Sort by (entry_id, snippet_index) so the file is in natural order
    # regardless of which batch each entry was annotated in.
    out_rows.sort(key=lambda r: (r["entry_id"], int(r["snippet_id"].split("-S")[1])))

    (OUT_DIR / "medqa.json").write_text(json.dumps(out_rows, indent=2, ensure_ascii=False))

    # Stats per subset
    by: dict[str, dict] = defaultdict(lambda: {
        "snippets": 0, "atomic_claims": 0,
        "human_general_false": 0, "human_contextual_false": 0,
        "atomic_false": 0, "ambiguous": 0,
    })
    for r in out_rows:
        b = by[r["subset"]]
        b["snippets"] += 1
        b["atomic_claims"] += len(r["atoms"])
        if not r["label_atomic"]:            b["atomic_false"] += 1
        if not r["label_human_general"]:     b["human_general_false"] += 1
        if not r["label_human_contextual"]:  b["human_contextual_false"] += 1
        if r["is_ambiguous"] == "yes":       b["ambiguous"] += 1

    summary = {}
    for s in ("consumer", "vignette"):
        b = by[s]
        n = b["snippets"]
        summary[s] = {
            "snippets":                    n,
            "atomic_claims":               b["atomic_claims"],
            "atomic_false_rate":           round(b["atomic_false"]           / n, 4) if n else 0,
            "human_general_false_rate":    round(b["human_general_false"]    / n, 4) if n else 0,
            "human_contextual_false_rate": round(b["human_contextual_false"] / n, 4) if n else 0,
            "ambiguous_rate":              round(b["ambiguous"]              / n, 4) if n else 0,
        }

    stats = {
        "total_entries":      len({r["entry_id"] for r in out_rows}),
        "total_snippets":     len(out_rows),
        "total_atomic_refs":  sum(len(r["atoms"]) for r in out_rows),
        "by_subset":          summary,
        "atoms_missing_in_source_claims": n_atoms_missing,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"wrote {len(out_rows)} snippet rows from {stats['total_entries']} entries")
    for s in ("consumer", "vignette"):
        info = summary[s]
        print(f"  {s:9s}  snippets={info['snippets']:4d}  "
              f"atom_refs={info['atomic_claims']:5d}  "
              f"human_general_false={info['human_general_false_rate']:.1%}  "
              f"human_contextual_false={info['human_contextual_false_rate']:.1%}  "
              f"ambig={info['ambiguous_rate']:.1%}")
    if n_atoms_missing:
        print(f"  WARN: {n_atoms_missing} atom references in source_claims had no matching atom row")


if __name__ == "__main__":
    main()
