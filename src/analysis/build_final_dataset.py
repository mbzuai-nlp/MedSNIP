"""Build the final dataset.json by joining the split file with pattern labels.

Inputs:
  - data/4-split/medsnip-bench.json  (one record per snippet with snippet_id, entry_id, atoms[], labels, split)
  - data/3-annotated/pattern/entries/entry_<N>_patterns.json  (per-entry pattern labels)

Output:
  - data/15-final/dataset.json  (same snippet records, with an added `pattern` field)

Matching strategy: for each snippet in the split file, look up the pattern
entry for its entry_id and find the pattern record whose `source_claims`
match the snippet's atom `index` set (as a multiset, ignoring order).

Usage:
    python -m src.analysis.build_final_dataset
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = ROOT / "data" / "4-split" / "medsnip-bench.json"
PATTERN_DIR = ROOT / "data" / "3-annotated" / "pattern" / "entries"
OUT_DIR = ROOT / "data" / "15-final"
OUT_FILE = OUT_DIR / "dataset.json"


def build_pattern_lookup() -> dict[int, list[dict]]:
    """Return {entry_id: [{source_claims:[...], pattern, ...}, ...]}."""
    lookup: dict[int, list[dict]] = {}
    for f in sorted(PATTERN_DIR.glob("entry_*_patterns.json")):
        eid = int(re.search(r"entry_(\d+)_patterns", f.name).group(1))
        d = json.load(open(f))
        lookup[eid] = d.get("snippets", [])
    return lookup


def find_pattern(snippet_records: list[dict],
                  atom_indices: list[int]) -> dict | None:
    """Find a pattern record matching the atom-index set."""
    target = sorted(atom_indices)
    for rec in snippet_records:
        if sorted(rec.get("source_claims", [])) == target:
            return rec
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = json.load(open(SPLIT_FILE))
    pat_lookup = build_pattern_lookup()
    print(f"loaded {len(split)} snippets from split file")
    print(f"loaded patterns for {len(pat_lookup)} entries")

    n_matched = 0
    n_missing = 0
    n_no_entry = 0
    missing_examples: list[tuple[int, list[int]]] = []
    for snip in split:
        eid = snip["entry_id"]
        atom_indices = [a["index"] for a in snip.get("atoms", [])]
        recs = pat_lookup.get(eid)
        if not recs:
            n_no_entry += 1
            snip["pattern"] = None
            if len(missing_examples) < 10:
                missing_examples.append((eid, atom_indices))
            continue
        match = find_pattern(recs, atom_indices)
        if match is None:
            n_missing += 1
            snip["pattern"] = None
            if len(missing_examples) < 10:
                missing_examples.append((eid, atom_indices))
            continue
        snip["pattern"] = match.get("pattern")
        n_matched += 1

    OUT_FILE.write_text(json.dumps(split, indent=2, ensure_ascii=False))

    print(f"\nmatched: {n_matched}/{len(split)}")
    print(f"no pattern entry for entry_id: {n_no_entry}")
    print(f"no matching source_claims:     {n_missing}")
    if missing_examples:
        print("\nfirst missing-match examples:")
        for eid, idx in missing_examples:
            print(f"  entry {eid} atoms={idx}")
    print(f"\nwrote {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
