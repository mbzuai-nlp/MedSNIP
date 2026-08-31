"""Pre-process annotated batches into per-entry files with atom texts inlined.

For each entry in each batch under data/3-annotated/batches/, write a
self-contained file at data/3-annotated/pattern/entries/<entry_id>.json
containing:
  - entry_id, batch, shared_context (passthrough)
  - atoms: { claim_id: claim_text } for every atom referenced in source_claims
  - snippets: original snippet records, unchanged

This lets a per-entry annotation agent operate on a tiny self-contained
file instead of looking up across 5,755 atoms.

Usage:
    python -m src.analysis.prep_pattern_entries
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCHES_DIR = ROOT / "data" / "3-annotated" / "batches"
ATOMS_FILE = ROOT / "data" / "2-subset" / "kim.json"
OUT_DIR = ROOT / "data" / "3-annotated" / "pattern" / "entries"


def build_atom_lookup() -> dict[int, dict[int, str]]:
    """Return {entry_id: {claim_id: claim_text}} from the atom file."""
    atoms = json.load(open(ATOMS_FILE))
    lookup: dict[int, dict[int, str]] = {}
    pat = re.compile(r"^(\d+)-(\d+)$")
    for a in atoms:
        m = pat.match(str(a["id"]))
        if not m:
            continue
        eid, cid = int(m.group(1)), int(m.group(2))
        lookup.setdefault(eid, {})[cid] = a["claim"]
    return lookup


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atom_lookup = build_atom_lookup()
    print(f"loaded atoms for {len(atom_lookup)} entries")

    n_batches = 0
    n_entries = 0
    n_snippets = 0
    for batch_file in sorted(BATCHES_DIR.glob("batch_*.json")):
        n_batches += 1
        batch_num = int(re.search(r"batch_(\d+)", batch_file.name).group(1))
        data = json.load(open(batch_file))
        for entry in data:
            eid = entry["entry_id"]
            atoms_for_entry = atom_lookup.get(eid, {})
            # Subset atoms to just those referenced in any snippet's source_claims.
            referenced: set[int] = set()
            for s in entry.get("snippets", []):
                for cid in s.get("source_claims", []):
                    referenced.add(int(cid))
            atoms_subset = {
                str(cid): atoms_for_entry[cid]
                for cid in sorted(referenced)
                if cid in atoms_for_entry
            }
            out = {
                "entry_id": eid,
                "batch": batch_num,
                "shared_context": entry.get("shared_context", {}),
                "atoms": atoms_subset,
                "snippets": entry.get("snippets", []),
            }
            out_path = OUT_DIR / f"entry_{eid}.json"
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            n_entries += 1
            n_snippets += len(entry.get("snippets", []))

    print(f"wrote {n_entries} entry files across {n_batches} batches")
    print(f"total snippets: {n_snippets}")
    print(f"output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
