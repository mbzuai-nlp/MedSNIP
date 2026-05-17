"""Precompute the gold-atom → sentence-index alignment for every entry.

Reads data/4-split/medqa.json (snippet rows; we aggregate per entry) and
writes data/6-snippet-processor/sentence_alignment.json:

  {
    "<entry_id>": {
      "subset": ...,
      "sentences": [<str>, ...],
      "atom_to_sentence": {<atom_index>: <sentence_index or -1>},
      "n_atoms_aligned": <int>,
      "n_atoms_total": <int>
    }
  }

Reports overall alignment coverage and lists problem entries (any atom -1).
"""
import json
from pathlib import Path

from .run import aggregate_entries
from .sentence_utils import align_atoms_to_sentences, split_sentences

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "data" / "4-split" / "medqa.json"
OUT_PATH = ROOT / "data" / "6-snippet-processor" / "sentence_alignment.json"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SPLIT_PATH.read_text())
    entries = aggregate_entries(rows)

    result: dict[str, dict] = {}
    total_atoms = 0
    total_aligned = 0
    problem_entries = []
    for e in entries:
        sents = split_sentences(e["full_text"])
        align = align_atoms_to_sentences(sents, e["full_claims"])
        n_total = len(e["full_claims"])
        n_aligned = sum(1 for v in align.values() if v >= 0)
        result[str(e["entry_id"])] = {
            "subset": e["subset"],
            "sentences": sents,
            "atom_to_sentence": {str(k): v for k, v in align.items()},
            "n_atoms_aligned": n_aligned,
            "n_atoms_total": n_total,
        }
        total_atoms += n_total
        total_aligned += n_aligned
        if n_aligned < n_total:
            problem_entries.append({
                "entry_id": e["entry_id"],
                "subset": e["subset"],
                "missing": [a["index"] for a in e["full_claims"]
                            if align[a["index"]] < 0],
                "n_sentences": len(sents),
            })

    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    pct = 100 * total_aligned / total_atoms if total_atoms else 0.0
    print(f"wrote alignment for {len(result)} entries -> {OUT_PATH}")
    print(f"overall atom alignment: {total_aligned}/{total_atoms} ({pct:.1f}%)")
    print(f"problem entries (any atom unmatched): {len(problem_entries)}")
    for p in problem_entries[:10]:
        print(f"  e{p['entry_id']:3d} [{p['subset']}]: "
              f"{len(p['missing'])} unmatched ({p['n_sentences']} sentences)")


if __name__ == "__main__":
    main()
