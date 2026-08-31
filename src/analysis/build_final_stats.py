"""Compute comprehensive dataset statistics for data/15-final/dataset.json.

Output: data/15-final/stats.json — a single JSON file with:
  - totals (entries, snippets, atoms)
  - per-split counts
  - per-subset counts
  - pattern distribution (overall + by split + by subset)
  - label distributions (label_atomic / label_human_general / label_human_contextual)
  - snippet-size distribution (single-atom vs multi-atom)
  - cross-tabs: pattern x split, pattern x subset, pattern x labels

Usage:
    python -m src.analysis.build_final_stats
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "15-final" / "dataset.json"
OUT_FILE = ROOT / "data" / "15-final" / "stats.json"

PATTERN_CODES = ["A", "B", "C", "D", "E", "F"]


def count_label(records: list[dict], key: str) -> dict[str, int]:
    """Count true / false / null / other values for a label field."""
    c: Counter = Counter()
    for r in records:
        v = r.get(key)
        if v is True or v == "true":
            c["true"] += 1
        elif v is False or v == "false":
            c["false"] += 1
        elif v is None or v == "" or v is None:
            c["null"] += 1
        else:
            c[str(v)] += 1
    return dict(c)


def main() -> None:
    data = json.load(open(IN_FILE))
    n_snippets = len(data)

    # Entry-level counts
    entry_ids = sorted({r["entry_id"] for r in data})
    n_entries = len(entry_ids)

    # Atom counts
    n_atoms_total = sum(len(r.get("atoms", [])) for r in data)
    multi_atom = [r for r in data if len(r.get("atoms", [])) >= 2]
    single_atom = [r for r in data if len(r.get("atoms", [])) == 1]
    atoms_per_snippet = [len(r.get("atoms", [])) for r in data]
    atoms_per_snippet.sort()
    median_aps = atoms_per_snippet[n_snippets // 2]
    max_aps = atoms_per_snippet[-1]

    # Per-split
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in data:
        by_split[r.get("split", "unknown")].append(r)

    # Per-subset
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in data:
        by_subset[r.get("subset", "unknown")].append(r)

    # Pattern distribution overall
    pat_overall = Counter(r.get("pattern") for r in data)
    pat_overall_pct = {
        k: round(100 * v / n_snippets, 2)
        for k, v in pat_overall.items()
    }

    # Pattern x split
    pat_by_split: dict[str, dict[str, int]] = {}
    for split, recs in by_split.items():
        c = Counter(r.get("pattern") for r in recs)
        pat_by_split[split] = {k: c.get(k, 0) for k in PATTERN_CODES + [None]}

    # Pattern x subset
    pat_by_subset: dict[str, dict[str, int]] = {}
    for sub, recs in by_subset.items():
        c = Counter(r.get("pattern") for r in recs)
        pat_by_subset[sub] = {k: c.get(k, 0) for k in PATTERN_CODES + [None]}

    # Pattern x label_human_general (true/false)
    pat_x_label_general: dict[str, dict[str, int]] = {}
    for p in PATTERN_CODES:
        recs = [r for r in data if r.get("pattern") == p]
        pat_x_label_general[p] = count_label(recs, "label_human_general")

    pat_x_label_contextual: dict[str, dict[str, int]] = {}
    for p in PATTERN_CODES:
        recs = [r for r in data if r.get("pattern") == p]
        pat_x_label_contextual[p] = count_label(recs, "label_human_contextual")

    # Dual-label divergence: label_in_general != label_with_patient_context
    n_divergent = sum(
        1 for r in data
        if r.get("label_human_general") != r.get("label_human_contextual")
    )
    divergent_by_pattern = Counter()
    for r in data:
        if r.get("label_human_general") != r.get("label_human_contextual"):
            divergent_by_pattern[r.get("pattern")] += 1

    # Multi-atom vs single-atom pattern breakdown
    multi_pat = Counter(r.get("pattern") for r in multi_atom)
    single_pat = Counter(r.get("pattern") for r in single_atom)

    # Label distributions overall
    label_atomic = count_label(data, "label_atomic")
    label_general = count_label(data, "label_human_general")
    label_contextual = count_label(data, "label_human_contextual")

    stats = {
        "source_file": str(IN_FILE.relative_to(ROOT)),
        "totals": {
            "entries": n_entries,
            "snippets": n_snippets,
            "atoms": n_atoms_total,
        },
        "atoms_per_snippet": {
            "mean": round(n_atoms_total / n_snippets, 3),
            "median": median_aps,
            "max": max_aps,
            "single_atom": len(single_atom),
            "multi_atom": len(multi_atom),
        },
        "by_split": {
            k: {
                "snippets": len(v),
                "atoms": sum(len(r.get("atoms", [])) for r in v),
                "entries": len({r["entry_id"] for r in v}),
            }
            for k, v in by_split.items()
        },
        "by_subset": {
            k: {
                "snippets": len(v),
                "atoms": sum(len(r.get("atoms", [])) for r in v),
                "entries": len({r["entry_id"] for r in v}),
            }
            for k, v in by_subset.items()
        },
        "pattern_distribution": {
            "overall_counts": {k: pat_overall.get(k, 0) for k in PATTERN_CODES + [None]},
            "overall_percent": {k: pat_overall_pct.get(k, 0.0) for k in PATTERN_CODES + [None]},
            "by_split": pat_by_split,
            "by_subset": pat_by_subset,
            "multi_atom_only": {k: multi_pat.get(k, 0) for k in PATTERN_CODES + [None]},
            "single_atom_only": {k: single_pat.get(k, 0) for k in PATTERN_CODES + [None]},
        },
        "labels": {
            "label_atomic": label_atomic,
            "label_human_general": label_general,
            "label_human_contextual": label_contextual,
            "divergent_general_vs_contextual": n_divergent,
            "divergent_by_pattern": {
                k: divergent_by_pattern.get(k, 0) for k in PATTERN_CODES + [None]
            },
        },
        "pattern_x_label_human_general": pat_x_label_general,
        "pattern_x_label_human_contextual": pat_x_label_contextual,
    }

    OUT_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)")
    print(f"\ntotals: {stats['totals']}")
    print(f"splits: {list(stats['by_split'].keys())}")
    print(f"subsets: {list(stats['by_subset'].keys())}")
    print(f"pattern overall: {stats['pattern_distribution']['overall_counts']}")
    print(f"dual-label divergent: {n_divergent} / {n_snippets} "
          f"({100*n_divergent/n_snippets:.1f}%)")


if __name__ == "__main__":
    main()
