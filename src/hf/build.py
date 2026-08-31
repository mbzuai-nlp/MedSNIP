#!/usr/bin/env python3
"""Stage !-hf — Hugging Face release snapshot for MedSNIP-Bench.

Reads  data/15-final/dataset.json     (the released benchmark, 2,524 snippets)
Writes data/!-hf/train.json           (1,599 snippets)
       data/!-hf/dev.json             (494)
       data/!-hf/test.json            (431)
       data/!-hf/stats.json

Records are split into one file per partition because the benchmark ships with
entry-level train/dev/test splits; the dataset card maps them onto the HF
split names train / validation / test. Record contents are unchanged from
data/15-final/dataset.json, so a published row is byte-comparable with the
repo's own artifact.

This directory is the ONLY published dataset artifact. It is uploaded to
huggingface.co/datasets/MBZUAI/MedSNIP by .github/workflows/hf-push.yml.
README.md and logo.png in that directory are maintained by hand.

Usage:
    python src/hf/build.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "data" / "15-final" / "dataset.json"
DST_DIR = REPO_ROOT / "data" / "!-hf"

# Repo split name -> published filename. The dataset card maps "dev" onto the
# HF split name "validation"; the file keeps the repo's own vocabulary.
SPLIT_FILES = {"train": "train.json", "dev": "dev.json", "test": "test.json"}

LABELS = ("label_atomic", "label_human_general", "label_human_contextual")


def compute_stats(by_split: dict[str, list[dict]]) -> dict:
    rows = [r for rs in by_split.values() for r in rs]
    entries = {r["entry_id"] for r in rows}

    def label_dist(rs: list[dict], field: str) -> dict:
        return dict(sorted(Counter(r[field] for r in rs).items(),
                           key=lambda kv: str(kv[0])))

    return {
        "total_snippets": len(rows),
        "total_entries": len(entries),
        "split_distribution": {s: len(rs) for s, rs in by_split.items()},
        "split_entries": {s: len({r["entry_id"] for r in rs})
                          for s, rs in by_split.items()},
        "subset_distribution": dict(Counter(r["subset"] for r in rows).most_common()),
        "pattern_distribution": dict(sorted(Counter(r["pattern"] for r in rows).items())),
        "label_distribution": {f: label_dist(rows, f) for f in LABELS},
        "atoms_per_snippet": {
            "total": sum(len(r["atoms"]) for r in rows),
            "single_atom": sum(1 for r in rows if len(r["atoms"]) == 1),
            "multi_atom": sum(1 for r in rows if len(r["atoms"]) > 1),
            "max": max(len(r["atoms"]) for r in rows),
        },
        "ambiguous_flagged": sum(1 for r in rows if r["is_ambiguous"] == "yes"),
    }


def main() -> None:
    rows = json.loads(SRC.read_text(encoding="utf-8"))

    by_split: dict[str, list[dict]] = {s: [] for s in SPLIT_FILES}
    for r in rows:
        split = r["split"]
        if split not in by_split:
            raise SystemExit(f"unexpected split {split!r} on {r['snippet_id']}")
        by_split[split].append(r)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    for split, name in SPLIT_FILES.items():
        path = DST_DIR / name
        path.write_text(json.dumps(by_split[split], ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"Wrote {path.relative_to(REPO_ROOT)} ({len(by_split[split]):,} snippets)")

    stats = compute_stats(by_split)
    (DST_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {(DST_DIR / 'stats.json').relative_to(REPO_ROOT)}")
    print(f"{stats['total_snippets']:,} snippets over "
          f"{stats['total_entries']} entries")


if __name__ == "__main__":
    main()
