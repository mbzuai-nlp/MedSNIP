"""Tag every MedQA row with a `split` key.

80/20 dev/test split at the entry level, stratified by `subset` so the
consumer/vignette mix is the same in dev and test. Seed-fixed for
reproducibility.

Inputs (2-subset):
  data/2-subset/medqa.json       (each entry has `subset`)
  data/2-subset/medqa_flat.json  (each row has `subset`)

Outputs (3-split):
  data/3-split/medqa.json       (same shape + `split` on each entry)
  data/3-split/medqa_flat.json  (same shape + `split` on each row)
  data/3-split/stats.json
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "2-subset"
OUT_DIR = ROOT / "data" / "3-split"

TEST_FRAC = 0.20
SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = json.loads((IN_DIR / "medqa.json").read_text())
    flat = json.loads((IN_DIR / "medqa_flat.json").read_text())

    by_subset = defaultdict(list)
    for e in entries:
        by_subset[e["subset"]].append(e)

    rng = random.Random(SEED)
    test_ids = set()
    for subset_name, group in sorted(by_subset.items()):
        n_test = max(1, round(len(group) * TEST_FRAC))
        for e in rng.sample(group, n_test):
            test_ids.add(e["id"])

    for e in entries:
        e["split"] = "test" if e["id"] in test_ids else "dev"
    for row in flat:
        entry_id = int(str(row["id"]).split("-")[0])
        row["split"] = "test" if entry_id in test_ids else "dev"

    (OUT_DIR / "medqa.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    (OUT_DIR / "medqa_flat.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False))

    by_cell = {
        (s, sp): {
            "entries": sum(1 for e in entries if e["subset"] == s and e["split"] == sp),
            "claims":  sum(1 for r in flat    if r["subset"] == s and r["split"] == sp),
            "false_claims": sum(
                1 for r in flat
                if r["subset"] == s and r["split"] == sp and not r["label"]
            ),
        }
        for s in ("consumer", "vignette") for sp in ("dev", "test")
    }

    stats = {
        "strategy": "stratified random by subset, entry-level, seed-fixed",
        "test_frac": TEST_FRAC,
        "seed": SEED,
        "totals": {
            "dev_entries":  sum(1 for e in entries if e["split"] == "dev"),
            "test_entries": sum(1 for e in entries if e["split"] == "test"),
            "dev_claims":   sum(1 for r in flat    if r["split"] == "dev"),
            "test_claims":  sum(1 for r in flat    if r["split"] == "test"),
        },
        "by_subset": {
            s: {
                "dev":  by_cell[(s, "dev")],
                "test": by_cell[(s, "test")],
            }
            for s in ("consumer", "vignette")
        },
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    for s in ("consumer", "vignette"):
        d, t = by_cell[(s, "dev")], by_cell[(s, "test")]
        print(f"{s:9s}  dev: {d['entries']:3d} entries / {d['claims']:5d} claims  "
              f"|  test: {t['entries']:3d} entries / {t['claims']:5d} claims")
    print(f"wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
