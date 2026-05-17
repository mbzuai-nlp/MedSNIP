"""Tag every MedQA claim with a `split` key: train | dev | test.

Three-way split at the entry level, stratified jointly by `subset` and
the per-entry false-claim count bucketed as 0 / 1-2 / 3-5 / 6+:

  - train: the rest    — the bulk; what you actually iterate on.
  - dev:   ~50 entries — small validation set for prompt/system tuning.
  - test:  ~56 entries — held out for final evaluation.

Stratifying on both axes keeps the consumer/vignette mix and the
false-claim rate balanced across all three buckets. Strata alone can't
push the false-rate spread to zero (entries inside a bucket vary in
exact false count, and the smaller buckets see strong rounding), so
the seed was picked by a sweep over seeds 0-499 minimizing the largest
per-subset (train/dev/test) false-rate spread — landing on 274, which
keeps every cell within ~0.4 pp of the subset's mean. Sampling within
each stratum is otherwise random.

Input:
  data/2-subset/medqa.json   (flat, with `subset` key)

Outputs:
  data/3-split/medqa.json    (same rows + `split` key in {train,dev,test})
  data/3-split/stats.json
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "2-subset" / "medqa.json"
OUT_DIR = ROOT / "data" / "3-split"

DEV_N = 50
TEST_N = 56
SEED = 274


def entry_id_of(row_id) -> int:
    return int(str(row_id).split("-")[0])


def false_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = json.loads(IN_PATH.read_text())

    entry_subset: dict[int, str] = {}
    entry_false_count: dict[int, int] = defaultdict(int)
    for r in flat:
        eid = entry_id_of(r["id"])
        entry_subset[eid] = r["subset"]
        if not r["label"]:
            entry_false_count[eid] += 1

    total_entries = len(entry_subset)
    dev_frac = DEV_N / total_entries
    test_frac = TEST_N / total_entries

    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for eid, sub in entry_subset.items():
        strata[(sub, false_bucket(entry_false_count[eid]))].append(eid)

    rng = random.Random(SEED)
    dev_ids: set[int] = set()
    test_ids: set[int] = set()
    for key, group in sorted(strata.items()):
        g = list(group)
        rng.shuffle(g)
        n_dev = max(1, round(len(g) * dev_frac))
        n_test = max(1, round(len(g) * test_frac))
        for eid in g[:n_dev]:
            dev_ids.add(eid)
        for eid in g[n_dev:n_dev + n_test]:
            test_ids.add(eid)

    def split_of(eid: int) -> str:
        if eid in dev_ids:
            return "dev"
        if eid in test_ids:
            return "test"
        return "train"

    for r in flat:
        r["split"] = split_of(entry_id_of(r["id"]))

    (OUT_DIR / "medqa.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False))

    SPLITS = ("train", "dev", "test")
    SUBSETS = ("consumer", "vignette")

    by_cell = {
        (s, sp): {
            "entries": sum(
                1 for eid, sub in entry_subset.items()
                if sub == s and split_of(eid) == sp
            ),
            "claims": sum(
                1 for r in flat
                if r["subset"] == s and r["split"] == sp
            ),
            "false_claims": sum(
                1 for r in flat
                if r["subset"] == s and r["split"] == sp and not r["label"]
            ),
        }
        for s in SUBSETS for sp in SPLITS
    }
    strata_sizes = {
        f"{s}_{bucket}": len(group)
        for (s, bucket), group in sorted(strata.items())
    }
    totals = {
        f"{sp}_entries": sum(1 for eid in entry_subset if split_of(eid) == sp)
        for sp in SPLITS
    }
    totals.update({
        f"{sp}_claims": sum(1 for r in flat if r["split"] == sp) for sp in SPLITS
    })
    stats = {
        "strategy": "stratified random by (subset, false_bucket 0|1-2|3-5|6+), entry-level, seed-fixed",
        "target_sizes": {"dev": DEV_N, "test": TEST_N},
        "seed": SEED,
        "strata_sizes": strata_sizes,
        "totals": totals,
        "by_subset": {
            s: {sp: by_cell[(s, sp)] for sp in SPLITS}
            for s in SUBSETS
        },
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"{'':9s}  " + "  |  ".join(f"{sp:>5s}: ent / claims / false%" for sp in SPLITS))
    for s in SUBSETS:
        cells = [by_cell[(s, sp)] for sp in SPLITS]
        rates = [100 * c["false_claims"] / c["claims"] if c["claims"] else 0.0 for c in cells]
        parts = " | ".join(
            f"{c['entries']:3d} / {c['claims']:5d} / {r:5.1f}%"
            for c, r in zip(cells, rates)
        )
        print(f"{s:9s}  {parts}")
    print(f"wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
