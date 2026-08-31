"""Tag every snippet with a `split` key: train | dev | test.

Three-way entry-level split: 50 dev entries, ~45 test (~80/20 of the
remaining 226), train = the rest. Snippets of the same entry stay in
the same split — no leakage through shared query / full_text /
shared_context.

Stratification: each entry is bucketed on (subset,
human_general_false_count_bucket) with bucket = 0 / 1-2 / 3-5 / 6+.
The seed is chosen by sweep to minimize the worst (subset × split)
false-rate gap *jointly* across:
  - label_atomic           (per snippet)
  - label_human_general    (per snippet)
  - label_human_contextual (per snippet)

So all three rates stay close in every (subset, split) cell.

Input:
  data/3-annotated/medsnip-bench.json   (snippet rows with subset + human labels)

Outputs:
  data/4-split/medsnip-bench.json       (same rows + `split` key)
  data/4-split/stats.json
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "3-annotated" / "medsnip-bench.json"
OUT_DIR = ROOT / "data" / "4-split"

DEV_N = 50
TEST_FRAC_OF_REST = 0.20
SEED_SWEEP_RANGE = range(500)

SPLITS = ("train", "dev", "test")
SUBSETS = ("consumer", "vignette")
RATES = ("label_atomic", "label_human_general", "label_human_contextual")


def false_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


def aggregate_per_entry(rows):
    """entry_id -> dict with subset, n_snippets, hg_false count, snippet rows."""
    per: dict[int, dict] = defaultdict(lambda: {
        "subset": None, "snippets": [], "hg_false": 0,
    })
    for r in rows:
        e = per[r["entry_id"]]
        e["subset"] = r["subset"]
        e["snippets"].append(r)
        if not r["label_human_general"]:
            e["hg_false"] += 1
    return per


def assign_splits(per_entry, seed: int, target_dev: int, target_test: int):
    """Stratified random split, returning (dev_ids, test_ids)."""
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for eid, e in per_entry.items():
        strata[(e["subset"], false_bucket(e["hg_false"]))].append(eid)

    total = len(per_entry)
    dev_frac = target_dev / total
    test_frac = target_test / total

    rng = random.Random(seed)
    dev_ids, test_ids = set(), set()
    for key, group in sorted(strata.items()):
        g = list(group)
        rng.shuffle(g)
        n_dev = max(1, round(len(g) * dev_frac))
        n_test = max(1, round(len(g) * test_frac))
        for x in g[:n_dev]:
            dev_ids.add(x)
        for x in g[n_dev:n_dev + n_test]:
            test_ids.add(x)
    return dev_ids, test_ids


def score(per_entry, dev_ids: set[int], test_ids: set[int]) -> dict:
    """Compute per-(subset, split) false rates for each of the three labels.

    Returns:
      {
        "rates": {(subset, split, rate_name): float, ...},
        "worst_spread": float (max spread across (train, dev, test) for any
                              (subset, rate_name) combination),
      }
    """
    bins: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "false": {r: 0 for r in RATES}})
    for eid, e in per_entry.items():
        sp = "dev" if eid in dev_ids else "test" if eid in test_ids else "train"
        for r in e["snippets"]:
            cell = bins[(e["subset"], sp)]
            cell["n"] += 1
            if not r["label_atomic"]:            cell["false"]["label_atomic"] += 1
            if not r["label_human_general"]:     cell["false"]["label_human_general"] += 1
            if not r["label_human_contextual"]:  cell["false"]["label_human_contextual"] += 1

    rates = {}
    for (sub, sp), cell in bins.items():
        for r_name in RATES:
            rates[(sub, sp, r_name)] = (cell["false"][r_name] / cell["n"]) if cell["n"] else 0.0

    worst = 0.0
    for sub in SUBSETS:
        for r_name in RATES:
            vals = [rates[(sub, sp, r_name)] for sp in SPLITS]
            worst = max(worst, max(vals) - min(vals))
    return {"rates": rates, "worst_spread": worst}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(IN_PATH.read_text())
    per_entry = aggregate_per_entry(rows)

    total = len(per_entry)
    rest = total - DEV_N
    target_test = round(rest * TEST_FRAC_OF_REST)

    # Sweep seeds; pick the one minimizing worst-spread
    best_seed = None
    best_score = None
    for seed in SEED_SWEEP_RANGE:
        dev_ids, test_ids = assign_splits(per_entry, seed, DEV_N, target_test)
        s = score(per_entry, dev_ids, test_ids)
        if best_score is None or s["worst_spread"] < best_score["worst_spread"]:
            best_seed = seed
            best_score = s
            best_dev, best_test = dev_ids, test_ids

    print(f"swept {len(SEED_SWEEP_RANGE)} seeds")
    print(f"best seed: {best_seed}  worst false-rate spread: {best_score['worst_spread']:.4f}")

    # Apply best
    for r in rows:
        eid = r["entry_id"]
        r["split"] = "dev" if eid in best_dev else "test" if eid in best_test else "train"

    rows.sort(key=lambda r: (r["entry_id"], int(r["snippet_id"].split("-S")[1])))
    (OUT_DIR / "medsnip-bench.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    # Stats
    by_cell = {(sub, sp): {"entries": set(), "snippets": 0,
                            "atomic_false": 0,
                            "hg_false": 0, "hc_false": 0}
               for sub in SUBSETS for sp in SPLITS}
    for r in rows:
        sub, sp = r["subset"], r["split"]
        c = by_cell[(sub, sp)]
        c["entries"].add(r["entry_id"])
        c["snippets"] += 1
        if not r["label_atomic"]:            c["atomic_false"] += 1
        if not r["label_human_general"]:     c["hg_false"] += 1
        if not r["label_human_contextual"]:  c["hc_false"] += 1

    summary = {
        sub: {
            sp: {
                "entries": len(by_cell[(sub, sp)]["entries"]),
                "snippets": by_cell[(sub, sp)]["snippets"],
                "atomic_false_rate":         round(by_cell[(sub, sp)]["atomic_false"] / by_cell[(sub, sp)]["snippets"], 4) if by_cell[(sub, sp)]["snippets"] else 0,
                "human_general_false_rate":  round(by_cell[(sub, sp)]["hg_false"] / by_cell[(sub, sp)]["snippets"], 4) if by_cell[(sub, sp)]["snippets"] else 0,
                "human_contextual_false_rate": round(by_cell[(sub, sp)]["hc_false"] / by_cell[(sub, sp)]["snippets"], 4) if by_cell[(sub, sp)]["snippets"] else 0,
            }
            for sp in SPLITS
        }
        for sub in SUBSETS
    }
    totals = {sp: {
        "entries":  sum(summary[sub][sp]["entries"]  for sub in SUBSETS),
        "snippets": sum(summary[sub][sp]["snippets"] for sub in SUBSETS),
    } for sp in SPLITS}
    stats = {
        "strategy": "stratified entry-level on (subset, human_general_false_bucket); seed picked by sweep to minimize worst (subset × split) false-rate spread across 3 labels",
        "target_sizes": {"dev": DEV_N, "test_frac_of_rest": TEST_FRAC_OF_REST},
        "seed": best_seed,
        "worst_false_rate_spread": round(best_score["worst_spread"], 4),
        "totals": totals,
        "by_subset": summary,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    # Print
    print()
    print(f"{'subset':9s} {'split':6s} {'ents':>4s} {'snips':>5s}  {'atomic':>7s}  {'human_gen':>10s}  {'human_ctx':>10s}")
    for sub in SUBSETS:
        for sp in SPLITS:
            info = summary[sub][sp]
            print(f"{sub:9s} {sp:6s} {info['entries']:>4d} {info['snippets']:>5d}  "
                  f"{info['atomic_false_rate']*100:6.2f}%  "
                  f"{info['human_general_false_rate']*100:9.2f}%  "
                  f"{info['human_contextual_false_rate']*100:9.2f}%")
    print()
    print(f"totals: train {totals['train']['entries']} ent / {totals['train']['snippets']} snips  "
          f"|  dev {totals['dev']['entries']} ent / {totals['dev']['snippets']} snips  "
          f"|  test {totals['test']['entries']} ent / {totals['test']['snippets']} snips")
    print(f"wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
