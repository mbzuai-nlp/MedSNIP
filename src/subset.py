"""Tag every Kim claim with a `subset` key.

Subsets are based on query length:
  - consumer:  short patient/public questions  (< 80 words)
  - vignette:  long clinical-presentation questions  (>= 80 words)

From this step on the pipeline carries only the flat schema (one row
per claim). The nested per-query JSON stays in 1-raw as the original
source; everything downstream filters/aggregates the flat file.

Input:
  data/1-raw/kim_flat.json  (5755 rows, one per claim)

Outputs:
  data/2-subset/kim.json    (same rows + `subset` key)
  data/2-subset/stats.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "1-raw" / "kim_flat.json"
OUT_DIR = ROOT / "data" / "2-subset"

WORD_THRESHOLD = 80


def label(query: str) -> str:
    return "consumer" if len(query.split()) < WORD_THRESHOLD else "vignette"


def entry_id_of(row_id) -> int:
    return int(str(row_id).split("-")[0])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flat = json.loads(IN_PATH.read_text())
    for row in flat:
        row["subset"] = label(row["query"])

    (OUT_DIR / "kim.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False))

    # Per-entry view (one row per entry_id) for entry-level counts
    seen_entries: dict[int, str] = {}
    seen_queries: dict[str, set[str]] = {"consumer": set(), "vignette": set()}
    for r in flat:
        eid = entry_id_of(r["id"])
        if eid not in seen_entries:
            seen_entries[eid] = r["subset"]
            seen_queries[r["subset"]].add(r["query"])

    per_subset = {}
    for s in ("consumer", "vignette"):
        s_claims = [r for r in flat if r["subset"] == s]
        n_false = sum(1 for r in s_claims if not r["label"])
        per_subset[s] = {
            "entries": sum(1 for sub in seen_entries.values() if sub == s),
            "unique_queries": len(seen_queries[s]),
            "claims": len(s_claims),
            "false_claims": n_false,
            "false_rate": round(n_false / len(s_claims), 4) if s_claims else 0.0,
        }

    stats = {
        "word_threshold": WORD_THRESHOLD,
        "total_entries": len(seen_entries),
        "total_claims": len(flat),
        "consumer": per_subset["consumer"],
        "vignette": per_subset["vignette"],
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    for s, info in per_subset.items():
        print(f"{s:9s}  entries={info['entries']:4d}  "
              f"unique_q={info['unique_queries']:3d}  "
              f"claims={info['claims']:5d}  "
              f"false={info['false_claims']:4d} ({info['false_rate']:.1%})")
    print(f"wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
