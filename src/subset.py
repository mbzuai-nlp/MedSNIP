"""Tag every MedQA row with a `subset` key.

Subsets are based on query length:
  - consumer:  short patient/public questions  (< 80 words)
  - vignette:  long clinical-presentation questions  (>= 80 words)

Inputs (1-raw):
  data/1-raw/medqa.json       (276 entries, one per query)
  data/1-raw/medqa_flat.json  (5755 rows, one per claim)

Outputs (2-subset):
  data/2-subset/medqa.json       (same shape + `subset` on each entry)
  data/2-subset/medqa_flat.json  (same shape + `subset` on each row)
  data/2-subset/stats.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "1-raw"
OUT_DIR = ROOT / "data" / "2-subset"

WORD_THRESHOLD = 80


def label(query: str) -> str:
    return "consumer" if len(query.split()) < WORD_THRESHOLD else "vignette"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = json.loads((RAW_DIR / "medqa.json").read_text())
    flat = json.loads((RAW_DIR / "medqa_flat.json").read_text())

    # entries: one per query
    by_id = {}
    for e in entries:
        e["subset"] = label(e["query"])
        by_id[e["id"]] = e["subset"]

    # flat: one per claim; id is "<entry_id>-<claim_idx>"
    for row in flat:
        entry_id = int(str(row["id"]).split("-")[0])
        row["subset"] = by_id[entry_id]

    (OUT_DIR / "medqa.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    (OUT_DIR / "medqa_flat.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False))

    per_subset = {}
    for s in ("consumer", "vignette"):
        s_entries = [e for e in entries if e["subset"] == s]
        s_claims = [r for r in flat if r["subset"] == s]
        n_false = sum(1 for r in s_claims if not r["label"])
        per_subset[s] = {
            "entries": len(s_entries),
            "unique_queries": len({e["query"] for e in s_entries}),
            "claims": len(s_claims),
            "false_claims": n_false,
            "false_rate": round(n_false / len(s_claims), 4) if s_claims else 0.0,
        }

    stats = {
        "word_threshold": WORD_THRESHOLD,
        "total_entries": len(entries),
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
