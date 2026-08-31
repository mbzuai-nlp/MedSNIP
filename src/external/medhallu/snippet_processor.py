"""Run our MedSNIP SnippetProcessor on MedHallu answers.

Each MedHallu row gives us TWO claims:
  - `ground_truth`        → gold_bool = True  (correct answer)
  - `hallucinated_answer` → gold_bool = False (controlled hallucination)

So we get 2000 binary claims from the 1000-row `pqa_labeled` split.

Uses `mode="atom-to-snippet"` — same pipeline as MedSNIP preprocessing.

Output: data/10-medhallu/snippet-processor/atom-to-snippet/medhallu.json

Usage:
    python -m src.external.medhallu.snippet_processor --workers 12 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from src.medsnip.snippet_processor.processor import SnippetProcessor

ROOT = Path(__file__).resolve().parents[3]
IN_PATH = ROOT / "data" / "1-raw" / "medhallu.json"
OUT_PATH = (ROOT / "data" / "10-medhallu" / "snippet-processor"
            / "atom-to-snippet" / "medhallu.json")


def flatten(rows: list[dict]) -> list[dict]:
    """Each MedHallu row → 2 items (ground_truth=True, hallucinated=False)."""
    out = []
    for r in rows:
        eid = r["id"]
        out.append({
            "id":         f"mh-{eid}-T",
            "medhallu_id": eid,
            "kind":       "ground_truth",
            "gold_bool":  True,
            "answer":     r["ground_truth"],
            "question":   r["question"],
            "difficulty": r["difficulty"],
        })
        out.append({
            "id":         f"mh-{eid}-H",
            "medhallu_id": eid,
            "kind":       "hallucinated",
            "gold_bool":  False,
            "answer":     r["hallucinated_answer"],
            "question":   r["question"],
            "difficulty": r["difficulty"],
            "hallucination_category": r.get("hallucination_category"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N items (smoke test)")
    ap.add_argument("--subset", default="vignette",
                    help="few-shot subset for the processor (default vignette — "
                         "PubMedQA-style scientific reasoning)")
    args = ap.parse_args()

    raw_rows = json.loads(IN_PATH.read_text())
    items = flatten(raw_rows)
    if args.limit:
        items = items[: args.limit]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if args.skip_existing and OUT_PATH.exists():
        for rec in json.loads(OUT_PATH.read_text()):
            existing[rec["id"]] = rec
        print(f"resuming: {len(existing)} already atomized")

    proc_kwargs = {"mode": "atom-to-snippet"}
    if args.model:
        proc_kwargs["model"] = args.model
    proc = SnippetProcessor(**proc_kwargs)

    lock = Lock()
    out_records: list[dict] = list(existing.values())
    totals = {"in": 0, "out": 0, "n": 0, "atoms": 0, "snippets": 0, "errors": 0}

    def worker(item: dict):
        sid = item["id"]
        if sid in existing:
            return
        t0 = time.time()
        try:
            result = proc(
                full_text=item["answer"],
                query=item["question"],
                subset=args.subset,
            )
        except Exception as e:
            with lock:
                totals["errors"] += 1
            print(f"  ERR {sid}: {type(e).__name__}: {e}")
            return
        atoms = result.get("generated_atoms") or []
        if not atoms:
            atoms = [{"index": 1, "text": item["answer"], "source_sentences": [0]}]
        snippets = result.get("snippets") or []
        if not snippets:
            snippets = [{"output": item["answer"],
                         "atom_indices": [a.get("index") for a in atoms]}]
        rec = {
            "id":              sid,
            "medhallu_id":     item["medhallu_id"],
            "kind":            item["kind"],
            "gold_bool":       item["gold_bool"],
            "question":        item["question"],
            "answer":          item["answer"],
            "difficulty":      item["difficulty"],
            "subset":          args.subset,
            "shared_context":  result.get("shared_context", {}),
            "sentences":       result.get("sentences", []),
            "generated_atoms": atoms,
            "snippets":        snippets,
            "model":           result.get("_model"),
            "usage":           result.get("_usage", {}),
            "usage_steps":     result.get("_usage_steps"),
            "wall_seconds":    round(time.time() - t0, 2),
        }
        if "hallucination_category" in item:
            rec["hallucination_category"] = item["hallucination_category"]
        with lock:
            out_records.append(rec)
            out_records.sort(key=lambda x: (x["medhallu_id"], x["kind"]))
            tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
            tmp.write_text(json.dumps(out_records, indent=2, ensure_ascii=False))
            tmp.replace(OUT_PATH)
            totals["n"] += 1
            u = result.get("_usage", {})
            totals["in"] += u.get("input_tokens", 0)
            totals["out"] += u.get("output_tokens", 0)
            totals["atoms"] += len(atoms)
            totals["snippets"] += len(snippets)
            if totals["n"] % 100 == 0 or totals["n"] == len(items):
                print(f"  {totals['n']}/{len(items)}  "
                      f"atoms={totals['atoms']} (avg {totals['atoms']/totals['n']:.2f}) "
                      f"snippets={totals['snippets']} (avg {totals['snippets']/totals['n']:.2f})")

    t_start = time.time()
    print(f"atomizing {len(items) - len(existing)} MedHallu answers "
          f"(subset={args.subset}, workers={args.workers})")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, it) for it in items]
        for _ in as_completed(futs):
            pass
    t_total = time.time() - t_start

    print(f"\n=== summary (wall: {t_total:.1f}s) ===")
    print(f"  atomized: {totals['n']}  errors: {totals['errors']}")
    print(f"  total atoms:    {totals['atoms']}  "
          f"(avg {totals['atoms']/max(totals['n'],1):.2f} per answer)")
    print(f"  total snippets: {totals['snippets']}  "
          f"(avg {totals['snippets']/max(totals['n'],1):.2f} per answer)")
    print(f"  tokens: in={totals['in']:,} out={totals['out']:,}")
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
