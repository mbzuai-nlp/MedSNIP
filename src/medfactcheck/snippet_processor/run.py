"""Run the snippet processor over the annotated dataset.

Reads data/4-split/medqa.json. Aggregates per entry, runs the
SnippetProcessor on each, saves data/6-snippet-processor/<mode>/e<id>.json.
Each mode gets its own subdirectory so both can coexist.

Usage:
  python -m src.medfactcheck.snippet_processor.run --mode atom-to-snippet
  python -m src.medfactcheck.snippet_processor.run --mode snippet-direct
  python -m src.medfactcheck.snippet_processor.run --mode atom-to-snippet --split dev
  python -m src.medfactcheck.snippet_processor.run --entry-ids 1 61
  python -m src.medfactcheck.snippet_processor.run --skip-existing
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from .processor import DEFAULT_MODE, MODES, SnippetProcessor

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "data" / "4-split" / "medqa.json"
OUT_ROOT = ROOT / "data" / "6-snippet-processor"


def aggregate_entries(rows: list[dict]) -> list[dict]:
    """Snippet-flat rows → one record per entry."""
    by_id: dict[int, dict] = {}
    for r in rows:
        eid = r["entry_id"]
        if eid not in by_id:
            by_id[eid] = {
                "entry_id":       eid,
                "batch_id":       r["batch_id"],
                "subset":         r["subset"],
                "split":          r["split"],
                "full_query":     r["query"],
                "full_text":      r["full_text"],
                "shared_context": r.get("shared_context") or {},
            }
    return [by_id[k] for k in sorted(by_id)]


def run_one(proc: SnippetProcessor, entry: dict) -> dict:
    result = proc(query=entry["full_query"], full_text=entry["full_text"],
                  subset=entry["subset"])
    return {
        "entry_id":        entry["entry_id"],
        "batch_id":        entry["batch_id"],
        "subset":          entry["subset"],
        "split":           entry["split"],
        "mode":            result["mode"],
        "detected_subset": result["subset"],
        "full_query":      entry["full_query"],
        "model":           result["_model"],
        "usage":           result["_usage"],
        "usage_steps":     result.get("_usage_steps"),
        "shared_context":  result["shared_context"],
        "sentences":       result.get("sentences"),
        "generated_atoms": result.get("generated_atoms"),
        "snippets":        result["snippets"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE,
                    help=f"processor mode (default: {DEFAULT_MODE})")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--entry-ids", type=int, nargs="*", default=None)
    ap.add_argument("--split", choices=["train", "dev", "test"], default=None,
                    help="run only on entries in this split")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--model", default=None, help="override default model")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent worker threads (default 8)")
    args = ap.parse_args()

    rows = json.loads(SPLIT_PATH.read_text())
    entries = aggregate_entries(rows)
    if args.entry_ids:
        wanted = set(args.entry_ids)
        entries = [e for e in entries if e["entry_id"] in wanted]
    if args.split:
        entries = [e for e in entries if e["split"] == args.split]
    if args.limit:
        entries = entries[: args.limit]

    kwargs = {"mode": args.mode}
    if args.model:
        kwargs["model"] = args.model
    proc = SnippetProcessor(**kwargs)

    out_dir = OUT_ROOT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for entry in entries:
        out_path = out_dir / f"e{entry['entry_id']}.json"
        if args.skip_existing and out_path.exists():
            continue
        tasks.append((entry, out_path))

    print(f"running {len(tasks)} entries across {args.workers} workers "
          f"(mode={proc.mode}, model={proc.model})")
    totals = {"input": 0, "output": 0, "cached": 0, "errors": [], "n": 0}
    totals_lock = Lock()
    print_lock = Lock()

    def worker(task):
        entry, out_path = task
        eid = entry["entry_id"]
        t0 = time.time()
        try:
            result = run_one(proc, entry)
        except Exception as exc:
            with print_lock:
                print(f"  e{eid:3d}: ERROR {exc}")
            with totals_lock:
                totals["errors"].append({"entry_id": eid, "error": str(exc)})
            return
        dt = time.time() - t0
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        u = result["usage"]
        with totals_lock:
            totals["input"]  += u["input_tokens"]
            totals["output"] += u["output_tokens"]
            totals["cached"] += u.get("cached_input_tokens", 0)
            totals["n"]      += 1
            n_so_far = totals["n"]
        with print_lock:
            print(f"  [{n_so_far}/{len(tasks)}] e{eid:3d}: "
                  f"{len(result['snippets'])} snippets "
                  f"sents={len(result.get('sentences') or [])} "
                  f"atoms={len(result.get('generated_atoms') or [])}  "
                  f"in={u['input_tokens']} (cached={u.get('cached_input_tokens', 0)}) "
                  f"out={u['output_tokens']} {dt:.1f}s")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for _ in as_completed(futures):
            pass
    t_total = time.time() - t_start

    print(f"\n=== totals (wall: {t_total:.1f}s) ===")
    print(f"  entries={totals['n']}  in={totals['input']:,} "
          f"(cached={totals['cached']:,})  out={totals['output']:,}  "
          f"errors={len(totals['errors'])}")
    if totals["errors"]:
        for e in totals["errors"]:
            print(f"    e{e['entry_id']}: {e['error']}")


if __name__ == "__main__":
    main()
