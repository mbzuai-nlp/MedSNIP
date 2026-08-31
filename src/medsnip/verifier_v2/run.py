"""Run the iterative verifier over a split.

Source of snippets is selectable:

  --source human            : human gold snippets from data/4-split/medsnip-bench.json
                              (the oracle path — what humans curated)
  --source atom-to-snippet  : auto snippets from
                              data/6-snippet-processor/atom-to-snippet/
                              (the deployable path; 2-step processor)
  --source snippet-direct   : auto snippets from
                              data/6-snippet-processor/snippet-direct/
                              (the deployable path; 1-pass processor)

Writes per-snippet verdicts to
data/8-verifier/verdicts_<split>_<source>.jsonl. Resumable.

Usage:
  python -m src.medsnip.verifier.run --source human --split dev --workers 16
  python -m src.medsnip.verifier.run --source atom-to-snippet --split dev
  python -m src.medsnip.verifier.run --split dev --limit 5           # smoke
  python -m src.medsnip.verifier.run --entry-ids 1 62 65             # specific
  python -m src.medsnip.verifier.run --snippet-ids 1-S2 62-S5
  python -m src.medsnip.verifier.run --disable-abstain
  python -m src.medsnip.verifier.run --max-iters 5
  python -m src.medsnip.verifier.run --skip-existing
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from .verifier import Verifier

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "data" / "4-split" / "medsnip-bench.json"
SNIPPET_PROC_ROOT = ROOT / "data" / "6-snippet-processor"
OUT_DIR = ROOT / "data" / "8-verifier"

SOURCES = ("human", "atom-to-snippet", "snippet-direct")


def _strip_correct_answer(ctx) -> dict:
    """Strip vignette correct_answer keys (gold leak)."""
    if isinstance(ctx, dict):
        out = dict(ctx)
    else:
        out = {"summary": str(ctx)}
    out.pop("correct_answer", None)
    return out


def load_tasks_human(split: str | None) -> list[dict]:
    """One task per human snippet from data/4-split/medsnip-bench.json."""
    rows = json.loads(SPLIT_PATH.read_text())
    out = []
    for r in rows:
        if split and r["split"] != split:
            continue
        out.append({
            "id":             r["snippet_id"],
            "entry_id":       r["entry_id"],
            "subset":         r["subset"],
            "split":          r["split"],
            "shared_context": _strip_correct_answer(r.get("shared_context")),
            "full_text":      r["full_text"],
            "query":          r["query"],
            "snippet":        r["snippet_text"],
        })
    out.sort(key=lambda t: (t["entry_id"], int(t["id"].split("-S")[1])))
    return out


def load_tasks_auto(source: str, split: str | None) -> list[dict]:
    """One task per auto snippet from data/6-snippet-processor/<source>/e*.json."""
    src_dir = SNIPPET_PROC_ROOT / source
    if not src_dir.exists():
        raise SystemExit(
            f"missing {src_dir}; run "
            f"`python -m src.medsnip.snippet_processor.run --mode {source}` first"
        )
    # Entry-level split lookup
    split_rows = json.loads(SPLIT_PATH.read_text())
    entry_split = {r["entry_id"]: r["split"] for r in split_rows}

    out = []
    for f in sorted(src_dir.glob("e*.json")):
        d = json.loads(f.read_text())
        eid = d["entry_id"]
        e_split = entry_split.get(eid)
        if split and e_split != split:
            continue
        sc = _strip_correct_answer(d.get("shared_context"))
        for i, s in enumerate(d["snippets"], 1):
            out.append({
                "id":             f"{eid}-S{i}",
                "entry_id":       eid,
                "subset":         d["subset"],
                "split":          e_split,
                "shared_context": sc,
                "full_text":      d.get("full_query", ""),  # not strictly the full text; we
                                                            # don't carry it on auto outputs
                "query":          d.get("full_query", ""),
                "snippet":        s["output"],
            })
    out.sort(key=lambda t: (t["entry_id"], int(t["id"].split("-S")[1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="human",
                    help="which snippet source to verify (default: human)")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--entry-ids", type=int, nargs="*", default=None)
    ap.add_argument("--snippet-ids", type=str, nargs="*", default=None)
    ap.add_argument("--disable-abstain", action="store_true",
                    help="force a final verdict even on otherwise-abstainable snippets")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--retrieval-k", type=int, default=3)
    ap.add_argument("--use-full-text", action="store_true",
                    help="pass the original full LLM response to the verifier")
    ap.add_argument("--use-query", action="store_true",
                    help="pass the literal user query to the verifier")
    ap.add_argument("--no-shared-context", action="store_true",
                    help="drop the curated shared_context dict from the prompt")
    ap.add_argument("--version", default=None,
                    help="optional subdir under data/8-verifier/ (e.g. 'v1') for the run")
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                    default=None,
                    help="reasoning_effort for reasoning-capable models (e.g. gpt-5.4)")
    ap.add_argument("--baseline-priors", default=None,
                    help="path to a baseline predictions.json (snippet-level) "
                         "to feed as a per-snippet prior into the verifier prompt")
    args = ap.parse_args()

    if args.source == "human":
        tasks = load_tasks_human(args.split)
    else:
        tasks = load_tasks_auto(args.source, args.split)

    priors: dict[str, bool] = {}
    if args.baseline_priors:
        for r in json.loads(Path(args.baseline_priors).read_text()):
            sid = r.get("snippet_id") or r.get("id")
            if sid is not None and "prediction" in r:
                priors[sid] = bool(r["prediction"])
        print(f"loaded {len(priors)} baseline priors from {args.baseline_priors}")

    if args.entry_ids:
        wanted = set(args.entry_ids)
        tasks = [t for t in tasks if t["entry_id"] in wanted]
    if args.snippet_ids:
        wanted = set()
        for item in args.snippet_ids:
            wanted.update(item.split())
        tasks = [t for t in tasks if t["id"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]

    out_dir = OUT_DIR / args.version if args.version else OUT_DIR
    out_path = out_dir / f"verdicts_{args.split}_{args.source}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if args.skip_existing and out_path.exists():
        records = json.loads(out_path.read_text())
        for r in records:
            existing[r["id"]] = r
        print(f"resuming: {len(existing)} verdicts already on disk")

    todo = [t for t in tasks if t["id"] not in existing]
    print(f"verifying {len(todo)} snippets "
          f"({args.source} / {args.split}, {args.workers} workers, "
          f"max_iters={args.max_iters}, disable_abstain={args.disable_abstain})")
    if not todo:
        print("nothing to do")
        return

    kw = {"model": args.model} if args.model else {}
    verifier = Verifier(
        max_iters=args.max_iters,
        retrieval_k=args.retrieval_k,
        disable_abstain=args.disable_abstain,
        use_full_text=args.use_full_text,
        use_query=args.use_query,
        use_shared_context=not args.no_shared_context,
        reasoning_effort=args.reasoning_effort,
        **kw,
    )

    out_lock = Lock()
    all_records: list[dict] = list(existing.values())
    counts = {
        "done": 0, "errors": 0, "abstained": 0, "cap_hit": 0,
        "true": 0, "false": 0,
        "src_web": 0, "src_pubmed": 0,
        "tok_in": 0, "tok_out": 0, "cached": 0,
        "iter_dist": {},
    }

    def worker(task):
        t0 = time.time()
        try:
            res = verifier(
                snippet=task["snippet"],
                subset=task["subset"],
                shared_context=task["shared_context"],
                full_text=task.get("full_text"),
                query=task.get("query"),
                prior=priors.get(task["id"]),
                cache_key=f"medsnip-entry-{task['entry_id']}",
            )
        except Exception as exc:
            with out_lock:
                counts["errors"] += 1
            print(f"  ERR {task['id']}: {exc}")
            return
        dt = time.time() - t0
        rec = {
            "id":           task["id"],
            "entry_id":     task["entry_id"],
            "subset":       task["subset"],
            "split":        task["split"],
            "source":       args.source,
            "snippet":      task["snippet"],
            "prediction":   res.prediction,
            "abstained":    res.abstained,
            "confidence":   res.confidence,
            "reasoning":    res.reasoning,
            "iterations":   res.iterations,
            "cap_hit":      res.cap_hit,
            "sources_used": res.sources_used,
            "evidence":     res.evidence,
            "model":        res.model,
            "usage":        res.usage,
            "error":        res.error,
            "wall_seconds": round(dt, 2),
        }
        with out_lock:
            all_records.append(rec)
            all_records.sort(key=lambda r: (r["entry_id"], int(r["id"].split("-S")[1])))
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(all_records, indent=2, ensure_ascii=False)
            )
            tmp_path.replace(out_path)
            counts["done"] += 1
            if res.abstained:
                counts["abstained"] += 1
            elif res.prediction is True:
                counts["true"] += 1
            elif res.prediction is False:
                counts["false"] += 1
            if res.cap_hit:
                counts["cap_hit"] += 1
            for src in res.sources_used:
                counts[f"src_{src}"] = counts.get(f"src_{src}", 0) + 1
            counts["iter_dist"][res.iterations] = counts["iter_dist"].get(res.iterations, 0) + 1
            counts["tok_in"]  += res.usage.get("input_tokens", 0)
            counts["tok_out"] += res.usage.get("output_tokens", 0)
            counts["cached"]  += res.usage.get("cached_input_tokens", 0)
            if counts["done"] % 25 == 0 or counts["done"] == len(todo):
                print(f"  {counts['done']}/{len(todo)}  "
                      f"true={counts['true']} false={counts['false']} "
                      f"abstain={counts['abstained']} cap_hit={counts['cap_hit']} "
                      f"errors={counts['errors']}")

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, t) for t in todo]
        for _ in as_completed(futs):
            pass
    t_total = time.time() - t_start

    print(f"\n=== summary (wall: {t_total:.1f}s) ===")
    print(f"  verified: {counts['done']}  errors: {counts['errors']}  "
          f"abstained: {counts['abstained']}  cap_hit: {counts['cap_hit']}")
    print(f"  predictions: true={counts['true']} false={counts['false']}")
    print(f"  search sources used: web={counts.get('src_web', 0)} "
          f"pubmed={counts.get('src_pubmed', 0)}")
    print(f"  iteration distribution (incl. forced-final):")
    for k in sorted(counts["iter_dist"]):
        print(f"    {k}: {counts['iter_dist'][k]}")
    print(f"  tokens — in={counts['tok_in']:,} (cached={counts['cached']:,}) "
          f"out={counts['tok_out']:,}")


if __name__ == "__main__":
    main()
