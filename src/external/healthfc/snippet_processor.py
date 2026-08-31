"""Run our MedSNIP SnippetProcessor on HealthFC claims.

Uses `mode="atom-to-snippet"` — the same pipeline MedSNIP's
preprocessing uses (data/6-snippet-processor/atom-to-snippet/). We keep
both `generated_atoms` and `snippets` outputs so the downstream baselines
can run at either granularity.

Output: data/9-healthfc/snippet-processor/atom-to-snippet/healthfc.json
(structure mirrors data/6-snippet-processor/atom-to-snippet/e*.json):
  {
    "id": "hfc-12", "en_claim": "...", "label": 0|1|2, "subset": "consumer",
    "shared_context": {...},
    "sentences":       [...],
    "generated_atoms": [{"text": "...", "index": 1, ...}, ...],
    "snippets":        [{"output": "...", "atom_indices": [...]}, ...],
    "model": "gpt-5.4",
    "usage": {...},
  }

Usage:
    python -m src.external.healthfc.snippet_processor --workers 12 --skip-existing
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
IN_PATH = ROOT / "data" / "1-raw" / "healthfc.json"
OUT_ROOT = (ROOT / "data" / "9-healthfc" / "snippet-processor"
            / "atom-to-snippet" / "healthfc.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="override SnippetProcessor default")
    ap.add_argument("--claim-source", default="raw", choices=["raw", "declarative"],
                    help="raw = en_claim as-is (99%% are questions, which the "
                         "extract prompt mishandles); declarative = the "
                         "statement form from declarativize.py")
    ap.add_argument("--mode", default="atom-to-snippet",
                    choices=["atom-to-snippet", "snippet-direct"],
                    help="snippet-direct writes to its own directory, so the "
                         "published Mode 1 outputs are never overwritten")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reasoning", default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="reasoning effort; the MedSNIP-Bench runs use high")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only atomize the first N claims (smoke test)")
    ap.add_argument("--subset", default="consumer",
                    help="few-shot subset for the processor (consumer | vignette). "
                         "HealthFC is consumer-health so default consumer.")
    args = ap.parse_args()

    global OUT_PATH
    OUT_PATH = Path(str(OUT_ROOT).replace("atom-to-snippet", args.mode))
    if args.claim_source == "declarative":
        # Separate tree so the published raw-claim decomposition is preserved.
        OUT_PATH = OUT_PATH.parent / "declarative" / OUT_PATH.name

    rows = json.loads(IN_PATH.read_text())
    if args.claim_source == "declarative":
        decl_path = ROOT / "data" / "9-healthfc" / "declarative.json"
        if not decl_path.exists():
            raise SystemExit(f"missing {decl_path}; run "
                             f"`python -m src.external.healthfc.declarativize` first")
        decl = {r["id"]: r["statement"] for r in json.loads(decl_path.read_text())}
        missing = 0
        for i, r in enumerate(rows):
            st = decl.get(f"hfc-{i}")
            if st:
                r["en_claim"] = st
            else:
                missing += 1
        print(f"claim source: declarative ({len(rows) - missing}/{len(rows)} substituted)")
    if args.limit:
        rows = rows[: args.limit]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if args.skip_existing and OUT_PATH.exists():
        for rec in json.loads(OUT_PATH.read_text()):
            existing[rec["id"]] = rec
        print(f"resuming: {len(existing)} already atomized")

    proc_kwargs = {"mode": args.mode}
    if args.model:
        proc_kwargs["model"] = args.model
    if args.reasoning:
        proc_kwargs["reasoning_effort"] = args.reasoning
    proc = SnippetProcessor(**proc_kwargs)

    lock = Lock()
    out_records: list[dict] = list(existing.values())
    totals = {"in": 0, "out": 0, "n": 0, "atoms": 0, "snippets": 0, "errors": 0}

    def worker(i: int, r: dict):
        sid = f"hfc-{i}"
        if sid in existing:
            return
        t0 = time.time()
        try:
            result = proc(
                full_text=r["en_claim"],
                query=r["en_claim"],  # claim is its own query
                subset=args.subset,
            )
        except Exception as e:
            with lock:
                totals["errors"] += 1
            print(f"  ERR {sid}: {type(e).__name__}: {e}")
            return
        atoms = result.get("generated_atoms") or []
        if not atoms:
            atoms = [{"index": 1, "text": r["en_claim"], "source_sentences": [0]}]
        snippets = result.get("snippets") or []
        if not snippets:
            # fallback: use the original claim as the sole snippet
            snippets = [{"output": r["en_claim"], "atom_indices": [a.get("index") for a in atoms]}]
        rec = {
            "id":              sid,
            "en_claim":        r["en_claim"],
            "label":           r["label"],
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
        with lock:
            out_records.append(rec)
            out_records.sort(key=lambda x: int(x["id"].split("-")[1]))
            tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
            tmp.write_text(json.dumps(out_records, indent=2, ensure_ascii=False))
            tmp.replace(OUT_PATH)
            totals["n"] += 1
            u = result.get("_usage", {})
            totals["in"] += u.get("input_tokens", 0)
            totals["out"] += u.get("output_tokens", 0)
            totals["atoms"] += len(atoms)
            totals["snippets"] += len(snippets)
            if totals["n"] % 25 == 0 or totals["n"] == len(rows):
                print(f"  {totals['n']}/{len(rows)}  "
                      f"atoms={totals['atoms']} (avg {totals['atoms']/totals['n']:.2f}/claim) "
                      f"snippets={totals['snippets']} (avg {totals['snippets']/totals['n']:.2f}/claim)")

    t_start = time.time()
    print(f"atomizing {len(rows) - len(existing)} claims with SnippetProcessor "
          f"(mode=atom-to-snippet, subset={args.subset}, workers={args.workers})")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, i, r) for i, r in enumerate(rows)]
        for _ in as_completed(futs):
            pass
    t_total = time.time() - t_start

    print(f"\n=== summary (wall: {t_total:.1f}s) ===")
    print(f"  atomized: {totals['n']}  errors: {totals['errors']}")
    print(f"  total atoms:    {totals['atoms']}  "
          f"(avg {totals['atoms']/max(totals['n'],1):.2f} per claim)")
    print(f"  total snippets: {totals['snippets']}  "
          f"(avg {totals['snippets']/max(totals['n'],1):.2f} per claim)")
    print(f"  tokens: in={totals['in']:,} out={totals['out']:,}")
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
