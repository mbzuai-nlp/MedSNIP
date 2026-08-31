"""Decomposition sweep: run Mode 1 over MedSNIP-Bench with each decomposer.

Drives the existing snippet-processor entry point once per model, so all the
resume, checkpointing and retry behaviour is inherited rather than
reimplemented. Each model writes to its own slug directory under
data/6-snippet-processor/atom-to-snippet/, so rows never clobber each other.

GPT-5.4 is excluded by default: that row already exists from the original
paper run (data/6-snippet-processor/atom-to-snippet/gpt-5.4-high/) and is
reused as the matrix reference. Pass --include-reference to regenerate it.

Resume
------
Entries already on disk are never re-generated, so the sweep is safe to stop
and restart at any point. Two levels of resume:

  - Within a model, `--skip-existing` on the inner runner skips finished
    entries, so an interrupted row picks up where it left off.
  - Across the sweep, `--passes N` repeats until every row is complete or no
    pass makes progress. Weak decomposers fail a scattering of entries to
    transient timeouts, and a second pass usually clears them without
    re-running the entries that already succeeded.

`--resume` skips models that are already complete, so the common case of
"continue what was interrupted" is a single flag with no model list.

Status accumulates in data/12-decomposer/decompose_status.json across
invocations rather than being overwritten, so coverage stays auditable.

Models run sequentially rather than concurrently, so per-provider rate limits
do not compound across rows.

Usage:
  python -m src.decomposer.run_decompose --resume
  python -m src.decomposer.run_decompose --passes 3
  python -m src.decomposer.run_decompose --models openai/gpt-4o
  python -m src.decomposer.run_decompose --workers 48
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .models import (
    DEFAULT_WORKERS,
    DISPLAY,
    MATRIX_MODELS,
    PROVIDER_PIN,
    REASONING,
    WORKERS,
    slug,
)

ROOT = Path(__file__).resolve().parents[2]
SP_ROOT = ROOT / "data" / "6-snippet-processor"
# Mode-aware: snippet-direct writes to its own tree, so resume counting
# must follow the mode rather than assuming atom-to-snippet.
DECOMP_ROOT = SP_ROOT / "atom-to-snippet"
OUT_DIR = ROOT / "data" / "12-decomposer"
LOG_DIR = OUT_DIR / "logs"
STATUS_PATH = OUT_DIR / "decompose_status.json"

REFERENCE_MODEL = "openai/gpt-5.4"
# The reference row predates the vendor-prefixed slug convention.
REFERENCE_DIR = "gpt-5.4-high"
N_ENTRIES = 276


def row_dir(model: str, mode: str = "atom-to-snippet") -> Path:
    root = SP_ROOT / mode
    if model == REFERENCE_MODEL and (root / REFERENCE_DIR).exists():
        return root / REFERENCE_DIR
    return root / slug(model)


def count_done(model: str, mode: str = "atom-to-snippet") -> int:
    d = row_dir(model, mode)
    return len(list(d.glob("e*.json"))) if d.exists() else 0


def workers_for(model: str, override: int | None) -> int:
    if override:
        return override
    return WORKERS.get(model, DEFAULT_WORKERS)


def load_status() -> dict[str, dict]:
    """Status keyed by model, carried across invocations."""
    if not STATUS_PATH.exists():
        return {}
    try:
        data = json.loads(STATUS_PATH.read_text())
    except Exception:
        return {}
    if isinstance(data, list):  # earlier format was a bare list of rows
        return {r["model"]: r for r in data if "model" in r}
    return data


def save_status(status: dict[str, dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def run_one(model: str, workers: int, mode: str, max_retries: int,
            pass_no: int) -> dict:
    log_path = LOG_DIR / f"decompose_{mode}_{slug(model)}.log"
    cmd = [
        sys.executable, "-u", "-m", "src.medsnip.snippet_processor.run",
        "--mode", mode,
        "--model", model,
        "--workers", str(workers),
        "--max-retries", str(max_retries),
        "--skip-existing",
    ]
    reasoning = REASONING.get(model)
    if reasoning:
        cmd += ["--reasoning", reasoning]
    pin = PROVIDER_PIN.get(model)
    if pin:
        cmd += ["--provider-pin", pin]

    before = count_done(model, mode)
    print(f"\n=== {DISPLAY.get(model, model)} (pass {pass_no}) === "
          f"{before}/{N_ENTRIES} on disk, {workers} workers", flush=True)
    t0 = time.time()
    # Append across passes so an earlier pass's diagnostics survive.
    with open(log_path, "a") as fh:
        fh.write(f"\n----- pass {pass_no} -----\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    after = count_done(model, mode)
    elapsed = time.time() - t0

    status = "ok" if after >= N_ENTRIES else f"INCOMPLETE ({after}/{N_ENTRIES})"
    print(f"    {status}  rc={proc.returncode}  {elapsed/60:.1f} min  "
          f"(+{after - before} entries)", flush=True)

    return {
        "model":          model,
        "display":        DISPLAY.get(model, model),
        "slug":           slug(model),
        "pin":            pin,
        "workers":        workers,
        "returncode":     proc.returncode,
        "entries_before": before,
        "entries_after":  after,
        "missing":        N_ENTRIES - after,
        "complete":       after >= N_ENTRIES,
        "gained":         after - before,
        "seconds":        round(elapsed, 1),
        "passes":         pass_no,
        "log":            str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None,
                    help=f"override per-model concurrency (default: per model, "
                         f"{DEFAULT_WORKERS} if unlisted)")
    ap.add_argument("--mode", default="atom-to-snippet",
                    choices=["atom-to-snippet", "snippet-direct"])
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--passes", type=int, default=2,
                    help="repeat incomplete rows up to N times; stops early "
                         "when a pass adds nothing (default 2)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip rows already at full coverage")
    ap.add_argument("--include-reference", action="store_true",
                    help="also regenerate the GPT-5.4 reference row")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    models = args.models or [
        m for m in MATRIX_MODELS
        if args.include_reference or m != REFERENCE_MODEL
    ]
    if args.resume:
        models = [m for m in models if count_done(m, args.mode) < N_ENTRIES]

    status = load_status()

    print(f"decomposition sweep: mode={args.mode}, max_retries={args.max_retries}, "
          f"passes={args.passes}", flush=True)
    for m in models:
        print(f"  {DISPLAY.get(m, m):16s} {count_done(m, args.mode):3d}/{N_ENTRIES} on disk, "
              f"{workers_for(m, args.workers)} workers", flush=True)
    if not models:
        print("nothing to do — every requested row is already complete.", flush=True)
        return

    pending = list(models)
    for pass_no in range(1, args.passes + 1):
        if not pending:
            break
        progressed = False
        for model in list(pending):
            row = run_one(model, workers_for(model, args.workers), args.mode,
                          args.max_retries, pass_no)
            prev = status.get(model, {})
            row["passes"] = prev.get("passes", 0) + 1
            row["seconds"] = round(prev.get("seconds", 0) + row["seconds"], 1)
            status[model] = row
            save_status(status)
            if row["gained"] > 0:
                progressed = True
            if row["complete"]:
                pending.remove(model)
        if pending and not progressed:
            print(f"\npass {pass_no} added no entries for the remaining "
                  f"{len(pending)} row(s); stopping early rather than "
                  f"re-burning the same failures.", flush=True)
            break

    print("\n=== sweep summary ===", flush=True)
    for m in models:
        r = status.get(m, {})
        mark = "ok" if r.get("complete") else "INCOMPLETE"
        print(f"  {r.get('display', m):16s} {r.get('entries_after', 0):3d}/{N_ENTRIES}  "
              f"{r.get('seconds', 0)/60:6.1f} min  {r.get('passes', 0)} pass(es)  {mark}",
              flush=True)
    if pending:
        print(f"\n{len(pending)} row(s) still incomplete. Resume with:", flush=True)
        print(f"  python -m src.decomposer.run_decompose --resume", flush=True)
        for m in pending:
            print(f"    {DISPLAY.get(m, m)}: missing {N_ENTRIES - count_done(m, args.mode)}, "
                  f"see {status.get(m, {}).get('log', '')}", flush=True)
    print(f"\nwrote {STATUS_PATH}", flush=True)


if __name__ == "__main__":
    main()
