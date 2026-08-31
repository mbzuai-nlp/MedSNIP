"""Verification sweep: every decomposer row x every verifier, atoms and snippets.

This is the expensive stage. For each decomposer row it projects human labels
onto that decomposer's own atoms and snippets, then verifies both arms with all
six verifiers under claim-only prompting.

Both arms come from the same decomposition call, so the atom-vs-snippet delta
within a row cannot be attributed to atom quality — only to the verification
unit. That symmetry is the point of the design: it isolates the granularity
effect from decomposer strength.

Outputs land under data/12-decomposer/predictions/ rather than
data/5-baselines/, so the paper's published cells are never touched.

Resume: predictions.json accumulates per cell and the underlying runner skips
ids it already has, so an interrupted sweep continues where it stopped.
Cells already complete are skipped outright with --resume.

Usage:
  python -m src.decomposer.run_verify --dry-run
  python -m src.decomposer.run_verify --resume
  python -m src.decomposer.run_verify --decomposers google/gemma-4-31b-it
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .models import DEFAULT_WORKERS, DISPLAY, MATRIX_MODELS, REASONING, WORKERS, slug

ROOT = Path(__file__).resolve().parents[2]
SP_ROOT = ROOT / "data" / "6-snippet-processor"
DECOMP_ROOT = SP_ROOT / "atom-to-snippet"
OUT_DIR = ROOT / "data" / "12-decomposer"
PRED_ROOT = OUT_DIR / "predictions"
LOG_DIR = OUT_DIR / "logs"
STATUS_PATH = OUT_DIR / "verify_status.json"

REFERENCE_MODEL = "openai/gpt-5.4"
REFERENCE_DIR = "gpt-5.4-high"
UNITS = ("snippet", "atom")


def row_slug(model: str) -> str:
    if model == REFERENCE_MODEL and (DECOMP_ROOT / REFERENCE_DIR).exists():
        return REFERENCE_DIR
    return slug(model)


def labeled_path(decomposer: str, unit: str,
                 decomp_mode: str = "atom-to-snippet") -> Path:
    return SP_ROOT / decomp_mode / row_slug(decomposer) / f"labeled_{unit}s.json"


def unit_tag(unit: str, decomp_mode: str) -> str:
    """Mode 2 cells get their own tag so Mode 1 predictions are never overwritten."""
    return unit if decomp_mode == "atom-to-snippet" else f"mode2-{unit}"


def ensure_labels(decomposer: str, unit: str, force: bool,
                  decomp_mode: str = "atom-to-snippet") -> int:
    """Project human labels onto this decomposer's own units. Returns row count."""
    path = labeled_path(decomposer, unit, decomp_mode)
    if path.exists() and not force:
        return len(json.loads(path.read_text()))
    cmd = [sys.executable, "-m", "src.baselines.prep_auto_labels",
           "--mode", decomp_mode, "--model-slug", row_slug(decomposer),
           "--unit", unit]
    log = LOG_DIR / f"prep_{decomp_mode}_{row_slug(decomposer)}_{unit}.log"
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not path.exists():
        raise SystemExit(f"label projection failed for {decomposer}/{unit}, see {log}")
    return len(json.loads(path.read_text()))


# run_snippet.py builds: <out_base>/<mode>/<source_tag>/<model_slug>/predictions.json
# so the mode segment sits between the root and our decomposer/unit tag.
MODE = "claim-only"


def cell_path(decomposer: str, unit: str, verifier: str,
              decomp_mode: str = "atom-to-snippet") -> Path:
    suffix = REASONING.get(verifier) or "none"
    return (PRED_ROOT / MODE / row_slug(decomposer) / unit_tag(unit, decomp_mode) /
            f"{verifier.replace('/', '__')}-{suffix}" / "predictions.json")


def cell_done(decomposer: str, unit: str, verifier: str,
              decomp_mode: str = "atom-to-snippet") -> int:
    p = cell_path(decomposer, unit, verifier, decomp_mode)
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text()))
    except Exception:
        return 0


def run_cell(decomposer: str, unit: str, verifier: str, workers: int | None,
             decomp_mode: str = "atom-to-snippet") -> dict:
    inp = labeled_path(decomposer, unit, decomp_mode)
    n_units = len(json.loads(inp.read_text()))
    ut = unit_tag(unit, decomp_mode)
    tag = f"{row_slug(decomposer)}/{ut}"
    log = LOG_DIR / f"verify_{row_slug(decomposer)}_{ut}_{verifier.replace('/', '__')}.log"
    cmd = [
        sys.executable, "-u", "-m", "src.baselines.run_snippet",
        "--mode", MODE,
        "--model", verifier,
        "--provider", "openrouter",
        "--input-path", str(inp),
        "--out-base", str(PRED_ROOT),
        "--source-tag", tag,
        "--workers", str(workers or WORKERS.get(verifier, DEFAULT_WORKERS)),
    ]
    reasoning = REASONING.get(verifier)
    if reasoning:
        cmd += ["--reasoning", reasoning]

    before = cell_done(decomposer, unit, verifier, decomp_mode)
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write(f"\n----- {time.strftime('%F %T')} -----\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    after = cell_done(decomposer, unit, verifier, decomp_mode)
    elapsed = time.time() - t0

    return {
        "decomposer": decomposer, "unit": unit, "verifier": verifier,
        "decomp_mode": decomp_mode,
        "n_units": n_units, "before": before, "after": after,
        "complete": after >= n_units, "returncode": proc.returncode,
        "seconds": round(elapsed, 1), "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomposers", nargs="*", default=None)
    ap.add_argument("--verifiers", nargs="*", default=None)
    ap.add_argument("--units", nargs="*", default=list(UNITS), choices=list(UNITS))
    ap.add_argument("--decomp-mode", default="atom-to-snippet",
                    choices=["atom-to-snippet", "snippet-direct"],
                    help="which decomposition to verify. snippet-direct emits "
                         "no atoms, so only the snippet arm exists")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already at full coverage")
    ap.add_argument("--force-labels", action="store_true",
                    help="rebuild label projections even if present")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cell plan and unit counts, run nothing")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)

    dm = args.decomp_mode
    units = ["snippet"] if dm == "snippet-direct" else args.units
    decomposers = args.decomposers or [
        m for m in MATRIX_MODELS
        if (SP_ROOT / dm / row_slug(m)).exists()
    ]
    verifiers = args.verifiers or MATRIX_MODELS

    print("building label projections …", flush=True)
    counts: dict[tuple[str, str], int] = {}
    for d in decomposers:
        for u in units:
            counts[(d, u)] = ensure_labels(d, u, args.force_labels, dm)
            print(f"  {DISPLAY.get(d, d):16s} {u:8s} {counts[(d, u)]:5d} units", flush=True)

    # Verifier is the OUTER loop so the cheap verifiers complete the whole
    # matrix before an expensive one starts. GPT-5.4 is ~83% of the budget;
    # ordering it last means the matrix shape is visible for a few dollars and
    # an early stop still leaves five usable columns.
    plan = [(d, u, v) for v in verifiers for d in decomposers for u in units]
    if args.resume:
        plan = [c for c in plan if cell_done(*c, dm) < counts[(c[0], c[1])]]

    total_calls = sum(counts[(d, u)] - cell_done(d, u, v, dm) for d, u, v in plan)
    print(f"\n{len(plan)} cells to run, {total_calls:,} verifier calls", flush=True)
    if args.dry_run:
        for d, u, v in plan:
            done = cell_done(d, u, v, dm)
            print(f"  {DISPLAY.get(d, d):16s} {u:8s} {DISPLAY.get(v, v):16s} "
                  f"{done}/{counts[(d, u)]}", flush=True)
        return

    status = {}
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text())
        except Exception:
            status = {}

    for i, (d, u, v) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {DISPLAY.get(d, d)} / {u} / {DISPLAY.get(v, v)}",
              flush=True)
        row = run_cell(d, u, v, args.workers, dm)
        key = f"{row_slug(d)}|{unit_tag(u, dm)}|{v}"
        status[key] = row
        STATUS_PATH.write_text(json.dumps(status, indent=2))
        mark = "ok" if row["complete"] else f"INCOMPLETE ({row['after']}/{row['n_units']})"
        print(f"    {mark}  {row['seconds']/60:.1f} min", flush=True)

    incomplete = [k for k, r in status.items() if not r.get("complete")]
    print(f"\n=== verification sweep done: {len(plan)} cells ===", flush=True)
    if incomplete:
        print(f"{len(incomplete)} cell(s) incomplete; resume with --resume", flush=True)
        for k in incomplete:
            print(f"  {k}", flush=True)
    print(f"wrote {STATUS_PATH}", flush=True)


if __name__ == "__main__":
    main()
