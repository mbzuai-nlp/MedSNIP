"""Viability pilot for the decomposer x verifier matrix.

Runs the Mode 1 (atom-to-snippet) pipeline over a handful of MedSNIP-Bench
entries with each of the six candidate decomposers, before any budget is
committed to the full sweep. What we are checking, per model:

  - does it hold the JSON schema across the two-call extract-then-cluster
    chain (the weaker models are the risk here);
  - how many atoms and snippets it produces relative to GPT-5.4, which is
    the first signal of whether its segmentation is usable at all;
  - measured cost and wall time per entry, to extrapolate the full sweep;
  - which upstream provider OpenRouter routes to, needed for reproducibility.

Usage:
  python -m src.decomposer.pilot --entries 3
  python -m src.decomposer.pilot --entries 3 --models openai/gpt-5.4
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from ..medsnip.snippet_processor.processor import SnippetProcessor
from ..medsnip.snippet_processor.run import SPLIT_PATH, aggregate_entries
from .models import DISPLAY, MATRIX_MODELS, REASONING

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "12-decomposer"
OUT_PATH = OUT_DIR / "pilot_report.json"


def run_model(model: str, entries: list[dict], mode: str) -> dict:
    proc = SnippetProcessor(model=model, mode=mode,
                            reasoning_effort=REASONING.get(model))
    ok = fail = atoms = snippets = 0
    cost = 0.0
    errors: list[str] = []
    t0 = time.time()

    for e in entries:
        try:
            r = proc(full_text=e["full_text"], query=e["full_query"],
                     subset=e["subset"])
            ok += 1
            atoms += len(r.get("generated_atoms") or [])
            snippets += len(r["snippets"])
            for step in (r.get("_usage_steps") or []):
                cost += step.get("cost") or 0.0
        except Exception as exc:
            fail += 1
            errors.append(f"e{e['entry_id']}: {type(exc).__name__}: {str(exc)[:160]}")

    elapsed = time.time() - t0
    return {
        "model":              model,
        "display":            DISPLAY.get(model, model),
        "reasoning":          REASONING.get(model),
        "mode":               mode,
        "entries":            len(entries),
        "ok":                 ok,
        "failed":             fail,
        "atoms":              atoms,
        "snippets":           snippets,
        "cost_usd":           round(cost, 6),
        "cost_per_entry_usd": round(cost / ok, 6) if ok else None,
        "seconds":            round(elapsed, 1),
        "sec_per_entry":      round(elapsed / len(entries), 1) if entries else None,
        "upstream_providers": sorted(proc.upstream_providers),
        "errors":             errors,
    }


def write_markdown(report: list[dict], n_entries: int) -> None:
    ref = next((r for r in report if r["model"] == "openai/gpt-5.4"), None)
    lines = [
        "# Decomposer matrix — viability pilot",
        "",
        f"Mode 1 (atom-to-snippet) over **{n_entries}** MedSNIP-Bench entries per model, "
        "via OpenRouter. Checks JSON-schema survival, segmentation sanity, and measured "
        "cost before the full sweep is funded.",
        "",
        "| Decomposer | ok | atoms | snippets | atoms/entry | snip/entry | $/entry | s/entry | upstream |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report:
        n = r["ok"] or 1
        lines.append(
            f"| {r['display']} | {r['ok']}/{r['entries']} | {r['atoms']} | {r['snippets']} | "
            f"{r['atoms']/n:.1f} | {r['snippets']/n:.1f} | "
            f"{('$%.4f' % r['cost_per_entry_usd']) if r['cost_per_entry_usd'] else 'n/a'} | "
            f"{r['sec_per_entry'] or 'n/a'} | {', '.join(r['upstream_providers']) or 'n/a'} |"
        )

    if ref and ref["ok"]:
        lines += ["", "## Full-sweep extrapolation (276 entries, Mode 1)", "",
                  "| Decomposer | decomp $ | atoms out | snippets out |", "|---|---:|---:|---:|"]
        for r in report:
            if not r["ok"]:
                lines.append(f"| {r['display']} | FAILED | — | — |")
                continue
            per = r["cost_per_entry_usd"] or 0.0
            lines.append(
                f"| {r['display']} | ${per*276:.2f} | "
                f"{round(r['atoms']/r['ok']*276):,} | {round(r['snippets']/r['ok']*276):,} |"
            )

    failures = [(r["display"], e) for r in report for e in r["errors"]]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- **{d}** — {e}" for d, e in failures]

    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", type=int, default=3,
                    help="how many entries to pilot per model (default 3)")
    ap.add_argument("--mode", default="atom-to-snippet",
                    choices=["atom-to-snippet", "snippet-direct"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of matrix models (default: all six)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = aggregate_entries(json.loads(SPLIT_PATH.read_text()))[: args.entries]
    models = args.models or MATRIX_MODELS

    print(f"pilot: {len(models)} decomposers x {len(entries)} entries "
          f"(ids {[e['entry_id'] for e in entries]}), mode={args.mode}\n", flush=True)

    report: list[dict] = []
    for model in models:
        print(f"--- {model} ---", flush=True)
        try:
            row = run_model(model, entries, args.mode)
        except Exception:
            traceback.print_exc()
            row = {"model": model, "display": DISPLAY.get(model, model),
                   "mode": args.mode, "entries": len(entries), "ok": 0,
                   "failed": len(entries), "atoms": 0, "snippets": 0,
                   "cost_usd": 0.0, "cost_per_entry_usd": None, "seconds": 0.0,
                   "sec_per_entry": None, "upstream_providers": [],
                   "errors": ["constructor failed: " + traceback.format_exc(limit=1)]}
        report.append(row)
        print(f"    ok={row['ok']}/{row['entries']} atoms={row['atoms']} "
              f"snippets={row['snippets']} ${row['cost_usd']:.4f} {row['seconds']}s "
              f"prov={row['upstream_providers']}", flush=True)
        for e in row["errors"]:
            print(f"    ! {e}", flush=True)
        # Checkpoint after every model so a crash mid-sweep keeps what ran.
        OUT_PATH.write_text(json.dumps(report, indent=2))
        write_markdown(report, len(entries))

    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
