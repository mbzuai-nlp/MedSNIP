"""Chunking fidelity for every decomposer row, against human gold boundaries.

Runs the existing snippet-processor evaluator once per row, writing each row's
results to its own directory so the paper's published GPT-5.4 numbers are never
overwritten, then collects the summary into one table.

Fidelity is the metric the camera-ready actually needs: the meta-review asks for
"chunking fidelity and downstream F1" from an open-weight decomposer, and
sentence-set F1 against human boundaries is the former. Merge ratio is not a
substitute — GPT-4o matches the human ratio almost exactly while agreeing less
with human boundaries.

Coverage is reported alongside every score. A row scored over fewer entries than
another is not directly comparable, and quietly averaging over survivors is the
failure mode this pipeline already had once.

Usage:
  python -m src.decomposer.run_fidelity
  python -m src.decomposer.run_fidelity --models openai/gpt-4o
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .models import DISPLAY, MATRIX_MODELS, slug

ROOT = Path(__file__).resolve().parents[2]
DECOMP_ROOT = ROOT / "data" / "6-snippet-processor" / "atom-to-snippet"
OUT_DIR = ROOT / "data" / "12-decomposer"
FIDELITY_DIR = OUT_DIR / "fidelity"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_JSON = OUT_DIR / "fidelity_summary.json"

REFERENCE_MODEL = "openai/gpt-5.4"
REFERENCE_DIR = "gpt-5.4-high"
N_ENTRIES = 276
HUMAN_SNIPPETS = 2524
HUMAN_ATOMS = 5755


def row_slug(model: str) -> str:
    if model == REFERENCE_MODEL and (DECOMP_ROOT / REFERENCE_DIR).exists():
        return REFERENCE_DIR
    return slug(model)


def row_stats(model: str, mode: str = "atom-to-snippet") -> dict:
    """Counts straight off disk, independent of the evaluator."""
    d = DECOMP_ROOT.parent / mode / row_slug(model)
    files = sorted(d.glob("e*.json"))
    atoms = snippets = malformed = empty = 0
    cost = 0.0
    for f in files:
        r = json.loads(f.read_text())
        s = r.get("snippets") or []
        atoms += len(r.get("generated_atoms") or [])
        snippets += len(s)
        if not s:
            empty += 1
        malformed += sum(1 for x in s if not isinstance(x, dict) or "output" not in x)
        for u in (r.get("usage_steps") or []):
            cost += u.get("cost") or 0.0
    return {"entries": len(files), "atoms": atoms, "snippets": snippets,
            "malformed": malformed, "empty": empty, "cost_usd": round(cost, 4),
            "merge_ratio": round(atoms / snippets, 3) if snippets else None}


def evaluate(model: str) -> dict | None:
    sl = row_slug(model)
    out = FIDELITY_DIR / sl
    out.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"fidelity_{sl}.log"
    cmd = [sys.executable, "-u", "-m", "src.medsnip.snippet_processor.evaluate",
           "--model-slug", sl, "--out-dir", str(out)]
    print(f"  evaluating {DISPLAY.get(model, model)} …", flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"    FAILED rc={proc.returncode}, see {log.relative_to(ROOT)}", flush=True)
        return None
    res = out / "results.json"
    if not res.exists():
        print(f"    no results.json written, see {log.relative_to(ROOT)}", flush=True)
        return None
    # evaluate.py scores every mode present under the slug, so one run yields
    # both atom-to-snippet and snippet-direct once Mode 2 exists on disk.
    return json.loads(res.read_text()).get("summary", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    FIDELITY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    models = args.models or MATRIX_MODELS

    rows = []
    for model in models:
        d = DECOMP_ROOT / row_slug(model)
        if not d.exists() or not list(d.glob("e*.json")):
            print(f"  skipping {DISPLAY.get(model, model)} — no decomposition on disk",
                  flush=True)
            continue
        fid_all = evaluate(model) or {}
        for mode in ("atom-to-snippet", "snippet-direct"):
            stats = row_stats(model, mode)
            if not stats.get("entries"):
                continue
            rows.append({"model": model, "display": DISPLAY.get(model, model),
                         "slug": row_slug(model), "mode": mode, **stats,
                         "fidelity": fid_all.get(mode, {})})
        SUMMARY_JSON.write_text(json.dumps(rows, indent=2))

    def g(r, key):
        return r["fidelity"].get(key)

    lines = [
        "# Decomposer chunking fidelity",
        "",
        f"Mode 1 (atom-to-snippet) and Mode 2 (snippet-direct) against human "
        f"gold boundaries on "
        f"MedSNIP-Bench ({HUMAN_ATOMS:,} atoms → {HUMAN_SNIPPETS:,} human snippets, "
        f"merge ratio {HUMAN_ATOMS/HUMAN_SNIPPETS:.2f}).",
        "",
        "`cov` is entries evaluated out of 276. Rows with unequal coverage are "
        "not directly comparable. Mode 2 emits no atoms, so its atom and merge "
        "columns are empty by construction.",
        "",
        "| Decomposer | mode | cov | atoms | snippets | merge | sent F1 | P | R | embed | cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        f1 = g(r, "sent_f1_mean")
        p = g(r, "sent_precision_mean")
        rc = g(r, "sent_recall_mean")
        emb = g(r, "embedding_cosine_mean")
        cons = (r["fidelity"].get("sent_f1_by_subset") or {}).get("consumer")
        vign = (r["fidelity"].get("sent_f1_by_subset") or {}).get("vignette")
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        mlabel = "1" if r.get("mode") == "atom-to-snippet" else "2"
        merge = r["merge_ratio"] if r.get("mode") == "atom-to-snippet" else "—"
        lines.append(
            f"| {r['display']} | {mlabel} | {r['entries']}/{N_ENTRIES} | "
            f"{r['atoms']:,} | {r['snippets']:,} | {merge} | {fmt(f1)} | {fmt(p)} | "
            f"{fmt(rc)} | {fmt(emb)} | "
            f"{('$%.2f' % r['cost_usd']) if r['cost_usd'] else '—'} |"
        )
    incomplete = [r for r in rows if r["entries"] < N_ENTRIES]
    if incomplete:
        lines += ["", "## Incomplete rows", ""]
        for r in incomplete:
            lines.append(f"- **{r['display']}** (Mode "
                         f"{'1' if r.get('mode') == 'atom-to-snippet' else '2'}): "
                         f"{r['entries']}/{N_ENTRIES} entries "
                         f"({N_ENTRIES - r['entries']} failed decomposition)")
    print("\n".join(lines))

    print(f"\nwrote {SUMMARY_JSON}", flush=True)
    print("\n" + "\n".join(lines[6:]), flush=True)


if __name__ == "__main__":
    main()
