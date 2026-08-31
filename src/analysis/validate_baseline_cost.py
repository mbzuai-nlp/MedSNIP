"""Sample-validate baseline cost estimates with measured token usage + wall time.

Runs 200 fresh LLM calls per critical cell (gpt-5.4-high × {snippet, atom} ×
{full-context, claim-only} on MedSNIP-Bench dev), captures `usage` and per-call
`wall_seconds`. Compares the measured aggregate cost to our tiktoken estimate
in `data/8-verifier/metrics.json`, plus reports actual wall time (which we don't have for baselines
otherwise).

Output: data/!-analysis/baseline_cost_validation.json

Usage:
    python -m src.analysis.validate_baseline_cost
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OUT_DIR = ROOT / "data" / "!-analysis"
OUT_JSON = OUT_DIR / "baseline_cost_validation.json"
PER_CALL_DIR = OUT_DIR / "baseline_cost_per_call"  # one file per cell

CLAIM_ONLY_PROMPT = (
    "You are a cautious medical fact-checking assistant. "
    "Given a claim, decide if the claim is true. "
    'Respond with exactly one word: "true" if the claim is fully supported, otherwise "false". '
    "No other words."
)
FULL_CONTEXT_PROMPT = (
    "You are a cautious medical fact-checking assistant. "
    "You will be given a user question, a full answer text, and a single extracted claim "
    "(the claim was decomposed from the full answer text). "
    "The extracted claim may be underspecified on its own (e.g., pronouns like 'this', missing entities). "
    "Use the full answer text ONLY to decontextualize/interpret what the claim is referring to. "
    "Then decide whether the interpreted claim is medically/factually correct in the real world. "
    "Do NOT treat the full answer text as evidence that the claim is true; it is just context for interpretation. "
    'Respond with exactly one word: "true" if the claim is correct, otherwise "false". '
    "No other words."
)

# OpenAI rates (per 1M tokens, May 2026)
RATES = {
    "gpt-5.4-high": {"in": 2.50, "cached": 0.25, "out": 15.0, "reasoning": "high"},
}


def build_user(item: dict, mode: str) -> str:
    if mode == "claim-only":
        return f"Claim:\n{item['claim_text']}"
    return (
        f"Question:\n{item['query']}\n\n"
        f"Full answer text (for claim interpretation only):\n{item['full_text']}\n\n"
        f"Extracted claim:\n{item['claim_text']}"
    )


def call_once(client: OpenAI, model: str, sys_prompt: str, user: str,
              reasoning: str | None) -> tuple[dict, float]:
    kwargs = dict(
        model=model.split("-high")[0].split("-none")[0],
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        max_output_tokens=2000,
    )
    if reasoning:
        kwargs["reasoning"] = {"effort": reasoning}
    t0 = time.time()
    resp = client.responses.create(**kwargs)
    wall = time.time() - t0
    u = resp.usage
    # Anthropic / OpenAI responses API: input_tokens, output_tokens,
    # input_tokens_details has cached_tokens
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cached = 0
    details = getattr(u, "input_tokens_details", None)
    if details:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {"input": inp, "cached": cached, "output": out, "wall": wall}, wall


def load_medsnip_bench_dev_snippet_items(n: int = 200, seed: int = 42) -> list[dict]:
    rows = [r for r in json.load(open(ROOT/"data/4-split/medsnip-bench.json"))
            if r["split"] == "dev"]
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    return [{
        "id": r["snippet_id"],
        "claim_text": r["snippet_text"],
        "query": r["query"],
        "full_text": r["full_text"],
    } for r in sample]


def load_medsnip_bench_dev_atom_items(n: int = 200, seed: int = 42) -> list[dict]:
    atoms = json.load(open(ROOT/"data/2-subset/kim.json"))
    gold = json.load(open(ROOT/"data/4-split/medsnip-bench.json"))
    entry_split = {r["entry_id"]: r["split"] for r in gold}
    dev_atoms = [a for a in atoms
                 if entry_split.get(int(str(a["id"]).split("-")[0])) == "dev"]
    rng = random.Random(seed)
    sample = rng.sample(dev_atoms, min(n, len(dev_atoms)))
    return [{
        "id": a["id"],
        "claim_text": a["claim"],
        "query": a["query"],
        "full_text": a["full_text"],
    } for a in sample]


def run_cell(items: list[dict], grain: str, mode: str, model: str,
             workers: int = 12, per_call_path: Path | None = None) -> dict:
    sys_p = FULL_CONTEXT_PROMPT if mode == "full-context" else CLAIM_ONLY_PROMPT
    rates = RATES[model]
    client = OpenAI()

    results = []
    t_start = time.time()
    def worker(it):
        try:
            usage, wall = call_once(client, model, sys_p, build_user(it, mode),
                                     rates.get("reasoning"))
            return {"id": it["id"], **usage}
        except Exception as e:
            return {"id": it["id"], "error": str(e)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, it) for it in items]
        for f in as_completed(futs):
            results.append(f.result())
    t_total = time.time() - t_start

    # Persist per-call records
    if per_call_path is not None:
        per_call_path.parent.mkdir(parents=True, exist_ok=True)
        per_call_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    ok = [r for r in results if "error" not in r]
    err = len(results) - len(ok)
    tot_in = sum(r["input"] for r in ok)
    tot_cached = sum(r["cached"] for r in ok)
    tot_out = sum(r["output"] for r in ok)
    nc = tot_in - tot_cached
    measured_cost = (nc * rates["in"] + tot_cached * rates["cached"]
                     + tot_out * rates["out"]) / 1_000_000

    walls = [r["wall"] for r in ok]
    walls.sort()
    n = len(walls)
    return {
        "grain": grain, "mode": mode, "model": model,
        "n_calls": len(ok), "n_errors": err,
        "tokens": {"input": tot_in, "cached": tot_cached, "output": tot_out,
                   "non_cached_input": nc},
        "measured_cost_total": round(measured_cost, 4),
        "per_call_cost_mean": round(measured_cost / max(len(ok), 1), 5),
        "wall_total_at_12workers": round(t_total, 2),
        "per_call_wall": {
            "mean": round(sum(walls)/n, 2) if n else 0,
            "median": round(walls[n//2], 2) if n else 0,
            "p95": round(walls[int(0.95*n)], 2) if n else 0,
            "max": round(max(walls), 2) if walls else 0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = [
        ("snippet", "full-context", "gpt-5.4-high"),
        ("snippet", "claim-only",   "gpt-5.4-high"),
        ("atom",    "full-context", "gpt-5.4-high"),
        ("atom",    "claim-only",   "gpt-5.4-high"),
    ]

    all_results = {}
    for grain, mode, model in cells:
        print(f"\n--- {grain} / {mode} / {model} ---")
        if grain == "snippet":
            items = load_medsnip_bench_dev_snippet_items(args.n_per_cell)
        else:
            items = load_medsnip_bench_dev_atom_items(args.n_per_cell)
        cell_key = f"{grain}/{mode}/{model}"
        per_call_file = PER_CALL_DIR / f"{grain}_{mode}_{model}.json"
        r = run_cell(items, grain, mode, model, args.workers,
                     per_call_path=per_call_file)
        r["per_call_file"] = str(per_call_file.relative_to(ROOT))
        all_results[cell_key] = r
        print(f"  n_calls={r['n_calls']} errors={r['n_errors']}")
        print(f"  tokens: in={r['tokens']['input']:,} cached={r['tokens']['cached']:,} "
              f"out={r['tokens']['output']:,}")
        print(f"  measured cost: ${r['measured_cost_total']:.2f} "
              f"({r['per_call_cost_mean']*1000:.2f}¢/call)")
        print(f"  per-call wall: mean={r['per_call_wall']['mean']}s "
              f"median={r['per_call_wall']['median']}s p95={r['per_call_wall']['p95']}s")
        print(f"  total wall @ {args.workers} workers: {r['wall_total_at_12workers']}s")

    # Extrapolate to full dev: per-call cost × full n
    full_dev_counts = {"snippet": 494, "atom": 1078}
    print("\n\n=== Extrapolation to full dev (n=494 snippets, 1078 atoms) ===")
    print(f"{'cell':<40s} {'measured/call':>13s} {'×n':>5s} {'extrapolated':>13s} "
          f"{'tiktoken_est':>13s}")
    tiktoken_estimates = {
        "snippet/full-context/gpt-5.4-high": 6.65,
        "snippet/claim-only/gpt-5.4-high":   6.02,
        "atom/full-context/gpt-5.4-high":   14.43,
        "atom/claim-only/gpt-5.4-high":     13.10,
    }
    for cell, r in all_results.items():
        grain = r["grain"]
        n_full = full_dev_counts[grain]
        per_call = r["per_call_cost_mean"]
        extrap = per_call * n_full
        est = tiktoken_estimates.get(cell, 0)
        print(f"{cell:<40s} ${per_call:>11.4f} {n_full:>5d} ${extrap:>11.2f} ${est:>11.2f}")

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
