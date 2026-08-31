"""Snippet-level LLM baseline. Two modes:

  - claim-only   : send only the human-edited snippet text.
  - full-context : also send the user question and the LLM's full
                   answer text from which the snippet was extracted.

This script is preserved for reproducibility. The predictions in
data/5-baselines/predictions/snippet/ were generated from previous
runs; no need to re-run unless changing a model or prompt.

Usage:
    python -m src.baselines.run_snippet --mode claim-only --model gpt-5.4 --reasoning high
    python -m src.baselines.run_snippet --mode full-context --model gpt-4o
    python -m src.baselines.run_snippet --mode claim-only \
        --provider hf --model meta-llama/Llama-3.3-70B-Instruct --hf-provider novita
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import _hf_runner, _openai_runner, _openrouter_runner

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "3-annotated" / "medsnip-bench.json"
OUT_BASE = ROOT / "data" / "5-baselines" / "predictions" / "snippet"

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


def build_claim_only(row: dict) -> str:
    return f"Claim:\n{row['snippet_text']}"


def build_full_context(row: dict) -> str:
    return (
        f"Question:\n{row['query']}\n\n"
        f"Full answer text (for claim interpretation only):\n{row['full_text']}\n\n"
        f"Extracted claim:\n{row['snippet_text']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["claim-only", "full-context"], required=True)
    ap.add_argument("--model", required=True, help="e.g. gpt-4o, gpt-5.4, meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--provider", choices=["openai", "hf", "openrouter"], default="openrouter",
                    help="openai = client.responses.create; hf = HF router (chat.completions, OpenAI-compatible)")
    ap.add_argument("--hf-provider", default="novita",
                    help="HF Inference provider when --provider=hf (e.g. novita, fireworks-ai)")
    ap.add_argument("--limit", type=int, default=None, help="optional: only run first N items (smoke test)")
    ap.add_argument("--reasoning", default=None, choices=["minimal", "low", "medium", "high", None],
                    help="reasoning effort for gpt-5.x; omit for gpt-4o / open-weight models")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--input-path", default=None,
                    help="override input file (default: data/3-annotated/medsnip-bench.json)")
    ap.add_argument("--out-base", default=None,
                    help="override output root (keeps side experiments out of "
                         "data/5-baselines/)")
    ap.add_argument("--source-tag", default=None,
                    help="extra tag to slot into output path (e.g. 'auto-a2s', "
                         "'auto-direct'); inserted between mode and model dirs")
    args = ap.parse_args()

    in_path = Path(args.input_path) if args.input_path else IN_PATH
    rows = json.loads(in_path.read_text())
    # Re-key by snippet_id since runner expects `id`
    for r in rows:
        r["id"] = r["snippet_id"]
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"{args.mode} / {args.model} / provider={args.provider} / "
          f"reasoning={args.reasoning} / input={in_path}")

    builder = build_claim_only if args.mode == "claim-only" else build_full_context
    prompt = CLAIM_ONLY_PROMPT if args.mode == "claim-only" else FULL_CONTEXT_PROMPT

    suffix = args.reasoning or "none"
    model_slug = args.model.replace("/", "__")
    out_base = Path(args.out_base) if args.out_base else OUT_BASE
    if args.source_tag:
        out_path = out_base / args.mode / args.source_tag / f"{model_slug}-{suffix}" / "predictions.json"
    else:
        out_path = out_base / args.mode / f"{model_slug}-{suffix}" / "predictions.json"

    def extras(row: dict) -> dict:
        return {
            "snippet_id":             row["snippet_id"],
            "entry_id":               row["entry_id"],
            "subset":                 row["subset"],
            "snippet_text":           row["snippet_text"],
            "label_atomic":           bool(row["label_atomic"]),
            "label_human_general":    bool(row["label_human_general"]),
            "label_human_contextual": bool(row["label_human_contextual"]),
        }

    common_kwargs = dict(
        items=rows,
        output_path=out_path,
        build_user_message=builder,
        system_prompt=prompt,
        model=args.model,
        reasoning=args.reasoning,
        temperature=args.temperature,
        workers=args.workers,
        save_every=args.save_every,
        extra_row_fields=extras,
    )
    if args.provider == "hf":
        _hf_runner.run(**common_kwargs, provider=args.hf_provider)
    elif args.provider == "openai":
        _openai_runner.run(**common_kwargs)
    else:
        _openrouter_runner.run(**common_kwargs)


if __name__ == "__main__":
    main()
