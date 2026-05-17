"""Atom-level LLM baseline. Two modes:

  - claim-only   : send only the atomic claim; the LLM judges from its
                   parametric knowledge alone.
  - full-context : also send the user question and the LLM's full
                   answer text. The full text is for decontextualizing
                   the atom (resolving pronouns / missing entities),
                   not as evidence the atom is true.

This script is preserved for reproducibility. The predictions in
data/5-baselines/predictions/atom/ were generated from previous runs;
no need to re-run unless changing a model or prompt.

Usage:
    python -m src.baselines.run_atom --mode claim-only --model gpt-5.4 --reasoning high
    python -m src.baselines.run_atom --mode full-context --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._openai_runner import run

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "2-subset" / "medqa.json"
OUT_BASE = ROOT / "data" / "5-baselines" / "predictions" / "atom"

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
    return f"Claim:\n{row['claim']}"


def build_full_context(row: dict) -> str:
    return (
        f"Question:\n{row['query']}\n\n"
        f"Full answer text (for claim interpretation only):\n{row['full_text']}\n\n"
        f"Extracted claim:\n{row['claim']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["claim-only", "full-context"], required=True)
    ap.add_argument("--model", required=True, help="e.g. gpt-4o, gpt-5-mini, gpt-5.4")
    ap.add_argument("--reasoning", default=None, choices=["minimal", "low", "medium", "high", None],
                    help="reasoning effort for gpt-5.x; omit for gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    rows = json.loads(IN_PATH.read_text())
    print(f"{args.mode} / {args.model} / reasoning={args.reasoning}")

    builder = build_claim_only if args.mode == "claim-only" else build_full_context
    prompt = CLAIM_ONLY_PROMPT if args.mode == "claim-only" else FULL_CONTEXT_PROMPT

    suffix = args.reasoning or "none"
    out_path = OUT_BASE / args.mode / f"{args.model}-{suffix}" / "predictions.json"

    def extras(row: dict) -> dict:
        # Carry the input fields onto the prediction row for self-containment
        return {
            "subset": row.get("subset"),
            "claim":  row["claim"],
            "label":  bool(row["label"]),
        }

    run(
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


if __name__ == "__main__":
    main()
