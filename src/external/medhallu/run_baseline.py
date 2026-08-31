"""Run the snippet / atom baselines on MedHallu (binary subset).

Mirrors src.external.healthfc.run_baseline. Same `claim-only` prompt as the
existing MedSNIP baselines.

Output: data/10-medhallu/baselines/predictions/{grain}/claim-only/{tag}/predictions.json

Usage:
    python -m src.external.medhallu.run_baseline --grain snippet --model gpt-5.4 --reasoning high
    python -m src.external.medhallu.run_baseline --grain atom    --model gpt-5.4 --reasoning high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.baselines._openai_runner import run

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

IN_RAW = ROOT / "data" / "1-raw" / "medhallu.json"
IN_ATOMS = (ROOT / "data" / "10-medhallu" / "snippet-processor"
            / "atom-to-snippet" / "medhallu.json")
OUT_BASE = ROOT / "data" / "10-medhallu" / "baselines" / "predictions"

CLAIM_ONLY_PROMPT = (
    "You are a cautious medical fact-checking assistant. "
    "Given a claim, decide if the claim is true. "
    'Respond with exactly one word: "true" if the claim is fully supported, otherwise "false". '
    "No other words."
)


def build_user(item: dict) -> str:
    return f"Claim:\n{item['claim_text']}"


def load_items_snippet() -> list[dict]:
    """One item per MedHallu answer: 2 per row (ground_truth + hallucinated)."""
    rows = json.loads(IN_RAW.read_text())
    out = []
    for r in rows:
        eid = r["id"]
        for kind, text, gold in [
            ("ground_truth", r["ground_truth"], True),
            ("hallucinated", r["hallucinated_answer"], False),
        ]:
            cid = f"mh-{eid}-{'T' if gold else 'H'}"
            out.append({
                "id":         cid,
                "claim_text": text,
                "claim_id":   cid,
                "medhallu_id": eid,
                "kind":       kind,
                "difficulty": r["difficulty"],
                "gold_bool":  gold,
                "is_binary":  True,
            })
    return out


def load_items_atom() -> list[dict]:
    """One item per atom, carrying claim_id for aggregation later."""
    atoms_rows = json.loads(IN_ATOMS.read_text())
    out = []
    for rec in atoms_rows:
        claim_id = rec["id"]
        for j, atom in enumerate(rec["generated_atoms"], 1):
            out.append({
                "id":          f"{claim_id}-A{j}",
                "claim_text":  atom["text"],
                "claim_id":    claim_id,
                "medhallu_id": rec["medhallu_id"],
                "kind":        rec["kind"],
                "difficulty":  rec["difficulty"],
                "gold_bool":   rec["gold_bool"],
                "is_binary":   True,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grain", choices=["snippet", "atom"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning", default=None,
                    choices=["minimal", "low", "medium", "high", None])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    if args.grain == "snippet":
        items = load_items_snippet()
    else:
        if not IN_ATOMS.exists():
            raise SystemExit(
                f"missing {IN_ATOMS}; run "
                f"`python -m src.external.medhallu.snippet_processor` first"
            )
        items = load_items_atom()

    print(f"loaded {len(items)} {args.grain}-level items")

    tag = f"{args.model}-{args.reasoning or 'none'}"
    out_dir = OUT_BASE / args.grain / "claim-only" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.json"

    def extra(item: dict) -> dict:
        return {
            "claim_id":    item["claim_id"],
            "claim_text":  item["claim_text"],
            "medhallu_id": item["medhallu_id"],
            "kind":        item["kind"],
            "difficulty":  item["difficulty"],
            "is_binary":   item["is_binary"],
            "gold_bool":   item["gold_bool"],
        }

    run(
        items=items,
        output_path=out_path,
        build_user_message=build_user,
        system_prompt=CLAIM_ONLY_PROMPT,
        model=args.model,
        reasoning=args.reasoning,
        temperature=args.temperature,
        workers=args.workers,
        save_every=args.save_every,
        extra_row_fields=extra,
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
