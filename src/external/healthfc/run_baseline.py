"""Run the snippet baseline on HealthFC at two grains:

  --grain snippet : predict T/F on each whole en_claim (one call per claim).
  --grain atom    : predict T/F on each atom from atoms.json (one call per atom);
                    atom predictions are aggregated to the claim level by
                    OR-False (any atom False → claim False) for evaluation.

Same `claim-only` prompt as the existing MedSNIP snippet/atom baselines.

Output: data/9-healthfc/baselines/predictions/{grain}/claim-only/{model}-{reasoning}/predictions.json

Usage:
    python -m src.external.healthfc.run_baseline --grain snippet --model gpt-5.4 --reasoning high
    python -m src.external.healthfc.run_baseline --grain atom    --model gpt-5.4 --reasoning high
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.baselines._openai_runner import run

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
IN_RAW = ROOT / "data" / "1-raw" / "healthfc.json"
IN_ATOMS = (ROOT / "data" / "9-healthfc" / "snippet-processor"
            / "atom-to-snippet" / "healthfc.json")
OUT_BASE = ROOT / "data" / "9-healthfc" / "baselines" / "predictions"

CLAIM_ONLY_PROMPT = (
    "You are a cautious medical fact-checking assistant. "
    "Given a claim, decide if the claim is true. "
    'Respond with exactly one word: "true" if the claim is fully supported, otherwise "false". '
    "No other words."
)


def build_user(item: dict) -> str:
    return f"Claim:\n{item['claim_text']}"


def load_items_snippet() -> list[dict]:
    """One item per en_claim. id = hfc-{i}, claim_text = en_claim."""
    rows = json.loads(IN_RAW.read_text())
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id":            f"hfc-{i}",
            "claim_text":    r["en_claim"],
            "claim_id":      f"hfc-{i}",
            "label":         r["label"],
            "is_binary":     r["label"] in (0, 2),
            "gold_bool":     True if r["label"] == 0 else (False if r["label"] == 2 else None),
        })
    return out


def load_items_atom() -> list[dict]:
    """One item per atom. id = hfc-{i}-A{j}, claim_text = atom text.
    Carries claim_id so we can aggregate atom predictions to the claim later.
    """
    atoms_rows = json.loads(IN_ATOMS.read_text())
    out = []
    for rec in atoms_rows:
        claim_id = rec["id"]
        for j, atom in enumerate(rec["generated_atoms"], 1):
            out.append({
                "id":          f"{claim_id}-A{j}",
                "claim_text":  atom["text"],
                "claim_id":    claim_id,
                "label":       rec["label"],
                "is_binary":   rec["label"] in (0, 2),
                "gold_bool":   True if rec["label"] == 0 else (False if rec["label"] == 2 else None),
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
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    if args.grain == "snippet":
        items = load_items_snippet()
    else:
        if not IN_ATOMS.exists():
            raise SystemExit(
                f"missing {IN_ATOMS}; run "
                f"`python -m src.external.healthfc.atomize` first"
            )
        items = load_items_atom()

    print(f"loaded {len(items)} {args.grain}-level items")

    tag = f"{args.model}-{args.reasoning or 'none'}"
    out_dir = OUT_BASE / args.grain / "claim-only" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.json"

    def extra(item: dict) -> dict:
        return {
            "claim_id":   item["claim_id"],
            "claim_text": item["claim_text"],
            "label":      item["label"],
            "is_binary":  item["is_binary"],
            "gold_bool":  item["gold_bool"],
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
