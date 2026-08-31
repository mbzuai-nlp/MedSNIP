"""Convert HealthFC's interrogative claims into the propositions they assert.

99% of HealthFC `en_claim` values are questions ("Can masks reduce corona
infections?"). The decomposition prompt assumes declarative input, and on 26%
of items it emits a meta-statement instead of a claim - "The user asks whether
Paxlovid protects..." - which is vacuously true regardless of the medical facts
and therefore cannot discriminate. 80 of the 327 scorable claims have an atom
baseline made *entirely* of such statements.

This converts each question to the assertion it presupposes, so both arms
verify a checkable proposition.

Label safety
------------
The prompt sees ONLY `en_claim`. It never sees `label`, `en_explanation` or
`en_top_sentences`. `en_explanation` in particular states the verdict in prose
("...the number of corona infections decreases when many people wear masks"),
so exposing it would leak gold and turn verification into transcription.

Polarity is preserved rather than resolved: "Can X?" becomes "X can", never
"X cannot". The assertion is what gets tested, so the conversion must stay
neutral about whether it is true - a converter that hedged or negated toward
the truth would leak the answer through phrasing.

Usage:
  python -m src.external.healthfc.declarativize --limit 5
  python -m src.external.healthfc.declarativize
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from ...baselines._openrouter_runner import build_kwargs, make_client, usage_row

ROOT = Path(__file__).resolve().parents[3]
IN_RAW = ROOT / "data" / "1-raw" / "healthfc.json"
OUT_PATH = ROOT / "data" / "9-healthfc" / "declarative.json"

MODEL = "openai/gpt-5.4"
REASONING = "high"

SYSTEM = """\
You convert a health question into the single declarative claim it asserts.

Rules:
  - Output the proposition the question puts forward, as a statement.
    "Can masks reduce corona infections?" -> "Masks can reduce corona infections."
  - Preserve the exact scope, population, intervention, outcome and any \
hedging words from the question. Do not broaden or narrow it.
  - Preserve polarity. Never negate, and never add words like "may", \
"probably" or "does not" that were not in the question.
  - Do NOT answer the question, judge it, or indicate whether it is true. \
You are restating it, not evaluating it.
  - Never write about the asker. "The user asks whether X" is wrong; write "X".
  - If the input is already declarative, return it unchanged.

Return JSON: {"statement": "<declarative claim>"}\
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    rows = json.loads(IN_RAW.read_text())
    if args.limit:
        rows = rows[: args.limit]

    existing = {}
    if OUT_PATH.exists():
        existing = {r["id"]: r for r in json.loads(OUT_PATH.read_text())}
    todo = [(i, r) for i, r in enumerate(rows) if f"hfc-{i}" not in existing]
    print(f"claims={len(rows)} done={len(existing)} todo={len(todo)}", flush=True)
    if not todo:
        return

    client = make_client()
    lock = Lock()
    out = list(existing.values())
    stats = {"n": 0, "changed": 0, "cost": 0.0, "errors": 0}

    def work(i: int, r: dict) -> dict:
        kw = build_kwargs(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": r["en_claim"]}],
            reasoning=REASONING, temperature=0.0, max_tokens=600)
        kw["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kw)
        text = json.loads((resp.choices[0].message.content or "").strip()).get("statement")
        return {
            "id": f"hfc-{i}",
            "en_claim": r["en_claim"],
            "statement": (text or r["en_claim"]).strip(),
            # carried through unchanged; never shown to the converter
            "label": r["label"],
            "usage": usage_row(resp),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, i, r): (i, r) for i, r in todo}
        for fut in as_completed(futs):
            i, r = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"id": f"hfc-{i}", "en_claim": r["en_claim"],
                       "statement": r["en_claim"], "label": r["label"],
                       "error": f"{type(e).__name__}: {str(e)[:140]}"}
                stats["errors"] += 1
            with lock:
                out.append(row)
                stats["n"] += 1
                if row["statement"].strip() != row["en_claim"].strip():
                    stats["changed"] += 1
                stats["cost"] += ((row.get("usage") or {}).get("cost") or 0.0)
                if stats["n"] % 100 == 0 or stats["n"] == len(todo):
                    print(f"  {stats['n']}/{len(todo)} changed={stats['changed']} "
                          f"errors={stats['errors']} ${stats['cost']:.2f}", flush=True)
                    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    out.sort(key=lambda r: int(r["id"].split("-")[1]))
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(out)} -> {OUT_PATH}")
    print(f"changed={stats['changed']} errors={stats['errors']} cost=${stats['cost']:.2f}")


if __name__ == "__main__":
    main()
