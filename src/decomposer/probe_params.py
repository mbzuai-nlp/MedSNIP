"""Isolate which request parameter blocks open-weight decomposers on OpenRouter.

The viability pilot found all four open-weight models returning 404
"No endpoints found that can handle the requested parameters", while both
OpenAI models succeeded. That error names the parameters collectively, so
this probe varies them one at a time to find the real blocker.

The three suspects, in the order they are worth ruling out:

  - `max_completion_tokens`. The processor sends this (an OpenAI-specific
    newer spelling) rather than `max_tokens`. Non-OpenAI upstreams may only
    accept the latter, and under `require_parameters` that alone 404s.
  - `response_format: {"type": "json_object"}`. Genuinely unsupported by
    some upstreams.
  - `provider: {require_parameters: true}` itself, which converts any
    unsupported parameter from a silent drop into a hard routing failure.

Each variant sends one trivial prompt, so the whole probe costs fractions
of a cent. A variant "passes" only if it both routes and returns parseable
JSON — routing success alone is not enough, since a provider may accept the
call and then ignore the JSON instruction.

Usage:
  python -m src.decomposer.probe_params
  python -m src.decomposer.probe_params --models meta-llama/llama-3.3-70b-instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..baselines._openrouter_runner import make_client
from .models import DISPLAY, MATRIX_MODELS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "12-decomposer"
OUT_PATH = OUT_DIR / "param_probe.json"

MESSAGES = [
    {"role": "system", "content": "You output only JSON."},
    {"role": "user", "content": 'Return {"atoms": ["a", "b"]} exactly.'},
]

# Ordered most-constrained to least. The first passing variant is the one to adopt.
VARIANTS: list[tuple[str, dict]] = [
    ("A: max_completion_tokens + json + require", {
        "max_completion_tokens": 200, "response_format": {"type": "json_object"},
        "extra_body": {"provider": {"require_parameters": True}, "usage": {"include": True}},
    }),
    ("B: max_tokens + json + require", {
        "max_tokens": 200, "response_format": {"type": "json_object"},
        "extra_body": {"provider": {"require_parameters": True}, "usage": {"include": True}},
    }),
    ("C: max_tokens + json, no require", {
        "max_tokens": 200, "response_format": {"type": "json_object"},
        "extra_body": {"usage": {"include": True}},
    }),
    ("D: max_tokens only, no json", {
        "max_tokens": 200,
        "extra_body": {"usage": {"include": True}},
    }),
]


def parses_as_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def probe(client, model: str, name: str, kwargs: dict) -> dict:
    try:
        resp = client.chat.completions.create(model=model, messages=MESSAGES, **kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        provider = getattr(resp, "provider", None) or \
            (getattr(resp, "model_extra", None) or {}).get("provider")
        return {"variant": name, "routed": True, "json": parses_as_json(raw),
                "provider": provider, "sample": raw[:80], "error": None}
    except Exception as e:
        return {"variant": name, "routed": False, "json": False, "provider": None,
                "sample": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()
    models = args.models or MATRIX_MODELS

    results: dict[str, list[dict]] = {}
    for model in models:
        print(f"--- {model} ---", flush=True)
        rows = []
        for name, kwargs in VARIANTS:
            r = probe(client, model, name, kwargs)
            rows.append(r)
            status = "ok  " if r["routed"] else "404 "
            jm = "json" if r["json"] else ("text" if r["routed"] else "----")
            print(f"  {status} {jm}  {name}"
                  f"{'  [' + r['provider'] + ']' if r['provider'] else ''}"
                  f"{'  ' + r['error'] if r['error'] else ''}", flush=True)
        results[model] = rows
        OUT_PATH.write_text(json.dumps(results, indent=2))

    lines = ["# OpenRouter parameter probe", "",
             "Which request shape routes *and* returns parseable JSON, per model. "
             "Variants run most-constrained first; the first passing row is the shape to adopt.",
             "", "| Model | " + " | ".join(n.split(":")[0] for n, _ in VARIANTS) + " |",
             "|---" * (len(VARIANTS) + 1) + "|"]
    for model, rows in results.items():
        cells = []
        for r in rows:
            cells.append("json" if r["json"] else ("routed, no json" if r["routed"] else "404"))
        lines.append(f"| {DISPLAY.get(model, model)} | " + " | ".join(cells) + " |")
    lines += ["", "## Variants", ""]
    lines += [f"- **{n}**" for n, _ in VARIANTS]
    print("\n".join(lines))

    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
