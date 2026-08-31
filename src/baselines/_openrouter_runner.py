"""OpenRouter runner (OpenAI-SDK compatible).

Mirrors the `run(...)` signature of `_openai_runner.py` and `_hf_runner.py`
so the calling scripts only differ in which runner they import. Uses the
OpenAI Python SDK pointed at `https://openrouter.ai/api/v1` with
chat.completions, which is the one call shape every model on the router
supports.

Two things this runner records that the older runners do not:

  - `upstream_provider`: which backend OpenRouter actually routed to.
    Open-weight models are served by several providers at different
    quantizations, so a run is not reproducible without it.
  - `usage.cost`: OpenRouter's own accounting, requested via
    `usage: {include: true}`. This removes the tiktoken-estimate step that
    `validate_baseline_cost.py` had to correct for.

`reasoning` is forwarded as OpenRouter's `reasoning: {effort: ...}` for
models that advertise support; for the rest it is recorded but not sent,
and `temperature` is used instead.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from openai import OpenAI

from ._hf_runner import call_with_retries, load_existing, parse_bool, save_json_atomic

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models that accept OpenRouter's `reasoning` block. Everything else gets
# `temperature` instead — sending both is rejected by some upstreams.
REASONING_MODELS = (
    "openai/gpt-5",
    "openai/gpt-oss",
    "google/gemma-4",
)


def supports_reasoning(model: str) -> bool:
    return model.startswith(REASONING_MODELS)


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment (.env)")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=300.0)


def build_kwargs(
    model: str,
    messages: list[dict],
    reasoning: Optional[str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Shared request shape, so the verifier and the decomposer stay in sync."""
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    extra: dict[str, Any] = {
        # Ask OpenRouter for its own cost accounting on every response.
        "usage": {"include": True},
        # Only route to upstreams that honour the parameters we send, so a
        # provider that silently drops response_format can't corrupt a run.
        "provider": {"require_parameters": True},
    }
    if reasoning and supports_reasoning(model):
        extra["reasoning"] = {"effort": reasoning}
    else:
        kwargs["temperature"] = temperature
    kwargs["extra_body"] = extra
    return kwargs


def usage_row(resp: Any) -> dict | None:
    usage = getattr(resp, "usage", None)
    if not usage:
        return None
    extra = getattr(usage, "model_extra", None) or {}
    return {
        "prompt_tokens":     getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens":      getattr(usage, "total_tokens", None),
        "cost":              extra.get("cost"),
    }


def upstream_provider(resp: Any) -> str | None:
    """Which backend OpenRouter actually served this call from."""
    direct = getattr(resp, "provider", None)
    if direct:
        return direct
    return (getattr(resp, "model_extra", None) or {}).get("provider")


def run(
    items: list[dict],
    output_path: Path,
    build_user_message: Callable[[dict], str],
    system_prompt: str,
    model: str,
    reasoning: Optional[str],
    temperature: float,
    workers: int,
    save_every: int,
    extra_row_fields: Callable[[dict], dict] | None = None,
) -> None:
    """Run the verifier over `items` via OpenRouter. Each `item` needs an `id`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output_path)
    todo = [it for it in items if it["id"] not in existing]
    print(f"total={len(items)} done={len(existing)} todo={len(todo)}")
    if not todo:
        return

    client = make_client()

    def process_one(item: dict) -> tuple[str, dict]:
        def _call():
            return client.chat.completions.create(
                **build_kwargs(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_message(item)},
                    ],
                    reasoning=reasoning,
                    temperature=temperature,
                    max_tokens=1000,
                )
            )

        try:
            resp = call_with_retries(_call)
            raw = (resp.choices[0].message.content or "").strip()
            parsed = parse_bool(raw) or "false"
            out = {
                "id":                item["id"],
                "prediction":        parsed == "true",
                "prediction_raw":    parsed,
                "model":             model,
                "route":             "openrouter",
                "upstream_provider": upstream_provider(resp),
                "reasoning":         reasoning,
                "usage":             usage_row(resp),
            }
        except Exception as e:
            out = {
                "id":                item["id"],
                "prediction":        False,
                "prediction_raw":    "false",
                "model":             model,
                "route":             "openrouter",
                "upstream_provider": None,
                "reasoning":         reasoning,
                "error":             str(e),
            }
        if extra_row_fields:
            out.update(extra_row_fields(item))
        return item["id"], out

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, it) for it in todo]
        for fut in as_completed(futures):
            row_id, row = fut.result()
            existing[row_id] = row
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}")
            if done % save_every == 0:
                save_json_atomic(output_path, list(existing.values()))

    save_json_atomic(output_path, list(existing.values()))
    print(f"wrote {len(existing)} rows → {output_path}")
