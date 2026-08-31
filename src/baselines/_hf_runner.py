"""HF Inference runner (OpenAI-SDK compatible via HF router).

Mirrors the `run(...)` signature of `_openai_runner.py` so the calling
scripts only differ in which runner they import. Uses the OpenAI Python
SDK pointed at `https://router.huggingface.co/v1` with chat.completions;
the model id is suffixed with `:<provider>` (e.g. `:novita`) to pin a
specific inference provider on HF's router.

`reasoning` is accepted but ignored — open-weight chat models on HF
Inference don't expose an OpenAI-style reasoning-effort knob.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


def parse_bool(text: str) -> Optional[str]:
    if not text:
        return None
    token = text.strip().split()[0].lower().strip('".,;:!()[]{}')
    return token if token in ("true", "false") else None


def call_with_retries(fn: Callable, max_retries: int = 5, base_delay: float = 0.6):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            print(f"  retry {attempt + 1}/{max_retries}: {type(e).__name__}: {e} ({delay:.1f}s)")
            time.sleep(delay)


def save_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return {item["id"]: item for item in data if "id" in item}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    provider: str = "novita",
) -> None:
    """Run the verifier over `items` via HF Inference router.

    `model` is the HF model id (e.g. `meta-llama/Llama-3.3-70B-Instruct`).
    `provider` is the HF Inference provider slug appended as `model:provider`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output_path)
    todo = [it for it in items if it["id"] not in existing]
    print(f"total={len(items)} done={len(existing)} todo={len(todo)}")
    if not todo:
        return

    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HF_TOKEN not set in environment (.env)")

    client = OpenAI(base_url=HF_ROUTER_BASE_URL, api_key=api_key)
    routed_model = f"{model}:{provider}"

    def process_one(item: dict) -> tuple[str, dict]:
        def _call():
            return client.chat.completions.create(
                model=routed_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_message(item)},
                ],
                temperature=temperature,
                max_tokens=1000,
            )

        try:
            resp = call_with_retries(_call)
            raw = (resp.choices[0].message.content or "").strip()
            parsed = parse_bool(raw) or "false"
            usage = getattr(resp, "usage", None)
            out = {
                "id":             item["id"],
                "prediction":     parsed == "true",
                "prediction_raw": parsed,
                "model":          model,
                "provider":       provider,
                "reasoning":      reasoning,
                "usage":          {
                    "prompt_tokens":     getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens":      getattr(usage, "total_tokens", None),
                } if usage else None,
            }
        except Exception as e:
            out = {
                "id":             item["id"],
                "prediction":     False,
                "prediction_raw": "false",
                "model":          model,
                "provider":       provider,
                "reasoning":      reasoning,
                "error":          str(e),
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
