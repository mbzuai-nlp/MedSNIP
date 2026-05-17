"""Shared OpenAI runner used by both atom and snippet baseline scripts.

Calls `client.responses.create` with optional `reasoning={"effort": ...}`
for gpt-5 variants and a plain `temperature` for older models. Threaded
with resume-on-restart based on the output file.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from openai import OpenAI


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
) -> None:
    """Run the verifier over `items`. Each `item` must have an `id`.

    `build_user_message(item)` produces the user content.
    `extra_row_fields(item)` produces extra fields to keep in the output row
    (e.g., snippet_id, subset, split, label_atomic, …).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output_path)
    todo = [it for it in items if it["id"] not in existing]
    print(f"total={len(items)} done={len(existing)} todo={len(todo)}")
    if not todo:
        return

    client = OpenAI()
    reasoning_arg = {"effort": reasoning} if reasoning else None

    def process_one(item: dict) -> tuple[str, dict]:
        def _call():
            kwargs: dict[str, Any] = dict(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_message(item)},
                ],
                max_output_tokens=1000,
            )
            if reasoning_arg:
                kwargs["reasoning"] = reasoning_arg
            else:
                kwargs["temperature"] = temperature
            return client.responses.create(**kwargs)

        try:
            resp = call_with_retries(_call)
            raw = (resp.output_text or "").strip()
            parsed = parse_bool(raw) or "false"
            out = {
                "id":             item["id"],
                "prediction":     parsed == "true",
                "prediction_raw": parsed,
                "model":          model,
                "reasoning":      reasoning,
            }
        except Exception as e:
            out = {
                "id":             item["id"],
                "prediction":     False,
                "prediction_raw": "false",
                "model":          model,
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
