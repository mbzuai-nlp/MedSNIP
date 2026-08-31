"""Snippet processor — two modes.

  - **atom-to-snippet** (default): two LLM calls.
      1. extract: query + numbered sentences  → atoms + shared_context
      2. cluster: query + atoms + shared_context → snippets
      Mirrors the human annotation workflow.

  - **snippet-direct**: one LLM call.
      query + numbered sentences → snippets + shared_context
      Cheaper; slightly weaker on consumer-style dense responses.

Input: a `full_text` (the model response to fact-check) and an optional
`query` (the user's question that elicited the response).
Output: snippets + shared_context.

Default model: OpenAI `gpt-5.4`. Subset (consumer / vignette) is
auto-detected from the query length when a query is supplied; with no
query the caller can pass `subset` explicitly; otherwise we default to
`vignette` (the heavier prompts — they degrade gracefully on consumer
input).
"""

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import (
    CLUSTER_SYSTEM_PROMPTS,
    EXTRACT_SYSTEM_PROMPTS,
    SNIPPET_DIRECT_SYSTEM_PROMPTS,
    classify_subset,
    few_shot,
    user_message_atom,
    user_message_extract,
    user_message_snippet_direct,
)
from .sentence_utils import split_sentences

# Repo root is three parents above this file (src/medsnip/snippet_processor/ → repo root).
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DEFAULT_MODEL = "gpt-5.4"
MAX_TOKENS = 8000
REQUEST_TIMEOUT = 90.0
MODES = ("atom-to-snippet", "snippet-direct")
DEFAULT_MODE = "atom-to-snippet"


class SnippetProcessor:
    """`(full_text, query=None) → snippets`.

    Mode chooses the algorithm:
      - `atom-to-snippet` (default): 2 LLM calls — extract atoms then cluster.
      - `snippet-direct`           : 1 LLM call — sentences directly to snippets.
    """

    def __init__(self, model: str = DEFAULT_MODEL, mode: str = DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.model = model
        self.mode = mode
        self.client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=0)

    def __call__(self, *, full_text: str, query: str | None = None,
                 subset: str | None = None, max_retries: int = 2) -> dict:
        if subset is None:
            subset = classify_subset(query) if query else "vignette"
        query = query or ""
        sentences = split_sentences(full_text)

        if self.mode == "snippet-direct":
            return self._run_snippet_direct(query, sentences, subset, max_retries)
        return self._run_atom_to_snippet(query, sentences, subset, max_retries)

    # ------------------------------------------------------------------
    def _run_atom_to_snippet(self, query, sentences, subset, max_retries):
        extract_result = self._extract(query, sentences, subset, max_retries)
        generated_atoms = extract_result["atoms"]
        shared_context = extract_result["shared_context"]
        cluster_result = self._cluster(query, generated_atoms, shared_context,
                                       subset, max_retries)

        total_usage = _sum_usage([extract_result["_usage"], cluster_result["_usage"]])
        return {
            "subset":          subset,
            "mode":            "atom-to-snippet",
            "shared_context":  shared_context,
            "snippets":        cluster_result["parsed"]["snippets"],
            "generated_atoms": generated_atoms,
            "sentences":       sentences,
            "_raw_extract":    extract_result["_raw"],
            "_raw_cluster":    cluster_result["_raw"],
            "_usage_steps": [
                {"step": "extract", **extract_result["_usage"]},
                {"step": "cluster", **cluster_result["_usage"]},
            ],
            "_usage":          total_usage,
            "_model":          self.model,
        }

    def _run_snippet_direct(self, query, sentences, subset, max_retries):
        fs_user, fs_assistant = few_shot(subset, "snippet_direct")
        messages = [
            {"role": "system", "content": SNIPPET_DIRECT_SYSTEM_PROMPTS[subset]},
            {"role": "user", "content": fs_user},
            {"role": "assistant", "content": fs_assistant},
            {"role": "user", "content": user_message_snippet_direct(query, sentences)},
        ]
        raw, parsed, usage = self._chat(messages, max_retries=max_retries, require_key="snippets")
        return {
            "subset":          subset,
            "mode":            "snippet-direct",
            "shared_context":  parsed.get("shared_context", {}),
            "snippets":        parsed["snippets"],
            "generated_atoms": None,
            "sentences":       sentences,
            "_raw_cluster":    raw,
            "_usage_steps":    [{"step": "snippet_direct", **usage}],
            "_usage":          usage,
            "_model":          self.model,
        }

    # ------------------------------------------------------------------
    def _extract(self, query: str, sentences: list[str], subset: str,
                 max_retries: int) -> dict:
        fs_user, fs_assistant = few_shot(subset, "extract")
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPTS[subset]},
            {"role": "user", "content": fs_user},
            {"role": "assistant", "content": fs_assistant},
            {"role": "user", "content": user_message_extract(query, sentences)},
        ]
        raw, parsed, usage = self._chat(messages, max_retries=max_retries, require_key="atoms")
        atoms = []
        for i, a in enumerate(parsed["atoms"], 1):
            atoms.append({
                "index":            int(a.get("index", i)),
                "text":             a["text"],
                "source_sentences": [int(s) for s in a.get("source_sentences", [])],
            })
        return {
            "atoms":          atoms,
            "shared_context": parsed.get("shared_context", {}),
            "_raw":           raw,
            "_usage":         usage,
        }

    def _cluster(self, query: str, atoms: list[dict], shared_context: dict,
                 subset: str, max_retries: int) -> dict:
        fs_user, fs_assistant = few_shot(subset, "atom")  # cluster reuses atom few-shot
        messages = [
            {"role": "system", "content": CLUSTER_SYSTEM_PROMPTS[subset]},
            {"role": "user", "content": fs_user},
            {"role": "assistant", "content": fs_assistant},
            {"role": "user", "content": user_message_atom(query, atoms, shared_context)},
        ]
        raw, parsed, usage = self._chat(messages, max_retries=max_retries, require_key="snippets")
        return {"_raw": raw, "parsed": parsed, "_usage": usage}

    # ------------------------------------------------------------------
    def _chat(self, messages, max_retries: int, require_key: str):
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                raw = resp.choices[0].message.content or ""
                parsed = _extract_json(raw)
                if require_key not in parsed:
                    raise ValueError(f"response missing '{require_key}'")
                usage = resp.usage
                cached = 0
                if usage and getattr(usage, "prompt_tokens_details", None):
                    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                return raw, parsed, {
                    "input_tokens":        usage.prompt_tokens if usage else 0,
                    "output_tokens":       usage.completion_tokens if usage else 0,
                    "cached_input_tokens": cached,
                }
            except Exception as e:
                if attempt < max_retries:
                    print(f"    retry {attempt + 1}/{max_retries}: "
                          f"{type(e).__name__}: {str(e)[:120]}")
                    time.sleep(2 ** attempt)
                    continue
                raise


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object found in response: {text[:200]}")
    return json.loads(text[start:end + 1])


def _sum_usage(usages):
    out = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    for u in usages:
        out["input_tokens"]        += u.get("input_tokens", 0)
        out["output_tokens"]       += u.get("output_tokens", 0)
        out["cached_input_tokens"] += u.get("cached_input_tokens", 0)
    return out
