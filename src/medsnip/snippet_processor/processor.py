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

Default route: OpenRouter, default model `openai/gpt-5.4`. Subset (consumer / vignette) is
auto-detected from the query length when a query is supplied; with no
query the caller can pass `subset` explicitly; otherwise we default to
`vignette` (the heavier prompts — they degrade gracefully on consumer
input).
"""

import json
import os
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

DEFAULT_MODEL = "openai/gpt-5.4"
MAX_TOKENS = 8000
REQUEST_TIMEOUT = 300.0  # bumped from 90s — long-context entries with reasoning=high need more time
MODES = ("atom-to-snippet", "snippet-direct")
DEFAULT_MODE = "atom-to-snippet"
ROUTES = ("openrouter", "openai")
DEFAULT_ROUTE = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models that accept OpenRouter's `reasoning` block; the rest get `temperature`.
REASONING_MODELS = ("openai/gpt-5", "openai/gpt-oss", "google/gemma-4")


class SnippetProcessor:
    """`(full_text, query=None) → snippets`.

    Mode chooses the algorithm:
      - `atom-to-snippet` (default): 2 LLM calls — extract atoms then cluster.
      - `snippet-direct`           : 1 LLM call — sentences directly to snippets.
    """

    def __init__(self, model: str = DEFAULT_MODEL, mode: str = DEFAULT_MODE,
                 reasoning_effort: str | None = None, route: str = DEFAULT_ROUTE,
                 provider_pin: str | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        self.model = model
        self.mode = mode
        self.reasoning_effort = reasoning_effort
        self.route = route
        # Pin a single upstream (fallbacks off) so a run cannot silently
        # straddle providers serving different quantizations.
        self.provider_pin = provider_pin
        # Which upstream OpenRouter actually served each call. Open-weight
        # models are offered by several providers at different quantizations,
        # so the run is not reproducible without recording this.
        self.upstream_providers: set[str] = set()
        if route == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set in environment (.env)")
            self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key,
                                 timeout=REQUEST_TIMEOUT, max_retries=0)
        else:
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
            "_model":           self.model,
            "_reasoning_effort": self.reasoning_effort,
            "_route":            self.route,
            "_provider_pin":     self.provider_pin,
            "_upstream_providers": sorted(self.upstream_providers),
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
            "_model":           self.model,
            "_reasoning_effort": self.reasoning_effort,
            "_route":            self.route,
            "_provider_pin":     self.provider_pin,
            "_upstream_providers": sorted(self.upstream_providers),
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
                kwargs = dict(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                # `max_completion_tokens` is OpenAI-specific. Non-OpenAI
                # upstreams on OpenRouter reject it, and under
                # `require_parameters` that fails routing outright — which is
                # what blocked all four open-weight decomposers until we
                # probed it (data/12-decomposer/param_probe.md).
                if self.route == "openrouter":
                    kwargs["max_tokens"] = MAX_TOKENS
                    extra = {
                        "usage": {"include": True},
                        # Refuse upstreams that would silently drop
                        # response_format — a dropped JSON mode corrupts the
                        # whole run rather than failing loudly.
                        "provider": {"require_parameters": True},
                    }
                    if self.provider_pin:
                        extra["provider"]["order"] = [self.provider_pin]
                        extra["provider"]["allow_fallbacks"] = False
                    if self.reasoning_effort and self.model.startswith(REASONING_MODELS):
                        extra["reasoning"] = {"effort": self.reasoning_effort}
                    kwargs["extra_body"] = extra
                else:
                    kwargs["max_completion_tokens"] = MAX_TOKENS
                    if self.reasoning_effort:
                        kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self.client.chat.completions.create(**kwargs)
                provider = getattr(resp, "provider", None) or \
                    (getattr(resp, "model_extra", None) or {}).get("provider")
                if provider:
                    self.upstream_providers.add(provider)
                raw = resp.choices[0].message.content or ""
                parsed = _extract_json(raw)
                if require_key not in parsed:
                    raise ValueError(f"response missing '{require_key}'")
                _validate_records(parsed, require_key)
                usage = resp.usage
                cached = 0
                if usage and getattr(usage, "prompt_tokens_details", None):
                    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                return raw, parsed, {
                    "input_tokens":        usage.prompt_tokens if usage else 0,
                    "output_tokens":       usage.completion_tokens if usage else 0,
                    "cached_input_tokens": cached,
                    # OpenRouter's own dollar accounting, when routed there.
                    "cost": (getattr(usage, "model_extra", None) or {}).get("cost") if usage else None,
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


# Keys a record must carry to be usable downstream. `pattern` and `notes` are
# optional, so a model that omits them is not penalised.
#
# Provenance is required but its key depends on the mode: Mode 1 snippets point
# back at atoms (`source_claims`), Mode 2 snippets point straight at sentences
# (`source_sentences`). Demanding one specific key would reject the other mode's
# perfectly valid output.
_SNIPPET_REQUIRED = ("output",)
_SNIPPET_PROVENANCE = ("source_claims", "source_sentences")
_ATOM_REQUIRED = ("text",)


def _coerce_record(item):
    """Repair the flattened key-value-pair shape some models emit.

    Weaker decomposers sometimes return a record as a flat array,
    `["source_claims", [1], "output", "text", ...]`, rather than an object.
    The pairs are intact, so this is losslessly recoverable without another
    API call — which is both cheaper than resampling and truer to what the
    model actually produced.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, list) and len(item) % 2 == 0:
        keys = item[0::2]
        if all(isinstance(k, str) for k in keys):
            return dict(zip(keys, item[1::2]))
    return item


def _validate_records(parsed: dict, key: str) -> None:
    """Reject responses whose contents are unusable, not merely present.

    Checking only that `key` exists lets a structurally wrong payload pass as
    a success and corrupt the run silently. Repair what is recoverable first,
    in place, then insist the result is actually readable.
    """
    records = parsed.get(key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"'{key}' is empty or not a list")

    parsed[key] = [_coerce_record(r) for r in records]
    required = _SNIPPET_REQUIRED if key == "snippets" else _ATOM_REQUIRED
    for i, r in enumerate(parsed[key]):
        if not isinstance(r, dict):
            raise ValueError(f"{key}[{i}] is {type(r).__name__}, not an object")
        missing = [f for f in required if f not in r]
        if missing:
            raise ValueError(f"{key}[{i}] missing {missing}; keys={sorted(r)[:6]}")
        if key == "snippets" and not any(f in r for f in _SNIPPET_PROVENANCE):
            raise ValueError(f"{key}[{i}] has no provenance "
                             f"({' or '.join(_SNIPPET_PROVENANCE)}); "
                             f"keys={sorted(r)[:6]}")


def _sum_usage(usages):
    out = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    for u in usages:
        out["input_tokens"]        += u.get("input_tokens", 0)
        out["output_tokens"]       += u.get("output_tokens", 0)
        out["cached_input_tokens"] += u.get("cached_input_tokens", 0)
    return out
