"""Unified iterative verifier.

One loop, one prompt. At each iteration the verifier picks one of:

  - final_answer   → verdict + confidence + reasoning
  - abstain        → no verdict (unless `disable_abstain` forces one)
  - search_query   → retrieve from chosen source (web / pubmed), then loop

No upfront bucket routing; parametric/abstain/source-selection all happen
inside the loop. Confidence is reported on every final_answer (calibrated
0.00–1.00) so a downstream threshold can flag low-confidence verdicts.
"""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ..retriever import Retriever
from .prompts import FORCE_FINAL_SYSTEM, SYSTEM_PROMPT, user_message

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_ITERS = 3
DEFAULT_RETRIEVAL_K = 3
REQUEST_TIMEOUT = 180.0
MAX_TOKENS = 8000
VALID_SOURCES = ("web", "pubmed")


@dataclass
class VerifierResult:
    prediction: bool | None         # None if abstained
    abstained: bool
    reasoning: str
    iterations: int                 # total LLM calls (incl. forced-final if any)
    confidence: float | None = None # model-reported confidence in [0,1] (None if abstained / not provided)
    evidence: list[dict] = field(default_factory=list)  # one entry per search step
    sources_used: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""
    error: str | None = None
    cap_hit: bool = False           # True if we had to force a decision


class Verifier:
    def __init__(self, model: str = DEFAULT_MODEL,
                 max_iters: int = DEFAULT_MAX_ITERS,
                 retrieval_k: int = DEFAULT_RETRIEVAL_K,
                 disable_abstain: bool = False,
                 use_full_text: bool = False,
                 use_query: bool = False,
                 use_shared_context: bool = True,
                 reasoning_effort: str | None = None):
        self.model = model
        self.max_iters = max_iters
        self.retrieval_k = retrieval_k
        self.disable_abstain = disable_abstain
        self.use_full_text = use_full_text
        self.use_query = use_query
        self.use_shared_context = use_shared_context
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=0)
        self._retrievers: dict[str, Retriever] = {}

    def _get_retriever(self, source: str) -> Retriever:
        if source not in self._retrievers:
            self._retrievers[source] = Retriever(index=source)
        return self._retrievers[source]

    # ------------------------------------------------------------------
    def __call__(self, *, snippet: str,
                 subset: str = "consumer",
                 shared_context: dict | None = None,
                 full_text: str | None = None,
                 query: str | None = None,
                 cache_key: str | None = None) -> VerifierResult:
        ft = full_text if self.use_full_text else None
        q_in = query if self.use_query else None
        sc = shared_context if self.use_shared_context else None
        self._current_cache_key = cache_key
        evidence: list[dict] = []
        sources_used: list[str] = []
        usage_total = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}

        for it in range(1, self.max_iters + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(
                    snippet, subset, sc, evidence, ft,
                    query=q_in, include_shared_context=self.use_shared_context,
                )},
            ]
            try:
                _raw, parsed, usage = self._chat(messages)
            except Exception as e:
                return VerifierResult(
                    prediction=None, abstained=False, reasoning="",
                    iterations=it - 1, evidence=evidence,
                    sources_used=sources_used, usage=usage_total,
                    model=self.model, error=f"step_failed: {e}",
                )
            for k in usage_total:
                usage_total[k] += usage.get(k, 0)

            if "final_answer" in parsed:
                conf = parsed.get("confidence")
                if conf is not None:
                    try:
                        conf = float(conf)
                    except Exception:
                        conf = None
                return VerifierResult(
                    prediction=bool(parsed["final_answer"]),
                    abstained=False,
                    reasoning=parsed.get("reasoning", ""),
                    confidence=conf,
                    iterations=it, evidence=evidence, sources_used=sources_used,
                    usage=usage_total, model=self.model,
                )
            if parsed.get("abstain"):
                if self.disable_abstain:
                    break  # force a verdict via the forced-final call below
                return VerifierResult(
                    prediction=None, abstained=True,
                    reasoning=parsed.get("reasoning", ""),
                    iterations=it, evidence=evidence, sources_used=sources_used,
                    usage=usage_total, model=self.model,
                )
            if "search_query" in parsed:
                sq = parsed["search_query"]
                src = parsed.get("source", "web").lower().strip()
                if src not in VALID_SOURCES:
                    src = "web"
                try:
                    hits = self._get_retriever(src).search(sq, k=self.retrieval_k)
                    evidence.append({
                        "query": sq,
                        "source": src,
                        "hits": [{"title": h.title, "text": h.text,
                                  "source_url": h.source_url} for h in hits],
                    })
                    sources_used.append(src)
                except Exception as e:
                    evidence.append({"query": sq, "source": src, "hits": [], "error": str(e)})
                continue
            # malformed output → force final
            break

        # Iteration cap hit / malformed / abstain-disabled override: force a decision.
        force_msgs = [
            {"role": "system", "content": FORCE_FINAL_SYSTEM},
            {"role": "user", "content": user_message(
                snippet, subset, sc, evidence, ft,
                query=q_in, include_shared_context=self.use_shared_context,
            )},
        ]
        try:
            _raw, parsed, usage = self._chat(force_msgs)
            for k in usage_total:
                usage_total[k] += usage.get(k, 0)
        except Exception as e:
            return VerifierResult(
                prediction=None, abstained=False, reasoning="force_failed",
                iterations=self.max_iters, evidence=evidence,
                sources_used=sources_used, usage=usage_total,
                model=self.model, error=str(e), cap_hit=True,
            )

        if parsed.get("abstain") and not self.disable_abstain:
            return VerifierResult(
                prediction=None, abstained=True,
                reasoning=parsed.get("reasoning", ""),
                iterations=self.max_iters + 1, evidence=evidence,
                sources_used=sources_used,
                usage=usage_total, model=self.model, cap_hit=True,
            )
        conf = parsed.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
            except Exception:
                conf = None
        return VerifierResult(
            prediction=bool(parsed.get("final_answer", False)),
            abstained=False,
            reasoning=parsed.get("reasoning", ""),
            confidence=conf,
            iterations=self.max_iters + 1, evidence=evidence,
            sources_used=sources_used,
            usage=usage_total, model=self.model, cap_hit=True,
        )

    # ------------------------------------------------------------------
    def _chat(self, messages):
        """prompt_cache_key hints OpenAI to route to the same cache shard for
        all calls sharing that key (e.g., all snippets of one entry), which
        improves prefix-cache hit rates across snippets sharing the per-entry
        context block.
        """
        kw = dict(
            model=self.model,
            max_completion_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=messages,
        )
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        ck = getattr(self, "_current_cache_key", None)
        if ck:
            kw["prompt_cache_key"] = ck
        resp = self.client.chat.completions.create(**kw)
        raw = resp.choices[0].message.content or ""
        parsed = _extract_json(raw)
        usage = resp.usage
        cached = 0
        if usage and getattr(usage, "prompt_tokens_details", None):
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        return raw, parsed, {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cached_input_tokens": cached,
        }


def _extract_json(text: str) -> dict:
    """Parse the FIRST JSON object in `text`. Tolerates trailing data, code
    fences, or multiple objects (the verifier occasionally emits a reasoning
    blob plus a JSON verdict)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object: {text[:200]}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj
