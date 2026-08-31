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

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

DEFAULT_MODEL = "gpt-4o"
DEFAULT_PROVIDER = "openai"
# Extended-thinking budgets for Anthropic. Map our reasoning_effort flag onto
# token budgets.
ANTHROPIC_THINK_BUDGET = {
    "minimal": 1024, "low": 2048, "medium": 4096, "high": 8000,
}
DEFAULT_MAX_ITERS = 5
DEFAULT_RETRIEVAL_K = 3
REQUEST_TIMEOUT = 180.0
REQUEST_TIMEOUT_ANTHROPIC = 600.0  # Anthropic extended thinking can take >180s
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
                 reasoning_effort: str | None = None,
                 min_confidence: float | None = None,
                 provider: str = DEFAULT_PROVIDER):
        self.model = model
        self.max_iters = max_iters
        self.retrieval_k = retrieval_k
        self.disable_abstain = disable_abstain
        self.use_full_text = use_full_text
        self.use_query = use_query
        self.use_shared_context = use_shared_context
        self.reasoning_effort = reasoning_effort
        # Phase B: in-loop confidence gate. If a final_answer is emitted with
        # confidence < min_confidence and iteration budget remains, re-prompt
        # the model to search more (or abstain).
        self.min_confidence = min_confidence
        self.provider = provider
        if provider == "openai":
            self.client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=0)
        elif provider == "anthropic":
            if Anthropic is None:
                raise SystemExit("anthropic SDK not installed. `uv add anthropic`")
            self.client = Anthropic(timeout=REQUEST_TIMEOUT_ANTHROPIC, max_retries=0)
        else:
            raise ValueError(f"unknown provider: {provider}")
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
                 prior: bool | None = None,
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
                    prior=prior,
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
                # Guardrail: do not allow final_answer=true on iter 1 with NO
                # evidence retrieved yet. The model has been shortcutting on
                # plausible-sounding vague claims (see wrong_cases.json FN set).
                # Force a search before accepting True.
                if (bool(parsed["final_answer"]) is True
                        and not evidence
                        and it == 1):
                    forced_search = {
                        "role": "user",
                        "content": (
                            "REMINDER: you cannot emit `final_answer: true` on "
                            "the first turn without retrieving any evidence. "
                            "Issue ONE refutation-biased search_query now."
                        ),
                    }
                    try:
                        _raw, parsed, usage = self._chat(messages + [
                            {"role": "assistant",
                             "content": json.dumps(parsed)},
                            forced_search,
                        ])
                        for k in usage_total:
                            usage_total[k] += usage.get(k, 0)
                    except Exception as e:
                        return VerifierResult(
                            prediction=None, abstained=False, reasoning="",
                            iterations=it, evidence=evidence,
                            sources_used=sources_used, usage=usage_total,
                            model=self.model, error=f"forced_search_failed: {e}",
                        )
                    # fall through; the new `parsed` will be handled below

                # Phase B: low-confidence guard. If the model commits with
                # confidence below threshold AND iteration budget remains,
                # re-prompt asking for one more search (or an abstain).
                if (self.min_confidence is not None
                        and "final_answer" in parsed
                        and parsed.get("confidence") is not None
                        and float(parsed["confidence"]) < self.min_confidence
                        and it < self.max_iters):
                    low_conf_msg = {
                        "role": "user",
                        "content": (
                            f"REMINDER: your last verdict had "
                            f"confidence={float(parsed['confidence']):.2f}, below "
                            f"the threshold {self.min_confidence:.2f}. Either "
                            f"issue ONE more refutation-biased search_query to "
                            f"raise confidence, OR emit `abstain` if you "
                            f"genuinely cannot decide. Do NOT recommit with "
                            f"the same low confidence."
                        ),
                    }
                    try:
                        _raw, parsed, usage = self._chat(messages + [
                            {"role": "assistant",
                             "content": json.dumps(parsed)},
                            low_conf_msg,
                        ])
                        for k in usage_total:
                            usage_total[k] += usage.get(k, 0)
                    except Exception:
                        # Fall through to accepting the original low-conf
                        # verdict rather than crashing.
                        pass
                    # Falls through to the if/elif chain below; new `parsed`
                    # may be search_query (handled below) / abstain / final.

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
        """Dispatch to provider-specific chat call.

        For OpenAI: `prompt_cache_key` hints route-to-cache-shard. JSON enforced
        via `response_format`.

        For Anthropic: cache control markers attached to system + early
        per-entry content. JSON enforced via prompt only. Extended thinking
        opt-in when `reasoning_effort` is set.
        """
        if self.provider == "anthropic":
            return self._chat_anthropic(messages)
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

    def _chat_anthropic(self, messages):
        """Anthropic call. `messages` follows OpenAI's [{"role","content"},...]
        format; we split system → top-level + user/assistant → messages list.
        Cache control: tag the system block as ephemeral so prefix caches.
        Extended thinking: opt in only when reasoning_effort is set.
        """
        # Split system from user/assistant turns
        system_blocks = []
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                # cache_control on system makes Anthropic cache the system prefix.
                system_blocks.append({
                    "type": "text",
                    "text": m["content"],
                    "cache_control": {"type": "ephemeral"},
                })
            else:
                chat_msgs.append({"role": m["role"], "content": m["content"]})

        kw = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            messages=chat_msgs,
        )
        if self.reasoning_effort:
            budget = ANTHROPIC_THINK_BUDGET.get(self.reasoning_effort, 4096)
            kw["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Anthropic requires max_tokens > thinking budget. Bump if needed.
            kw["max_tokens"] = max(MAX_TOKENS, budget + 2000)
            # When thinking is enabled, temperature must be 1.0; we leave default.

        # Use streaming so long extended-thinking calls don't hit per-request
        # timeouts at proxies. Anthropic docs explicitly recommend streaming
        # for calls > a few minutes. We accumulate the final message and
        # return the same shape as the non-streaming path.
        with self.client.messages.stream(**kw) as stream:
            for _ in stream:
                pass  # drain the event stream
            resp = stream.get_final_message()

        # Output: list of content blocks. Skip thinking blocks; grab first text.
        raw = ""
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                raw = block.text
                break
        parsed = _extract_json(raw)
        usage = resp.usage
        # Anthropic usage: input_tokens, output_tokens, cache_creation_input_tokens,
        # cache_read_input_tokens. Map onto our OpenAI-style schema.
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return raw, parsed, {
            "input_tokens": (usage.input_tokens + cache_read + cache_create),
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": cache_read,
            "cache_creation_tokens": cache_create,
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
