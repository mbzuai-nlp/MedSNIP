"""The six models forming the decomposer x verifier matrix.

Every model appears on both axes, so the diagonal is the fully-local case
where one model both decomposes and verifies. All ids are OpenRouter ids;
every one of them advertises `response_format` support, which the
decomposition prompts require.
"""
from __future__ import annotations

# Order is strongest-first, matching the row order of Table 3 in the paper.
MATRIX_MODELS: list[str] = [
    "openai/gpt-5.4",
    "openai/gpt-4o",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

# Reasoning effort per model, where the model accepts one. Mirrors the
# paper's verifier configuration: GPT-5.4 at high, everything else default.
REASONING: dict[str, str | None] = {
    "openai/gpt-5.4": "high",
}

# Upstream provider to pin per model, with fallbacks disabled.
#
# The viability pilot showed OpenRouter load-balancing a single model's calls
# across up to five upstreams (Llama-3.3-70B hit Cloudflare, CoreWeave, Crusoe,
# DeepInfra and Parasail in six calls). Providers serve different quantizations,
# so an unpinned row silently mixes backends and is not reproducible.
#
# DeepInfra was the one upstream that served every open-weight model in the
# pilot, so pinning there keeps the four open-weight rows on a single backend.
# The OpenAI models are left unpinned: OpenRouter already routes them to OpenAI
# under our request shape, and pinning would add no information.
PROVIDER_PIN: dict[str, str | None] = {
    "openai/gpt-5.4":                    None,
    "openai/gpt-4o":                     None,
    "google/gemma-4-31b-it":             "deepinfra",
    "openai/gpt-oss-20b":                "deepinfra",
    "meta-llama/llama-3.3-70b-instruct": "deepinfra",
    "meta-llama/llama-3.1-8b-instruct":  "deepinfra",
}

# Concurrent in-flight requests per model.
#
# Tuned to the upstream rather than the model. OpenAI absorbs high concurrency
# comfortably. DeepInfra was the source of the timeout burst in the first
# Gemma run at 10 workers, but those were slow responses rather than rate-limit
# rejections, so more requests in flight raises throughput instead of hurting
# it — the per-call timeout is what protects us.
WORKERS: dict[str, int] = {
    "openai/gpt-5.4":                    24,
    "openai/gpt-4o":                     32,
    "google/gemma-4-31b-it":             32,
    "openai/gpt-oss-20b":                32,
    "meta-llama/llama-3.3-70b-instruct": 32,
    "meta-llama/llama-3.1-8b-instruct":  32,
}
DEFAULT_WORKERS = 24

# Display names for tables, keyed by OpenRouter id.
DISPLAY: dict[str, str] = {
    "openai/gpt-5.4":                    "GPT-5.4-high",
    "openai/gpt-4o":                     "GPT-4o",
    "google/gemma-4-31b-it":             "gemma-4-31B-it",
    "openai/gpt-oss-20b":                "gpt-oss-20b",
    "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B",
    "meta-llama/llama-3.1-8b-instruct":  "Llama-3.1-8B",
}


def slug(model: str, reasoning: str | None = None) -> str:
    """Path-safe directory name, matching the existing `<model>-<effort>` convention."""
    suffix = reasoning if reasoning is not None else (REASONING.get(model) or "none")
    return f"{model.replace('/', '__')}-{suffix}"
