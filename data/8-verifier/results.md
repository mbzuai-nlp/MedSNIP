# Dev results — gt=label_atomic (snippet) / label (atom)

## Systems

**Verifier versions** (iterative loop: retrieve evidence → reason → final_answer / abstain / search; max 5 iters):
- **verifier-v1** — gpt-5.4-high; original iterative verifier with refutation-biased retrieval (serper.dev + PubMed); base prompt only.
- **verifier-v2** — v1 + **anti-example prompt engineering** (concrete "do not reject because…" / "do not accept because…" patterns drawn from v1 failure cases).
- **verifier-v3** — v2 + **in-loop confidence gate** (FIRE-style): if model emits `final_answer` with conf < 0.85, force one more retrieval round; force-final biases toward abstain on residual uncertainty.
- **verifier-v4** — gpt-5-mini with `reasoning_effort=minimal` (cheap-model ablation; same v3 architecture).
- **verifier-v5** — claude-sonnet-4-6 + extended-thinking budget 8000 (cross-family ablation; same v3 architecture).

**Derived strategies** (post-hoc combinations of a verifier's output, no new LLM calls):
- **calibrated-only @conf≥τ (NA dropped)** — keep only verifier verdicts where `confidence ≥ τ` AND not abstained AND not cap-hit; drop the rest from eval (lower coverage).
- **calibrated-ensemble @conf≥τ else baseline** — same threshold rule, but for dropped records fall back to the snippet baseline's prediction (full coverage).
- **hybrid-router** — subset-conditional: consumer snippets → verifier-v2's verdict; vignette snippets → baseline (a "vignette OR-False" variant predicts False if either system says False). ⚠️ dev finding only — inverts on test.

**Baselines** (one shot LLM call per snippet or atom, no retrieval):
- **snippet/full-context** — system + user question + full LLM answer + snippet → "true"/"false".
- **snippet/claim-only** — system + snippet only → "true"/"false".
- **atom/full-context** — same as snippet/full-context but on atomic claims (1078 dev atoms).
- **atom/claim-only** — same as snippet/claim-only on atomic claims.

| # | grain | mode | model | n | acc | F1_F | F1_T | macro | LLM $ | retr $ | base $ | total $ |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | snippet | calibrated-ensemble | v3 @conf>=0.90 else baseline | 494 | 0.826 | 0.566 | 0.891 | 0.728 | $19.97 | $0.60 | $6.65 | $27.22 |
| 2 | snippet | calibrated-only | v3 @conf>=0.85 (NA dropped) | 411 | 0.844 | 0.562 | 0.905 | 0.733 | $19.97 | $0.60 | — | $20.57 |
| 3 | snippet | calibrated-only | v5 @conf>=0.90 (NA dropped) | 294 | 0.915 | 0.561 | 0.953 | 0.757 | $15.24 | $0.21 | — | $15.45 |
| 4 | snippet | calibrated-ensemble | v2 @conf>=0.90 else baseline | 494 | 0.826 | 0.561 | 0.891 | 0.726 | $16.15 | $0.50 | $6.65 | $23.30 |
| 5 | snippet | calibrated-only | v2 @conf>=0.85 (NA dropped) | 378 | 0.849 | 0.558 | 0.909 | 0.734 | $16.15 | $0.50 | — | $16.65 |
| 6 | snippet | calibrated-ensemble | v3 @conf>=0.85 else baseline | 494 | 0.822 | 0.551 | 0.889 | 0.720 | $19.97 | $0.60 | $6.65 | $27.22 |
| 7 | snippet | calibrated-ensemble | v2 @conf>=0.85 else baseline | 494 | 0.818 | 0.550 | 0.886 | 0.718 | $16.15 | $0.50 | $6.65 | $23.30 |
| 8 | snippet | hybrid-router | consumer→verifier-v2, vignette OR-False | 494 | 0.810 | 0.539 | 0.880 | 0.710 | $16.15 | $0.50 | $6.65 | $23.30 |
| 9 | snippet | calibrated-ensemble | v5 @conf>=0.85 else baseline | 494 | 0.832 | 0.536 | 0.897 | 0.717 | $15.24 | $0.21 | $6.65 | $22.10 |
| 10 | snippet | calibrated-only | v5 @conf>=0.85 (NA dropped) | 423 | 0.856 | 0.534 | 0.915 | 0.725 | $15.24 | $0.21 | — | $15.45 |
| 11 | snippet | hybrid-router | consumer→verifier-v2, vignette→baseline | 494 | 0.816 | 0.533 | 0.885 | 0.709 | $16.15 | $0.50 | $6.65 | $23.30 |
| 12 | snippet | calibrated-ensemble | v5 @conf>=0.90 else baseline | 494 | 0.822 | 0.532 | 0.890 | 0.711 | $15.24 | $0.21 | $6.65 | $22.10 |
| 13 | snippet | verifier-v5 | claude-sonnet-4-6 (extended thinking + conf-gate) | 478 | 0.839 | 0.528 | 0.903 | 0.715 | $15.24 | $0.21 | — | $15.45 |
| 14 | snippet | verifier-v3 | gpt-5.4-high (anti-examples + conf-gate) | 474 | 0.831 | 0.518 | 0.898 | 0.708 | $19.97 | $0.60 | — | $20.57 |
| 15 | snippet | full-context | gpt-5.4-high | 494 | 0.816 | 0.518 | 0.886 | 0.702 | — | — | $6.65 | $6.65 |
| 16 | snippet | calibrated-ensemble | v4 @conf>=0.90 else baseline | 494 | 0.822 | 0.506 | 0.891 | 0.699 | $0.64 | $0.30 | $6.65 | $7.59 |
| 17 | snippet | verifier-v2 | gpt-5.4-high (anti-examples) | 474 | 0.825 | 0.491 | 0.894 | 0.693 | $16.15 | $0.50 | — | $16.65 |
| 18 | atom | full-context | gpt-5.4-high | 1078 | 0.890 | 0.485 | 0.938 | 0.712 | — | — | $14.43 | $14.43 |
| 19 | snippet | verifier-v1 | gpt-5.4-high | 480 | 0.792 | 0.484 | 0.870 | 0.677 | $11.22 | $0.31 | — | $11.53 |
| 20 | snippet | calibrated-only | v4 @conf>=0.90 (NA dropped) | 333 | 0.874 | 0.432 | 0.929 | 0.681 | $0.64 | $0.30 | — | $0.94 |
| 21 | snippet | calibrated-ensemble | v4 @conf>=0.85 else baseline | 494 | 0.822 | 0.429 | 0.894 | 0.661 | $0.64 | $0.30 | $6.65 | $7.59 |
| 22 | snippet | claim-only | gpt-5.4-high | 494 | 0.729 | 0.407 | 0.824 | 0.616 | — | — | $6.02 | $6.02 |
| 23 | snippet | calibrated-only | v4 @conf>=0.85 (NA dropped) | 460 | 0.822 | 0.359 | 0.896 | 0.628 | $0.64 | $0.30 | — | $0.94 |
| 24 | snippet | verifier-v4 | gpt-5-mini-minimal (anti-examples + conf-gate) | 476 | 0.817 | 0.356 | 0.893 | 0.625 | $0.64 | $0.30 | — | $0.94 |
| 25 | atom | claim-only | gpt-5.4-high | 1078 | 0.772 | 0.356 | 0.861 | 0.609 | — | — | $13.10 | $13.10 |

**calibrated-ensemble / calibrated-only inspiration**:
- Abstain-as-its-own-class — MedHallu (EMNLP 2025), reports +38% relative F1 from a "not sure" bucket
- Three-label output {supported / not-supported / not-addressed} — VeriFact (NEJM AI 2025)
- Confidence-gated stopping — FIRE (NAACL Findings 2025)
- Baseline fallback after NA — novel synthesis (no direct prior; closest is cascade / selective-prediction abstain-and-defer)

See [survey.md](survey.md) for full citations.

**cost** (approx $ for one full dev sweep, n in column 5):
- gpt-5.4: $2.50/M input, $0.25/M cached (90% off), $15/M output (output includes reasoning tokens).
- serper.dev: $0.001/query (web retrieval). PubMed E-utilities: free.
- Verifier rows: LLM tokens from stored usage logs; retrieval counts from `sources_used`.
- Baseline rows: input via tiktoken `o200k_base` on actual prompts; caching modeled per-entry (prefix ≥1024 tokens). Output assumed at ~800 reasoning tokens/call for `-high`; ~30 for `-none`. Reasonable but unverified.
- Calibrated-ensemble: LLM verifier + LLM baseline + retrieval (baseline runs on all snippets, used only on fallback).
- Hybrid-router: LLM verifier-v2 + LLM baseline + retrieval (both run; could be reduced if deployed with per-subset routing).

## Test results — gt=label_atomic (snippet) / label (atom)

| # | grain | mode | model | n | acc | F1_F | F1_T | macro | LLM $ | retr $ | base $ | total $ |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | snippet | calibrated-only | v3 @conf>=0.90 (NA dropped) | 261 | 0.851 | 0.552 | 0.910 | 0.731 | $19.87 | $0.65 | — | $20.52 |
| 2 | snippet | calibrated-only | v5 @conf>=0.90 (NA dropped) | 253 | 0.925 | 0.537 | 0.959 | 0.748 | $13.92 | $0.23 | — | $14.15 |
| 3 | snippet | calibrated-only | v3 @conf>=0.85 (NA dropped) | 358 | 0.830 | 0.504 | 0.897 | 0.701 | $19.87 | $0.65 | — | $20.52 |
| 4 | snippet | verifier-v3 | gpt-5.4-high (anti-examples + conf-gate) | 418 | 0.816 | 0.497 | 0.887 | 0.692 | $19.87 | $0.65 | — | $20.52 |
| 5 | atom | full-context | gpt-5.4-high | 942 | 0.875 | 0.491 | 0.929 | 0.710 | — | — | $12.74 | $12.74 |
| 6 | snippet | full-context | gpt-5.4-high | 431 | 0.803 | 0.491 | 0.878 | 0.684 | — | — | $5.87 | $5.87 |
| 7 | snippet | calibrated-ensemble | v3 @conf>=0.90 else baseline | 431 | 0.794 | 0.491 | 0.870 | 0.681 | $19.87 | $0.65 | $5.87 | $26.38 |
| 8 | snippet | calibrated-ensemble | v3 @conf>=0.85 else baseline | 431 | 0.789 | 0.486 | 0.867 | 0.677 | $19.87 | $0.65 | $5.87 | $26.38 |
| 9 | snippet | calibrated-ensemble | v5 @conf>=0.90 else baseline | 431 | 0.803 | 0.465 | 0.879 | 0.672 | $13.92 | $0.23 | $5.87 | $20.02 |
| 10 | snippet | claim-only | gpt-5.4-high | 431 | 0.752 | 0.451 | 0.840 | 0.645 | — | — | $5.26 | $5.26 |
| 11 | snippet | calibrated-only | v5 @conf>=0.85 (NA dropped) | 343 | 0.866 | 0.439 | 0.924 | 0.681 | $13.92 | $0.23 | — | $14.15 |
| 12 | snippet | calibrated-ensemble | v5 @conf>=0.85 else baseline | 431 | 0.807 | 0.435 | 0.884 | 0.660 | $13.92 | $0.23 | $5.87 | $20.02 |
| 13 | snippet | verifier-v5 | claude-sonnet-4-6 (extended thinking + conf-gate) | 400 | 0.853 | 0.427 | 0.915 | 0.671 | $13.92 | $0.23 | — | $14.15 |
| 14 | atom | claim-only | gpt-5.4-high | 942 | 0.741 | 0.351 | 0.838 | 0.595 | — | — | $11.45 | $11.45 |

### Replication summary (dev → test)

| dev finding | dev F1_F | test F1_F | bootstrap CI (test gap vs baseline) | status |
|---|---:|---:|---|---|
| v3 calibrated ensemble @0.90 + fallback beats baseline at full cov | 0.566 | 0.491 | [−0.044, +0.047], P(>0)=0.487 | ❌ does NOT replicate |
| Hybrid router (consumer→v3, vignette→baseline) beats baseline | 0.533 | 0.475 | [−0.056, +0.024], P(>0)=0.205 | ❌ INVERTS (subset gaps flip sign) |
| Coverage/F1 Pareto: tighter threshold → higher F1_F at lower cov | 0.491 → 0.609 | 0.491 → 0.552 | direction holds; wide CIs | ✅ replicates qualitatively |
| Verifier matches baseline at full coverage (parity) | 0.518 vs 0.518 | 0.497 vs 0.491 | [−0.044, +0.044] | ✅ replicates (parity, not gain) |

### Per-subset on test (the dev subset story inverts)

| subset | system | n | F1_F |
|---|---|---:|---:|
| consumer | baseline | 198 | 0.507 |
| consumer | verifier-v3 (NA drop) | 189 | 0.479 |
| consumer | v3 @0.90 NA drop | 119 (60% cov) | 0.526 |
| vignette | baseline | 233 | 0.479 |
| vignette | verifier-v3 (NA drop) | 229 | 0.512 |
| vignette | v3 @0.90 NA drop | 142 (61% cov) | **0.571** |

Dev had consumer→v3 wins / vignette→baseline wins. **Test has the opposite.** Bootstrap on per-subset @0.90 calibrated gain vs baseline: consumer +0.019 [−0.161, +0.185]; vignette +0.090 [−0.066, +0.238]. Neither is 95%-significant.

### Paper-defensible claims (post-test)

- Iterative verifier with retrieval matches strong parametric baseline at full coverage (parity, no significant difference either direction).
- Per-claim confidence enables a calibrated coverage/F1 Pareto: F1_F 0.491 → 0.552 over 61% coverage on test. Direction replicates across both splits and both subsets; magnitude has wide CIs.
- Per-claim retrieved evidence is an interpretability artifact the baseline lacks.
- Methodology ablations (anti-examples reduce over-skepticism; in-loop conf-gate reduces forced verdicts).

### NOT defensible

- "Our calibrated ensemble beats baseline at full coverage F1_F" — dev-only.
- "Hybrid subset routing helps" — dev-only, inverts on test.

## Headline: the only system that wins on both dev AND test

**v3 (gpt-5.4-high + anti-examples + in-loop conf-gate) with NA-dropped calibrated abstention.**

| system | dev F1_F | test F1_F | dev cov | test cov |
|---|---:|---:|---:|---:|
| baseline (snippet full-ctx gpt-5.4-high) | 0.518 | 0.491 | 100% | 100% |
| **v3 @conf≥0.90 NA-dropped** | **0.609** (+0.091) | **0.552** (+0.061) | **63%** | **61%** |
| v3 @conf≥0.85 NA-dropped | 0.562 (+0.044) | 0.504 (+0.013) | 83% | 83% |
| v3 @conf≥0.90 + baseline fallback (full cov) | 0.566 (+0.047) | 0.491 (+0.000) | 100% | 100% |

The **full-coverage ensemble win evaporated on test** (0.566 dev → 0.491 test). The **NA-dropped variants held in direction**, though magnitude shrinks on test and CIs are wide (not 95%-significant on test).

**Paper framing**: iterative verifier + confidence-gated abstention provides a calibrated coverage/F1 Pareto curve. Pick the operating point:
- ~83% coverage at +0.013 F1_F over baseline (test)
- ~61% coverage at +0.061 F1_F over baseline (test)

Grounded in selective-prediction literature (MedHallu, VeriFact, FIRE — see [survey.md](survey.md)).

## Version flowcharts

Each chart shows one snippet's path through the verifier. Diffs from the prior version are marked **bold**.

Color legend (darkest → lightest = early step → late/derivative):
**dark navy = input** · **blue = LLM call** · **mid-blue = retrieval** · **light blue = decision** · **pale = output** · 🟧 orange = new in this version

### v1 — base iterative verifier (gpt-5.4-high)
```mermaid
flowchart LR
  S([snippet]) --> L[LLM call]
  L -->|search_query| R[retrieve web/PubMed]
  R --> L
  L -->|abstain| A[abstain]
  L -->|final_answer| F[T/F + conf]
  L -->|iter ≥ 3| FF[force-final]
  classDef inp fill:#08306b,color:#fff,stroke-width:0
  classDef llm fill:#2171b5,color:#fff,stroke-width:0
  classDef retr fill:#6baed6,color:#000,stroke-width:0
  classDef out fill:#c6dbef,color:#000,stroke-width:0
  classDef tail fill:#f7fbff,color:#000,stroke-width:0
  class S inp
  class L llm
  class R retr
  class A,F out
  class FF tail
```

### v2 — v1 + 🟧 anti-example prompt (gpt-5.4-high)
```mermaid
flowchart LR
  S([snippet]) --> L["LLM call<br/>+ anti-example rules"]
  L -->|search_query| R[retrieve web/PubMed]
  R --> L
  L -->|abstain| A[abstain]
  L -->|final_answer| F[T/F + conf]
  L -->|iter ≥ 3| FF[force-final]
  classDef inp fill:#08306b,color:#fff,stroke-width:0
  classDef new fill:#f4a261,color:#000,stroke-width:0
  classDef retr fill:#6baed6,color:#000,stroke-width:0
  classDef out fill:#c6dbef,color:#000,stroke-width:0
  classDef tail fill:#f7fbff,color:#000,stroke-width:0
  class S inp
  class L new
  class R retr
  class A,F out
  class FF tail
```

### v3 — v2 + 🟧 in-loop confidence gate (gpt-5.4-high)
```mermaid
flowchart LR
  S([snippet]) --> L[LLM call w/ anti-examples]
  L -->|search_query| R[retrieve]
  R --> L
  L -->|abstain| A[abstain]
  L -->|final_answer| C{"conf ≥ 0.85?"}
  C -->|yes| F[T/F + conf]
  C -->|no, iters left| R2[force one more search]
  R2 --> L
  L -->|iter ≥ 5| FF["force-final<br/>bias to abstain"]
  classDef inp fill:#08306b,color:#fff,stroke-width:0
  classDef llm fill:#2171b5,color:#fff,stroke-width:0
  classDef retr fill:#6baed6,color:#000,stroke-width:0
  classDef new fill:#f4a261,color:#000,stroke-width:0
  classDef out fill:#c6dbef,color:#000,stroke-width:0
  classDef tail fill:#f7fbff,color:#000,stroke-width:0
  class S inp
  class L llm
  class R retr
  class C,R2,FF new
  class A,F out
```

### v4 — v3 + 🟧 smaller model (gpt-5-mini, reasoning_effort=minimal)
```mermaid
flowchart LR
  S([snippet]) --> L[gpt-5-mini minimal-reasoning]
  L -->|search_query| R[retrieve]
  R --> L
  L -->|abstain| A[abstain]
  L -->|final_answer| C{conf ≥ 0.85?}
  C -->|yes| F[T/F + conf]
  C -->|no| R2[force search]
  R2 --> L
  L -->|iter ≥ 5| FF[force-final]
  classDef inp fill:#08306b,color:#fff,stroke-width:0
  classDef new fill:#f4a261,color:#000,stroke-width:0
  classDef retr fill:#6baed6,color:#000,stroke-width:0
  classDef decide fill:#9ecae1,color:#000,stroke-width:0
  classDef out fill:#c6dbef,color:#000,stroke-width:0
  classDef tail fill:#f7fbff,color:#000,stroke-width:0
  class S inp
  class L new
  class R,R2 retr
  class C decide
  class A,F out
  class FF tail
```

### v5 — v3 + 🟧 cross-family (claude-sonnet-4-6 + extended thinking, budget 8000)
```mermaid
flowchart LR
  S([snippet]) --> L["Sonnet 4.6 (streaming)<br/>extended-thinking 8K"]
  L -->|search_query| R[retrieve]
  R --> L
  L -->|abstain| A[abstain]
  L -->|final_answer| C{conf ≥ 0.85?}
  C -->|yes| F[T/F + conf]
  C -->|no| R2[force search]
  R2 --> L
  L -->|iter ≥ 5| FF[force-final]
  classDef inp fill:#08306b,color:#fff,stroke-width:0
  classDef new fill:#f4a261,color:#000,stroke-width:0
  classDef retr fill:#6baed6,color:#000,stroke-width:0
  classDef decide fill:#9ecae1,color:#000,stroke-width:0
  classDef out fill:#c6dbef,color:#000,stroke-width:0
  classDef tail fill:#f7fbff,color:#000,stroke-width:0
  class S inp
  class L new
  class R,R2 retr
  class C decide
  class A,F out
  class FF tail
```

### Derived strategies (post-hoc, no new LLM calls)
```mermaid
flowchart LR
  V([verifier verdict]) --> X{"abstain / cap-hit / conf < τ?"}
  X -->|no| K[keep verdict]
  X -->|yes — NA-dropped| D[drop from eval<br/>↓ coverage, ↑ F1_F]
  X -->|yes — baseline-fallback| B[use baseline pred<br/>full coverage]
  classDef inp fill:#08306b,color:#fff,stroke-width:0
  classDef decide fill:#9ecae1,color:#000,stroke-width:0
  classDef out fill:#c6dbef,color:#000,stroke-width:0
  classDef alt fill:#deebf7,color:#000,stroke-width:0
  class V inp
  class X decide
  class K out
  class D,B alt
```

