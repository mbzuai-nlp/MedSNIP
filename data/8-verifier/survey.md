# Medical Fact-Checking Literature Survey (2024–2026)

## Landscape

The field has consolidated around **decompose-then-verify** with retrieval, but
2025 work shows the bottleneck has shifted from retrieval coverage to
**verifier calibration and decomposition quality**. Two themes dominate:

1. **Domain-adapted decomposition** for medical text (MedScore, FActBench-Medical)
   showing generic FActScore-style decomposition wastes most claims on medical
   prose.
2. **Abstention / uncertainty mechanisms** (MedAbstain, SelectLLM, UQ heads) as
   the primary lever for improving F1 in high-stakes clinical QA.

Consistent negative result that mirrors our hybrid-router finding: **RAG hurts
or is flat on clinical-vignette / USMLE-style reasoning**, where retrieval
covers only ~26-33% of must-have statements, while "over-criticism" from
advanced reasoning models inflates false-positive False labels — exactly our
over-skeptical compound-claim symptom.

## References

| # | Title | Venue / Year | Relevance |
|---|---|---|---|
| 1 | MedScore: Generalizable Factuality Evaluation of Free-Form Medical Answers (Zhao et al., arXiv:2505.18452) | 2025 | **HIGH** — domain-adapted decomposition + 7-error taxonomy; 74% valid claims vs FActScore's 17%. Directly attacks our compound-claim weakness. |
| 2 | MedFact (Chinese, EMNLP 2025) | 2025 | **HIGH** — 1.3K Qs / 7.4K claims; documents "over-criticism" where stronger reasoning models flip correct text to False; multi-agent collab *decreased* precision. Explains our v2 behavior. |
| 3 | Optimizing Decomposition for Optimal Claim Verification (Wang et al., ACL 2025, arXiv:2503.15354) | 2025 | **HIGH** — learns verifier-preferred atomicity via RL; +0.12 accuracy on average. Formalizes our atom-vs-snippet question. |
| 4 | FIRE: Fact-checking with Iterative Retrieval and Verification (NAACL Findings 2025) | 2025 | **HIGH** — direct cousin to our iterative verifier; confidence-gated stop; 7.6× cheaper at equivalent F1. |
| 5 | MedHallu (EMNLP 2025) | 2025 | **HIGH** — 10K PubMedQA pairs; adding "not sure" abstain bucket → **+38% relative F1**. Direct fit for our cap_hit problem. |
| 6 | FAITH: Assessing AFC for Medical LLM Responses (Tang et al., AAAI 2026, arXiv:2511.12817) | 2026 | MED — KG-grounded atomic scoring; better correlation with clinician judgments. KG tie-breaker for vignettes worth trying. |
| 7 | Pre-trained UQ Heads for Hallucination Detection (EMNLP 2025) | 2025 | MED — auxiliary heads on open-weight models; claim-level SOTA. Useful if we run an open-weight verifier alongside gpt-5.4. |
| 8 | Rethinking RAG for Medicine: Large-Scale Expert Evaluation (arXiv:2511.06738) | 2025 | **HIGH** — empirical: retrievers cover only 26% of must-haves on USMLE-style queries. Empirical justification for our hybrid router. |
| 9 | Resolving Conflicting Evidence in Automated Fact-Checking (IJCAI 2025) | 2025 | MED — CONFACT dataset; RAG under contradictory snippets. Relevant to our refutation-biased retrieval. |
| 10 | VeriFact (NEJM AI 2025) | 2025 | MED — RAG + LLM-judge against EHR; per-proposition {supported / not-supported / not-addressed} labels. Three-way label is a cleaner abstain target than binary. |
| 11 | HealthFC (LREC-COLING 2024) | 2024 | MED — 750 expert-labeled health claims w/ systematic-review evidence (DE/EN/TR/ZH). Use as external eval for consumer subset. |
| 12 | FActBench-Medical (arXiv:2509.02198) | 2025 | MED — task-diverse atomic fact-checking; CoT+NLI ensemble correlates best with medical expert ratings. Ensemble of CoT+NLI heads is a cheap path beyond gpt-5.4 alone. |

## Concrete techniques to consider

1. **Medical-aware decomposition prompt (MedScore)** — taxonomy with context-dependent,
   condition-dependent, subjective markers so the verifier ignores peripheral
   imprecisions on compound claims. **Highest ROI estimate.**
2. **Three-label output {True, False, Abstain/Not-addressed}** with abstain
   mapped explicitly (not collapsed to False). Attacks our force-final cap_hit
   problem head-on. MedHallu reports +38% relative F1.
3. **Confidence-gated iteration stop** (FIRE) — drop fixed max=5; stop when
   confidence > τ. 7.6× cheaper at equivalent F1. We already emit confidence.
4. **NLI head as second verifier** (FActBench): gpt-5.4 only flips to False if
   both judges agree. Cheap; proven on medical text.
5. **Verifier-aware atomicity** (Optimizing Decomposition): coarser atomization
   for vignettes, finer for consumer-health.

## External eval candidates

- **MedHallu** — closest in spirit to our snippet task; binary hallucination.
  **Highest priority.**
- **HealthFC** — expert-curated consumer-health claims; natural external test.
- **MedFact-EMNLP / Xunfei** (Chinese) — would need translation; their
  over-criticism analysis is the unique contribution.
- **FActBench-Medical** — for expert-correlation rather than F1.
- **VeriFact / EHR-based** — out-of-scope for our setup.

## Sources

- [MedScore](https://arxiv.org/html/2505.18452)
- [MedFact (Chinese)](https://arxiv.org/html/2509.12440)
- [MedFact EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1646/)
- [Optimizing Decomposition](https://arxiv.org/html/2503.15354v2)
- [FIRE NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.158/)
- [MedHallu](https://aclanthology.org/2025.emnlp-main.143/)
- [FAITH AAAI 2026](https://arxiv.org/abs/2511.12817)
- [UQ Heads EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1809/)
- [Rethinking RAG for Medicine](https://arxiv.org/html/2511.06738)
- [Resolving Conflicting Evidence IJCAI 2025](https://www.ijcai.org/proceedings/2025/1073.pdf)
- [VeriFact NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIdbp2500418)
- [HealthFC LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.709/)
- [FActBench-Medical](https://arxiv.org/html/2509.02198)
