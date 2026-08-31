<p align="center">
  <img src="assets/logo.svg" alt="MedSNIP" width="350">
</p>

<p align="center">
  Building and Benchmarking Snippet-Level Granularity for Medical Fact Verification
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=1vYUGLZnvr"><img src="https://img.shields.io/badge/Paper-OpenReview-8C1B13?logo=openreview&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/MBZUAI/MedSNIP"><img src="https://img.shields.io/badge/Dataset-MedSNIP--Bench-FFD21E?logo=huggingface&logoColor=black" alt="Dataset"></a>
  <a href="https://mbzuai-nlp.github.io/MedSNIP/"><img src="https://img.shields.io/badge/Project-Website-1F6FEB?logo=googlechrome&logoColor=white" alt="Project Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-4B9E46" alt="License"></a>
</p>

---

A medical claim's correctness often depends not on the claim alone, but on the clinical structure around it. Atom-level decomposition fragments those dependencies, leaving the verifier with claims that are formally standalone but clinically incomplete.

This repository reformulates medical fact-checking around **snippet-level** verification, where clause-grouped units preserve local clinical structure. It contains **MedSNIP-Bench**, a human-annotated benchmark of 2,524 snippets over 276 responses, and **MedSNIP**, the pipeline that generates snippets automatically.

## Setup

```bash
uv sync
cp .env.template .env   # fill in API keys
```

## Dataset

The benchmark is published on the Hugging Face Hub:

```python
from datasets import load_dataset

ds = load_dataset("MBZUAI/MedSNIP")   # train / validation / test
```

The same records live in `data/15-final/dataset.json`, with per-split copies staged in `data/!-hf/`.

## Layout

```
src/medsnip/        the pipeline: snippet_processor, retriever, verifier
src/baselines/      atom- and snippet-level verification baselines
src/decomposer/     decomposer x verifier matrix
src/external/       HealthFC and MedHallu
src/analysis/       the analyses behind the paper's tables
data/<N>-<step>/    each stage's outputs, in pipeline order
web/annotation/     the annotation dashboard
```

Pipeline stages run in order, each reading the previous stage's directory:

| | Stage | Output |
|---|---|---|
| 1 | Raw corpora | `data/1-raw/` |
| 2 | Subset routing | `data/2-subset/` |
| 3 | Human annotation | `data/3-annotated/` |
| 4 | Train/dev/test split | `data/4-split/` |
| 5 | Atom & snippet baselines | `data/5-baselines/` |
| 6 | Snippet processor | `data/6-snippet-processor/` |
| 7 | Retriever | `data/7-retriever/` |
| 8 | Retrieval-augmented verifier | `data/8-verifier/` |
| 9–10 | HealthFC, MedHallu | `data/9-healthfc/`, `data/10-medhallu/` |
| 11–14 | Analyses, decomposer matrix, cost, ablations | `data/11-analysis/` … `data/14-normalization/` |
| 15 | Released dataset | `data/15-final/` |

## Reproducing

Stages 1–10 call paid APIs. The analyses in 11–15 read committed outputs and run offline:

```bash
python -m src.analysis.external_results      # external corpora results
python -m src.analysis.msb_column_cis        # MedSNIP-Bench confidence intervals
python -m src.analysis.pattern_f1_gap        # per-pattern snippet-atom gap
python -m src.analysis.cost_accounting       # end-to-end cost
python -m src.decomposer.build_gold_matrix   # decomposer x verifier matrix
```

To rebuild the released dataset and its Hugging Face export:

```bash
python -m src.analysis.build_final_dataset
python -m src.analysis.build_final_stats
python src/hf/build.py
```

## Intended use

This work supports research on automated factuality evaluation for medical text generation. It is not a source of medical advice, diagnosis, or treatment recommendations, and should not be used for autonomous clinical decision support.

## License

Code is released under [GPL-3.0](LICENSE). The external corpora, the retrieval cache, and the source responses retain their original terms.

## Citation

```bibtex
@inproceedings{
anonymous2026medsnip,
title={Med{SNIP}: Building and Benchmarking Snippet-Level Granularity for Medical Fact Verification},
author={Anonymous},
booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing},
year={2026},
url={https://openreview.net/forum?id=1vYUGLZnvr}
}
```
