---
license: cc-by-4.0
language:
  - en
pretty_name: MedSNIP-Bench
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
tags:
  - medical
  - fact-verification
  - fact-checking
  - factuality
  - hallucination-detection
  - consumer-health
  - clinical-vignettes
  - benchmark
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.json
      - split: validation
        path: dev.json
      - split: test
        path: test.json
---

<p align="center">
  <img src="logo.svg" alt="MedSNIP" width="200">
</p>

<h1 align="center">MedSNIP: Building and Benchmarking Snippet-Level Granularity for Medical Fact Verification</h1>

<p align="center">
  <b>Hasan Iqbal</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Sarfraz Ahmad</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Hyunjae Kim</b><sup>2</sup> &nbsp;·&nbsp;
  <b>Sihyeon Park</b><sup>3</sup> &nbsp;·&nbsp;
  <b>Junjie Liao</b><sup>4,5</sup> &nbsp;·&nbsp;
  <b>Qingyu Chen</b><sup>2</sup> &nbsp;·&nbsp;
  <b>Preslav Nakov</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Yuxia Wang</b><sup>5</sup>
</p>

<p align="center">
  <sup>1</sup>MBZUAI &nbsp;·&nbsp;
  <sup>2</sup>Yale University &nbsp;·&nbsp;
  <sup>3</sup>Korea University &nbsp;·&nbsp;
  <sup>4</sup>Beijing Normal University &nbsp;·&nbsp;
  <sup>5</sup>INSAIT, Sofia University "St. Kliment Ohridski"
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=1vYUGLZnvr"><img src="https://img.shields.io/badge/Paper-OpenReview-8C1B13?logo=openreview&logoColor=white" alt="Paper"></a>
  <a href="https://mbzuai-nlp.github.io/MedSNIP/"><img src="https://img.shields.io/badge/Project-Website-1F6FEB?logo=googlechrome&logoColor=white" alt="Project Website"></a>
  <a href="https://github.com/mbzuai-nlp/MedSNIP"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub"></a>
</p>

A medical claim's correctness often depends not on the claim alone, but on the clinical structure around it. A claim may require a lab reference range, a causal or conditional link, or patient-specific details to be judged correctly, and atom-level decomposition can fragment these dependencies, leaving the verifier with clinically incomplete claims.

**MedSNIP-Bench** reformulates medical fact-checking around *snippet-level* verification, where clause-grouped units preserve local clinical structure. It covers 276 consumer-health and clinical-vignette responses, segmented by trained annotators into 2,524 snippets with dual in-general and in-patient-context labels and six structural pattern codes.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("MBZUAI/MedSNIP")
ds["train"], ds["validation"], ds["test"]
```

## Splits

Splits are made at the entry level, so all snippets from one response stay together.

| Split | Entries | Snippets |
|---|---:|---:|
| train | 180 | 1,599 |
| validation | 51 | 494 |
| test | 45 | 431 |
| **total** | **276** | **2,524** |

## Fields

| Field | Description |
|---|---|
| `snippet_id` | Stable identifier, `<entry_id>-S<n>` |
| `entry_id` | Source response this snippet came from |
| `query` | The original user question |
| `full_text` | The complete model response the snippet was cut from |
| `subset` | `consumer` (short patient questions) or `vignette` (clinical presentations) |
| `batch_id` | Annotation batch |
| `shared_context` | Entry-level context an annotator recorded, such as topic or patient details |
| `snippet_text` | The verification unit |
| `atoms` | The atomic claims grouped into this snippet |
| `pattern` | Structural pattern code, `A`–`F` |
| `is_ambiguous` | Annotator flag; all flagged snippets were adjudicated to binary labels |
| `notes` | Annotator rationale, where recorded |
| `label_atomic` | Label projected from the source physician atom labels |
| `label_human_general` | Annotator label, judged in general |
| `label_human_contextual` | Annotator label, judged for this patient |
| `split` | `train`, `dev`, or `test` |

Because the corpus skews toward true snippets, the paper reports **F1 on the false class** as its primary metric.

## Structural patterns

Each snippet carries one of six pattern codes. `A`, `B` and `C` merge multiple atoms; `D`, `E` and `F` keep an atom standalone.

| Code | Pattern | Snippets |
|---|---|---:|
| A | Enumeration over a shared subject | 1,008 |
| B | Causal or conditional chain | 225 |
| C | Conclusion plus its premises | 321 |
| D | Standalone claim | 617 |
| E | Distinct alternative, deliberately separated | 39 |
| F | Isolated claim kept apart from its neighbours | 314 |

Of the 2,524 snippets, 1,554 group two or more atoms and 970 correspond to a single atom, over 5,838 atom references in total. 37 snippets carry the `is_ambiguous` flag.

## Intended use

This dataset supports research on automated factuality evaluation for medical text generation. It is **not** a source of medical advice, diagnosis, or treatment recommendations, and should not be used for autonomous clinical decision support. Automated verification is imperfect: a verifier may accept false claims, reject correct ones, or abstain on clinically important cases.

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
