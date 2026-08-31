"""Evaluate snippet processors against human gold via SENTENCE-SET F1.

Evaluation currency: sentence indices into a deterministic segmentation of
`full_text`. Gold atoms are pre-mapped to sentence indices once
(see precompute_alignment.py). Then:

  - human snippet → set of sentence indices (via atom_to_sentence)
  - ft   snippet  → set of sentence indices (output directly by pipeline)
  - atom snippet  → set of sentence indices (via atom_to_sentence)

No embedding-based alignment for the matching step itself — differences in
F1 reflect true grouping disagreements, not artifacts. Text similarity
(embedding cosine + ROUGE-L) is computed on the best-matched pair for
context only.

Reads:
  data/3-annotated/medsnip-bench.json                       (gold, snippet rows)
  data/6-snippet-processor/outputs/<mode>/e<id>.json (predicted per-entry)
  data/6-snippet-processor/sentence_alignment.json   (atom → sentence index)

Writes:
  data/6-snippet-processor/results.json
  data/6-snippet-processor/results.md
  data/6-snippet-processor/plots/*.png
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[3]
ANNOTATED_PATH = ROOT / "data" / "3-annotated" / "medsnip-bench.json"
DATA_DIR = ROOT / "data" / "6-snippet-processor"
ALIGN_PATH = DATA_DIR / "sentence_alignment.json"

MODES = ("atom-to-snippet", "snippet-direct")
MODE_COLORS = {"atom-to-snippet": "#4C72B0", "snippet-direct": "#55A868"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_prf(a_human: set[int], a_pred: set[int]) -> tuple[float, float, float]:
    if not a_human or not a_pred:
        return 0.0, 0.0, 0.0
    inter = len(a_human & a_pred)
    p = inter / len(a_pred)
    r = inter / len(a_human)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def aggregate_human(rows: list[dict]) -> dict[int, dict]:
    """Snippet rows → {entry_id: {entry_id, subset, batch_id, snippets:[{source_claims, output}]}}."""
    by: dict[int, dict] = {}
    for r in rows:
        eid = r["entry_id"]
        if eid not in by:
            by[eid] = {
                "entry_id": eid,
                "subset":   r["subset"],
                "batch_id": r["batch_id"],
                "snippets": [],
            }
        by[eid]["snippets"].append({
            "source_claims": [a["index"] for a in r.get("atoms") or []],
            "output":        r["snippet_text"],
        })
    return by


def load_outputs(mode: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    mode_dir = DATA_DIR / mode
    if not mode_dir.exists():
        return out
    for f in sorted(mode_dir.glob("e*.json")):
        d = json.loads(f.read_text())
        out[d["entry_id"]] = d
    return out


def predicted_sentence_sets(mode: str, pred_entry: dict,
                            atom_to_sent: dict[int, int]) -> list[tuple[set[int], str]]:
    """Return [(sentence_index_set, output_text), ...] per predicted snippet.

    atom-to-snippet: snippet.source_claims are generated-atom indices; each
    generated atom carries its own source_sentences. Union them per snippet.
    snippet-direct: snippets carry source_sentences directly.
    """
    out = []
    if mode == "atom-to-snippet":
        gen_atoms = pred_entry.get("generated_atoms") or []
        gen_idx_to_sents = {
            int(a["index"]): {int(s) for s in a.get("source_sentences", [])}
            for a in gen_atoms
        }
        for s in pred_entry["snippets"]:
            sents: set[int] = set()
            for a in s.get("source_claims", []):
                sents |= gen_idx_to_sents.get(int(a), set())
            out.append((sents, s.get("output", "")))
    else:  # snippet-direct
        for s in pred_entry["snippets"]:
            sents = {int(si) for si in s.get("source_sentences", [])}
            out.append((sents, s.get("output", "")))
    return out


def human_sentence_sets(human: dict, atom_to_sent: dict[int, int]) -> list[tuple[set[int], str]]:
    out = []
    for s in human["snippets"]:
        sents = {atom_to_sent.get(int(a), -1) for a in s["source_claims"]}
        sents.discard(-1)
        out.append((sents, s["output"]))
    return out


# ---------------------------------------------------------------------------
# per-entry evaluation
# ---------------------------------------------------------------------------

def evaluate_entry(human: dict, pred_snippet_sets, human_snippet_sets,
                   embed_model, scorer) -> dict:
    pairs = []
    for pi, (psents, ptext) in enumerate(pred_snippet_sets):
        best = (-1.0, -1.0, -1.0, None, -1, "")
        for hi, (hsents, htext) in enumerate(human_snippet_sets):
            p, r, f1 = set_prf(hsents, psents)
            if f1 > best[0]:
                best = (f1, p, r, hsents, hi, htext)
        f1, p, r, hsents, hi, htext = best
        if hsents is None:
            continue
        pairs.append({
            "pred_idx":     pi, "human_idx": hi,
            "sent_precision": p, "sent_recall": r, "sent_f1": f1,
            "pred_text":    ptext, "human_text": htext,
            "pred_sents":   sorted(psents),
            "human_sents":  sorted(hsents),
        })

    if not pairs:
        return {"entry_id": human["entry_id"],
                "subset":   human["subset"],
                "batch_id": human["batch_id"],
                "n_pred":   len(pred_snippet_sets),
                "n_human":  len(human_snippet_sets),
                "pairs":    []}

    h_texts = [pr["human_text"] for pr in pairs]
    p_texts = [pr["pred_text"] for pr in pairs]
    h_emb = embed_model.encode(h_texts, batch_size=32, show_progress_bar=False,
                               convert_to_tensor=True)
    p_emb = embed_model.encode(p_texts, batch_size=32, show_progress_bar=False,
                               convert_to_tensor=True)
    cos = np.diag(util.cos_sim(h_emb, p_emb).cpu().numpy())
    for pr, c in zip(pairs, cos):
        pr["embedding_cosine"] = float(c)
        s = scorer.score(pr["human_text"], pr["pred_text"])["rougeL"]
        pr["rouge_l_f1"] = s.fmeasure

    return {
        "entry_id": human["entry_id"],
        "subset":   human["subset"],
        "batch_id": human["batch_id"],
        "n_pred":   len(pred_snippet_sets),
        "n_human":  len(human_snippet_sets),
        "pairs":    pairs,
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate(per_entry: list[dict]) -> dict:
    pairs = [p for e in per_entry for p in e.get("pairs", [])]
    mean = lambda xs: statistics.mean(xs) if xs else None
    med  = lambda xs: statistics.median(xs) if xs else None
    if not pairs:
        return {"n_pairs": 0}

    def by_subset(field):
        out = {}
        for sub in ("consumer", "vignette"):
            ps = [p for e in per_entry if e.get("subset") == sub for p in e["pairs"]]
            if ps:
                out[sub] = mean([p[field] for p in ps])
        return out

    return {
        "n_entries":                    len(per_entry),
        "n_pairs":                      len(pairs),
        "n_pred_snippets_total":        sum(e["n_pred"]  for e in per_entry),
        "n_human_snippets_total":       sum(e["n_human"] for e in per_entry),
        "sent_precision_mean":          mean([p["sent_precision"] for p in pairs]),
        "sent_recall_mean":             mean([p["sent_recall"]    for p in pairs]),
        "sent_f1_mean":                 mean([p["sent_f1"]        for p in pairs]),
        "sent_f1_median":               med ([p["sent_f1"]        for p in pairs]),
        "embedding_cosine_mean":        mean([p["embedding_cosine"] for p in pairs]),
        "rouge_l_f1_mean":              mean([p["rouge_l_f1"]      for p in pairs]),
        "sent_f1_by_subset":            by_subset("sent_f1"),
        "embedding_cosine_by_subset":   by_subset("embedding_cosine"),
    }


# ---------------------------------------------------------------------------
# plots + markdown
# ---------------------------------------------------------------------------

def _write_plots(per_entry):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for m in MODES:
        if m not in per_entry:
            continue
        f1s = [p["sent_f1"] for e in per_entry[m] for p in e.get("pairs", [])]
        if f1s:
            ax.hist(f1s, bins=30, alpha=0.6, label=f"{m} (n={len(f1s)})",
                    color=MODE_COLORS[m])
    ax.set_xlabel("Sentence-set F1 (predicted snippet ↔ best human snippet)")
    ax.set_ylabel("# snippet pairs")
    ax.set_title("Clustering agreement with human gold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / "sentence_f1_distribution.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MODES), figsize=(5 * len(MODES), 4))
    if len(MODES) == 1:
        axes = [axes]
    for ax, m in zip(axes, MODES):
        if m not in per_entry:
            continue
        xs = [e["n_human"] for e in per_entry[m]]
        ys = [e["n_pred"]  for e in per_entry[m]]
        ax.scatter(xs, ys, alpha=0.6, color=MODE_COLORS[m])
        lim = max(max(xs, default=1), max(ys, default=1)) + 1
        ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("# human snippets")
        ax.set_ylabel("# predicted snippets")
        ax.set_title(f"{m}: snippet count per entry")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "snippet_count_scatter.png", dpi=130)
    plt.close(fig)


def _write_md(summary, n_entries):
    md = []
    md.append("# Snippet Processor — Sentence-Grounded Evaluation\n")
    md.append(f"Evaluated **{n_entries}** entries against human gold "
              f"(data/3-annotated/medsnip-bench.json).\n")
    md.append("Evaluation currency: **sentence indices** into a deterministic "
              "segmentation of `full_text` (see sentence_utils.py). Gold atoms are "
              "pre-mapped to sentence indices once (sentence_alignment.json), so "
              "all snippet comparisons are on the same currency — no embedding "
              "alignment.\n")
    md.append("## Pipelines\n")
    md.append("| Mode | Input | API calls | Role |")
    md.append("|---|---|---|---|")
    md.append("| **atom-to-snippet** | `(query, full_text)` | **2** | Default. Sentence-split → extract atoms (+ shared_context) → cluster atoms into snippets. Mirrors the human annotation workflow. |")
    md.append("| **snippet-direct** | `(query, full_text)` | **1** | Single-pass: sentences directly to snippets + shared_context. Cheaper; slightly weaker on dense consumer responses. |")
    md.append("")
    md.append("## Headline numbers (sentence-set F1)\n")
    md.append("| Mode | sent F1 | sent P | sent R | embed cos | ROUGE-L | n pred / human |")
    md.append("|---|---|---|---|---|---|---|")
    for m in MODES:
        s = summary.get(m)
        if not s or not s.get("n_pairs"):
            continue
        md.append(f"| {m} | **{s['sent_f1_mean']:.3f}** | {s['sent_precision_mean']:.3f} "
                  f"| {s['sent_recall_mean']:.3f} | {s['embedding_cosine_mean']:.3f} "
                  f"| {s['rouge_l_f1_mean']:.3f} "
                  f"| {s['n_pred_snippets_total']} / {s['n_human_snippets_total']} |")
    md.append("")
    md.append("![Sentence F1](sentence_f1_distribution.png)\n")
    md.append("![Snippet count](snippet_count_scatter.png)\n")
    md.append("## By subset\n")
    md.append("| Mode | consumer F1 | vignette F1 |")
    md.append("|---|---|---|")
    for m in MODES:
        s = summary.get(m)
        if not s or not s.get("n_pairs"):
            continue
        bs = s["sent_f1_by_subset"]
        md.append(f"| {m} | {bs.get('consumer', 0.0):.3f} | {bs.get('vignette', 0.0):.3f} |")
    md.append("")
    (DATA_DIR / "results.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if not ALIGN_PATH.exists():
        raise SystemExit(
            f"missing {ALIGN_PATH}; run "
            f"`python -m src.medsnip.snippet_processor.precompute_alignment` first"
        )
    rows = json.loads(ANNOTATED_PATH.read_text())
    human_entries = aggregate_human(rows)
    alignment = json.loads(ALIGN_PATH.read_text())
    print(f"loaded {len(human_entries)} human entries, "
          f"{len(alignment)} alignment records")

    pred = {m: load_outputs(m) for m in MODES}
    for m in MODES:
        print(f"  {m}: {len(pred[m])} entries on disk")

    pred_ids = [set(p) for p in pred.values() if p]
    common = sorted(set.intersection(*pred_ids) & set(human_entries)) if pred_ids else []
    print(f"common entries: {len(common)}")
    if not common:
        print("nothing to evaluate")
        return

    print("loading sentence-transformer all-MiniLM-L6-v2 …")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    per_entry: dict[str, list[dict]] = defaultdict(list)
    for eid in common:
        h = human_entries[eid]
        a2s = {int(k): int(v) for k, v in alignment[str(eid)]["atom_to_sentence"].items()}
        human_sets = human_sentence_sets(h, a2s)
        for m in MODES:
            if eid not in pred[m]:
                continue
            pred_sets = predicted_sentence_sets(m, pred[m][eid], a2s)
            per_entry[m].append(evaluate_entry(h, pred_sets, human_sets, embed_model, scorer))

    summary = {m: aggregate(per_entry[m]) for m in MODES}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "per_entry": per_entry}, indent=2)
    )
    _write_plots(per_entry)
    _write_md(summary, len(common))

    print("\n=== summary (sentence-set F1) ===")
    for m in MODES:
        s = summary[m]
        if not s.get("n_pairs"):
            print(f"  {m}: no pairs")
            continue
        print(f"  {m:5s}: F1={s['sent_f1_mean']:.3f}  "
              f"P={s['sent_precision_mean']:.3f} R={s['sent_recall_mean']:.3f}  "
              f"emb={s['embedding_cosine_mean']:.3f}  "
              f"rougeL={s['rouge_l_f1_mean']:.3f}  "
              f"(pred={s['n_pred_snippets_total']} vs human {s['n_human_snippets_total']})")


if __name__ == "__main__":
    main()
