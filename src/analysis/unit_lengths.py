"""How much structure is there to group, per corpus?

MedSNIP's motivation is that long, context-rich medical answers fragment badly
under atomization: a claim loses the reference range, the causal antecedent, or
the patient framing that sits in a neighbouring clause. That argument only bites
when the source text HAS neighbouring clauses.

This measures the raw material each corpus offers:

  - source length in words and sentences
  - units produced per source, and the merge ratio (atoms per snippet)
  - unit length

A corpus whose sources are single sentences yielding ~1 atom and ~1 snippet
gives the method nothing to work with, and a null result there is what the
thesis predicts rather than evidence against it.

Usage:
  python -m src.analysis.unit_lengths
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from ..medsnip.snippet_processor.sentence_utils import split_sentences

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "data" / "11-analysis" / "structure-analysis" / "unit_lengths.json"


def words(s: str) -> int:
    return len((s or "").split())


def sents(s: str) -> int:
    try:
        return len(split_sentences(s or ""))
    except Exception:
        return 0


def med(xs):
    return round(st.median(xs), 1) if xs else 0.0


def mean(xs):
    return round(st.mean(xs), 2) if xs else 0.0


def from_decomp(path: Path, text_key: str, label: str) -> dict | None:
    """Corpora whose decomposition file holds source text, atoms and snippets."""
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    src_w, src_s, na, ns, aw, sw = [], [], [], [], [], []
    for r in rows:
        t = r.get(text_key) or ""
        if not t and r.get("sentences"):
            # auto-decomposition records keep the source as a sentence list
            t = " ".join(r["sentences"])
        src_w.append(words(t))
        src_s.append(sents(t))
        atoms = r.get("generated_atoms") or []
        snips = [x for x in (r.get("snippets") or []) if isinstance(x, dict)]
        na.append(len(atoms))
        ns.append(len(snips))
        aw += [words(a.get("text")) for a in atoms]
        sw += [words(x.get("output") or x.get("text")) for x in snips]
    return {
        "corpus": label, "n_sources": len(rows),
        "src_words_med": med(src_w), "src_sents_med": med(src_s),
        "atoms_per_src": mean(na), "snips_per_src": mean(ns),
        "merge_ratio": round(sum(na) / sum(ns), 2) if sum(ns) else None,
        "atom_words_med": med(aw), "snip_words_med": med(sw),
        "pct_single_sentence_src": round(100 * sum(1 for x in src_s if x <= 1) / len(src_s), 1),
    }


def from_medsnip() -> dict:
    """MedSNIP-Bench: human snippets, expert atoms, answers as source."""
    rows = json.loads((ROOT / "data" / "15-final" / "dataset.json").read_text())
    by_entry: dict[int, dict] = {}
    for r in rows:
        e = by_entry.setdefault(r["entry_id"], {"text": r["full_text"], "sn": [], "at": 0})
        e["sn"].append(r["snippet_text"])
        e["at"] += len(r.get("atoms") or [])
    src_w = [words(e["text"]) for e in by_entry.values()]
    src_s = [sents(e["text"]) for e in by_entry.values()]
    na = [e["at"] for e in by_entry.values()]
    ns = [len(e["sn"]) for e in by_entry.values()]
    aw = [words(a.get("text")) for r in rows for a in (r.get("atoms") or [])]
    sw = [words(r["snippet_text"]) for r in rows]
    return {
        "corpus": "MedSNIP-Bench (human)", "n_sources": len(by_entry),
        "src_words_med": med(src_w), "src_sents_med": med(src_s),
        "atoms_per_src": mean(na), "snips_per_src": mean(ns),
        "merge_ratio": round(sum(na) / sum(ns), 2),
        "atom_words_med": med(aw), "snip_words_med": med(sw),
        "pct_single_sentence_src": round(100 * sum(1 for x in src_s if x <= 1) / len(src_s), 1),
    }


def main() -> None:
    D = ROOT / "data"
    rows = [from_medsnip()]
    for path, key, label in [
        (D / "6-snippet-processor/atom-to-snippet/gpt-5.4-high", "full_text",
         "MedSNIP-Bench (auto M1)"),
        (D / "9-healthfc/snippet-processor/atom-to-snippet/healthfc.json", "en_claim",
         "HealthFC (raw question)"),
        (D / "9-healthfc/snippet-processor/atom-to-snippet/declarative/healthfc.json",
         "en_claim", "HealthFC (declarative)"),
        (D / "10-medhallu/snippet-processor/atom-to-snippet/medhallu.json", "answer",
         "MedHallu"),
    ]:
        if path.is_dir():  # MedSNIP auto: one file per entry
            recs = []
            for f in sorted(path.glob("e*.json")):
                recs.append(json.loads(f.read_text()))
            tmp = path.parent / "_tmp_lengths.json"
            tmp.write_text(json.dumps(recs))
            r = from_decomp(tmp, key, label)
            tmp.unlink()
        else:
            r = from_decomp(path, key, label)
        if r:
            rows.append(r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    L = ["# How much structure does each corpus offer?", "",
         "MedSNIP's argument is that atomization fragments long, context-rich "
         "answers. That only applies where the source has multiple clauses to "
         "fragment. `merge ratio` is atoms per snippet: 1.0 means no grouping "
         "happened at all.", "",
         "| Corpus | sources | src words (med) | src sents (med) | % 1-sentence | "
         "atoms/src | snips/src | merge | atom words | snip words |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['corpus']} | {r['n_sources']} | {r['src_words_med']} | "
                 f"{r['src_sents_med']} | {r['pct_single_sentence_src']}% | "
                 f"{r['atoms_per_src']} | {r['snips_per_src']} | {r['merge_ratio']} | "
                 f"{r['atom_words_med']} | {r['snip_words_med']} |")
    print("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
