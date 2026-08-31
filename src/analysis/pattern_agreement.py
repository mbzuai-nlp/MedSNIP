"""Do automatic decomposers reproduce the human A-F pattern taxonomy?

The pipeline asks each decomposer to tag every snippet it produces with one of
the six structural pattern codes. Those tags have never been checked against
the human annotations, so this asks two things:

  1. Does the decomposer's pattern *distribution* match the human one? A
     decomposer that never emits Pattern B is not seeing causal chains.
  2. On snippets whose sentence coverage matches a human snippet closely
     enough to be the same unit, does the assigned code agree?

Only (2) is agreement in the usual sense; (1) is a weaker distributional check
that still works when boundaries differ, which they usually do.

Matching is by sentence-set Jaccard over the sentences each snippet covers,
reusing the alignment the fidelity evaluation already computes. A pair counts
as the same unit at Jaccard >= MATCH_THRESHOLD; unmatched auto snippets are
reported rather than dropped silently.

Usage:
  python -m src.analysis.pattern_agreement
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SP_ROOT = ROOT / "data" / "6-snippet-processor"
ALIGN = SP_ROOT / "sentence_alignment.json"
FINAL = ROOT / "data" / "!-final" / "dataset.json"
OUT_DIR = ROOT / "data" / "11-analysis" / "pattern-analysis"
OUT_JSON = OUT_DIR / "pattern_agreement.json"

PATTERNS = ["A", "B", "C", "D", "E", "F"]
MATCH_THRESHOLD = 0.5


def human_units(alignment: dict) -> dict[int, list[dict]]:
    """Human snippets with their covered sentence sets and pattern codes."""
    out: dict[int, list[dict]] = defaultdict(list)
    for r in json.loads(FINAL.read_text()):
        pat = (r.get("pattern") or "").strip().upper()[:1]
        if pat not in PATTERNS:
            continue
        eid = r["entry_id"]
        a2s = (alignment.get(str(eid)) or {}).get("atom_to_sentence", {}) or {}
        # atom_to_sentence maps an atom index to ONE sentence index (an int),
        # not a list, so collect rather than update.
        sents = set()
        for a in (r.get("atoms") or []):
            v = a2s.get(str(a.get("index")))
            if v is None:
                continue
            sents.update(v if isinstance(v, (list, tuple, set)) else [v])
        out[eid].append({"pattern": pat, "sents": sents,
                         "snippet_id": r["snippet_id"]})
    return out


def auto_units(slug_dir: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(slug_dir.glob("e*.json")):
        rec = json.loads(f.read_text())
        eid = rec["entry_id"]
        atom_sents = {int(a["index"]): set(a.get("source_sentences") or [])
                      for a in (rec.get("generated_atoms") or [])}
        for s in (rec.get("snippets") or []):
            if not isinstance(s, dict):
                continue
            pat = (s.get("pattern") or "").strip().upper()[:1]
            if atom_sents:
                sents = set()
                for i in (s.get("source_claims") or []):
                    sents |= atom_sents.get(int(i), set())
            else:
                sents = {int(i) for i in (s.get("source_sentences") or [])}
            out[eid].append({"pattern": pat, "sents": sents})
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None)
    args = ap.parse_args()

    if not ALIGN.exists():
        raise SystemExit(f"missing {ALIGN}")
    alignment = json.loads(ALIGN.read_text())
    human = human_units(alignment)
    human_dist = Counter(u["pattern"] for us in human.values() for u in us)

    slugs = args.slugs or [d.name for d in sorted((SP_ROOT / "atom-to-snippet").iterdir())
                           if d.is_dir()]
    rows = []
    for slug in slugs:
        auto = auto_units(SP_ROOT / "atom-to-snippet" / slug)
        if not auto:
            continue
        dist = Counter()
        matched = agree = 0
        confusion: Counter = Counter()
        unmatched = 0
        for eid, units in auto.items():
            hu = human.get(eid, [])
            for u in units:
                dist[u["pattern"]] += 1
                best, bj = None, 0.0
                for h in hu:
                    j = jaccard(u["sents"], h["sents"])
                    if j > bj:
                        best, bj = h, j
                if best is None or bj < MATCH_THRESHOLD:
                    unmatched += 1
                    continue
                matched += 1
                confusion[(best["pattern"], u["pattern"])] += 1
                if best["pattern"] == u["pattern"]:
                    agree += 1
        total = sum(dist.values())
        rows.append({
            "slug": slug, "snippets": total, "matched": matched,
            "unmatched": unmatched,
            "accuracy": round(agree / matched, 4) if matched else None,
            "distribution": {p: dist.get(p, 0) for p in PATTERNS},
            "distribution_pct": {p: round(100 * dist.get(p, 0) / total, 1)
                                 for p in PATTERNS} if total else {},
            "confusion": {f"{h}->{a}": n for (h, a), n in sorted(confusion.items())},
        })

    payload = {"human_distribution": {p: human_dist.get(p, 0) for p in PATTERNS},
               "match_threshold": MATCH_THRESHOLD, "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    ht = sum(human_dist.values())
    L = ["# Do decomposers reproduce the human A-F pattern taxonomy?", "",
         "Mode 1 snippets. `agree` is pattern accuracy on auto snippets whose "
         f"covered-sentence set overlaps a human snippet at Jaccard >= "
         f"{MATCH_THRESHOLD} (i.e. plausibly the same unit); `unmatched` are auto "
         "snippets with no such counterpart, which is a boundary disagreement "
         "rather than a labelling one.", "",
         "| Decomposer | snippets | matched | agree | " +
         " | ".join(f"%{p}" for p in PATTERNS) + " |",
         "|---|---:|---:|---:|" + "---:|" * len(PATTERNS)]
    L.append("| *human gold* | " + f"{ht} | — | — | " +
             " | ".join(f"{100*human_dist.get(p,0)/ht:.1f}" for p in PATTERNS) + " |")
    for r in rows:
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "—"
        L.append(f"| {r['slug']} | {r['snippets']} | {r['matched']} | {acc} | " +
                 " | ".join(f"{r['distribution_pct'].get(p, 0):.1f}" for p in PATTERNS) + " |")
    print("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
