"""Deterministic sentence segmentation + gold-atom→sentence alignment.

The whole evaluation rides on these two operations. Both are intentionally
plain Python (no LLM, no ML) so the comparison is reproducible.

Public API
----------
split_sentences(text) -> list[str]
    Split `text` into sentences. Citation markers ([1], [2,3]) and section
    headers (###, **bold**) are stripped/skipped. Returns the cleaned
    sentences in order.

align_atoms_to_sentences(sentences, atoms) -> dict[int, int]
    For each atom (with a `text` field), return the sentence index whose
    text best matches it. Uses normalized-token Jaccard + substring tests.
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[\s*[\d,\s\-]+\s*\]")          # [1], [1,2], [1-3]
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")                   # **bold**
_HEADER_RE = re.compile(r"^\s*#+\s+", re.MULTILINE)         # ### Header
_LIST_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)  # - bullet
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)    # 1. item
_WS_RE = re.compile(r"\s+")

# Patterns / phrases that mark a line as a meta-header rather than a claim.
_META_HEADERS = {
    "step-by-step explanation", "step-by-step analysis",
    "patient presentation and initial assessment",
    "differential diagnosis", "differential diagnosis analysis",
    "immediate management focus", "option analysis", "options analysis",
    "most appropriate initial step", "most appropriate next step",
    "final answer", "conclusion", "summary", "overview",
    "introduction", "background", "discussion",
    "key considerations", "physical exam findings",
    "medical history and risk factors",
    "statements requiring authoritative references",
    "statement requiring authoritative reference",
}

# A line is a header if any of:
#   - it ends with a colon and has <= 6 words
#   - lowercased text matches a known meta-header phrase
#   - <= 5 words, every word starts uppercase, contains no period/comma
_HEADER_KEYWORDS = {"step", "analysis", "presentation", "diagnosis", "management",
                    "answer", "conclusion", "summary", "overview", "rationale",
                    "discussion", "considerations", "findings", "recommendations"}

# Sentence boundary: ., !, ? followed by whitespace and a capital letter or end.
# Avoid splitting on common abbreviations.
_ABBREV = {"e.g", "i.e", "etc", "vs", "mg", "kg", "lb", "yr", "dr", "mr",
           "ms", "mrs", "st", "no", "fig", "u.s", "u.k", "approx", "mmhg",
           "cm", "mm", "ml", "min", "max"}

_SENT_BOUND_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")


def _strip_inline(text: str) -> str:
    text = _CITATION_RE.sub("", text)
    text = _BOLD_RE.sub(r"\1", text)
    return text


def _is_header(line: str) -> bool:
    """True if the line is a section header / meta-text rather than a claim."""
    s = line.strip(" .:;").rstrip(":")
    if not s:
        return True
    low = s.lower()
    if low in _META_HEADERS:
        return True
    words = s.split()
    # short title-case line with no internal punctuation -> header
    if (len(words) <= 6
            and "." not in s and "," not in s
            and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1)):
        # only flag if any known header keyword present OR ends with colon
        if line.rstrip().endswith(":") or any(w.lower() in _HEADER_KEYWORDS for w in words):
            return True
    # short trailing-colon line
    if line.rstrip().endswith(":") and len(words) <= 8:
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Return cleaned sentences in order. Drops headers / bullets / meta-text."""
    text = _strip_inline(text)
    pieces: list[str] = []
    for line in text.split("\n"):
        line = _HEADER_RE.sub("", line)
        line = _LIST_BULLET_RE.sub("", line)
        line = _NUMBERED_RE.sub("", line)
        line = line.strip()
        if not line:
            continue
        if _is_header(line):
            continue
        chunks = _SENT_BOUND_RE.split(line)
        for c in chunks:
            c = _WS_RE.sub(" ", c).strip(" .;:")
            if not c or len(c.split()) < 2:
                continue
            if _is_header(c):
                continue
            pieces.append(c)
    return pieces


# ---------------------------------------------------------------------------
# Atom-to-sentence alignment
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b[\w']+\b")


def _toks(s: str) -> set[str]:
    return set(t.lower() for t in _TOKEN_RE.findall(s or ""))


def _score(atom_text: str, sent_text: str) -> float:
    a, s = _toks(atom_text), _toks(sent_text)
    if not a or not s:
        return 0.0
    inter = len(a & s)
    # asymmetric: how much of the ATOM is covered by the sentence
    return inter / len(a)


def align_atoms_to_sentences(sentences: list[str],
                              atoms: list[dict]) -> dict[int, int]:
    """Return {atom_index: best_sentence_index}. If no sentence covers the
    atom well, returns -1 for that atom so the caller can flag it."""
    out: dict[int, int] = {}
    for a in atoms:
        best_i, best_s = -1, 0.0
        for i, sent in enumerate(sentences):
            s = _score(a["text"], sent)
            if s > best_s:
                best_s, best_i = s, i
        # require >= 0.5 token recall to call it a match
        out[a["index"]] = best_i if best_s >= 0.5 else -1
    return out


# ---------------------------------------------------------------------------
# Helpers for evaluation
# ---------------------------------------------------------------------------

def snippet_sentence_set(source_claims: Iterable[int],
                          atom_to_sent: dict[int, int]) -> set[int]:
    """Convert a human snippet's source_claims (atom indices) into the set of
    sentence indices it covers. Atoms with no matched sentence (-1) are
    dropped from the set."""
    return {atom_to_sent[a] for a in source_claims
            if atom_to_sent.get(a, -1) >= 0}


if __name__ == "__main__":
    # quick demo on Hasan's e1
    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    data = {e["entry_id"]: e for e in json.load(
        open(os.path.join(here, "..", "5-annotated", "dataset.json")))}
    for eid in (1, 61, 63):
        e = data[eid]
        sents = split_sentences(e["full_text"])
        align = align_atoms_to_sentences(sents, e["full_claims"])
        n_match = sum(1 for v in align.values() if v >= 0)
        print(f"e{eid} [{e['subset']}]: {len(sents)} sentences, "
              f"{n_match}/{len(e['full_claims'])} atoms aligned")
        miss = [a for a in e["full_claims"] if align[a["index"]] < 0]
        for a in miss[:3]:
            print(f"   UNMATCHED atom {a['index']}: {a['text'][:100]}")
