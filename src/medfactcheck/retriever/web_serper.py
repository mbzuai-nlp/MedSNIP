"""serper.dev (Google Search API) wrapper.

Returns ranked organic results plus optionally the knowledge-graph snippet
and answer box. Flattened into a uniform list of Hit dicts.
"""
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Repo root is four parents above this file (src/medfactcheck/retriever/web_serper.py)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

SERPER_URL = "https://google.serper.dev/search"
TIMEOUT = 30.0


def search_serper(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return up to k hits from serper.dev.

    Each hit: {doc_id, score, title, text, source_url, source}.
    `score` is descending integer rank-derived (k - position).
    Includes the answerBox / knowledgeGraph as synthetic top hits when present.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set")

    resp = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": min(k, 10)},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    hits: list[dict[str, Any]] = []

    # answerBox often has the most direct evidence
    if ab := data.get("answerBox"):
        text = ab.get("answer") or ab.get("snippet") or ab.get("snippetHighlighted") or ""
        if text:
            hits.append({
                "doc_id":     ab.get("link", "answer_box"),
                "title":      ab.get("title", "Google Answer Box"),
                "text":       text if isinstance(text, str) else " ".join(text),
                "source_url": ab.get("link", ""),
                "source":     "serper:answer_box",
                "score":      k + 1,  # promote above organic
            })

    # knowledgeGraph: structured entity info
    if kg := data.get("knowledgeGraph"):
        text = kg.get("description") or ""
        if text:
            hits.append({
                "doc_id":     kg.get("descriptionLink", "knowledge_graph"),
                "title":      kg.get("title", "Knowledge Graph"),
                "text":       text,
                "source_url": kg.get("descriptionLink", ""),
                "source":     "serper:knowledge_graph",
                "score":      k + 1,
            })

    # organic results
    for pos, item in enumerate(data.get("organic", [])[:k]):
        hits.append({
            "doc_id":     item.get("link", f"organic_{pos}"),
            "title":      item.get("title", ""),
            "text":       item.get("snippet", ""),
            "source_url": item.get("link", ""),
            "source":     "serper:organic",
            "score":      k - pos,
        })

    return hits[:k]


if __name__ == "__main__":
    res = search_serper("ibuprofen maximum daily dose adults", k=5)
    print(f"got {len(res)} hits")
    for h in res[:3]:
        print(f"  [{h['source']}] {h['title']}")
        print(f"    {h['text'][:140]}")
        print(f"    {h['source_url']}")
