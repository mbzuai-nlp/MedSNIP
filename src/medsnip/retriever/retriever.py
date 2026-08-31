"""Unified retrieval interface.

    from src.medsnip.retriever import Retriever
    r = Retriever(index="web")     # serper.dev
    r = Retriever(index="pubmed")  # NCBI E-utilities
    hits = r.search("...", k=5)

Hits are cached to disk so reruns are free.
"""
from dataclasses import dataclass
from pathlib import Path

from .cache import JsonCache
from .pubmed_eutils import search_pubmed
from .web_serper import search_serper

INDEX_BACKENDS = {
    "web":    search_serper,
    "pubmed": search_pubmed,
}

# data/7-retriever/cache/<index>.jsonl by default
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "7-retriever" / "cache"


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    source_url: str
    source: str
    score: float

    @classmethod
    def from_raw(cls, d: dict) -> "Hit":
        return cls(
            doc_id=str(d["doc_id"]),
            title=d.get("title", ""),
            text=d.get("text", ""),
            source_url=d.get("source_url", ""),
            source=d.get("source", ""),
            score=float(d.get("score", 0.0)),
        )


class Retriever:
    def __init__(self, index: str, cache_path: str | Path | None = None):
        if index not in INDEX_BACKENDS:
            raise ValueError(f"unknown index {index!r}; must be one of {list(INDEX_BACKENDS)}")
        self.index = index
        self.backend = INDEX_BACKENDS[index]
        self.cache = JsonCache(cache_path or DEFAULT_CACHE_ROOT / f"{index}.jsonl")

    def search(self, query: str, k: int = 5, refresh: bool = False) -> list[Hit]:
        if not refresh:
            cached = self.cache.get(self.index, query, k)
            if cached is not None:
                return [Hit.from_raw(h) for h in cached]
        raw = self.backend(query, k=k)
        self.cache.set(self.index, query, k, data=raw)
        return [Hit.from_raw(h) for h in raw]
