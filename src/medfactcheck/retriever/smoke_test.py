"""Quick sanity check for both retriever backends.

Usage:
    python -m src.medfactcheck.retriever.smoke_test
"""
from .retriever import Retriever


def main():
    for idx in ("web", "pubmed"):
        r = Retriever(index=idx)
        hits = r.search("ibuprofen maximum daily dose adults", k=3)
        print(f"\n=== {idx} ===  ({len(hits)} hits, cache size={len(r.cache)})")
        for h in hits:
            print(f"  [{h.source}] {h.title[:80]}")
            print(f"    {h.text[:160]}")


if __name__ == "__main__":
    main()
