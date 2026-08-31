"""NCBI E-utilities wrapper for PubMed retrieval.

Two-step: esearch (query -> PMIDs) then efetch (PMIDs -> abstracts).
Free, no API key required for our request volume. With an email registered
the rate cap goes from 3 req/s to 10 req/s.
"""
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

NCBI_EMAIL = os.environ.get("NCBI_EMAIL")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")

TIMEOUT = 30.0
# crude rate-limit cushion: 0.35s between requests = ~3 req/s
_RATE_DELAY = 0.35 if not NCBI_API_KEY else 0.11
_last_request = 0.0


def _pace():
    global _last_request
    now = time.time()
    wait = _RATE_DELAY - (now - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def _params(extra: dict[str, Any]) -> dict[str, Any]:
    p = dict(extra)
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    p["tool"] = "medsnip"
    return p


def _esearch(query: str, retmax: int) -> list[str]:
    _pace()
    r = requests.get(ESEARCH, params=_params({
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    }), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: list[str]) -> list[dict[str, Any]]:
    if not pmids:
        return []
    _pace()
    r = requests.get(EFETCH, params=_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }), timeout=TIMEOUT)
    r.raise_for_status()
    return _parse_pubmed_xml(r.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        # abstract may have multiple <AbstractText> sections with labels
        abs_pieces = []
        for at in art.findall(".//Abstract/AbstractText"):
            label = at.get("Label")
            text = "".join(at.itertext())
            if not text:
                continue
            abs_pieces.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abs_pieces).strip()
        year_el = art.find(".//PubDate/Year") or art.find(".//PubDate/MedlineDate")
        year = year_el.text if year_el is not None else ""
        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""
        out.append({
            "pmid": pmid,
            "title": title.strip(),
            "abstract": abstract,
            "year": year,
            "journal": journal,
        })
    return out


def search_pubmed(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return up to k PubMed hits sorted by relevance."""
    pmids = _esearch(query, retmax=k)
    articles = _efetch(pmids)
    out = []
    for pos, a in enumerate(articles):
        if not a.get("abstract") and not a.get("title"):
            continue
        out.append({
            "doc_id":     f"pmid:{a['pmid']}",
            "title":      a["title"],
            "text":       a["abstract"] or a["title"],
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/",
            "source":     "pubmed",
            "score":      k - pos,
            "year":       a.get("year", ""),
            "journal":    a.get("journal", ""),
        })
    return out
