"""On-demand indexing: if a question is about a drug that's not in the index,
fetch its label from DailyMed, index it, then let the normal pipeline run.

First question about a new drug takes ~30-60s (download + embed once);
every question after that is instant.
"""
import logging
import re
import threading

from ..config import settings
from . import dailymed
from .chunk import chunk_sections
from .spl_parser import parse_spl
from ..store import vectorstore

log = logging.getLogger(__name__)

_lock = threading.Lock()
_WORD = re.compile(r"[a-zA-Z][a-zA-Z\-]{3,}")


def _candidate_terms(question: str) -> list[str]:
    seen, out = set(), []
    for w in _WORD.findall(question):
        w = w.lower()
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def ensure_drug_indexed(question: str, embedder) -> str | None:
    """Index the asked-about drug if missing. Returns matched drug name or None."""
    indexed = {d.lower() for d in vectorstore.list_drugs()}
    terms = _candidate_terms(question)
    for term in terms:
        if term in indexed:
            return term  # already indexed, nothing to do

    with _lock:
        # Re-check inside the lock: another request may have indexed it.
        indexed = {d.lower() for d in vectorstore.list_drugs()}
        for term in terms:
            if term in indexed:
                return term
            try:
                names = dailymed.search_drug_names(term, pagesize=5)
            except Exception as e:
                log.warning("drug name search failed for %s: %s", term, e)
                continue
            # Prefer an exact name match over combo products.
            names.sort(key=lambda n: n.lower() != term)
            for name in names:
                drug = name.lower()
                if drug in indexed:
                    return drug
                try:
                    spls = dailymed.search_spls(name, pagesize=3)
                    if not spls:
                        continue
                    spl = spls[0]
                    log.info("on-demand indexing: %s (%s)", name, spl.setid)
                    xml = dailymed.download_spl_xml(spl.setid)
                    sections = parse_spl(xml, drug_name=drug, setid=spl.setid)
                    chunks = chunk_sections(
                        sections,
                        max_chars=settings.chunk_max_chars,
                        overlap_chars=settings.chunk_overlap_chars,
                    )
                    if not chunks:
                        continue
                    embeddings = embedder.embed([c.content for c in chunks])
                    vectorstore.upsert_chunks(
                        chunks, embeddings, delete_setids={spl.setid}
                    )
                    log.info("indexed %d chunks for %s", len(chunks), drug)
                    return drug
                except Exception as e:
                    log.warning("on-demand index failed for %s: %s", name, e)
                    continue
    return None
