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

STOPWORDS = {
    # Standard English function words & question words
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Specific query / conversational words
    "help", "helps", "helping", "with", "uses", "use", "using", "used", "drug", "drugs",
    "child", "children", "needs", "need", "needed", "needing", "tell", "tells", "telling",
    "about", "what", "which", "dose", "doses", "dosage", "dosages", "dosing", "safe", "safety",
    "please", "find", "search", "lookup", "query", "question", "ask", "asking", "give", "gives",
    "show", "showing", "know", "knowing", "take", "takes", "taking", "taken",
    "prescribe", "prescribed", "prescription",
    "information", "info", "detail", "details", "label", "labels", "fda", "package", "insert",
    "section", "sections", "contraindication", "contraindications", "warning", "warnings",
    "precaution", "precautions", "indication", "indications", "interaction", "interactions",
    "adverse", "reaction", "reactions", "effect", "effects", "side", "symptom", "symptoms",
    "treatment", "treat", "treats", "treating", "treated", "cure", "cures", "prevent", "prevention",
    "patient", "patients", "adult", "adults", "person", "people", "man", "men", "woman", "women",
    "baby", "babies", "infant", "infants", "elderly", "senior", "seniors",
    "disease", "diseases", "condition", "conditions", "illness", "illnesses",
    "pain", "relief", "ache", "fever", "cough", "cold", "flu", "allergy", "allergies",
    "recommend", "recommended", "recommendation", "guideline", "guidelines", "guide",
    "storage", "store", "overdose", "missed", "miss", "start", "stop", "change",
    "many", "much", "often", "long", "good", "bad", "better", "best", "work", "works",
}

_ALLOWED_FORM_TOKENS = {
    # Strengths and units
    "mg", "mcg", "ug", "g", "kg", "ml", "l", "meq", "iu", "unit", "units", "grain", "grains", "ct", "count",
    "percent", "%",
    # Salts & chemical forms
    "calcium", "hydrochloride", "hcl", "sodium", "potassium", "dihydrate", "trihydrate",
    "monohydrate", "hemihydrate", "maleate", "succinate", "tartrate", "citrate", "phosphate",
    "sulfate", "fumarate", "mesylate", "besylate", "acetate", "valerate", "dipropionate",
    "propionate", "nitrate", "oxide", "magnesium", "zinc", "aluminum", "carbonate", "chloride",
    "gluconate", "lactate", "salicylate",
    # Dosage forms & routes
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps", "caplet", "caplets",
    "solution", "suspension", "syrup", "elixir", "liquid", "drops", "drop", "spray", "inhalation",
    "aerosol", "powder", "granules", "cream", "ointment", "gel", "lotion", "patch", "patches",
    "injection", "injectable", "oral", "topical", "ophthalmic", "otic", "nasal", "sublingual",
    "buccal", "rectal", "vaginal", "chewable", "effervescent", "film", "wafer", "wafers", "lozenge", "lozenges",
    "packet", "packets", "vial", "vials", "ampule", "ampules", "syringe", "syringes",
    # Release types & modifiers
    "er", "xr", "sr", "cr", "dr", "ec", "ir", "xl", "la", "cd",
    "extended", "delayed", "sustained", "controlled", "immediate", "release",
    "enteric", "coated", "uncoated", "low", "dose", "high", "regular", "extra", "maximum",
    "strength", "adult", "pediatric", "dye", "free", "dye-free", "sugar-free", "preservative-free",
    "buffered", "nsaid", "usp", "nf", "bp",
}

_NUM_UNIT_RE = re.compile(r"^\d+(\.\d+)?(mg|mcg|ug|g|ml|meq|iu|ct|%|grain|grains)?$", re.IGNORECASE)


def matches_drug_term(term: str, drug_name: str) -> bool:
    """Only accept a DailyMed result if the returned drug name matches the term
    exactly (case-insensitive), or is exactly '<term> <strength/form>' (e.g. term 'aspirin'
    may match 'ASPIRIN 81 MG'). Reject everything else (no fuzzy contains-matching).
    """
    t = term.strip().lower()
    n = drug_name.strip().lower()
    if n == t:
        return True
    if not (n.startswith(t + " ") or n.startswith(t + "-")):
        return False
    remainder = n[len(t):].strip(" -")
    # Reject combination products or multiple active ingredients
    if re.search(r"(\band\b|\bwith\b|\bw/\b|[,&+/])", remainder):
        return False
    raw_tokens = re.findall(r"[a-zA-Z0-9%.-]+", remainder)
    if not raw_tokens:
        return False
    for raw in raw_tokens:
        parts = [p.strip(".-") for p in re.split(r"[-/]", raw) if p.strip(".-")]
        for tok in parts:
            if _NUM_UNIT_RE.match(tok):
                continue
            if tok in _ALLOWED_FORM_TOKENS:
                continue
            return False
    return True


def _candidate_terms(question: str) -> list[str]:
    seen = set()
    out = []
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]*[a-zA-Z]|[a-zA-Z]", question)
    for w in words:
        w_low = w.lower().strip("-")
        if len(w_low) >= 3 and w_low not in STOPWORDS and w_low not in seen:
            seen.add(w_low)
            out.append(w_low)
    # Try candidate terms longest-first so specific words win over short ones
    out.sort(key=lambda t: (len(t), t), reverse=True)
    return out


def _resolve_spls(term: str) -> list:
    """Find DailyMed SPLs for a drug term, best first.

    Path 1: the drugnames index with strict exact/<term> <strength/form>
    matching (avoids combination products and wrong drugs).
    Path 2 (fallback): search SPLs directly. The drugnames endpoint sometimes
    buries the plain generic name — e.g. "ibuprofen" returns only brand and
    combination names in the first 100 — while spls.json reliably returns that
    drug's labels.

    Candidates from both paths are de-duplicated; single-ingredient SPLs
    (title starts with the term, no " AND ") are preferred. If none qualify,
    the first candidate is returned (preserves the old take-spls[0] behavior).
    """
    candidates: list = []
    seen: set = set()

    def _add(spls) -> None:
        for s in spls or []:
            if s.setid not in seen:
                seen.add(s.setid)
                candidates.append(s)

    try:
        names = dailymed.search_drug_names(term, pagesize=100)
    except Exception as e:
        log.warning("drug name search failed for %s: %s", term, e)
        names = []

    valid_names = [n for n in names if matches_drug_term(term, n)]
    valid_names.sort(key=lambda n: (n.strip().lower() != term, len(n)))
    for canonical_name in valid_names[:3]:
        try:
            _add(dailymed.search_spls(canonical_name, pagesize=3))
        except Exception as e:
            log.warning("spl search failed for %s: %s", canonical_name, e)

    try:
        _add(dailymed.search_spls(term, pagesize=10))
    except Exception as e:
        log.warning("spl search failed for %s: %s", term, e)

    t = term.upper()
    single = [
        s
        for s in candidates
        if s.title.strip().upper().startswith(t) and " AND " not in s.title.upper()
    ]
    return single or candidates[:1]


def ensure_drug_indexed(question: str, embedder) -> str | None:
    """Index the asked-about drug if missing. Returns matched canonical drug name or None."""
    indexed = {d.lower() for d in vectorstore.list_drugs()}
    terms = _candidate_terms(question)
    if not terms:
        return None

    for term in terms:
        if term in indexed:
            return None  # already indexed, nothing to do

    with _lock:
        # Re-check inside the lock: another request may have indexed it.
        indexed = {d.lower() for d in vectorstore.list_drugs()}
        for term in terms:
            if term in indexed:
                return None
            spls = _resolve_spls(term)
            if not spls:
                continue
            for spl in spls[:3]:
                try:
                    log.info("on-demand indexing: %s -> %s (%s)", term, spl.title, spl.setid)
                    xml = dailymed.download_spl_xml(spl.setid)
                    sections = parse_spl(xml, drug_name=term, setid=spl.setid)
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
                    log.info("indexed %d chunks for %s", len(chunks), term)
                    return term
                except Exception as e:
                    log.warning("on-demand index failed for %s (%s): %s", term, spl.setid, e)
                    continue
    return None

