"""DailyMed v2 API client.

Docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
No auth required. Be polite: ~5 req/s max.

Transient failures (network blips, 5xx, 429) are retried with exponential
backoff + jitter so a single hiccup doesn't kill an on-demand download.
"""
import logging
import os
import random
import time
from dataclasses import dataclass

import httpx

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
_RATE_LIMIT_DELAY = 0.25  # seconds between requests

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubled each attempt + up to 1s jitter

log = logging.getLogger(__name__)


@dataclass
class SplSummary:
    setid: str
    title: str
    published_date: str


def _transient(exc: BaseException) -> bool:
    """True if the error is worth retrying (network/5xx/429, not 4xx)."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    return False


def _get_raw(url: str, params: dict | None = None, timeout: float = 30) -> httpx.Response:
    """GET with retry: transient failures retry up to _RETRY_ATTEMPTS times."""
    last_exc: BaseException | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        time.sleep(_RATE_LIMIT_DELAY)
        try:
            resp = httpx.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - retry policy needs broad catch
            last_exc = exc
            if not _transient(exc) or attempt == _RETRY_ATTEMPTS:
                raise
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
            log.warning(
                "DailyMed request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                _RETRY_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _get(url: str, params: dict | None = None) -> dict:
    return _get_raw(url, params=params, timeout=30).json()


def search_spls(drug_name: str, pagesize: int = 10) -> list[SplSummary]:
    """Return SPL summaries for a drug name, newest first."""
    data = _get(
        f"{BASE_URL}/spls.json",
        params={"drug_name": drug_name, "pagesize": pagesize},
    )
    results = []
    for item in data.get("data", []):
        results.append(
            SplSummary(
                setid=item["setid"],
                title=item.get("title", ""),
                published_date=item.get("published_date", ""),
            )
        )
    return results


def download_spl_xml(setid: str, cache_dir: str | None = None,
                     ttl_days: float = 7.0) -> bytes:
    """Download the full Structured Product Label XML for a setid.

    When cache_dir is given, the XML is cached on disk (keyed by setid) and a
    fresh cache hit (younger than ttl_days) skips the HTTP request entirely —
    re-asked drugs re-index from the local copy instead of re-downloading.
    Stale cache files are silently refreshed.
    """
    if cache_dir:
        cached = _read_xml_cache(setid, cache_dir, ttl_days)
        if cached is not None:
            log.info("DailyMed cache hit for %s", setid)
            return cached
    resp = _get_raw(f"{BASE_URL}/spls/{setid}.xml", timeout=60)
    content = resp.content
    if cache_dir:
        _write_xml_cache(setid, cache_dir, content)
    return content


def _cache_file(setid: str, cache_dir: str) -> str:
    safe = "".join(c for c in setid if c.isalnum() or c in "-_")
    return os.path.join(cache_dir, f"{safe}.xml")


def _read_xml_cache(setid: str, cache_dir: str, ttl_days: float) -> bytes | None:
    """Return cached XML if present and fresher than ttl_days, else None."""
    path = _cache_file(setid, cache_dir)
    try:
        age_days = (time.time() - os.path.getmtime(path)) / 86400
    except OSError:
        return None  # no cache file
    if age_days > ttl_days:
        log.info("DailyMed cache stale for %s (age %.1fd > %.1fd) — refetching",
                 setid, age_days, ttl_days)
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        log.warning("DailyMed cache read failed for %s: %s", setid, e)
        return None


def _write_xml_cache(setid: str, cache_dir: str, content: bytes) -> None:
    """Persist downloaded XML atomically (tmp + rename); failures are logged only."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = _cache_file(setid, cache_dir) + f".tmp-{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(content)
        os.replace(tmp, _cache_file(setid, cache_dir))
    except OSError as e:
        log.warning("DailyMed cache write failed for %s: %s", setid, e)


def search_drug_names(query: str, pagesize: int = 20) -> list[str]:
    """Normalize a free-text query against the DailyMed drug name index."""
    data = _get(
        f"{BASE_URL}/drugnames.json",
        params={"drug_name": query, "pagesize": pagesize},
    )
    return [item["drug_name"] for item in data.get("data", [])]
