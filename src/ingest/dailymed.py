"""DailyMed v2 API client.

Docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
No auth required. Be polite: ~5 req/s max.

Transient failures (network blips, 5xx, 429) are retried with exponential
backoff + jitter so a single hiccup doesn't kill an on-demand download.
"""
import logging
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


def download_spl_xml(setid: str) -> bytes:
    """Download the full Structured Product Label XML for a setid."""
    resp = _get_raw(f"{BASE_URL}/spls/{setid}.xml", timeout=60)
    return resp.content


def search_drug_names(query: str, pagesize: int = 20) -> list[str]:
    """Normalize a free-text query against the DailyMed drug name index."""
    data = _get(
        f"{BASE_URL}/drugnames.json",
        params={"drug_name": query, "pagesize": pagesize},
    )
    return [item["drug_name"] for item in data.get("data", [])]
