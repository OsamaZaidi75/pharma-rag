"""DailyMed v2 API client.

Docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
No auth required. Be polite: ~5 req/s max.
"""
import time
from dataclasses import dataclass

import httpx

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
_RATE_LIMIT_DELAY = 0.25  # seconds between requests


@dataclass
class SplSummary:
    setid: str
    title: str
    published_date: str


def _get(url: str, params: dict | None = None) -> dict:
    time.sleep(_RATE_LIMIT_DELAY)
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


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
    time.sleep(_RATE_LIMIT_DELAY)
    resp = httpx.get(f"{BASE_URL}/spls/{setid}.xml", timeout=60)
    resp.raise_for_status()
    return resp.content


def search_drug_names(query: str, pagesize: int = 20) -> list[str]:
    """Normalize a free-text query against the DailyMed drug name index."""
    data = _get(
        f"{BASE_URL}/drugnames.json",
        params={"drug_name": query, "pagesize": pagesize},
    )
    return [item["drug_name"] for item in data.get("data", [])]
