"""Unit tests for DailyMed client retry behavior and download caching."""
import os
import time

import httpx
import pytest

from src.ingest import dailymed


def _resp(status: int, url: str = "https://dailymed.nlm.nih.gov/x") -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", url))


def test_transient_network_error_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom")
        return _resp(200)

    monkeypatch.setattr(dailymed.httpx, "get", fake_get)
    monkeypatch.setattr(dailymed.time, "sleep", lambda s: None)
    monkeypatch.setattr(dailymed.random, "uniform", lambda a, b: 0)
    resp = dailymed._get_raw("https://example.com")
    assert resp.status_code == 200
    assert calls["n"] == 3


def test_transient_503_gives_up_after_max_attempts(monkeypatch):
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return _resp(503)

    monkeypatch.setattr(dailymed.httpx, "get", fake_get)
    monkeypatch.setattr(dailymed.time, "sleep", lambda s: None)
    monkeypatch.setattr(dailymed.random, "uniform", lambda a, b: 0)
    with pytest.raises(httpx.HTTPStatusError):
        dailymed._get_raw("https://example.com")
    assert calls["n"] == dailymed._RETRY_ATTEMPTS


def test_client_error_does_not_retry(monkeypatch):
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return _resp(404)

    monkeypatch.setattr(dailymed.httpx, "get", fake_get)
    monkeypatch.setattr(dailymed.time, "sleep", lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        dailymed._get_raw("https://example.com")
    assert calls["n"] == 1


# --- download_spl_xml on-disk cache (TTL) ---

SETID = "a" * 8 + "-" + "b" * 4 + "-" + "c" * 4 + "-" + "d" * 4 + "-" + "e" * 12
XML_BODY = b"<?xml version='1.0'?><document><title>TEST</title></document>"


class _XmlResponse(httpx.Response):
    """httpx.Response with a preset .content payload."""

    def __init__(self, xml: bytes):
        super().__init__(200, content=xml,
                         request=httpx.Request("GET", "https://dailymed.nlm.nih.gov/x"))


def _fresh_cache(tmp_path, setid: str, xml: bytes) -> str:
    path = dailymed._cache_file(setid, str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(xml)
    return path


def test_download_cache_miss_downloads_and_writes_cache(monkeypatch, tmp_path):
    calls = {"n": 0}
    monkeypatch.setattr(dailymed.httpx, "get",
                        lambda *a, **k: (_XmlResponse(XML_BODY), calls.__setitem__("n", calls["n"] + 1))[0])
    out = dailymed.download_spl_xml(SETID, cache_dir=str(tmp_path), ttl_days=7)
    assert out == XML_BODY
    assert calls["n"] == 1
    assert os.path.isfile(dailymed._cache_file(SETID, str(tmp_path)))


def test_download_cache_hit_skips_http(monkeypatch, tmp_path):
    _fresh_cache(tmp_path, SETID, XML_BODY)
    monkeypatch.setattr(dailymed.httpx, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("HTTP should not be called")))
    out = dailymed.download_spl_xml(SETID, cache_dir=str(tmp_path), ttl_days=7)
    assert out == XML_BODY


def test_download_stale_cache_refetches(monkeypatch, tmp_path):
    path = _fresh_cache(tmp_path, SETID, b"<old/>")
    old_mtime = time.time() - 10 * 86400  # 10 days old
    os.utime(path, (old_mtime, old_mtime))
    calls = {"n": 0}
    monkeypatch.setattr(dailymed.httpx, "get",
                        lambda *a, **k: (_XmlResponse(XML_BODY), calls.__setitem__("n", calls["n"] + 1))[0])
    out = dailymed.download_spl_xml(SETID, cache_dir=str(tmp_path), ttl_days=7)
    assert out == XML_BODY
    assert calls["n"] == 1
    with open(path, "rb") as f:
        assert f.read() == XML_BODY  # stale file refreshed


def test_download_no_cache_dir_behaves_as_before(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(dailymed.httpx, "get",
                        lambda *a, **k: (_XmlResponse(XML_BODY), calls.__setitem__("n", calls["n"] + 1))[0])
    out = dailymed.download_spl_xml(SETID)
    assert out == XML_BODY
    assert calls["n"] == 1
