"""Unit tests for DailyMed client retry behavior."""
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
