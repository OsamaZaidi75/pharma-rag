"""Unit tests for Settings.llm_health_info() (the GET /health LLM status fields).

Covers {provider, model, configured} for each provider; Ollama needs no key
so it is always configured, while Gemini/OpenAI require their API keys.
Asserts no secret values leak into the status dict.
"""
import pytest

from src.config import Settings


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    # Ambient keys would flip `configured`; the tests pin each case explicitly.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_ollama_is_always_configured():
    info = Settings(llm_provider="ollama").llm_health_info()
    assert info == {"provider": "ollama", "model": "llama3.1", "configured": True}


def test_gemini_unconfigured_without_key():
    info = Settings(llm_provider="gemini", gemini_api_key=None).llm_health_info()
    assert info["provider"] == "gemini"
    assert info["model"] == Settings().gemini_model
    assert info["configured"] is False


def test_gemini_configured_with_key():
    info = Settings(llm_provider="gemini", gemini_api_key="secret").llm_health_info()
    assert info["configured"] is True
    assert "secret" not in str(info.values())


def test_openai_unconfigured_without_key():
    info = Settings(llm_provider="openai", openai_api_key=None).llm_health_info()
    assert info["provider"] == "openai"
    assert info["model"] == "gpt-4o-mini"
    assert info["configured"] is False


def test_openai_configured_with_key():
    info = Settings(llm_provider="openai", openai_api_key="secret").llm_health_info()
    assert info["configured"] is True
    assert "secret" not in str(info.values())


def test_status_has_exactly_three_fields():
    info = Settings().llm_health_info()
    assert set(info) == {"provider", "model", "configured"}
