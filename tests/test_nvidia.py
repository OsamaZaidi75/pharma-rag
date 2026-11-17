"""Unit tests for the NVIDIA API LLM provider (the hosted default).

Health-info tests run in CI (pydantic-settings only). Client tests need the
`openai` package and are skipped where it is not installed.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.generation.answer import generate_answer, get_llm_client
from src.retrieval.retriever import RetrievedChunk


def _need_openai():
    pytest.importorskip("openai", reason="openai not installed in CI env")


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)


def test_nvidia_is_the_default_provider():
    assert Settings().llm_provider == "nvidia"


def test_nvidia_defaults_pinned():
    settings = Settings()
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.nvidia_model == "nvidia/nemotron-3-ultra-550b-a55b"


def test_nvidia_unconfigured_without_key():
    info = Settings(llm_provider="nvidia", nvidia_api_key=None).llm_health_info()
    assert info["provider"] == "nvidia"
    assert info["model"] == Settings().nvidia_model
    assert info["configured"] is False


def test_nvidia_configured_with_key():
    info = Settings(
        llm_provider="nvidia", nvidia_api_key="nvapi-test"
    ).llm_health_info()
    assert info["configured"] is True
    assert "nvapi-test" not in str(info.values())


def test_get_llm_client_requires_key(monkeypatch):
    _need_openai()
    import src.generation.answer as answer_mod

    monkeypatch.setattr(answer_mod.settings, "llm_provider", "nvidia")
    monkeypatch.setattr(answer_mod.settings, "nvidia_api_key", None)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is not set"):
        get_llm_client()


def test_get_llm_client_uses_nvidia_endpoint(monkeypatch):
    _need_openai()
    import src.generation.answer as answer_mod

    monkeypatch.setattr(answer_mod.settings, "llm_provider", "nvidia")
    monkeypatch.setattr(answer_mod.settings, "nvidia_api_key", "nvapi-test")
    with patch("openai.OpenAI") as mock_openai:
        client, model = get_llm_client()
        assert client is mock_openai.return_value
        assert model == "nvidia/nemotron-3-ultra-550b-a55b"
        _, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert kwargs["api_key"] == "nvapi-test"


def test_generate_answer_falls_back_without_key(monkeypatch):
    """No key -> the missing-key error is caught and the cited extractive
    fallback answers instead of raising."""
    import src.generation.answer as answer_mod

    monkeypatch.setattr(answer_mod.settings, "llm_provider", "nvidia")
    monkeypatch.setattr(answer_mod.settings, "nvidia_api_key", None)
    chunks = [
        RetrievedChunk(
            id=1,
            drug_name="atorvastatin",
            section_title="Contraindications",
            content="Contraindicated in pregnancy and hypersensitivity.",
            score=0.92,
        )
    ]
    ans = generate_answer("What are the contraindications?", chunks)
    assert "extractive-fallback" in ans.model
    assert "nvidia" in ans.model
    assert "[1]" in ans.text  # still cited
    assert ans.sources and ans.sources[0]["drug_name"] == "atorvastatin"


def test_generate_answer_uses_nvidia_client_when_configured(monkeypatch):
    """With a key set, the NVIDIA client is called (mocked — no network)."""
    _need_openai()
    import src.generation.answer as answer_mod

    monkeypatch.setattr(answer_mod.settings, "llm_provider", "nvidia")
    monkeypatch.setattr(answer_mod.settings, "nvidia_api_key", "nvapi-test")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Cited answer [1]."))]
    )
    chunks = [
        RetrievedChunk(
            id=1,
            drug_name="atorvastatin",
            section_title="Contraindications",
            content="Contraindicated in pregnancy.",
            score=0.92,
        )
    ]
    with patch(
        "src.generation.answer.get_llm_client",
        return_value=(mock_client, "nvidia/nemotron-3-ultra-550b-a55b"),
    ):
        ans = generate_answer("What are the contraindications?", chunks)
    assert ans.model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert "Cited answer [1]" in ans.text
    mock_client.chat.completions.create.assert_called_once()
    # Reasoning-model budget: the model thinks before answering and its
    # thinking consumes completion tokens first, so a tight max_tokens can
    # return no content at all. The answer call must stay generous.
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["max_tokens"] >= 2048
    assert kwargs["temperature"] == pytest.approx(0.2)
