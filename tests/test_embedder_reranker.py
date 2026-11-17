"""Unit tests for the NVIDIA API embedder and the (removed) reranker.

The embedder calls the NVIDIA embeddings endpoint through the `openai`
client — no local model, no torch. Client tests mock `openai.OpenAI`
(no network) and are skipped where `openai` is not installed, mirroring
tests/test_nvidia.py. The cross-encoder reranker was removed with the
torch removal (2026-11-18); the retrieval test below pins the replacement
behavior — RRF order is final, and the `use_reranker` flag is a no-op.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import Settings
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever

EMBED_MODEL = "nvidia/nemotron-3.5-embed-8b"
EMBED_DIM = 4096


def _need_openai():
    pytest.importorskip("openai", reason="openai not installed in CI env")


def _fake_embedding_response(vectors):
    """Mimic the openai client's embeddings.create response shape."""
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v) for v in vectors]
    )


def _mock_client_with(vectors_per_call):
    """Mock OpenAI client whose embeddings.create returns canned vectors.

    `vectors_per_call` is a list of per-call vector lists, so batched calls
    each get their own response.
    """
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        _fake_embedding_response(vs) for vs in vectors_per_call
    ]
    return mock_client


@pytest.fixture(autouse=True)
def _clean_embedding_env(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)


def test_embedding_defaults_pinned():
    settings = Settings()
    assert settings.embedding_model == EMBED_MODEL
    assert settings.embedding_dim == EMBED_DIM


def test_embedder_model_configurable_via_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
    monkeypatch.setenv("EMBEDDING_DIM", "2048")
    settings = Settings()
    assert settings.embedding_model == "nvidia/nemotron-3-embed-1b"
    assert settings.embedding_dim == 2048
    emb = Embedder(settings.embedding_model, dim=settings.embedding_dim)
    assert emb.model_name == "nvidia/nemotron-3-embed-1b"
    assert emb.dim == 2048


def test_embedder_dim_default():
    assert Embedder().dim == EMBED_DIM


def test_embedder_empty_list(monkeypatch):
    """No texts -> no client, no API call, no key needed."""
    import src.embeddings.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod.settings, "nvidia_api_key", None)
    assert Embedder().embed([]) == []


def test_embedder_requires_api_key(monkeypatch):
    _need_openai()
    import src.embeddings.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod.settings, "nvidia_api_key", None)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is not set"):
        Embedder().embed(["some text"])


def test_embedder_uses_nvidia_endpoint(monkeypatch):
    """Embeddings go to the NVIDIA endpoint with the configured key."""
    _need_openai()
    import src.embeddings.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod.settings, "nvidia_api_key", "nvapi-test")
    mock_client = _mock_client_with([[[1.0] + [0.0] * (EMBED_DIM - 1)]])
    with patch("openai.OpenAI", return_value=mock_client) as mock_openai:
        vecs = Embedder().embed(["Atorvastatin is a statin."])
    _, kwargs = mock_openai.call_args
    assert kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert kwargs["api_key"] == "nvapi-test"
    assert kwargs["timeout"] >= 120  # cold-start-tolerant
    assert len(vecs) == 1
    assert len(vecs[0]) == EMBED_DIM


def test_embedder_sends_passage_input_type_for_documents(monkeypatch):
    """Indexing batches are passages; the 8b model requires input_type."""
    _need_openai()
    import src.embeddings.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod.settings, "nvidia_api_key", "nvapi-test")
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    mock_client = _mock_client_with([[vec, vec]])
    with patch("openai.OpenAI", return_value=mock_client):
        Embedder().embed(["chunk one", "chunk two"])
    _, kwargs = mock_client.embeddings.create.call_args
    assert kwargs["model"] == EMBED_MODEL
    assert kwargs["input"] == ["chunk one", "chunk two"]  # batched list input
    assert kwargs["extra_body"] == {"input_type": "passage"}


def test_embed_one_sends_query_input_type(monkeypatch):
    """Search queries are embedded with input_type=query."""
    _need_openai()
    import src.embeddings.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod.settings, "nvidia_api_key", "nvapi-test")
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    mock_client = _mock_client_with([[vec]])
    with patch("openai.OpenAI", return_value=mock_client):
        out = Embedder().embed_one("What are the contraindications?")
    _, kwargs = mock_client.embeddings.create.call_args
    assert kwargs["input"] == ["What are the contraindications?"]
    assert kwargs["extra_body"] == {"input_type": "query"}
    assert len(out) == EMBED_DIM


def test_embedder_batches_large_inputs(monkeypatch):
    """130 texts at batch_size=64 -> 3 API calls (64/64/2)."""
    _need_openai()
    import src.embeddings.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod.settings, "nvidia_api_key", "nvapi-test")
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    mock_client = _mock_client_with(
        [[vec] * 64, [vec] * 64, [vec] * 2]
    )
    with patch("openai.OpenAI", return_value=mock_client):
        vecs = Embedder().embed([f"chunk {i}" for i in range(130)])
    assert mock_client.embeddings.create.call_count == 3
    assert len(vecs) == 130
    sizes = [
        len(call.kwargs["input"])
        for call in mock_client.embeddings.create.call_args_list
    ]
    assert sizes == [64, 64, 2]


def test_embedder_normalizes_vectors(monkeypatch):
    """Returned vectors are L2-normalized (cosine-ready), as before."""
    _need_openai()
    import src.embeddings.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod.settings, "nvidia_api_key", "nvapi-test")
    # Non-unit input vector: norm 5.0.
    vec = [3.0, 4.0] + [0.0] * (EMBED_DIM - 2)
    mock_client = _mock_client_with([[vec]])
    with patch("openai.OpenAI", return_value=mock_client):
        out = Embedder().embed_one("some query")
    assert pytest.approx(np.linalg.norm(np.array(out)), 0.001) == 1.0
    assert pytest.approx(out[0], 0.001) == 0.6
    assert pytest.approx(out[1], 0.001) == 0.8


def test_retriever_uses_rrf_order_without_reranker(monkeypatch):
    """Reranking is gone: retrieve() returns hybrid-search (RRF) order as-is,
    with the RRF score, whether or not use_reranker is set."""
    import src.retrieval.retriever as retriever_mod

    mock_embedder = MagicMock()
    mock_embedder.embed_one.return_value = [0.1] * EMBED_DIM
    candidates = [
        {"id": 1, "drug_name": "atorvastatin", "section_title": "How Supplied",
         "content": "Bottles of 100 tablets.", "rrf_score": 0.9},
        {"id": 2, "drug_name": "atorvastatin", "section_title": "Contraindications",
         "content": "Contraindicated in pregnancy.", "rrf_score": 0.8},
    ]
    with patch.object(
        retriever_mod.vectorstore, "hybrid_search", return_value=candidates
    ) as mock_search:
        for flag in (True, False):
            retriever = Retriever(embedder=mock_embedder, use_reranker=flag)
            chunks = retriever.retrieve("contraindications?", top_k=20, top_n=2)
    assert mock_search.call_count == 2
    assert [c.id for c in chunks] == [1, 2]  # RRF order preserved
    assert chunks[0].score == pytest.approx(0.9)
    assert chunks[1].score == pytest.approx(0.8)
    # The query is embedded once per retrieve (input_type=query inside).
    assert mock_embedder.embed_one.call_count == 2
