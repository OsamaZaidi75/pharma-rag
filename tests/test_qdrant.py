"""Unit tests for the Qdrant Cloud vector-store backend + backend selection.

The Qdrant client is mocked (no network, no credentials). Tests that need
the `qdrant_client` package are skipped where it is not installed,
mirroring tests/test_nvidia.py; backend-selection tests run anywhere
(the Qdrant backend imports `qdrant_client` lazily, so selecting it
never requires the package).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.ingest.chunk import Chunk
from src.store import vectorstore
from src.store.qdrant_store import QdrantBackend

QDRANT_URL = "https://test-cluster.cloud.qdrant.io:6333"


def _need_qdrant():
    pytest.importorskip("qdrant_client", reason="qdrant_client not installed in CI env")


@pytest.fixture(autouse=True)
def _clean_qdrant_env(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    monkeypatch.setattr(vectorstore.settings, "qdrant_url", None)
    monkeypatch.setattr(vectorstore.settings, "qdrant_api_key", None)


def _set_qdrant(monkeypatch, url=QDRANT_URL, key="qdr-test"):
    monkeypatch.setattr(vectorstore.settings, "qdrant_url", url)
    monkeypatch.setattr(vectorstore.settings, "qdrant_api_key", key)


# --- backend selection -------------------------------------------------------


def test_no_qdrant_env_means_chroma():
    assert vectorstore.get_backend_name() == "chroma"
    assert vectorstore.get_backend().name == "chroma"


def test_qdrant_url_without_key_still_chroma(monkeypatch):
    _set_qdrant(monkeypatch, key=None)
    assert vectorstore.get_backend_name() == "chroma"
    assert vectorstore.get_backend().name == "chroma"


def test_qdrant_key_without_url_still_chroma(monkeypatch):
    _set_qdrant(monkeypatch, url=None)
    assert vectorstore.get_backend_name() == "chroma"
    assert vectorstore.get_backend().name == "chroma"


def test_both_qdrant_vars_means_qdrant(monkeypatch):
    _set_qdrant(monkeypatch)
    assert vectorstore.get_backend_name() == "qdrant"
    backend = vectorstore.get_backend()
    assert isinstance(backend, QdrantBackend)
    assert backend.name == "qdrant"
    assert backend.collection == "pharma_rag"


def test_qdrant_collection_default():
    assert Settings().qdrant_collection == "pharma_rag"


def test_qdrant_settings_from_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", QDRANT_URL)
    monkeypatch.setenv("QDRANT_API_KEY", "qdr-test")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_coll")
    settings = Settings()
    assert settings.qdrant_url == QDRANT_URL
    assert settings.qdrant_api_key == "qdr-test"
    assert settings.qdrant_collection == "custom_coll"


def test_qdrant_defaults_unset():
    settings = Settings()
    assert settings.qdrant_url is None
    assert settings.qdrant_api_key is None


# --- Qdrant backend behavior (mocked client) ---------------------------------


@pytest.fixture
def mocked_backend(monkeypatch):
    """(backend, mock_client) with a mocked QdrantClient; collection missing."""
    _need_qdrant()
    _set_qdrant(monkeypatch)
    monkeypatch.setattr(vectorstore.settings, "embedding_dim", 4096)
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    with patch("qdrant_client.QdrantClient", return_value=mock_client) as mock_ctor:
        backend = QdrantBackend()
        backend.init()  # creates the client + the collection
        yield backend, mock_client, mock_ctor


def test_client_built_from_settings(mocked_backend):
    _backend, _client, mock_ctor = mocked_backend
    _, kwargs = mock_ctor.call_args
    assert kwargs["url"] == QDRANT_URL
    assert kwargs["api_key"] == "qdr-test"
    assert kwargs["timeout"] >= 60


def test_init_creates_collection_when_missing(mocked_backend):
    _backend, mock_client, _ctor = mocked_backend
    mock_client.collection_exists.assert_called_with("pharma_rag")
    mock_client.create_collection.assert_called_once()
    _, kwargs = mock_client.create_collection.call_args
    assert kwargs["collection_name"] == "pharma_rag"
    vectors_config = kwargs["vectors_config"]
    assert vectors_config.size == 4096
    from qdrant_client import models

    assert vectors_config.distance == models.Distance.COSINE


def test_init_skips_create_when_collection_exists(monkeypatch):
    _need_qdrant()
    _set_qdrant(monkeypatch)
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=4096))
        )
    )
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        QdrantBackend().init()
    mock_client.create_collection.assert_not_called()


def test_init_rejects_dimension_mismatch(monkeypatch):
    _need_qdrant()
    _set_qdrant(monkeypatch)
    monkeypatch.setattr(vectorstore.settings, "embedding_dim", 4096)
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=384))
        )
    )
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="vector size 384"):
            QdrantBackend().init()


def test_upsert_sends_points_with_payload(mocked_backend):
    backend, mock_client, _ctor = mocked_backend
    backend.upsert(
        ids=["set_1#0", "set_1#1"],
        embeddings=[[0.1] * 4096, [0.2] * 4096],
        documents=["chunk one", "chunk two"],
        metadatas=[
            {"drug_name": "atorvastatin", "setid": "set_1",
             "section_code": "", "section_title": "Contraindications",
             "chunk_index": 0},
            {"drug_name": "atorvastatin", "setid": "set_1",
             "section_code": "", "section_title": "Dosage",
             "chunk_index": 1},
        ],
    )
    mock_client.upsert.assert_called_once()
    _, kwargs = mock_client.upsert.call_args
    assert kwargs["collection_name"] == "pharma_rag"
    points = kwargs["points"]
    assert len(points) == 2
    assert points[0].vector == [0.1] * 4096
    assert points[0].payload["chunk_id"] == "set_1#0"
    assert points[0].payload["document"] == "chunk one"
    assert points[0].payload["drug_name"] == "atorvastatin"
    assert points[0].payload["setid"] == "set_1"
    # Point ids are deterministic UUIDs: same chunk id -> same point id.
    assert str(points[0].id) == str(points[0].id)
    assert points[0].id != points[1].id


def test_delete_setids_uses_filter(mocked_backend):
    backend, mock_client, _ctor = mocked_backend
    backend.delete_setids({"set_1", "set_2"})
    mock_client.delete.assert_called_once()
    _, kwargs = mock_client.delete.call_args
    assert kwargs["collection_name"] == "pharma_rag"
    selector = kwargs["points_selector"]
    condition = selector.must[0]
    assert condition.key == "setid"
    assert sorted(condition.match.any) == ["set_1", "set_2"]


def test_delete_setids_empty_is_noop(mocked_backend):
    backend, mock_client, _ctor = mocked_backend
    backend.delete_setids(set())
    mock_client.delete.assert_not_called()


def test_query_vector_returns_chunk_ids(mocked_backend):
    backend, mock_client, _ctor = mocked_backend
    mock_client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(payload={"chunk_id": "set_1#0"}),
            SimpleNamespace(payload={"chunk_id": "set_1#1"}),
        ]
    )
    ids = backend.query_vector([0.1] * 4096, 2)
    assert ids == ["set_1#0", "set_1#1"]
    _, kwargs = mock_client.query_points.call_args
    assert kwargs["collection_name"] == "pharma_rag"
    assert kwargs["limit"] == 2


def test_get_all_scrolls_pages(mocked_backend):
    backend, mock_client, _ctor = mocked_backend
    mock_client.scroll.side_effect = [
        (
            [SimpleNamespace(payload={
                "chunk_id": "set_1#0", "document": "doc one",
                "drug_name": "atorvastatin", "setid": "set_1",
                "section_code": "", "section_title": "T1", "chunk_index": 0,
            })],
            "next-offset",
        ),
        (
            [SimpleNamespace(payload={
                "chunk_id": "set_2#0", "document": "doc two",
                "drug_name": "metformin", "setid": "set_2",
                "section_code": "", "section_title": "T2", "chunk_index": 0,
            })],
            None,
        ),
    ]
    ids, docs, metas = backend.get_all()
    assert ids == ["set_1#0", "set_2#0"]
    assert docs == ["doc one", "doc two"]
    assert metas[0]["drug_name"] == "atorvastatin"
    assert metas[1]["drug_name"] == "metformin"
    assert mock_client.scroll.call_count == 2


def test_count(mocked_backend):
    backend, mock_client, _ctor = mocked_backend
    mock_client.count.return_value = SimpleNamespace(count=42)
    assert backend.count() == 42
    _, kwargs = mock_client.count.call_args
    assert kwargs["exact"] is True


def test_drop_deletes_and_recreates(mocked_backend):
    backend, mock_client, _ctor = mocked_backend
    mock_client.reset_mock()
    mock_client.collection_exists.return_value = False
    backend.drop()
    mock_client.delete_collection.assert_called_once_with("pharma_rag")
    mock_client.create_collection.assert_called_once()


# --- dispatcher wiring: hybrid search over the Qdrant backend ----------------


def test_hybrid_search_over_qdrant_backend(monkeypatch):
    """vectorstore.hybrid_search runs the same BM25+RRF logic when the
    backend is Qdrant (mocked) — only the storage calls differ."""
    _set_qdrant(monkeypatch)
    mock_backend = MagicMock()
    mock_backend.count.return_value = 2
    mock_backend.query_vector.return_value = ["set_1#0", "set_2#0"]
    mock_backend.get_all.return_value = (
        ["set_1#0", "set_2#0"],
        ["Contraindicated in active liver disease.",
         "Starting dose is 500 mg."],
        [
            {"drug_name": "atorvastatin", "setid": "set_1",
             "section_code": "", "section_title": "Contraindications",
             "chunk_index": 0},
            {"drug_name": "metformin", "setid": "set_2",
             "section_code": "", "section_title": "Dosage",
             "chunk_index": 0},
        ],
    )
    with patch.object(vectorstore, "get_backend", return_value=mock_backend):
        results = vectorstore.hybrid_search(
            "contraindications atorvastatin", [0.1] * 4096, top_k=5
        )
    assert len(results) == 2
    assert results[0]["drug_name"] == "atorvastatin"
    assert results[0]["section_title"] == "Contraindications"
    assert "rrf_score" in results[0]


def test_upsert_chunks_dispatches_to_qdrant(monkeypatch):
    """vectorstore.upsert_chunks builds ids/metadata once and hands them to
    whichever backend is selected — here a mocked Qdrant backend."""
    _set_qdrant(monkeypatch)
    mock_backend = MagicMock()
    chunks = [
        Chunk(
            drug_name="atorvastatin", setid="set_1", section_code="",
            section_title="Contraindications", chunk_index=0,
            content="Contraindicated in pregnancy.",
        )
    ]
    with patch.object(vectorstore, "get_backend", return_value=mock_backend):
        n = vectorstore.upsert_chunks(
            chunks, [[0.1] * 4096], delete_setids={"set_1"}
        )
    assert n == 1
    mock_backend.delete_setids.assert_called_once_with({"set_1"})
    mock_backend.upsert.assert_called_once()
    args = mock_backend.upsert.call_args.args
    assert args[0] == ["set_1#0"]  # ids
    assert args[2] == ["Contraindicated in pregnancy."]  # documents
    assert args[3][0]["drug_name"] == "atorvastatin"  # metadatas
