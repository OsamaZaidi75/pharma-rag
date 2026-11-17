"""Unit tests for the POST /index pre-indexing endpoint."""
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")  # CI test env has no fastapi (like test_api.py)

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _post(drugs):
    return client.post("/index", json={"drugs": drugs})


def test_index_new_drug_success():
    mock_retriever = MagicMock()
    with patch("src.api.main.get_retriever", return_value=mock_retriever), \
         patch("src.api.main.vectorstore.list_drugs", return_value=[]), \
         patch("src.api.main.ensure_drug_indexed", return_value="atorvastatin"), \
         patch("src.api.main.vectorstore.count_chunks", return_value=120):
        response = _post(["atorvastatin"])
        assert response.status_code == 200
        assert response.json() == {
            "indexed": ["atorvastatin"],
            "already_indexed": [],
            "failed": [],
            "total_chunks": 120,
        }


def test_index_skips_already_indexed_drug():
    mock_retriever = MagicMock()
    with patch("src.api.main.get_retriever", return_value=mock_retriever), \
         patch("src.api.main.vectorstore.list_drugs",
               return_value=["atorvastatin"]), \
         patch("src.api.main.ensure_drug_indexed") as mock_ensure, \
         patch("src.api.main.vectorstore.count_chunks", return_value=120):
        response = _post(["atorvastatin"])
        assert response.status_code == 200
        assert response.json()["already_indexed"] == ["atorvastatin"]
        assert response.json()["indexed"] == []
        mock_ensure.assert_not_called()


def test_index_reports_drug_with_no_label():
    mock_retriever = MagicMock()
    with patch("src.api.main.get_retriever", return_value=mock_retriever), \
         patch("src.api.main.vectorstore.list_drugs", return_value=[]), \
         patch("src.api.main.ensure_drug_indexed", return_value=None), \
         patch("src.api.main.vectorstore.count_chunks", return_value=0):
        response = _post(["notarealdrugname"])
        assert response.status_code == 200
        assert response.json()["failed"] == ["notarealdrugname"]
        assert response.json()["total_chunks"] == 0


def test_index_mixed_results():
    mock_retriever = MagicMock()

    def _ensure(name, _embedder):
        return name if name == "metformin" else None

    with patch("src.api.main.get_retriever", return_value=mock_retriever), \
         patch("src.api.main.vectorstore.list_drugs",
               return_value=["atorvastatin"]), \
         patch("src.api.main.ensure_drug_indexed", side_effect=_ensure), \
         patch("src.api.main.vectorstore.count_chunks", return_value=250):
        response = _post(["atorvastatin", "metformin", "notarealdrugname"])
        assert response.status_code == 200
        data = response.json()
        assert data["already_indexed"] == ["atorvastatin"]
        assert data["indexed"] == ["metformin"]
        assert data["failed"] == ["notarealdrugname"]
        assert data["total_chunks"] == 250


def test_index_validation_error_empty_list():
    response = _post([])
    assert response.status_code == 422


def test_index_db_failure():
    mock_retriever = MagicMock()
    with patch("src.api.main.get_retriever", return_value=mock_retriever), \
         patch("src.api.main.vectorstore.list_drugs",
               side_effect=RuntimeError("DB error")):
        response = _post(["atorvastatin"])
        assert response.status_code == 503
        assert "DB unreachable" in response.json()["detail"]
