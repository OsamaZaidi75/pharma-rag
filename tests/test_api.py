"""Unit tests for FastAPI endpoints."""
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.generation.answer import Answer
from src.retrieval.retriever import RetrievedChunk

client = TestClient(app)


def test_health_endpoint_success():
    with patch("src.api.main.vectorstore.count_chunks", return_value=120):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "indexed_chunks": 120}


def test_health_endpoint_db_failure():
    with patch("src.api.main.vectorstore.count_chunks", side_effect=RuntimeError("DB disconnected")):
        response = client.get("/health")
        assert response.status_code == 503
        assert "DB unreachable" in response.json()["detail"]


def test_drugs_endpoint_success():
    with patch("src.api.main.vectorstore.list_drugs", return_value=["atorvastatin", "lisinopril"]):
        response = client.get("/drugs")
        assert response.status_code == 200
        assert response.json() == {"drugs": ["atorvastatin", "lisinopril"]}


def test_drugs_endpoint_db_failure():
    with patch("src.api.main.vectorstore.list_drugs", side_effect=RuntimeError("DB error")):
        response = client.get("/drugs")
        assert response.status_code == 503
        assert "DB unreachable" in response.json()["detail"]


def test_ask_endpoint_success():
    mock_retriever = MagicMock()
    mock_chunk = RetrievedChunk(
        id=1,
        drug_name="atorvastatin",
        section_title="Contraindications",
        content="Contraindicated in pregnancy and hypersensitivity.",
        score=0.92,
    )
    mock_retriever.retrieve.return_value = [mock_chunk]

    mock_answer = Answer(
        text="Atorvastatin is contraindicated in pregnancy [1].\n\nThis is not medical advice. Consult a healthcare professional.",
        sources=[
            {
                "ref": 1,
                "drug_name": "atorvastatin",
                "section_title": "Contraindications",
                "score": 0.92,
                "preview": "Contraindicated in pregnancy...",
            }
        ],
        model="llama3.1",
    )

    with patch("src.api.main.get_retriever", return_value=mock_retriever), \
         patch("src.api.main.generate_answer", return_value=mock_answer):
        response = client.post(
            "/ask",
            json={"question": "What are contraindications for atorvastatin?", "top_k": 10, "top_n": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert "contraindicated in pregnancy" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["drug_name"] == "atorvastatin"
        assert data["model"] == "llama3.1"


def test_ask_endpoint_validation_error():
    # Question too short (< 3 chars)
    response = client.post("/ask", json={"question": "ab"})
    assert response.status_code == 422
