"""Unit tests for DailyMed API client and evaluation metrics."""
from unittest.mock import MagicMock, patch

from src.ingest.dailymed import SplSummary, search_drug_names, search_spls
from src.evals.run_evals import is_hit
from src.retrieval.retriever import RetrievedChunk


def test_dailymed_search_spls():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {
                "setid": "set123",
                "title": "Metformin HCl Tablets",
                "published_date": "2024-01-01",
            }
        ]
    }
    with patch("src.ingest.dailymed.httpx.get", return_value=mock_resp):
        spls = search_spls("metformin", pagesize=1)
        assert len(spls) == 1
        assert isinstance(spls[0], SplSummary)
        assert spls[0].setid == "set123"
        assert spls[0].title == "Metformin HCl Tablets"


def test_dailymed_search_drug_names():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"drug_name": "METFORMIN HYDROCHLORIDE"},
            {"drug_name": "METFORMIN AND GLIP値IDE"},
        ]
    }
    with patch("src.ingest.dailymed.httpx.get", return_value=mock_resp):
        names = search_drug_names("metformin")
        assert len(names) == 2
        assert "METFORMIN HYDROCHLORIDE" in names


def test_evals_is_hit_found():
    chunks = [
        RetrievedChunk(id=1, drug_name="lisinopril", section_title="Indications", content="", score=0.9),
        RetrievedChunk(id=2, drug_name="atorvastatin", section_title="Contraindications", content="", score=0.8),
    ]
    rank = is_hit(chunks, expected_drug="atorvastatin", expected_section="Contraindications")
    assert rank == 2


def test_evals_is_hit_not_found():
    chunks = [
        RetrievedChunk(id=1, drug_name="lisinopril", section_title="Indications", content="", score=0.9),
    ]
    rank = is_hit(chunks, expected_drug="atorvastatin", expected_section="Contraindications")
    assert rank is None
