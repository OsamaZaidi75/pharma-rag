"""Unit tests for on-demand indexing and term validation."""
from unittest.mock import MagicMock, patch

from src.ingest.ondemand import _candidate_terms, ensure_drug_indexed, matches_drug_term


def test_candidate_terms_stopword_filtering():
    question = "HELP ME WITH USES OF DRUG ASPIRIN"
    terms = _candidate_terms(question)
    assert terms == ["aspirin"]


def test_candidate_terms_longest_first():
    question = "Can I take amoxicillin with omeprazole?"
    terms = _candidate_terms(question)
    assert terms == ["amoxicillin", "omeprazole"]


def test_matches_drug_term_exact():
    assert matches_drug_term("aspirin", "ASPIRIN") is True
    assert matches_drug_term("aspirin", "aspirin") is True
    assert matches_drug_term("atorvastatin", "Atorvastatin") is True


def test_matches_drug_term_dosage_and_form():
    assert matches_drug_term("aspirin", "ASPIRIN 81 MG") is True
    assert matches_drug_term("aspirin", "ASPIRIN 325 MG TABLET") is True
    assert matches_drug_term("aspirin", "ASPIRIN ENTERIC COATED DELAYED RELEASE TABLETS") is True
    assert matches_drug_term("atorvastatin", "ATORVASTATIN CALCIUM 10 MG TABLET") is True
    assert matches_drug_term("metformin", "METFORMIN HYDROCHLORIDE EXTENDED-RELEASE") is True


def test_matches_drug_term_rejects_combos_and_fuzzy():
    # Combination products rejected
    assert matches_drug_term("aspirin", "ASPIRIN AND CAFFEINE") is False
    assert matches_drug_term("aspirin", "ASPIRIN WITH DIPHENHYDRAMINE") is False
    assert matches_drug_term("aspirin", "ASPIRIN / CITRIC ACID") is False
    # Random phrases containing term rejected
    assert matches_drug_term("help", "ALLERGIES MY CHILD NEEDS HELP WITH") is False
    # Substring prefix without boundary rejected
    assert matches_drug_term("aspirin", "ASPIRINLIKE") is False


def test_ensure_drug_indexed_already_present():
    mock_embedder = MagicMock()
    with patch("src.ingest.ondemand.vectorstore.list_drugs", return_value=["atorvastatin", "aspirin"]):
        result = ensure_drug_indexed("What is the dose of aspirin?", mock_embedder)
        assert result is None
        mock_embedder.embed.assert_not_called()


def test_ensure_drug_indexed_no_valid_terms():
    mock_embedder = MagicMock()
    with patch("src.ingest.ondemand.vectorstore.list_drugs", return_value=["atorvastatin"]):
        # All words are stopwords
        result = ensure_drug_indexed("Help with what drug dose is safe?", mock_embedder)
        assert result is None
        mock_embedder.embed.assert_not_called()
