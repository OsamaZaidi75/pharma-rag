"""Unit tests for generation prompt construction and answering logic."""
from unittest.mock import MagicMock, patch
import pytest

from src.generation.answer import (
    DISCLAIMER,
    Answer,
    build_prompt,
    generate_answer,
)
from src.retrieval.retriever import RetrievedChunk


def test_build_prompt_formatting():
    chunks = [
        RetrievedChunk(
            id=1,
            drug_name="metformin",
            section_title="Dosage and Administration",
            content="Starting dose is 500 mg twice daily.",
            score=0.88,
        ),
        RetrievedChunk(
            id=2,
            drug_name="metformin",
            section_title="Contraindications",
            content="Severe renal impairment.",
            score=0.75,
        ),
    ]
    messages = build_prompt("How to take metformin?", chunks)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "[1] Drug: metformin | Section: Dosage and Administration" in messages[1]["content"]
    assert "[2] Drug: metformin | Section: Contraindications" in messages[1]["content"]
    assert "Question: How to take metformin?" in messages[1]["content"]


def test_generate_answer_empty_chunks():
    ans = generate_answer("What is drug X?", [])
    assert isinstance(ans, Answer)
    assert "couldn't find relevant information" in ans.text
    assert DISCLAIMER in ans.text
    assert ans.sources == []
    assert ans.model == "none"


def test_generate_answer_with_llm():
    chunks = [
        RetrievedChunk(
            id=1,
            drug_name="atorvastatin",
            section_title="Contraindications",
            content="Active liver disease.",
            score=0.91,
        )
    ]
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Atorvastatin is contraindicated in active liver disease [1]."
    mock_resp = MagicMock(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("src.generation.answer._get_client", return_value=(mock_client, "gpt-4o-mini")):
        ans = generate_answer("What are contraindications for atorvastatin?", chunks)
        assert "Active liver disease" in ans.text or "active liver disease" in ans.text
        assert DISCLAIMER in ans.text
        assert len(ans.sources) == 1
        assert ans.sources[0]["ref"] == 1
        assert ans.sources[0]["drug_name"] == "atorvastatin"
        assert ans.sources[0]["score"] == 0.91
        assert ans.model == "gpt-4o-mini"


def test_generate_answer_llm_fallback():
    chunks = [
        RetrievedChunk(
            id=1,
            drug_name="atorvastatin",
            section_title="Contraindications",
            content="Drug: atorvastatin | Section: Contraindications\nActive liver disease.",
            score=0.91,
        )
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Connection refused")

    with patch("src.generation.answer._get_client", return_value=(mock_client, "llama3.1")):
        ans = generate_answer("What are contraindications?", chunks)
        assert "Active liver disease" in ans.text or "atorvastatin" in ans.text.lower()
        assert DISCLAIMER in ans.text
        assert "extractive-fallback" in ans.model
