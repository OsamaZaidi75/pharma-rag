"""Unit tests for ChromaDB vectorstore module."""
import os
import shutil
import tempfile
import pytest

from src.config import settings
from src.ingest.chunk import Chunk
from src.store import vectorstore


@pytest.fixture(autouse=True)
def temp_chroma_dir(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr(settings, "chroma_dir", temp_dir)
    vectorstore.drop_all()
    yield temp_dir
    vectorstore.drop_all()
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_init_and_count_empty():
    vectorstore.init_db()
    assert vectorstore.count_chunks() == 0
    assert vectorstore.list_drugs() == []


def test_upsert_and_hybrid_search():
    chunks = [
        Chunk(
            drug_name="atorvastatin",
            setid="set_atorva_1",
            section_code="34070-3",
            section_title="Contraindications",
            chunk_index=0,
            content="Drug: atorvastatin | Section: Contraindications\nContraindicated in active liver disease.",
        ),
        Chunk(
            drug_name="metformin",
            setid="set_metform_1",
            section_code="34068-7",
            section_title="Dosage and Administration",
            chunk_index=0,
            content="Drug: metformin | Section: Dosage\nStarting dose is 500 mg.",
        ),
    ]
    # Simple 384-dim dummy embeddings
    emb1 = [0.1] * 384
    emb2 = [0.0] * 384
    emb1[0] = 1.0

    n = vectorstore.upsert_chunks(chunks, [emb1, emb2])
    assert n == 2
    assert vectorstore.count_chunks() == 2
    assert vectorstore.list_drugs() == ["atorvastatin", "metformin"]

    # Hybrid search for atorvastatin
    results = vectorstore.hybrid_search("contraindications atorvastatin", emb1, top_k=5)
    assert len(results) == 2
    assert results[0]["drug_name"] == "atorvastatin"
    assert results[0]["section_title"] == "Contraindications"
    assert "rrf_score" in results[0]
    assert results[0]["id"] == "set_atorva_1#0"


def test_delete_setids():
    chunks = [
        Chunk(
            drug_name="atorvastatin",
            setid="set_1",
            section_code="",
            section_title="Title 1",
            chunk_index=0,
            content="Content 1",
        ),
        Chunk(
            drug_name="lisinopril",
            setid="set_2",
            section_code="",
            section_title="Title 2",
            chunk_index=0,
            content="Content 2",
        ),
    ]
    emb = [[0.0] * 384, [0.0] * 384]
    vectorstore.upsert_chunks(chunks, emb)
    assert vectorstore.count_chunks() == 2

    # Upsert new chunk with delete_setids
    new_chunks = [
        Chunk(
            drug_name="atorvastatin",
            setid="set_1",
            section_code="",
            section_title="Title 1 Updated",
            chunk_index=0,
            content="Content 1 Updated",
        )
    ]
    vectorstore.upsert_chunks(new_chunks, [[0.1] * 384], delete_setids={"set_1"})
    assert vectorstore.count_chunks() == 2
    assert vectorstore.list_drugs() == ["atorvastatin", "lisinopril"]
