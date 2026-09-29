"""Unit tests for embedder and reranker modules."""
import numpy as np
import pytest

from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Reranker


def test_embedder_dimension_and_normalization():
    emb = Embedder()
    vec = emb.embed_one("Metformin is an antihyperglycemic agent.")
    assert len(vec) == 384
    # Check L2 normalization (norm should be ~1.0)
    norm = np.linalg.norm(np.array(vec))
    assert pytest.approx(norm, 0.01) == 1.0


def test_embedder_empty_list():
    emb = Embedder()
    assert emb.embed([]) == []


def test_reranker_reordering():
    reranker = Reranker()
    chunks = [
        {"id": 1, "content": "How supplied: Bottles of 100 tablets."},
        {"id": 2, "content": "Contraindications: Atorvastatin is contraindicated in pregnancy and lactation."},
    ]
    reranked = reranker.rerank("What are contraindications in pregnancy?", chunks)
    assert len(reranked) == 2
    # The contraindication chunk should be ranked first
    assert reranked[0]["id"] == 2
    assert "rerank_score" in reranked[0]
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
