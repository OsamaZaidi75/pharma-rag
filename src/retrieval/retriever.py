"""Retrieval: embed query -> hybrid search (vector + BM25, RRF fusion).

The cross-encoder reranker was removed with the torch removal (2026-11-18):
reranking needed sentence-transformers, which dragged torch into the image.
RRF ordering is now the final ranking. The `use_reranker` / `rerank` flags
are kept as accepted-but-ignored no-ops so existing callers (API, evals,
Streamlit) and the API contract don't break.
"""
from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..embeddings.embedder import Embedder
from ..store import vectorstore


@dataclass
class RetrievedChunk:
    id: Any
    drug_name: str
    section_title: str
    content: str
    score: float  # RRF score


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        use_reranker: bool | None = None,
    ):
        self.embedder = embedder or Embedder(settings.embedding_model)
        # Deprecated no-op: reranking was removed with the torch removal.
        # Kept so existing callers keep working; the value is ignored.
        self.use_reranker = settings.use_reranker if use_reranker is None else use_reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k
        top_n = top_n or settings.rerank_top_n

        q_emb = self.embedder.embed_one(query)
        candidates = vectorstore.hybrid_search(query, q_emb, top_k=top_k)

        out = []
        for c in candidates[:top_n]:
            out.append(
                RetrievedChunk(
                    id=c["id"],
                    drug_name=c["drug_name"],
                    section_title=c["section_title"],
                    content=c["content"],
                    score=c["rrf_score"],
                )
            )
        return out
