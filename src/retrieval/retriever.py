"""Retrieval: embed query -> hybrid search -> cross-encoder rerank."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import settings
from ..embeddings.embedder import Embedder
from ..store import vectorstore

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@dataclass
class RetrievedChunk:
    id: Any
    drug_name: str
    section_title: str
    content: str
    score: float  # rerank score if reranked, else RRF score


class Reranker:
    """Cross-encoder reranker, loaded lazily (local, free)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: "CrossEncoder | None" = None

    def _load(self) -> "CrossEncoder":
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, c["content"]) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)
        chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        return chunks


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        use_reranker: bool | None = None,
    ):
        self.embedder = embedder or Embedder(settings.embedding_model)
        self.use_reranker = settings.use_reranker if use_reranker is None else use_reranker
        self._reranker: Reranker | None = None

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

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
        if self.use_reranker:
            candidates = self._get_reranker().rerank(query, candidates)

        out = []
        for c in candidates[:top_n]:
            out.append(
                RetrievedChunk(
                    id=c["id"],
                    drug_name=c["drug_name"],
                    section_title=c["section_title"],
                    content=c["content"],
                    score=c.get("rerank_score", c["rrf_score"]),
                )
            )
        return out
