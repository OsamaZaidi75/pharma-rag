"""Vector storage with hybrid (vector + BM25 keyword) search.

Two backends, one interface:
- Qdrant Cloud when QDRANT_URL and QDRANT_API_KEY are both set
  (src/store/qdrant_store.py; collection `pharma_rag`, Cosine).
- Local ChromaDB otherwise (the default; ./data/chroma, collection
  `label_chunks`).

Backend selection is explicit (get_backend_name) and evaluated on every
call, so tests can flip it by monkeypatching settings. Hybrid search
(vector + BM25 fused with RRF) is identical on both backends — only the
storage calls differ. All embeddings come from the NVIDIA API client in
both backends; Chroma's default embedding function is never used
(vectors are always passed explicitly to upsert/query).
"""
import os
from typing import Any
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from ..config import settings
from ..ingest.chunk import Chunk

COLLECTION_NAME = "label_chunks"


def get_backend_name() -> str:
    """'qdrant' when Qdrant is configured, else 'chroma' (the fallback).

    Both QDRANT_URL and QDRANT_API_KEY must be set; either one missing
    means Chroma. No credentials exist yet, so the deployed app stays on
    Chroma until they are added to the server env file.
    """
    if settings.qdrant_url and settings.qdrant_api_key:
        return "qdrant"
    return "chroma"


def get_backend():
    """Backend instance for the current settings (fresh each call, so a
    settings change takes effect immediately and tests stay isolated)."""
    if get_backend_name() == "qdrant":
        from .qdrant_store import QdrantBackend

        return QdrantBackend()
    return ChromaBackend()


class ChromaBackend:
    """Local ChromaDB backend (the default / fallback)."""

    name = "chroma"

    def __init__(self):
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            os.makedirs(settings.chroma_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=settings.chroma_dir)
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def init(self) -> None:
        """Create the ChromaDB collection. Safe to run repeatedly."""
        self._get_collection()

    def drop(self) -> None:
        """Delete and recreate the collection."""
        os.makedirs(settings.chroma_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._get_collection().count()

    def delete_setids(self, setids: set[str]) -> None:
        """Delete all chunks whose metadata setid is in `setids`."""
        if not setids:
            return
        collection = self._get_collection()
        setids_list = sorted(setids)
        if len(setids_list) == 1:
            where_clause = {"setid": setids_list[0]}
        else:
            where_clause = {"setid": {"$in": setids_list}}
        try:
            collection.delete(where=where_clause)
        except Exception:
            pass

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        collection = self._get_collection()
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def query_vector(self, embedding, n: int) -> list[str]:
        """Top-n chunk ids by cosine similarity, best first."""
        query_res = self._get_collection().query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas"],
        )
        return query_res["ids"][0] if query_res["ids"] else []

    def get_all(self):
        """Return (ids, documents, metadatas) for every chunk."""
        all_data = self._get_collection().get(include=["documents", "metadatas"])
        return all_data["ids"], all_data["documents"], all_data["metadatas"]


def init_db() -> None:
    """Create the vector store collection. Safe to run repeatedly."""
    get_backend().init()


def drop_all() -> None:
    """Delete and recreate the collection."""
    get_backend().drop()


def rrf_fuse(ranked_id_lists: list[list[Any]], k: int = 60) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion over ranked id lists. Returns (id, score) sorted desc."""
    scores: dict[Any, float] = {}
    for ranked in ranked_id_lists:
        for rank, _id in enumerate(ranked, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def upsert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    delete_setids: set[str] | None = None,
) -> int:
    """Replace any existing chunks for the given setids, then insert. Returns count."""
    assert len(chunks) == len(embeddings), "chunks/embeddings length mismatch"
    if not chunks:
        return 0

    backend = get_backend()

    if delete_setids:
        backend.delete_setids(delete_setids)

    ids = []
    seen_ids = set()
    for c in chunks:
        base_id = f"{c.setid}#{c.chunk_index}"
        unique_id = base_id
        suffix = 1
        while unique_id in seen_ids:
            unique_id = f"{c.setid}#{c.chunk_index}_{suffix}"
            suffix += 1
        seen_ids.add(unique_id)
        ids.append(unique_id)

    documents = [c.content for c in chunks]
    metadatas = [
        {
            "drug_name": c.drug_name,
            "setid": c.setid,
            "section_code": c.section_code or "",
            "section_title": c.section_title,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    backend.upsert(ids, embeddings, documents, metadatas)
    return len(chunks)


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    top_k: int = 20,
    candidate_multiplier: int = 3,
) -> list[dict]:
    """Vector + BM25 keyword search fused with RRF. Returns chunk dicts with rrf_score."""
    backend = get_backend()
    total_count = backend.count()
    if total_count == 0:
        return []

    n = min(top_k * candidate_multiplier, total_count)

    # 1. Vector candidates
    vec_ids = backend.query_vector(query_embedding, n)

    # 2. BM25 keyword candidates over all documents in the store
    all_ids, all_docs, all_metas = backend.get_all()

    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = query_text.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    top_kw_indices = np.argsort(-bm25_scores)[:n]
    kw_ids = [all_ids[idx] for idx in top_kw_indices]

    # 3. Reciprocal Rank Fusion
    fused = rrf_fuse([vec_ids, kw_ids])[:top_k]
    if not fused:
        return []

    id_to_score = dict(fused)
    id_to_doc = {doc_id: (doc, meta) for doc_id, doc, meta in zip(all_ids, all_docs, all_metas)}

    results = []
    for doc_id, score in fused:
        if doc_id in id_to_doc:
            doc, meta = id_to_doc[doc_id]
            results.append(
                {
                    "id": doc_id,
                    "drug_name": meta.get("drug_name", ""),
                    "setid": meta.get("setid", ""),
                    "section_code": meta.get("section_code", ""),
                    "section_title": meta.get("section_title", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "content": doc,
                    "rrf_score": score,
                }
            )

    results.sort(key=lambda r: r["rrf_score"], reverse=True)
    return results


def list_drugs() -> list[str]:
    """Return distinct drug names in the store."""
    backend = get_backend()
    if backend.count() == 0:
        return []
    _, _, all_metas = backend.get_all()
    drugs = {m["drug_name"] for m in all_metas if m and "drug_name" in m}
    return sorted(list(drugs))


def count_chunks() -> int:
    """Return total number of chunks in the store."""
    return get_backend().count()
