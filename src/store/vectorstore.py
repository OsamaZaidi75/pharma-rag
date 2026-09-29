"""ChromaDB storage with hybrid (vector + BM25 keyword) search.

Hybrid strategy: fetch candidates from BOTH a cosine-similarity vector query
and BM25 keyword matching over document texts, then fuse the two ranked
lists with Reciprocal Rank Fusion (RRF).
"""
import os
from typing import Any
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from ..config import settings
from ..ingest.chunk import Chunk

COLLECTION_NAME = "label_chunks"


def _get_client():
    os.makedirs(settings.chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_dir)


def _get_collection(client=None):
    client = client or _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def init_db() -> None:
    """Create ChromaDB client and collection. Safe to run repeatedly."""
    client = _get_client()
    _get_collection(client)


def drop_all() -> None:
    """Delete and recreate the collection."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _get_collection(client)


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

    collection = _get_collection()

    if delete_setids:
        setids_list = sorted(delete_setids)
        if len(setids_list) == 1:
            where_clause = {"setid": setids_list[0]}
        else:
            where_clause = {"setid": {"$in": setids_list}}
        try:
            collection.delete(where=where_clause)
        except Exception:
            pass

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

    # Batch upsert if large
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    return len(chunks)


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    top_k: int = 20,
    candidate_multiplier: int = 3,
) -> list[dict]:
    """Vector + BM25 keyword search fused with RRF. Returns chunk dicts with rrf_score."""
    collection = _get_collection()
    total_count = collection.count()
    if total_count == 0:
        return []

    n = min(top_k * candidate_multiplier, total_count)

    # 1. Vector candidates
    query_res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )
    vec_ids = query_res["ids"][0] if query_res["ids"] else []

    # 2. BM25 keyword candidates over all documents in collection
    all_data = collection.get(include=["documents", "metadatas"])
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

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
    """Return distinct drug names in the collection."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    all_data = collection.get(include=["metadatas"])
    drugs = {m["drug_name"] for m in all_data["metadatas"] if m and "drug_name" in m}
    return sorted(list(drugs))


def count_chunks() -> int:
    """Return total number of chunks in the collection."""
    collection = _get_collection()
    return collection.count()
