"""Qdrant Cloud vector-store backend.

Used when QDRANT_URL and QDRANT_API_KEY are both set; otherwise the app
falls back to local ChromaDB (see vectorstore.get_backend). Same data
model as the Chroma backend: one point per chunk, vector = NVIDIA API
embedding (4096-dim for the default model), payload carries the chunk
text + metadata. Distance is Cosine; the collection is created on first
use if missing.

`qdrant_client` is imported lazily (inside the methods that need it) so
importing this module never requires the package — the CI test env
doesn't install it, and the Chroma fallback must work without it.
"""
import uuid

from ..config import settings

QDRANT_TIMEOUT_S = 60


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk id (Qdrant point ids must be int/UUID).

    Same chunk id -> same point id, so re-upserting a chunk replaces it
    instead of duplicating it. The original string id is also stored in
    the payload as `chunk_id` (BM25/RRF identity).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pharma-rag:{chunk_id}"))


class QdrantBackend:
    """Qdrant Cloud backend. Created per call by vectorstore.get_backend()."""

    name = "qdrant"

    def __init__(self, client=None):
        # `client` is injectable for tests; otherwise built lazily from
        # settings on first use.
        self._client = client

    @property
    def collection(self) -> str:
        return settings.qdrant_collection

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=QDRANT_TIMEOUT_S,
            )
        return self._client

    def init(self) -> None:
        """Create the collection if missing; fail loudly on a dimension mismatch."""
        from qdrant_client import models

        client = self._get_client()
        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            # Collection exists: verify its vector size matches the embedding
            # model, so a stale collection fails here with a clear message
            # instead of failing later on upsert.
            info = client.get_collection(self.collection)
            params = info.config.params.vectors
            size = params.size if hasattr(params, "size") else None
            if size is not None and size != settings.embedding_dim:
                raise RuntimeError(
                    f"Qdrant collection {self.collection!r} has vector size {size} "
                    f"but the embedding model needs {settings.embedding_dim} — "
                    "delete the collection and re-index"
                )
        # Qdrant Cloud runs in strict mode: filtered queries are rejected
        # unless the filtered field has a payload index — delete_setids()
        # filters on `setid`, so a fresh collection fails its first /index
        # without this. Creating an index that already exists is a no-op
        # server-side, so this is safe to run on every init.
        client.create_payload_index(
            collection_name=self.collection,
            field_name="setid",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def drop(self) -> None:
        """Delete the collection, then recreate it empty."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection)
        except Exception:
            pass
        self.init()

    def count(self) -> int:
        return self._get_client().count(self.collection, exact=True).count

    def delete_setids(self, setids: set[str]) -> None:
        """Delete all points whose payload setid is in `setids`."""
        if not setids:
            return
        from qdrant_client import models

        self._get_client().delete(
            collection_name=self.collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="setid",
                        match=models.MatchAny(any=sorted(setids)),
                    )
                ]
            ),
        )

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=_point_id(chunk_id),
                vector=vec,
                payload={
                    "chunk_id": chunk_id,
                    "document": doc,
                    **meta,
                },
            )
            for chunk_id, vec, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]
        batch_size = 500
        client = self._get_client()
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch_size],
            )

    def query_vector(self, embedding, n: int) -> list[str]:
        """Top-n chunk ids by cosine similarity, best first."""
        resp = self._get_client().query_points(
            collection_name=self.collection,
            query=embedding,
            limit=n,
            with_payload=["chunk_id"],
        )
        return [p.payload["chunk_id"] for p in resp.points]

    def get_all(self):
        """Return (ids, documents, metadatas) for every point (scroll pages)."""
        client = self._get_client()
        ids, documents, metadatas = [], [], []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = dict(p.payload)
                ids.append(payload.pop("chunk_id"))
                documents.append(payload.pop("document"))
                metadatas.append(payload)
            if offset is None:
                break
        return ids, documents, metadatas
