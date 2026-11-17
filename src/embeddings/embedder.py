"""Embeddings via the NVIDIA API (OpenAI-compatible embeddings endpoint).

Replaces the old local sentence-transformers model: no torch, no model
download, and the Docker image drops from ~3.5 GB to a few hundred MB.
The same NVIDIA_API_KEY the LLM calls use authenticates the embeddings
calls; the model is env-configurable (EMBEDDING_MODEL, default
nvidia/nemotron-3.5-embed-8b, 4096-dim).

The 8b model requires an `input_type`: "query" for search queries,
"passage" for documents being indexed. embed() defaults to "passage"
(indexing); embed_one() uses "query" (retrieval). NOTE: the fallback
model nvidia/nemotron-3-embed-1b (2048-dim) works with no input_type —
if you switch EMBEDDING_MODEL to it, drop the extra_body below.

The first API call can be slow (cold start), so the client timeout is a
generous 120s. The OpenAI client is imported lazily so importing this
module never needs the `openai` package (the CI test env doesn't install
it) and never makes a network call.
"""
import numpy as np

from ..config import settings

EMBED_TIMEOUT_S = 120  # first call can be slow (cold start)


class Embedder:
    def __init__(self, model_name: str | None = None, dim: int | None = None):
        self.model_name = model_name or settings.embedding_model
        self._dim = dim if dim is not None else settings.embedding_dim
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not settings.nvidia_api_key:
                raise RuntimeError(
                    "NVIDIA_API_KEY is not set — embeddings need the NVIDIA API key"
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.nvidia_api_key,
                base_url=settings.nvidia_base_url,
                timeout=EMBED_TIMEOUT_S,
                max_retries=3,
            )
        return self._client

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        input_type: str = "passage",
    ) -> list[list[float]]:
        """Embed texts -> L2-normalized vectors (cosine-ready).

        Sent to the API in batches (input as a list); `input_type` is
        "passage" for documents being indexed, "query" for search queries.
        """
        if not texts:
            return []
        client = self._get_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                model=self.model_name,
                input=batch,
                extra_body={"input_type": input_type},
            )
            for item in resp.data:
                vec = np.asarray(item.embedding, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                out.append(vec.tolist())
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text], input_type="query")[0]

    @property
    def dim(self) -> int:
        return self._dim
