"""Local embeddings via sentence-transformers. Free, no API key, CPU-friendly.

The model is loaded lazily so importing this module never downloads anything.
"""
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed texts -> L2-normalized vectors (cosine-ready)."""
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return [v.astype(np.float32).tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()
