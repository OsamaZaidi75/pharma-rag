"""Central configuration. Everything is overridable via environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chroma_dir: str = "./data/chroma"

    # Vector store: Qdrant Cloud when BOTH qdrant_url and qdrant_api_key
    # are set, else local ChromaDB (the default / fallback). No credentials
    # exist yet, so the deployed app stays on Chroma until they arrive —
    # cutover is just adding QDRANT_URL + QDRANT_API_KEY to the env file
    # and re-indexing (the Qdrant collection starts empty).
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "pharma_rag"

    # Embeddings go through the NVIDIA API (same key + base URL as the LLM
    # below) — no local model, no torch. The 8b model is 4096-dim; if you
    # switch EMBEDDING_MODEL, set EMBEDDING_DIM to match, and note the old
    # ChromaDB index is incompatible with a new dimension (rebuild it).
    embedding_model: str = "nvidia/nemotron-3.5-embed-8b"
    embedding_dim: int = 4096

    # Hosted default is NVIDIA API (OpenAI-compatible). Ollama / OpenAI /
    # Gemini remain as local alternatives. With no LLM reachable (no key
    # set and no Ollama server), /ask falls back to the built-in extractive
    # answerer — still cited; only /ask-agent needs a working LLM.
    # NOTE: embeddings always need NVIDIA_API_KEY (no local fallback since
    # the torch removal), so retrieval/indexing fail without the key even
    # when the LLM provider is Ollama/OpenAI/Gemini.
    llm_provider: str = "nvidia"  # "nvidia" | "ollama" | "openai" | "gemini"
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"  # 2.5-flash is retired for new API keys

    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    # Deprecated no-op: cross-encoder reranking was removed with the torch
    # removal (2026-11-18); RRF order is final. Kept so existing .env files
    # and callers don't break.
    use_reranker: bool = False

    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    def llm_health_info(self) -> dict:
        """LLM provider status for GET /health: {provider, model, configured}.

        No secrets — only the provider name, the configured model id, and
        whether the provider is usable. Ollama needs no API key, so it is
        always configured.
        """
        provider = self.llm_provider
        if provider == "nvidia":
            return {"provider": provider, "model": self.nvidia_model,
                    "configured": bool(self.nvidia_api_key)}
        if provider == "gemini":
            return {"provider": provider, "model": self.gemini_model,
                    "configured": bool(self.gemini_api_key)}
        if provider == "openai":
            return {"provider": provider, "model": self.openai_model,
                    "configured": bool(self.openai_api_key)}
        return {"provider": provider, "model": self.ollama_model, "configured": True}

    # On-demand DailyMed download cache: downloaded SPL XML is cached on disk
    # so re-asked drugs skip the HTTP round-trip. TTL in days.
    spl_cache_dir: str = "./data/spl_cache"
    spl_cache_ttl_days: float = 7.0


settings = Settings()
