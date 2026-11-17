"""Central configuration. Everything is overridable via environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chroma_dir: str = "./data/chroma"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Hosted default is NVIDIA API (OpenAI-compatible). Ollama / OpenAI /
    # Gemini remain as local alternatives. With no LLM reachable (no key
    # set and no Ollama server), /ask falls back to the built-in extractive
    # answerer — still cited; only /ask-agent needs a working LLM.
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
    use_reranker: bool = True

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
