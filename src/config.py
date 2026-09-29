"""Central configuration. Everything is overridable via environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chroma_dir: str = "./data/chroma"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    llm_provider: str = "ollama"  # "ollama" | "openai"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    use_reranker: bool = True

    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200


settings = Settings()
