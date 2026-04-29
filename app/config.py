from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API metadata
    app_name: str = "Indicnode RAG Assistant"
    app_version: str = "1.0.0"
    debug: bool = False

    # OpenRouter (OpenAI-compatible)
    openrouter_api_key: str = ""
    # fast_model  : used for simple/short queries
    # smart_model : used for complex/comparative queries (auto-routed)
    # fallback_model: used when primary fails
    fast_model: str = "anthropic/claude-3-haiku"
    smart_model: str = "openai/gpt-4o-mini"
    fallback_model: str = "openai/gpt-4o-mini"
    generation_model: str = "anthropic/claude-3-haiku"  # kept for backwards compat

    # Embeddings (sentence-transformers, runs locally)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval
    top_k_dense: int = 5
    top_k_sparse: int = 5
    top_k_final: int = 3

    # Chunking
    chunk_size: int = 400   # words
    chunk_overlap: int = 50  # words

    # Security
    # Minimum cosine similarity of the top retrieved chunk for a query to be
    # considered in-domain.  Tune this on your eval set.
    domain_similarity_threshold: float = 0.28

    # Cache
    cache_max_size: int = 1000
    cache_semantic_threshold: float = 0.93

    # Paths
    knowledge_base_dir: str = "knowledge_base/documents"
    index_dir: str = "knowledge_base/index"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
