# ============================================================
# FILE: app/config.py
# PURPOSE: Centralised settings loaded from .env via pydantic BaseSettings
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- OpenAI ---
    OPENAI_API_KEY: str

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379"

    # --- FAISS ---
    FAISS_INDEX_DIR: str = "./faiss_indexes"

    # --- Chunking ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # --- Cache ---
    CACHE_TTL_SECONDS: int = 3600

    # --- Retrieval ---
    TOP_K_RESULTS: int = 4

    # --- Logging ---
    LOG_LEVEL: str = "INFO"


settings = Settings()
