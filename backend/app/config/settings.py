from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str
    voyage_api_key: str

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prod: str = "mse_kb_prod"
    qdrant_collection_staging: str = "mse_kb_staging"

    voyage_embedding_model: str = "voyage-3"
    anthropic_generation_model: str = "claude-sonnet-5"
    anthropic_classification_model: str = "claude-haiku-4-5-20251001"


@lru_cache
def get_settings() -> Settings:
    return Settings()
