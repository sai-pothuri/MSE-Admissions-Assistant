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

    # Single shared faculty password (CLAUDE.md: "simple password auth for
    # now"), plus the key used to sign admin session cookies. Both required
    # with no default — an admin console with a guessable/default password
    # is worse than one that fails to start.
    admin_password: str
    admin_session_secret: str


@lru_cache
def get_settings() -> Settings:
    # Required fields are supplied by env vars / .env at runtime, not by the
    # caller — mypy can't see that through pydantic-settings' BaseSettings.
    return Settings()  # type: ignore[call-arg]
