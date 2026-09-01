import pytest

from app.config.settings import get_settings


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Tests never call real APIs, but Settings requires these fields to be
    present to construct at all — stub values keep tests independent of a
    real .env file."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
