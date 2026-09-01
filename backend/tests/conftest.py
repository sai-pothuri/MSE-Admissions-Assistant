import pytest

from app.clients.anthropic_client import get_anthropic_client
from app.clients.qdrant_client import get_qdrant_client
from app.clients.voyage_client import get_voyage_client
from app.config.settings import get_settings

_CACHED_GETTERS = (get_settings, get_qdrant_client, get_voyage_client, get_anthropic_client)


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Tests never call real APIs, but Settings requires these fields to be
    present to construct at all — stub values keep tests independent of a
    real .env file. All @lru_cache'd getters (settings + the three clients,
    which are each built from settings) are cleared before and after every
    test so a real, unmocked call to one of them can't return a stale
    instance built under a different test's env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    for getter in _CACHED_GETTERS:
        getter.cache_clear()
    yield
    for getter in _CACHED_GETTERS:
        getter.cache_clear()
