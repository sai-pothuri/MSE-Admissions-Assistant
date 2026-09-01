from functools import lru_cache

import voyageai

from app.config.settings import get_settings


@lru_cache
def get_voyage_client() -> voyageai.Client:
    settings = get_settings()
    return voyageai.Client(api_key=settings.voyage_api_key)
