from typing import cast

import voyageai

from app.config.settings import get_settings


def embed_documents(client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    result = client.embed(texts, model=settings.voyage_embedding_model, input_type="document")
    return cast(list[list[float]], result.embeddings)


def embed_query(client: voyageai.Client, text: str) -> list[float]:
    settings = get_settings()
    result = client.embed([text], model=settings.voyage_embedding_model, input_type="query")
    return cast(list[float], result.embeddings[0])
