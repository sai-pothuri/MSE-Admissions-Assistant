from dataclasses import dataclass

from app.clients.qdrant_client import get_qdrant_client
from app.clients.voyage_client import get_voyage_client
from app.config.settings import get_settings
from app.models.schemas import ChunkPayload
from app.services.ingestion.embedding import embed_query

TOP_K = 5


@dataclass
class SearchResult:
    chunk: ChunkPayload
    score: float


def search(question: str) -> list[SearchResult]:
    settings = get_settings()
    qdrant = get_qdrant_client()
    voyage = get_voyage_client()

    query_vector = embed_query(voyage, question)
    hits = qdrant.query_points(
        collection_name=settings.qdrant_collection_prod,
        query=query_vector,
        limit=TOP_K,
    ).points

    return [
        SearchResult(chunk=ChunkPayload.model_validate(hit.payload), score=hit.score)
        for hit in hits
    ]
