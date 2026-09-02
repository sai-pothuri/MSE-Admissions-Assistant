from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.clients.qdrant_client import get_qdrant_client
from app.clients.voyage_client import get_voyage_client
from app.config.settings import get_settings
from app.config.thresholds import TOP_K
from app.models.schemas import ChunkPayload
from app.services.ingestion.embedding import embed_query


class CollectionNotReadyError(RuntimeError):
    """Raised when the Qdrant collection hasn't been created/ingested yet."""


@dataclass
class SearchResult:
    chunk: ChunkPayload
    score: float


def embed(question: str) -> list[float]:
    return embed_query(get_voyage_client(), question)


def search(
    question: str, category: str | None = None, query_vector: list[float] | None = None
) -> list[SearchResult]:
    """`query_vector` lets a caller that needs both a filtered and an
    unfiltered search for the same question (see `query_pipeline`) embed
    the question once via `embed()` and reuse it, instead of paying for a
    Voyage embedding call on every `search()` invocation."""
    settings = get_settings()
    qdrant = get_qdrant_client()

    if not qdrant.collection_exists(settings.qdrant_collection_prod):
        raise CollectionNotReadyError(
            f"Collection '{settings.qdrant_collection_prod}' does not exist yet. "
            "Run `python -m app.scripts.init_collection` and ingest documents first."
        )

    if query_vector is None:
        query_vector = embed(question)
    query_filter = (
        Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
        if category is not None
        else None
    )
    hits = qdrant.query_points(
        collection_name=settings.qdrant_collection_prod,
        query=query_vector,
        query_filter=query_filter,
        limit=TOP_K,
    ).points

    return [
        SearchResult(chunk=ChunkPayload.model_validate(hit.payload), score=hit.score)
        for hit in hits
    ]


def top1_and_gap(results: list[SearchResult]) -> tuple[float, float]:
    """Returns (top1_score, score_gap) for the confidence gate. With no
    runner-up, the gap defaults to the top score itself — there's nothing
    to be ambiguous against, so a missing second result shouldn't fail a
    gap check on its own."""
    if not results:
        return 0.0, 0.0
    top1 = results[0].score
    gap = top1 - (results[1].score if len(results) > 1 else 0.0)
    return top1, gap
