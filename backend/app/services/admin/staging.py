from dataclasses import dataclass
from typing import cast

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

from app.clients.qdrant_client import get_qdrant_client
from app.config.settings import get_settings
from app.services.admin import manual_edit_log

SCROLL_LIMIT = 1000  # small corpus (CLAUDE.md) — no pagination needed


@dataclass(frozen=True)
class PromotionResult:
    source_file: str
    promoted_count: int


def promote_file(source_file: str) -> PromotionResult:
    """Copies every staged chunk for `source_file` into the production
    collection, replacing whatever prod chunks currently exist for that
    file (delete-then-upsert, same semantics as re-indexing), then removes
    the promoted points from staging. Promoting is how staged content
    becomes the live version of that file."""
    settings = get_settings()
    qdrant = get_qdrant_client()
    source_filter = Filter(
        must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
    )

    points, _ = qdrant.scroll(
        collection_name=settings.qdrant_collection_staging,
        scroll_filter=source_filter,
        limit=SCROLL_LIMIT,
        with_payload=True,
        with_vectors=True,
    )
    if not points:
        return PromotionResult(source_file=source_file, promoted_count=0)

    qdrant.delete(
        collection_name=settings.qdrant_collection_prod,
        points_selector=FilterSelector(filter=source_filter),
    )
    qdrant.upsert(
        collection_name=settings.qdrant_collection_prod,
        points=[
            PointStruct(id=point.id, vector=cast(list[float], point.vector), payload=point.payload)
            for point in points
        ],
    )
    qdrant.delete(
        collection_name=settings.qdrant_collection_staging,
        points_selector=FilterSelector(filter=source_filter),
    )

    manual_edit_log.append_entry(
        "promote", f"{source_file}: {len(points)} chunk(s) staging -> prod"
    )
    return PromotionResult(source_file=source_file, promoted_count=len(points))
