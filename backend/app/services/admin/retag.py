from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from app.clients.qdrant_client import get_qdrant_client
from app.config.taxonomy import CATEGORIES
from app.services.admin import manual_edit_log


def _source_file_filter(source_file: str) -> Filter:
    return Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))])


LIST_CHUNKS_LIMIT = 1000  # small corpus (CLAUDE.md) — no pagination needed


@dataclass(frozen=True)
class ChunkRecord:
    point_id: str
    text: str
    source_file: str
    page_number: int
    category: str
    auto_tagged: bool


class InvalidCategoryError(ValueError):
    pass


def list_chunks(collection: str, source_file: str) -> list[ChunkRecord]:
    qdrant = get_qdrant_client()
    points, _ = qdrant.scroll(
        collection_name=collection,
        scroll_filter=_source_file_filter(source_file),
        limit=LIST_CHUNKS_LIMIT,
        with_payload=True,
    )
    return [
        ChunkRecord(
            point_id=str(point.id),
            text=point.payload["text"],
            source_file=point.payload["source_file"],
            page_number=point.payload["page_number"],
            category=point.payload["category"],
            auto_tagged=point.payload["auto_tagged"],
        )
        for point in points
        if point.payload is not None
    ]


def retag_chunk(collection: str, point_id: str, new_category: str) -> None:
    """Updates a single chunk's category in place — no re-embedding, since
    the chunk's text/vector don't change, only its classification.
    Overriding a category is inherently a manual correction, so
    `auto_tagged` is always set to False regardless of its prior value."""
    if new_category not in CATEGORIES:
        raise InvalidCategoryError(f"'{new_category}' is not a known category: {CATEGORIES}")
    qdrant = get_qdrant_client()
    qdrant.set_payload(
        collection_name=collection,
        payload={"category": new_category, "auto_tagged": False},
        points=[point_id],
    )
    manual_edit_log.append_entry("retag", f"{point_id} -> {new_category} (collection={collection})")


def delete_chunks(collection: str, source_file: str) -> None:
    """Removes every chunk for `source_file` from `collection` — used when
    a file is deleted entirely, to clear it out of staging and/or prod
    rather than leaving stale chunks behind."""
    qdrant = get_qdrant_client()
    qdrant.delete(
        collection_name=collection,
        points_selector=FilterSelector(filter=_source_file_filter(source_file)),
    )
