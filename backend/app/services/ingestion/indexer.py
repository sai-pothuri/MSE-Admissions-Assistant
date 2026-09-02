import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

from app.clients.qdrant_client import get_qdrant_client
from app.clients.voyage_client import get_voyage_client
from app.config.settings import get_settings
from app.models.schemas import ChunkPayload
from app.services.ingestion.auto_tagging import classify_chunk
from app.services.ingestion.chunking import chunk_blocks
from app.services.ingestion.embedding import embed_documents
from app.services.ingestion.pdf_extraction import extract_blocks


@dataclass
class IndexResult:
    chunk_count: int
    error: str | None = None


def _delete_existing_chunks(qdrant: QdrantClient, collection: str, source_file: str) -> None:
    qdrant.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            )
        ),
    )


def index_file(pdf_path: Path, collection: str | None = None) -> int:
    """Extract, chunk, embed, auto-tag, and upsert a single PDF. Each
    chunk's category is independently classified by Claude (one call per
    chunk, per CLAUDE.md) rather than inherited from the source file's
    folder — a single document can genuinely span multiple categories.

    `collection` defaults to the production collection (the CLI ingestion
    script's behavior); the admin console (Phase 4) passes the staging
    collection explicitly so uploads land there first, not live.

    Re-indexing a file fully replaces its existing chunks (delete by
    source_file, then re-upsert), but the delete only happens once the new
    chunks are fully prepared (embedded and classified) — if per-chunk
    classification fails partway through, the file's existing chunks are
    left untouched rather than being wiped with nothing to replace them."""
    settings = get_settings()
    resolved_collection = collection or settings.qdrant_collection_prod
    qdrant = get_qdrant_client()
    voyage = get_voyage_client()

    blocks = extract_blocks(pdf_path)
    chunks = chunk_blocks(blocks)
    if not chunks:
        # A genuinely empty re-extraction (not a mid-process failure) —
        # reflect that the file now has no content.
        _delete_existing_chunks(qdrant, resolved_collection, pdf_path.name)
        return 0

    embeddings = embed_documents(voyage, [chunk.text for chunk in chunks])
    categories = [classify_chunk(chunk.text) for chunk in chunks]

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=ChunkPayload(
                text=chunk.text,
                source_file=pdf_path.name,
                page_number=chunk.page_number,
                category=category,
                auto_tagged=True,
            ).model_dump(),
        )
        for chunk, embedding, category in zip(chunks, embeddings, categories, strict=True)
    ]

    _delete_existing_chunks(qdrant, resolved_collection, pdf_path.name)
    qdrant.upsert(collection_name=resolved_collection, points=points)
    return len(points)


def index_directory(root: Path, collection: str | None = None) -> dict[str, IndexResult]:
    """Index every PDF found under `root` (recursively), sequentially —
    the corpus is small enough that async/batch processing isn't warranted.
    A failure on one file is captured and skipped rather than aborting the
    whole batch."""
    results: dict[str, IndexResult] = {}
    for pdf_path in sorted(root.rglob("*.pdf")):
        try:
            results[str(pdf_path)] = IndexResult(chunk_count=index_file(pdf_path, collection))
        except Exception as exc:  # noqa: BLE001 - report and continue, don't abort the batch
            results[str(pdf_path)] = IndexResult(chunk_count=0, error=str(exc))
    return results
