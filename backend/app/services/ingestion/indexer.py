import uuid
from dataclasses import dataclass
from pathlib import Path

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


def index_file(pdf_path: Path) -> int:
    """Extract, chunk, auto-tag, embed, and upsert a single PDF. Each
    chunk's category is independently classified by Claude (one call per
    chunk, per CLAUDE.md) rather than inherited from the source file's
    folder — a single document can genuinely span multiple categories.

    Re-indexing a file fully replaces its existing chunks (delete by
    source_file, then re-upsert) so repeated runs stay idempotent."""
    settings = get_settings()
    qdrant = get_qdrant_client()
    voyage = get_voyage_client()

    qdrant.delete(
        collection_name=settings.qdrant_collection_prod,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=pdf_path.name))]
            )
        ),
    )

    blocks = extract_blocks(pdf_path)
    chunks = chunk_blocks(blocks)
    if not chunks:
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

    qdrant.upsert(collection_name=settings.qdrant_collection_prod, points=points)
    return len(points)


def index_directory(root: Path) -> dict[str, IndexResult]:
    """Index every PDF found under `root` (recursively), sequentially —
    the corpus is small enough that async/batch processing isn't warranted.
    A failure on one file is captured and skipped rather than aborting the
    whole batch."""
    results: dict[str, IndexResult] = {}
    for pdf_path in sorted(root.rglob("*.pdf")):
        try:
            results[str(pdf_path)] = IndexResult(chunk_count=index_file(pdf_path))
        except Exception as exc:  # noqa: BLE001 - report and continue, don't abort the batch
            results[str(pdf_path)] = IndexResult(chunk_count=0, error=str(exc))
    return results
