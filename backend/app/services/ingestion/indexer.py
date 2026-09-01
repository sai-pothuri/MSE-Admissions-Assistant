import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from app.clients.qdrant_client import get_qdrant_client
from app.clients.voyage_client import get_voyage_client
from app.config.settings import get_settings
from app.models.schemas import ChunkPayload
from app.services.ingestion.chunking import chunk_blocks
from app.services.ingestion.embedding import embed_documents
from app.services.ingestion.pdf_extraction import extract_blocks


def index_file(pdf_path: Path) -> int:
    """Extract, chunk, embed, and upsert a single PDF. Category is
    folder-derived (the parent directory name under data/knowledge_base/) —
    a Phase 1 placeholder; Phase 2 replaces it with real per-chunk
    auto-tagging via `app.clients.anthropic_client`."""
    settings = get_settings()
    qdrant = get_qdrant_client()
    voyage = get_voyage_client()
    category = pdf_path.parent.name

    blocks = extract_blocks(pdf_path)
    chunks = chunk_blocks(blocks)
    if not chunks:
        return 0

    embeddings = embed_documents(voyage, [chunk.text for chunk in chunks])

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=ChunkPayload(
                text=chunk.text,
                source_file=pdf_path.name,
                page_number=chunk.page_number,
                category=category,
                auto_tagged=False,
            ).model_dump(),
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]

    qdrant.upsert(collection_name=settings.qdrant_collection_prod, points=points)
    return len(points)


def index_directory(root: Path) -> dict[str, int]:
    """Index every PDF found under `root` (recursively), sequentially —
    the corpus is small enough that async/batch processing isn't warranted."""
    results: dict[str, int] = {}
    for pdf_path in sorted(root.rglob("*.pdf")):
        results[str(pdf_path)] = index_file(pdf_path)
    return results
