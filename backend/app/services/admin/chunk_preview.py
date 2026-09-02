from dataclasses import dataclass
from pathlib import Path

from app.services.ingestion.auto_tagging import classify_chunk
from app.services.ingestion.chunking import chunk_blocks
from app.services.ingestion.pdf_extraction import extract_blocks


@dataclass(frozen=True)
class ChunkPreview:
    text: str
    page_number: int
    category: str


def preview_file(pdf_path: Path) -> list[ChunkPreview]:
    """Dry run: extract + chunk + auto-tag, with no embedding and no
    Qdrant write, so an admin can see how a file will be chunked and
    categorized before committing to indexing it. Auto-tagging still makes
    real Claude calls (this is what actually shows the real category, not
    a placeholder) — only the embed+upsert step is skipped."""
    blocks = extract_blocks(pdf_path)
    chunks = chunk_blocks(blocks)
    return [
        ChunkPreview(
            text=chunk.text, page_number=chunk.page_number, category=classify_chunk(chunk.text)
        )
        for chunk in chunks
    ]
