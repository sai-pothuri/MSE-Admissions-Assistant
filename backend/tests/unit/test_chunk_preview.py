from pathlib import Path

from app.services.admin import chunk_preview
from app.services.ingestion.chunking import Chunk
from app.services.ingestion.pdf_extraction import ExtractedBlock


def test_preview_file_extracts_chunks_with_category_and_no_qdrant_write(monkeypatch):
    fake_blocks = [ExtractedBlock(text="Tuition is $26,250.", page_number=1)]
    fake_chunks = [Chunk(text="Tuition is $26,250.", page_number=1)]

    monkeypatch.setattr(chunk_preview, "extract_blocks", lambda path: fake_blocks)
    monkeypatch.setattr(chunk_preview, "chunk_blocks", lambda blocks: fake_chunks)
    monkeypatch.setattr(chunk_preview, "classify_chunk", lambda text: "tuition")

    previews = chunk_preview.preview_file(Path("data/knowledge_base/general/handbook.pdf"))

    assert previews == [
        chunk_preview.ChunkPreview(text="Tuition is $26,250.", page_number=1, category="tuition")
    ]


def test_preview_file_returns_empty_list_for_empty_document(monkeypatch):
    monkeypatch.setattr(chunk_preview, "extract_blocks", lambda path: [])
    monkeypatch.setattr(chunk_preview, "chunk_blocks", lambda blocks: [])

    assert chunk_preview.preview_file(Path("data/knowledge_base/general/empty.pdf")) == []
