from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.ingestion import indexer
from app.services.ingestion.chunking import Chunk
from app.services.ingestion.pdf_extraction import ExtractedBlock


def test_index_file_upserts_one_point_per_chunk(monkeypatch):
    fake_blocks = [ExtractedBlock(text="Some admissions info.", page_number=1)]
    fake_chunks = [Chunk(text="Some admissions info.", page_number=1)]
    fake_embeddings = [[0.1, 0.2, 0.3]]

    mock_qdrant = MagicMock()
    mock_voyage = MagicMock()

    monkeypatch.setattr(indexer, "extract_blocks", lambda path: fake_blocks)
    monkeypatch.setattr(indexer, "chunk_blocks", lambda blocks: fake_chunks)
    monkeypatch.setattr(indexer, "embed_documents", lambda client, texts: fake_embeddings)
    monkeypatch.setattr(indexer, "classify_chunk", lambda text: "admissions")
    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(indexer, "get_voyage_client", lambda: mock_voyage)

    count = indexer.index_file(Path("data/knowledge_base/general/handbook.pdf"))

    assert count == 1
    mock_qdrant.delete.assert_called_once()
    mock_qdrant.upsert.assert_called_once()
    _, kwargs = mock_qdrant.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["source_file"] == "handbook.pdf"
    assert points[0].payload["category"] == "admissions"
    assert points[0].payload["auto_tagged"] is True


def test_index_file_deletes_existing_chunks_for_the_file_before_upserting(monkeypatch):
    """Re-indexing must be idempotent: old chunks for this source_file are
    cleared first so re-running ingestion doesn't duplicate them."""
    fake_blocks = [ExtractedBlock(text="Some admissions info.", page_number=1)]
    fake_chunks = [Chunk(text="Some admissions info.", page_number=1)]
    mock_qdrant = MagicMock()

    monkeypatch.setattr(indexer, "extract_blocks", lambda path: fake_blocks)
    monkeypatch.setattr(indexer, "chunk_blocks", lambda blocks: fake_chunks)
    monkeypatch.setattr(indexer, "embed_documents", lambda client, texts: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(indexer, "classify_chunk", lambda text: "admissions")
    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(indexer, "get_voyage_client", lambda: MagicMock())

    indexer.index_file(Path("data/knowledge_base/general/handbook.pdf"))

    call_order = [call[0] for call in mock_qdrant.method_calls]
    assert call_order.index("delete") < call_order.index("upsert")


def test_index_file_preserves_existing_chunks_when_classification_fails_partway(monkeypatch):
    """Regression test: if classify_chunk raises partway through the
    per-chunk loop, the file's existing Qdrant chunks must be left intact
    rather than deleted with nothing ready to replace them."""
    fake_blocks = [
        ExtractedBlock(text="First chunk.", page_number=1),
        ExtractedBlock(text="Second chunk.", page_number=1),
    ]
    fake_chunks = [
        Chunk(text="First chunk.", page_number=1),
        Chunk(text="Second chunk.", page_number=1),
    ]
    mock_qdrant = MagicMock()

    def failing_classify_chunk(text: str) -> str:
        if text == "Second chunk.":
            raise RuntimeError("classification API error")
        return "admissions"

    monkeypatch.setattr(indexer, "extract_blocks", lambda path: fake_blocks)
    monkeypatch.setattr(indexer, "chunk_blocks", lambda blocks: fake_chunks)
    monkeypatch.setattr(
        indexer, "embed_documents", lambda client, texts: [[0.1, 0.2], [0.3, 0.4]]
    )
    monkeypatch.setattr(indexer, "classify_chunk", failing_classify_chunk)
    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(indexer, "get_voyage_client", lambda: MagicMock())

    with pytest.raises(RuntimeError):
        indexer.index_file(Path("data/knowledge_base/general/handbook.pdf"))

    mock_qdrant.delete.assert_not_called()
    mock_qdrant.upsert.assert_not_called()


def test_index_file_returns_zero_for_empty_document(monkeypatch):
    monkeypatch.setattr(indexer, "extract_blocks", lambda path: [])
    monkeypatch.setattr(indexer, "chunk_blocks", lambda blocks: [])
    mock_qdrant = MagicMock()
    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(indexer, "get_voyage_client", lambda: MagicMock())

    count = indexer.index_file(Path("data/knowledge_base/general/empty.pdf"))

    assert count == 0
    mock_qdrant.upsert.assert_not_called()
    mock_qdrant.delete.assert_called_once()


def test_index_directory_continues_after_one_file_fails(monkeypatch, tmp_path):
    good_pdf = tmp_path / "good.pdf"
    bad_pdf = tmp_path / "bad.pdf"
    good_pdf.write_bytes(b"")
    bad_pdf.write_bytes(b"")

    def fake_index_file(path: Path) -> int:
        if path.name == "bad.pdf":
            raise ValueError("corrupt PDF")
        return 3

    monkeypatch.setattr(indexer, "index_file", fake_index_file)

    results = indexer.index_directory(tmp_path)

    assert results[str(bad_pdf)].error == "corrupt PDF"
    assert results[str(bad_pdf)].chunk_count == 0
    assert results[str(good_pdf)].error is None
    assert results[str(good_pdf)].chunk_count == 3
