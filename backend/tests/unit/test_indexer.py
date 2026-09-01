from pathlib import Path
from unittest.mock import MagicMock

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
    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(indexer, "get_voyage_client", lambda: mock_voyage)

    count = indexer.index_file(Path("data/knowledge_base/general/handbook.pdf"))

    assert count == 1
    mock_qdrant.upsert.assert_called_once()
    _, kwargs = mock_qdrant.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["source_file"] == "handbook.pdf"
    assert points[0].payload["category"] == "general"
    assert points[0].payload["auto_tagged"] is False


def test_index_file_returns_zero_for_empty_document(monkeypatch):
    monkeypatch.setattr(indexer, "extract_blocks", lambda path: [])
    monkeypatch.setattr(indexer, "chunk_blocks", lambda blocks: [])
    mock_qdrant = MagicMock()
    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr(indexer, "get_voyage_client", lambda: MagicMock())

    count = indexer.index_file(Path("data/knowledge_base/general/empty.pdf"))

    assert count == 0
    mock_qdrant.upsert.assert_not_called()
