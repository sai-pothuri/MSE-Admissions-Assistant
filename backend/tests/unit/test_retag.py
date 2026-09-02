from unittest.mock import MagicMock

import pytest

from app.services.admin import retag


def _record(
    point_id="p1",
    text="Tuition is $26,250.",
    source_file="faq.pdf",
    page=1,
    category="tuition",
    auto_tagged=True,
):
    record = MagicMock()
    record.id = point_id
    record.payload = {
        "text": text,
        "source_file": source_file,
        "page_number": page,
        "category": category,
        "auto_tagged": auto_tagged,
    }
    return record


def test_list_chunks_maps_qdrant_records_to_chunk_records(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll.return_value = ([_record()], None)
    monkeypatch.setattr(retag, "get_qdrant_client", lambda: mock_qdrant)

    chunks = retag.list_chunks("mse_kb_prod", "faq.pdf")

    assert chunks == [
        retag.ChunkRecord(
            point_id="p1",
            text="Tuition is $26,250.",
            source_file="faq.pdf",
            page_number=1,
            category="tuition",
            auto_tagged=True,
        )
    ]
    _, kwargs = mock_qdrant.scroll.call_args
    assert kwargs["collection_name"] == "mse_kb_prod"


def test_list_chunks_returns_empty_list_when_no_matches(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll.return_value = ([], None)
    monkeypatch.setattr(retag, "get_qdrant_client", lambda: mock_qdrant)

    assert retag.list_chunks("mse_kb_prod", "nonexistent.pdf") == []


def test_retag_chunk_updates_payload_and_logs(monkeypatch):
    mock_qdrant = MagicMock()
    monkeypatch.setattr(retag, "get_qdrant_client", lambda: mock_qdrant)
    log_mock = MagicMock()
    monkeypatch.setattr(retag.manual_edit_log, "append_entry", log_mock)

    retag.retag_chunk("mse_kb_prod", "p1", "tuition")

    mock_qdrant.set_payload.assert_called_once_with(
        collection_name="mse_kb_prod",
        payload={"category": "tuition", "auto_tagged": False},
        points=["p1"],
    )
    log_mock.assert_called_once()


def test_retag_chunk_rejects_an_unknown_category(monkeypatch):
    mock_qdrant = MagicMock()
    monkeypatch.setattr(retag, "get_qdrant_client", lambda: mock_qdrant)
    log_mock = MagicMock()
    monkeypatch.setattr(retag.manual_edit_log, "append_entry", log_mock)

    with pytest.raises(retag.InvalidCategoryError):
        retag.retag_chunk("mse_kb_prod", "p1", "not_a_real_category")

    mock_qdrant.set_payload.assert_not_called()
    log_mock.assert_not_called()
