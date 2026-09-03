from unittest.mock import MagicMock

from app.services.admin import staging


def _point(point_id="p1", vector=None, payload=None):
    point = MagicMock()
    point.id = point_id
    point.vector = vector or [0.1, 0.2]
    point.payload = payload or {"source_file": "faq.pdf", "category": "tuition"}
    return point


def test_promote_file_copies_staged_points_to_prod_and_removes_from_staging(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll.return_value = ([_point()], None)
    monkeypatch.setattr(staging, "get_qdrant_client", lambda: mock_qdrant)
    log_mock = MagicMock()
    monkeypatch.setattr(staging.manual_edit_log, "append_entry", log_mock)

    result = staging.promote_file("faq.pdf")

    assert result == staging.PromotionResult(source_file="faq.pdf", promoted_count=1)
    mock_qdrant.scroll.assert_called_once()
    _, scroll_kwargs = mock_qdrant.scroll.call_args
    assert scroll_kwargs["collection_name"] == "mse_kb_staging"

    delete_calls = mock_qdrant.delete.call_args_list
    assert len(delete_calls) == 2
    assert delete_calls[0].kwargs["collection_name"] == "mse_kb_prod"
    assert delete_calls[1].kwargs["collection_name"] == "mse_kb_staging"

    mock_qdrant.upsert.assert_called_once()
    _, upsert_kwargs = mock_qdrant.upsert.call_args
    assert upsert_kwargs["collection_name"] == "mse_kb_prod"
    assert len(upsert_kwargs["points"]) == 1
    log_mock.assert_called_once()


def test_promote_file_returns_zero_and_does_nothing_when_staging_is_empty(monkeypatch):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll.return_value = ([], None)
    monkeypatch.setattr(staging, "get_qdrant_client", lambda: mock_qdrant)
    log_mock = MagicMock()
    monkeypatch.setattr(staging.manual_edit_log, "append_entry", log_mock)

    result = staging.promote_file("nonexistent.pdf")

    assert result == staging.PromotionResult(source_file="nonexistent.pdf", promoted_count=0)
    mock_qdrant.delete.assert_not_called()
    mock_qdrant.upsert.assert_not_called()
    log_mock.assert_not_called()
