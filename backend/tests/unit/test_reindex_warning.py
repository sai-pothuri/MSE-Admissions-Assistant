from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.admin import reindex
from app.services.admin.retag import ChunkRecord


def _chunk(point_id="p1", auto_tagged=True):
    return ChunkRecord(
        point_id=point_id,
        text="Some text.",
        source_file="faq.pdf",
        page_number=1,
        category="tuition",
        auto_tagged=auto_tagged,
    )


def test_check_reindex_warning_flags_manually_corrected_chunks(monkeypatch):
    monkeypatch.setattr(
        reindex, "list_chunks", lambda collection, source_file: [_chunk(auto_tagged=False)]
    )

    warning = reindex.check_reindex_warning("mse_kb_prod", "faq.pdf")

    assert warning.has_manual_corrections is True
    assert warning.corrected_chunk_count == 1


def test_check_reindex_warning_is_clean_when_all_chunks_are_auto_tagged(monkeypatch):
    monkeypatch.setattr(
        reindex, "list_chunks", lambda collection, source_file: [_chunk(auto_tagged=True)]
    )

    warning = reindex.check_reindex_warning("mse_kb_prod", "faq.pdf")

    assert warning.has_manual_corrections is False
    assert warning.corrected_chunk_count == 0


def test_check_reindex_warning_is_clean_when_file_has_no_existing_chunks(monkeypatch):
    monkeypatch.setattr(reindex, "list_chunks", lambda collection, source_file: [])

    warning = reindex.check_reindex_warning("mse_kb_prod", "new.pdf")

    assert warning.has_manual_corrections is False


def test_reindex_file_raises_when_manual_corrections_exist_and_not_confirmed(monkeypatch):
    monkeypatch.setattr(
        reindex, "list_chunks", lambda collection, source_file: [_chunk(auto_tagged=False)]
    )
    index_file_mock = MagicMock()
    monkeypatch.setattr(reindex, "index_file", index_file_mock)

    with pytest.raises(reindex.ReindexConfirmationRequiredError) as exc_info:
        reindex.reindex_file(Path("data/knowledge_base/general/faq.pdf"), "mse_kb_prod")

    assert exc_info.value.warning.has_manual_corrections is True
    index_file_mock.assert_not_called()


def test_reindex_file_proceeds_when_manual_corrections_exist_and_confirmed(monkeypatch):
    monkeypatch.setattr(
        reindex, "list_chunks", lambda collection, source_file: [_chunk(auto_tagged=False)]
    )
    index_file_mock = MagicMock(return_value=5)
    monkeypatch.setattr(reindex, "index_file", index_file_mock)

    count = reindex.reindex_file(
        Path("data/knowledge_base/general/faq.pdf"), "mse_kb_prod", confirm=True
    )

    assert count == 5
    index_file_mock.assert_called_once_with(
        Path("data/knowledge_base/general/faq.pdf"), collection="mse_kb_prod"
    )


def test_reindex_file_proceeds_without_confirmation_when_no_manual_corrections(monkeypatch):
    monkeypatch.setattr(reindex, "list_chunks", lambda collection, source_file: [])
    index_file_mock = MagicMock(return_value=3)
    monkeypatch.setattr(reindex, "index_file", index_file_mock)

    count = reindex.reindex_file(Path("data/knowledge_base/general/new.pdf"), "mse_kb_prod")

    assert count == 3
