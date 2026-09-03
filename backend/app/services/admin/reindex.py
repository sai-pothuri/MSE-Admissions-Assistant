from dataclasses import dataclass
from pathlib import Path

from app.services.admin.retag import list_chunks
from app.services.ingestion.indexer import index_file


@dataclass(frozen=True)
class ReindexWarning:
    has_manual_corrections: bool
    corrected_chunk_count: int


class ReindexConfirmationRequiredError(Exception):
    """Raised when re-indexing would discard manually-corrected chunks and
    the caller hasn't confirmed — the API route turns this into a 409 with
    the warning payload for the frontend to render as a confirm modal
    (CLAUDE.md's admin console spec)."""

    def __init__(self, warning: ReindexWarning) -> None:
        self.warning = warning
        super().__init__("Re-indexing this file will discard manual category corrections.")


def check_reindex_warning(collection: str, source_file: str) -> ReindexWarning:
    existing = list_chunks(collection, source_file)
    corrected = [chunk for chunk in existing if not chunk.auto_tagged]
    return ReindexWarning(
        has_manual_corrections=bool(corrected), corrected_chunk_count=len(corrected)
    )


def reindex_file(pdf_path: Path, collection: str, confirm: bool = False) -> int:
    """Re-indexes `pdf_path`, fully replacing its existing chunks in
    `collection`. Manual category corrections on the old chunks are NOT
    preserved (CLAUDE.md) — if any exist, this raises
    `ReindexConfirmationRequiredError` unless `confirm=True`, so discarding
    them is always a deliberate, confirmed action."""
    warning = check_reindex_warning(collection, pdf_path.name)
    if warning.has_manual_corrections and not confirm:
        raise ReindexConfirmationRequiredError(warning)
    return index_file(pdf_path, collection=collection)
