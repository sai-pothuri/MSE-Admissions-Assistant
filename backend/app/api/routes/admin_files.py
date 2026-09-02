from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import current_user
from app.config.settings import get_settings
from app.services.admin import file_manager, git_log, manual_edit_log, reindex, retag
from app.services.admin.git_utils import KNOWLEDGE_BASE_DIR
from app.services.ingestion.indexer import index_file

router = APIRouter(
    prefix="/admin/files", tags=["admin-files"], dependencies=[Depends(current_user)]
)


def _require_filename(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File has no name")
    return file.filename


@router.get("")
def list_files() -> list[str]:
    return file_manager.list_files()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Commits the file to git, then indexes it into the staging
    collection (never straight to prod) — CLAUDE.md's staging workflow:
    faculty review/query it there before explicitly promoting."""
    settings = get_settings()
    filename = _require_filename(file)
    try:
        commit = file_manager.upload_file(filename, file.file.read())
    except FileExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    chunk_count = index_file(
        KNOWLEDGE_BASE_DIR / filename, collection=settings.qdrant_collection_staging
    )
    return {
        "commit_hash": commit.commit_hash,
        "chunks_indexed": chunk_count,
        "collection": settings.qdrant_collection_staging,
    }


@router.put("/{filename}")
def replace(filename: str, file: UploadFile = File(...), confirm: bool = False) -> dict[str, Any]:
    """Replaces the file's content and re-indexes it into staging. If any
    of its current staged chunks were manually re-tagged, this requires
    `confirm=true` first (see `reindex.py`) — replacing changes the
    content entirely, so those corrections can't be preserved."""
    settings = get_settings()
    if filename != _require_filename(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file name doesn't match URL"
        )
    try:
        commit = file_manager.replace_file(filename, file.file.read())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        chunk_count = reindex.reindex_file(
            KNOWLEDGE_BASE_DIR / filename, settings.qdrant_collection_staging, confirm=confirm
        )
    except reindex.ReindexConfirmationRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=asdict(exc.warning)
        ) from exc

    return {
        "commit_hash": commit.commit_hash,
        "chunks_indexed": chunk_count,
        "collection": settings.qdrant_collection_staging,
    }


@router.delete("/{filename}")
def delete(filename: str) -> dict[str, Any]:
    """Deletes the file from git and clears its chunks from both staging
    and prod, so nothing stale is left behind on either side."""
    settings = get_settings()
    try:
        commit = file_manager.delete_file(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    retag.delete_chunks(settings.qdrant_collection_staging, filename)
    retag.delete_chunks(settings.qdrant_collection_prod, filename)

    return {"commit_hash": commit.commit_hash}


@router.get("/edit-log")
def edit_log() -> dict[str, Any]:
    """Merges git history on the knowledge base directory with the
    supplemental log of Qdrant-only changes (retags, promotions) —
    CLAUDE.md: derive primarily from git, supplement for what git can't
    see."""
    return {
        "file_changes": [asdict(entry) for entry in git_log.parse_git_log()],
        "manual_changes": [asdict(entry) for entry in manual_edit_log.read_entries()],
    }
