from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import current_user
from app.config.settings import get_settings
from app.services.admin import chunk_preview, retag
from app.services.admin.git_utils import (
    KNOWLEDGE_BASE_DIR,
    UnsafeFilenameError,
    resolve_safe_pdf_path,
)

router = APIRouter(
    prefix="/admin/chunks", tags=["admin-chunks"], dependencies=[Depends(current_user)]
)

CollectionName = Literal["staging", "prod"]


def _resolve_collection(name: CollectionName) -> str:
    settings = get_settings()
    return (
        settings.qdrant_collection_staging if name == "staging" else settings.qdrant_collection_prod
    )


@router.get("/preview")
def preview(filename: str) -> list[dict[str, Any]]:
    try:
        path = resolve_safe_pdf_path(KNOWLEDGE_BASE_DIR, filename)
    except UnsafeFilenameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{filename}' not found")
    return [asdict(chunk) for chunk in chunk_preview.preview_file(path)]


@router.get("")
def list_chunks(filename: str, collection: CollectionName = "staging") -> list[dict[str, Any]]:
    chunks = retag.list_chunks(_resolve_collection(collection), filename)
    return [asdict(chunk) for chunk in chunks]


@router.patch("/{point_id}")
def retag_chunk(
    point_id: str, category: str, collection: CollectionName = "staging"
) -> dict[str, str]:
    try:
        retag.retag_chunk(_resolve_collection(collection), point_id, category)
    except retag.InvalidCategoryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"status": "ok"}
