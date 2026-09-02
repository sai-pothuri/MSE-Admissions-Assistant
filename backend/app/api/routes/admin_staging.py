from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import current_user
from app.services.admin import staging

router = APIRouter(
    prefix="/admin/staging", tags=["admin-staging"], dependencies=[Depends(current_user)]
)


@router.post("/promote/{filename}")
def promote(filename: str) -> dict[str, Any]:
    result = staging.promote_file(filename)
    if result.promoted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No staged chunks found for '{filename}'",
        )
    return asdict(result)
