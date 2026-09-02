from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel

from app.auth.password_auth import get_auth_provider
from app.auth.session import SESSION_COOKIE_NAME, SESSION_MAX_AGE_SECONDS

router = APIRouter(prefix="/admin", tags=["admin-auth"])


class LoginRequest(BaseModel):
    password: str


@router.post("/login")
def login(request: LoginRequest, response: Response) -> dict[str, str]:
    auth_provider = get_auth_provider()
    if not auth_provider.authenticate(request.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")

    token = auth_provider.create_session()
    # `secure=False` for now — this runs over plain HTTP in local dev and
    # until Phase 6/7 puts a real deployment behind HTTPS. Flip to True
    # once that's in place (tracked as a Phase 7 hardening item).
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="strict",
        secure=False,
    )
    return {"status": "ok"}


@router.post("/logout")
def logout(response: Response) -> dict[str, str]:
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return {"status": "ok"}
