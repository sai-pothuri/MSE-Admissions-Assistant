from fastapi import Cookie, HTTPException, status

from app.auth.password_auth import get_auth_provider
from app.auth.session import SESSION_COOKIE_NAME


def current_user(
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> None:
    """FastAPI dependency every admin route requires — wraps the active
    `AuthProvider` (see `app.auth.base`) rather than any concrete
    implementation, so swapping to CMU SSO later only means changing what
    `get_auth_provider` returns."""
    auth_provider = get_auth_provider()
    if session is None or not auth_provider.validate_session(session):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
