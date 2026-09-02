import secrets
from functools import lru_cache

from app.auth.base import AuthProvider
from app.auth.session import sign_session, verify_session
from app.config.settings import get_settings


class PasswordAuthProvider(AuthProvider):
    """Single shared faculty password (CLAUDE.md: "simple password auth
    for now"). Session tokens are signed/verified via `app.auth.session`,
    scoped to this instance's secret key."""

    def __init__(self, password: str, session_secret: str) -> None:
        self._password = password
        self._session_secret = session_secret

    def authenticate(self, password: str) -> bool:
        # Constant-time comparison — the password is short-lived shared
        # faculty knowledge, not a per-user hashed credential, but a
        # timing side-channel is a free thing to not have.
        return secrets.compare_digest(password, self._password)

    def create_session(self) -> str:
        return sign_session(self._session_secret)

    def validate_session(self, token: str) -> bool:
        return verify_session(token, self._session_secret)


@lru_cache
def get_auth_provider() -> AuthProvider:
    """The active `AuthProvider` — `app/api/deps.py` and the admin auth
    routes depend on this factory, not on `PasswordAuthProvider` directly.
    Swapping to CMU SSO later means adding a new provider module and
    changing this one function's return value."""
    settings = get_settings()
    return PasswordAuthProvider(settings.admin_password, settings.admin_session_secret)
