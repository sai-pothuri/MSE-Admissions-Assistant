from abc import ABC, abstractmethod


class AuthProvider(ABC):
    """Every admin route depends on this interface via FastAPI dependency
    injection (see `app/api/deps.py`), never on a concrete implementation
    directly — swapping password auth for CMU SSO later means adding a new
    implementation and changing which one `password_auth.get_auth_provider`
    (or its successor) returns, not touching route handlers."""

    @abstractmethod
    def authenticate(self, password: str) -> bool:
        """True if `password` is the current admin credential."""

    @abstractmethod
    def create_session(self) -> str:
        """Returns an opaque, signed session token to store in a cookie."""

    @abstractmethod
    def validate_session(self, token: str) -> bool:
        """True if `token` is a currently-valid session issued by
        `create_session`."""
