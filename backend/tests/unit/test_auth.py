from app.auth.password_auth import PasswordAuthProvider
from app.auth.session import sign_session, verify_session


def test_authenticate_accepts_the_configured_password():
    provider = PasswordAuthProvider(password="correct-horse", session_secret="secret")

    assert provider.authenticate("correct-horse") is True


def test_authenticate_rejects_a_wrong_password():
    provider = PasswordAuthProvider(password="correct-horse", session_secret="secret")

    assert provider.authenticate("wrong") is False


def test_create_session_then_validate_session_round_trips():
    provider = PasswordAuthProvider(password="correct-horse", session_secret="secret")

    token = provider.create_session()

    assert provider.validate_session(token) is True


def test_validate_session_rejects_a_garbage_token():
    provider = PasswordAuthProvider(password="correct-horse", session_secret="secret")

    assert provider.validate_session("not-a-real-token") is False


def test_validate_session_rejects_a_token_signed_with_a_different_secret():
    issuer = PasswordAuthProvider(password="correct-horse", session_secret="secret-a")
    verifier = PasswordAuthProvider(password="correct-horse", session_secret="secret-b")

    token = issuer.create_session()

    assert verifier.validate_session(token) is False


def test_verify_session_rejects_an_expired_token(monkeypatch):
    import app.auth.session as session_module

    monkeypatch.setattr(session_module, "SESSION_MAX_AGE_SECONDS", -1)
    token = sign_session("secret")

    assert verify_session(token, "secret") is False
