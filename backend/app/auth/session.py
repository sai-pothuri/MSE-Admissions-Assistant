from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SESSION_COOKIE_NAME = "mse_admin_session"
SESSION_SALT = "mse-admin-session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 12  # 12 hours

# Signed, stateless session tokens (itsdangerous — the same primitive
# Flask's session cookies use) rather than hand-rolled HMAC: this is
# exactly the kind of code where reaching for a small, well-vetted,
# purpose-built library beats rolling it, even at this project's scale.
# No server-side session store is needed — the signature plus embedded
# timestamp is enough to validate and expire a token.


def _serializer(secret_key: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key, salt=SESSION_SALT)


def sign_session(secret_key: str) -> str:
    token: str = _serializer(secret_key).dumps({"role": "admin"})
    return token


def verify_session(token: str, secret_key: str) -> bool:
    try:
        _serializer(secret_key).loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return False
    return True
