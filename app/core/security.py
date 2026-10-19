"""JWT authentication and password hashing utilities."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, TokenExpiredError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Password hashing ─────────────────────────────────────────────────────────
# Try passlib first (preferred); fall back to bcrypt directly if passlib is
# incompatible with the installed bcrypt version (e.g. bcrypt >= 4.1 with
# passlib 1.7.4 which lacks the __about__ attribute check).

def _make_pwd_context():
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        # Quick smoke-test: passlib may load but fail on first use
        ctx.hash("probe")
        return ctx
    except Exception:
        return None


_pwd_context = _make_pwd_context()


def hash_password(plain: str) -> str:
    if _pwd_context is not None:
        return _pwd_context.hash(plain)
    import bcrypt  # direct fallback
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    if _pwd_context is not None:
        return _pwd_context.verify(plain, hashed)
    import bcrypt  # direct fallback
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(
    subject: str,
    tenant_id: str,
    scopes: list[str] | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "tenant_id": tenant_id,
        "scopes": scopes or [],
        "iat": now,
        "exp": expire,
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(subject: str, tenant_id: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.jwt_refresh_token_expire_days)
    payload: dict[str, Any] = {
        "sub": subject,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Access token has expired")
    except JWTError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc


def decode_access_token(token: str) -> dict[str, Any]:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise AuthenticationError("Token is not an access token")
    return payload
