"""Unit tests for core config, security, and exceptions."""
from __future__ import annotations

import pytest

from app.core.config import Environment, get_settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DocumentNotFoundError,
    FileTooLargeError,
)
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestConfig:
    def test_environment_is_test(self):
        settings = get_settings()
        assert settings.environment == Environment.test

    def test_embedding_dimensions_default(self):
        settings = get_settings()
        assert settings.openai_embedding_dimensions == 1536

    def test_invalid_embedding_dimensions_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "999")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        from app.core.config import invalidate_settings
        invalidate_settings()
        with pytest.raises(Exception):
            get_settings()
        invalidate_settings()

    def test_max_upload_size_bytes(self):
        settings = get_settings()
        assert settings.max_upload_size_bytes == settings.max_upload_size_mb * 1024 * 1024


class TestSecurity:
    def test_password_hash_and_verify(self):
        password = "SecurePass123!"   # short enough for all bcrypt backends
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("WrongPass!", hashed)

    def test_create_and_decode_access_token(self):
        import uuid
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        token = create_access_token(user_id, tenant_id, scopes=["customer"])
        payload = decode_access_token(token)
        assert payload["sub"] == user_id
        assert payload["tenant_id"] == tenant_id
        assert "customer" in payload["scopes"]
        assert payload["type"] == "access"

    def test_expired_token_raises(self, monkeypatch):
        from datetime import timedelta
        import uuid
        from jose import jwt

        settings = get_settings()
        payload = {
            "sub": str(uuid.uuid4()),
            "tenant_id": str(uuid.uuid4()),
            "scopes": [],
            "type": "access",
            # already expired
            "exp": 0,
            "iat": 0,
        }
        token = jwt.encode(
            payload,
            settings.secret_key.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )
        from app.core.exceptions import TokenExpiredError
        with pytest.raises(TokenExpiredError):
            decode_access_token(token)

    def test_wrong_token_type_raises(self):
        import uuid
        from app.core.security import create_refresh_token

        token = create_refresh_token(str(uuid.uuid4()), str(uuid.uuid4()))
        with pytest.raises(AuthenticationError):
            decode_access_token(token)


class TestExceptions:
    def test_exception_hierarchy(self):
        from app.core.exceptions import AppError
        exc = DocumentNotFoundError("not found")
        assert isinstance(exc, AppError)
        assert exc.status_code == 404
        assert exc.error_code == "document_not_found"

    def test_file_too_large_error(self):
        exc = FileTooLargeError("too big", detail="100 MB")
        assert exc.status_code == 422
        assert exc.detail == "100 MB"
