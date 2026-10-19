"""Security tests — auth boundaries, cross-tenant isolation, injection."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.security import create_access_token, decode_access_token
from app.core.exceptions import AuthenticationError, TokenExpiredError


class TestJWTSecurity:
    def test_tampered_token_is_rejected(self):
        token = create_access_token(str(uuid.uuid4()), str(uuid.uuid4()), scopes=["customer"])
        # Flip a character in the signature
        tampered = token[:-4] + "xxxx"
        with pytest.raises(AuthenticationError):
            decode_access_token(tampered)

    def test_empty_token_is_rejected(self):
        with pytest.raises(AuthenticationError):
            decode_access_token("")

    def test_malformed_token_is_rejected(self):
        with pytest.raises(AuthenticationError):
            decode_access_token("not.a.jwt")

    def test_token_without_bearer_prefix_rejected(self, sync_client):
        resp = sync_client.get(
            "/api/v1/documents",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401

    def test_no_auth_header_rejected(self, sync_client):
        resp = sync_client.get("/api/v1/documents")
        assert resp.status_code == 401


class TestCrossTenantIsolation:
    def test_token_for_tenant_a_cannot_claim_tenant_b_scope(self):
        """Tokens carry a fixed tenant_id; they cannot be used for another tenant."""
        tenant_a = str(uuid.uuid4())
        tenant_b = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        token = create_access_token(user_id, tenant_a, scopes=["customer"])
        payload = decode_access_token(token)
        assert payload["tenant_id"] == tenant_a
        assert payload["tenant_id"] != tenant_b

    def test_admin_scope_not_granted_to_customer_token(self):
        token = create_access_token(str(uuid.uuid4()), str(uuid.uuid4()), scopes=["customer"])
        payload = decode_access_token(token)
        assert "admin" not in payload.get("scopes", [])
        assert "super_admin" not in payload.get("scopes", [])


class TestInputValidation:
    def test_chat_message_too_long_rejected(self, sync_client, auth_headers):
        with patch("app.storage.database.get_session") as mock_sess:
            db = AsyncMock()
            db.commit = AsyncMock()
            db.rollback = AsyncMock()

            async def fake_session():
                yield db

            mock_sess.return_value = fake_session()
            resp = sync_client.post(
                "/api/v1/chat",
                headers=auth_headers,
                json={"message": "x" * 4001},
            )
        # FastAPI/pydantic validation should reject this
        assert resp.status_code == 422

    def test_chat_empty_message_rejected(self, sync_client, auth_headers):
        with patch("app.storage.database.get_session") as mock_sess:
            db = AsyncMock()
            db.commit = AsyncMock()
            db.rollback = AsyncMock()

            async def fake_session():
                yield db

            mock_sess.return_value = fake_session()
            resp = sync_client.post(
                "/api/v1/chat",
                headers=auth_headers,
                json={"message": ""},
            )
        assert resp.status_code == 422

    def test_uuid_path_param_validates(self, sync_client, auth_headers):
        resp = sync_client.get("/api/v1/documents/not-a-uuid", headers=auth_headers)
        assert resp.status_code == 422
