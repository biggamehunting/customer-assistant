"""Shared pytest fixtures for all test suites.

Design principles:
- No live OpenAI, S3, or Textract calls in unit/integration tests
- PostgreSQL integration tests use a real DB (TEST_DATABASE_URL)
- Unit tests use SQLite or mocks
- All secrets are dummies — nothing real is required to run the suite
"""
from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# ── Force test environment before any app imports ─────────────────────────────
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-not-real")
os.environ.setdefault("DATABASE_URL",
    os.getenv("TEST_DATABASE_URL",
              "postgresql+asyncpg://postgres:postgres@localhost:5432/customer_assistant_test"))
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("LOCAL_STORAGE_PATH", "/tmp/test_storage")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production-32chars!!")

from app.core.config import invalidate_settings
from app.core.security import create_access_token
from app.main import create_app


@pytest.fixture(autouse=True)
def reset_settings():
    """Ensure settings cache is cleared between tests."""
    invalidate_settings()
    yield
    invalidate_settings()


# ── App / HTTP clients ────────────────────────────────────────────────────────

@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def sync_client(app):
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ── Auth tokens ───────────────────────────────────────────────────────────────

@pytest.fixture
def tenant_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture
def admin_user_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000003")


@pytest.fixture
def customer_token(user_id, tenant_id) -> str:
    return create_access_token(
        subject=str(user_id),
        tenant_id=str(tenant_id),
        scopes=["customer"],
    )


@pytest.fixture
def admin_token(admin_user_id, tenant_id) -> str:
    return create_access_token(
        subject=str(admin_user_id),
        tenant_id=str(tenant_id),
        scopes=["admin"],
    )


@pytest.fixture
def auth_headers(customer_token) -> dict[str, str]:
    return {"Authorization": f"Bearer {customer_token}"}


@pytest.fixture
def admin_headers(admin_token) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}


# ── Mock storage ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.upload.return_value = "local://test/document.pdf"
    storage.download.return_value = b"%PDF-1.4 mock content"
    storage.exists.return_value = True
    storage.presigned_url.return_value = "http://example.com/test.pdf"
    storage.delete.return_value = None
    return storage


# ── Mock OpenAI ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai_embeddings():
    """Returns a fixed 1536-dim zero vector for all embedding requests."""
    with patch("openai.AsyncOpenAI") as mock_cls:
        instance = AsyncMock()
        mock_cls.return_value = instance
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.0] * 1536, index=0)]
        instance.embeddings.create = AsyncMock(return_value=embedding_response)
        yield instance


@pytest.fixture
def mock_openai_chat():
    """Returns a fixed chat completion for all chat requests."""
    with patch("openai.AsyncOpenAI") as mock_cls:
        instance = AsyncMock()
        mock_cls.return_value = instance
        choice = MagicMock()
        choice.message.content = "This is a test answer based on [1] the retrieved evidence."
        completion = MagicMock()
        completion.choices = [choice]
        completion.usage.prompt_tokens = 100
        completion.usage.completion_tokens = 50
        completion.usage.total_tokens = 150
        instance.chat.completions.create = AsyncMock(return_value=completion)
        yield instance


# ── Sample data ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Minimal valid PDF bytes for upload tests."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"
    )


@pytest.fixture
def sample_passages() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "content": "The refund policy allows returns within 30 days of purchase.",
            "document_title": "Refund Policy v2.pdf",
            "page_start": 1,
            "page_end": 1,
            "section_path": ["Policies", "Returns"],
            "fused_score": 0.85,
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "content": "Products must be returned in original packaging with receipt.",
            "document_title": "Refund Policy v2.pdf",
            "page_start": 2,
            "page_end": 2,
            "section_path": ["Policies", "Returns", "Requirements"],
            "fused_score": 0.72,
        },
    ]
