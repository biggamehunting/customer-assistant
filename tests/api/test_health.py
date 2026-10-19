"""API tests — health endpoints."""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest


class TestHealthEndpoints:
    def test_liveness_returns_200(self, sync_client):
        resp = sync_client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_readiness_returns_200_when_db_up(self, sync_client):
        # Mock the DB check so tests don't need a real PostgreSQL
        with patch("app.api.routes.health.get_db_context") as mock_ctx:
            session = AsyncMock()
            session.execute = AsyncMock()
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = sync_client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert any(d["name"] == "postgresql" for d in data["dependencies"])

    def test_health_summary_alias(self, sync_client):
        with patch("app.api.routes.health.get_db_context") as mock_ctx:
            session = AsyncMock()
            session.execute = AsyncMock()
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = sync_client.get("/health")
        assert resp.status_code == 200
