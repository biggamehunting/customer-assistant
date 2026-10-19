"""FastAPI dependency injection providers.

All route-level dependencies go here so they can be overridden in tests.
"""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import decode_access_token
from app.storage.database import AsyncSession, get_session
from app.storage.s3 import get_storage_backend
from app.domain.interfaces import StorageBackend


_bearer = HTTPBearer(auto_error=False)


class CurrentUser:
    """Extracted and validated JWT claims for the current request."""

    def __init__(
        self,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID,
        email: str,
        scopes: list[str],
    ) -> None:
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.email = email
        self.scopes = scopes

    def require_scope(self, scope: str) -> None:
        if scope not in self.scopes:
            raise AuthorizationError(f"Scope '{scope}' required")

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes or "super_admin" in self.scopes


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> CurrentUser:
    """Validate Bearer token and return the current user context."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_access_token(credentials.credentials)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return CurrentUser(
        user_id=uuid.UUID(payload["sub"]),
        tenant_id=uuid.UUID(payload["tenant_id"]),
        email=payload.get("email", ""),
        scopes=payload.get("scopes", []),
    )


async def get_admin_user(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> CurrentUser:
    """Require admin scope."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


# Type aliases for use in route signatures
DbSession = Annotated[AsyncSession, Depends(get_session)]
AuthUser = Annotated[CurrentUser, Depends(get_current_user)]
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]
Storage = Annotated[StorageBackend, Depends(get_storage_backend)]
