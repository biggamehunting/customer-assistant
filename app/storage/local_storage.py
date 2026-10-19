"""Local filesystem storage backend (development only).

Implements the StorageBackend interface using the local filesystem.
In production, replace with S3StorageBackend.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import aiofiles
import aiofiles.os

from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.domain.interfaces import StorageBackend

logger = get_logger(__name__)


class LocalStorageBackend(StorageBackend):
    """Stores files in a local directory tree.

    Key format: <bucket>/<path>  →  <base_dir>/<bucket>/<path>
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self._base = Path(base_dir or settings.local_storage_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Prevent path traversal
        resolved = (self._base / key).resolve()
        if not str(resolved).startswith(str(self._base.resolve())):
            raise StorageError(f"Path traversal detected: {key}")
        return resolved

    async def upload(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        logger.debug("storage.uploaded", key=key, size=len(data))
        return f"local://{key}"

    async def download(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"Object not found: {key}")
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                await aiofiles.os.remove(path)
        logger.debug("storage.deleted", key=key)

    async def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        # In local mode return a placeholder — the real URL is served by the API
        return f"/api/v1/documents/download/{key}"

    async def exists(self, key: str) -> bool:
        return self._resolve(key).exists()
