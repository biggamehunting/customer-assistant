"""AWS S3 storage backend (production).

Uses boto3 with IAM role-based credentials (no hardcoded keys).
Presigned URLs use short-lived credentials tied to the caller's IAM identity.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.core.telemetry import span
from app.domain.interfaces import StorageBackend

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=10)


class S3StorageBackend(StorageBackend):
    """S3-backed storage using the documents and artifacts buckets from config."""

    def __init__(self) -> None:
        settings = get_settings()
        session = boto3.Session(region_name=settings.aws_region)
        self._s3 = session.client("s3")
        self._documents_bucket = settings.s3_documents_bucket
        self._artifacts_bucket = settings.s3_artifacts_bucket

    def _bucket_and_key(self, key: str) -> tuple[str, str]:
        """Key format: 'documents/<path>' or 'artifacts/<path>'."""
        if key.startswith("documents/"):
            return self._documents_bucket, key[len("documents/"):]
        if key.startswith("artifacts/"):
            return self._artifacts_bucket, key[len("artifacts/"):]
        return self._documents_bucket, key

    async def _run(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, partial(func, *args, **kwargs))

    async def upload(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        bucket, s3_key = self._bucket_and_key(key)
        with span("s3.upload", bucket=bucket, key=s3_key, size=len(data)):
            try:
                await self._run(
                    self._s3.put_object,
                    Bucket=bucket,
                    Key=s3_key,
                    Body=data,
                    ContentType=content_type,
                    ServerSideEncryption="AES256",
                )
                logger.info("s3.uploaded", bucket=bucket, key=s3_key, size_bytes=len(data))
                return f"s3://{bucket}/{s3_key}"
            except ClientError as e:
                raise StorageError(f"S3 upload failed: {e}") from e

    async def download(self, key: str) -> bytes:
        bucket, s3_key = self._bucket_and_key(key)
        with span("s3.download", bucket=bucket, key=s3_key):
            try:
                response = await self._run(
                    self._s3.get_object, Bucket=bucket, Key=s3_key
                )
                return response["Body"].read()
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise StorageError(f"Object not found: {key}") from e
                raise StorageError(f"S3 download failed: {e}") from e

    async def delete(self, key: str) -> None:
        bucket, s3_key = self._bucket_and_key(key)
        try:
            await self._run(self._s3.delete_object, Bucket=bucket, Key=s3_key)
            logger.info("s3.deleted", bucket=bucket, key=s3_key)
        except ClientError as e:
            raise StorageError(f"S3 delete failed: {e}") from e

    async def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        bucket, s3_key = self._bucket_and_key(key)
        try:
            url: str = await self._run(
                self._s3.generate_presigned_url,
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expires_seconds,
            )
            return url
        except ClientError as e:
            raise StorageError(f"S3 presigned URL failed: {e}") from e

    async def exists(self, key: str) -> bool:
        bucket, s3_key = self._bucket_and_key(key)
        try:
            await self._run(self._s3.head_object, Bucket=bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise StorageError(f"S3 head_object failed: {e}") from e


def get_storage_backend() -> StorageBackend:
    """Factory: returns the correct backend based on config."""
    from app.core.config import get_settings
    from app.storage.local_storage import LocalStorageBackend

    settings = get_settings()
    if settings.storage_backend == "s3":
        return S3StorageBackend()
    return LocalStorageBackend()
