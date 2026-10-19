"""Centralised, validated application configuration via Pydantic Settings.

All settings are read from environment variables (or a .env file).
The application fails fast on startup if a required value is absent or invalid.
Secrets are never printed in validation errors.
"""
from __future__ import annotations

import secrets
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"
    test = "test"


class OCRProvider(str, Enum):
    textract = "textract"
    tesseract = "tesseract"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    environment: Environment = Environment.development
    app_name: str = "customer-assistant"
    app_version: str = "0.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_hex(32)))

    # ── Database ───────────────────────────────────────────────────────────
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/customer_assistant"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_query_timeout: int = 30  # seconds

    # ── Redis / Celery ─────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    celery_concurrency: int = 4
    celery_max_tasks_per_child: int = 100

    # ── OpenAI ─────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openai_chat_model: str = "gpt-4o"
    openai_router_model: str = "gpt-4o-mini"  # cheaper model for classification
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_max_retries: int = 3
    openai_timeout: float = 60.0

    # ── OCR ────────────────────────────────────────────────────────────────
    ocr_provider: OCRProvider = OCRProvider.tesseract
    textract_region: str = "us-east-1"
    tesseract_cmd: str = "tesseract"
    ocr_confidence_threshold: float = 60.0  # Textract confidence % below which page is flagged

    # ── AWS ────────────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    s3_documents_bucket: str = "customer-assistant-documents"
    s3_artifacts_bucket: str = "customer-assistant-artifacts"
    sqs_ingestion_queue_url: str = ""
    sqs_dlq_url: str = ""

    # ── Auth ───────────────────────────────────────────────────────────────
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 7

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 20          # candidates before reranking
    retrieval_rerank_top_k: int = 6    # passages passed to LLM context
    retrieval_score_threshold: float = 0.0
    retrieval_lexical_weight: float = 0.4
    retrieval_dense_weight: float = 0.6

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100
    table_chunk_max_rows: int = 50

    # ── LangGraph agent ────────────────────────────────────────────────────
    agent_max_steps: int = 20
    agent_max_retries: int = 3
    agent_timeout_seconds: float = 120.0

    # ── Rate limiting ──────────────────────────────────────────────────────
    rate_limit_chat_per_minute: int = 30
    rate_limit_upload_per_hour: int = 200

    # ── Upload limits ──────────────────────────────────────────────────────
    max_upload_size_mb: int = 100
    max_pages_per_document: int = 2000

    # ── Token budgets ──────────────────────────────────────────────────────
    max_context_tokens: int = 100_000
    max_completion_tokens: int = 4096

    # ── Observability ──────────────────────────────────────────────────────
    log_level: str = "INFO"
    otel_endpoint: str = ""
    otel_service_name: str = "customer-assistant"
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "customer-assistant"
    langsmith_tracing: bool = False

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: list[AnyHttpUrl] = Field(default=["http://localhost:3000", "http://localhost:8501"])

    # ── Storage mode ──────────────────────────────────────────────────────
    # "local" uses the filesystem; "s3" uses AWS S3
    storage_backend: str = "local"
    local_storage_path: str = "./data/documents"
    local_artifacts_path: str = "./data/artifacts"

    @field_validator("openai_embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, v: int, info: Any) -> int:
        model = info.data.get("openai_embedding_model", "")
        valid = {
            "text-embedding-3-small": [512, 1536],
            "text-embedding-3-large": [256, 1024, 3072],
            "text-embedding-ada-002": [1536],
        }
        if model in valid and v not in valid[model]:
            raise ValueError(
                f"Embedding dimensions {v} not valid for model {model}. "
                f"Valid: {valid[model]}"
            )
        return v

    @model_validator(mode="after")
    def validate_production_requirements(self) -> Settings:
        if self.environment == Environment.production:
            if not self.openai_api_key.get_secret_value():
                raise ValueError("OPENAI_API_KEY is required in production")
            if self.storage_backend != "s3":
                raise ValueError("storage_backend must be 's3' in production")
            if not self.sqs_ingestion_queue_url:
                raise ValueError("SQS_INGESTION_QUEUE_URL is required in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.production

    @property
    def is_test(self) -> bool:
        return self.environment == Environment.test

    @property
    def sync_database_url(self) -> str:
        """Synchronous URL for Alembic migrations."""
        return str(self.database_url).replace("+asyncpg", "+psycopg2")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.  Call invalidate_settings() in tests."""
    return Settings()


def invalidate_settings() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()
