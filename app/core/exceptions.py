"""Domain exception hierarchy.

All custom exceptions inherit from AppError, which carries an HTTP status code
and a machine-readable error_code.  FastAPI exception handlers translate these
to structured JSON responses without leaking internal stack traces.
"""
from __future__ import annotations

from http import HTTPStatus


class AppError(Exception):
    """Base for all application errors."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ── Authentication / Authorisation ────────────────────────────────────────────

class AuthenticationError(AppError):
    status_code = HTTPStatus.UNAUTHORIZED
    error_code = "authentication_failed"


class AuthorizationError(AppError):
    status_code = HTTPStatus.FORBIDDEN
    error_code = "authorization_failed"


class TokenExpiredError(AuthenticationError):
    error_code = "token_expired"


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationError(AppError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "validation_error"


class FileTooLargeError(ValidationError):
    error_code = "file_too_large"


class InvalidFileTypeError(ValidationError):
    error_code = "invalid_file_type"


class InvalidDocumentError(ValidationError):
    error_code = "invalid_document"


# ── Not Found ─────────────────────────────────────────────────────────────────

class NotFoundError(AppError):
    status_code = HTTPStatus.NOT_FOUND
    error_code = "not_found"


class DocumentNotFoundError(NotFoundError):
    error_code = "document_not_found"


class ConversationNotFoundError(NotFoundError):
    error_code = "conversation_not_found"


class ChartNotFoundError(NotFoundError):
    error_code = "chart_not_found"


# ── Conflict ──────────────────────────────────────────────────────────────────

class ConflictError(AppError):
    status_code = HTTPStatus.CONFLICT
    error_code = "conflict"


class DuplicateDocumentError(ConflictError):
    error_code = "duplicate_document"


# ── Rate Limiting ─────────────────────────────────────────────────────────────

class RateLimitError(AppError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "rate_limit_exceeded"


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionError(AppError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "ingestion_error"


class OCRError(IngestionError):
    error_code = "ocr_error"


class PDFParseError(IngestionError):
    error_code = "pdf_parse_error"


class EmbeddingError(IngestionError):
    error_code = "embedding_error"


class StorageError(IngestionError):
    error_code = "storage_error"


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievalError(AppError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "retrieval_error"


class CitationValidationError(RetrievalError):
    error_code = "citation_validation_failed"


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentError(AppError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "agent_error"


class AgentTimeoutError(AgentError):
    status_code = HTTPStatus.GATEWAY_TIMEOUT
    error_code = "agent_timeout"


class AgentMaxStepsError(AgentError):
    error_code = "agent_max_steps_exceeded"


class PromptInjectionSuspectedError(AgentError):
    status_code = HTTPStatus.BAD_REQUEST
    error_code = "prompt_injection_suspected"


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsError(AppError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "analytics_error"


class ChartGenerationError(AnalyticsError):
    error_code = "chart_generation_failed"


class CalculationError(AnalyticsError):
    error_code = "calculation_error"


# ── External services ─────────────────────────────────────────────────────────

class ExternalServiceError(AppError):
    status_code = HTTPStatus.BAD_GATEWAY
    error_code = "external_service_error"


class LLMProviderError(ExternalServiceError):
    error_code = "llm_provider_error"


class LLMRateLimitError(LLMProviderError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "llm_rate_limit"
