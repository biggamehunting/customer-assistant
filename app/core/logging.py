"""Structured JSON logging using structlog.

Fields included on every log line:
  timestamp, level, service, environment, event
Optional context fields are injected via bind_contextvars():
  request_id, trace_id, conversation_id, document_id, tenant_id, job_id

NEVER log: API keys, access tokens, full document text, raw customer messages.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.config import Environment, get_settings


def _add_service_context(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    settings = get_settings()
    event_dict["service"] = settings.app_name
    event_dict["environment"] = settings.environment.value
    event_dict["version"] = settings.app_version
    return event_dict


def _drop_color_message_key(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Remove uvicorn's color_message to avoid duplicates."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_service_context,
        _drop_color_message_key,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.environment in (Environment.development, Environment.test):
        # Pretty console output in dev / test
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Machine-readable JSON in staging / production
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[return-value]
