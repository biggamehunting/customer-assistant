"""OpenTelemetry distributed tracing setup.

Instruments FastAPI, SQLAlchemy, and provides helpers for manual spans.
Configures OTLP export when OTEL_ENDPOINT is set; otherwise uses a no-op
exporter so the application runs locally without a collector.

Trace attributes must NEVER include: API keys, secrets, raw document content,
customer PII, or full conversation text.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from app.core.config import Environment, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_tracer: trace.Tracer | None = None


def setup_telemetry() -> None:
    global _tracer
    settings = get_settings()

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.environment.value,
        }
    )

    # In production sample 10 % of traces; in dev/test record everything.
    if settings.is_production:
        sampler = ParentBased(TraceIdRatioBased(0.1))
    else:
        sampler = ParentBased(TraceIdRatioBased(1.0))

    provider = TracerProvider(resource=resource, sampler=sampler)

    if settings.otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("telemetry.otlp_configured", endpoint=settings.otel_endpoint)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telemetry.otlp_setup_failed", error=str(exc))
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif settings.environment == Environment.development:
        # Only print traces in dev when explicitly debugging
        pass
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.app_name, settings.app_version)

    SQLAlchemyInstrumentor().instrument()
    logger.info("telemetry.configured", environment=settings.environment.value)


def instrument_app(app: object) -> None:
    """Attach FastAPI auto-instrumentation after setup_telemetry() is called."""
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        return trace.get_tracer("customer-assistant")
    return _tracer


@contextmanager
def span(name: str, **attributes: str | int | float | bool) -> Generator[trace.Span, None, None]:
    """Context manager for a manual child span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            s.set_attribute(key, value)
        yield s
