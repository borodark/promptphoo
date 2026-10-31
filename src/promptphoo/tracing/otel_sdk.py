"""Set up, flush and shut down tracing of promptfoo's own spans. Ports src/tracing/otelSdk.ts.

`initialize_otel` registers a tracer provider (the in-house SDK in otel_trace_sdk.py, see D56)
with the `service.name` / `service.version` resource, a BatchSpanProcessor writing to the
local TraceStore, and an optional one exporting OTLP JSON to `config.endpoint`.

Differences from Node:
- No SIGTERM/SIGINT/beforeExit handlers. The evaluator flushes and shuts down in its
  `finally`, which also runs on KeyboardInterrupt and cancellation.
- Shutting down unregisters the provider, so a later `initialize_otel` in the same process
  installs a fresh one. In Node the first provider stays registered globally.
- `debug: true` sets this package's tracing loggers to DEBUG instead of OTel's diag logger.
"""

from __future__ import annotations

import logging

from promptphoo import UPSTREAM_VERSION
from promptphoo.tracing.otel_api import propagation, trace
from promptphoo.tracing.otel_config import OtelConfig
from promptphoo.tracing.otel_trace_sdk import (
    BasicTracerProvider,
    BatchSpanProcessor,
    SpanProcessor,
    resource_from_attributes,
)

logger = logging.getLogger("promptphoo.tracing")

ATTR_SERVICE_NAME = "service.name"
ATTR_SERVICE_VERSION = "service.version"

_provider: BasicTracerProvider | None = None
_initialized = False


def initialize_otel(config: OtelConfig) -> None:
    """Register the tracer provider for promptfoo's spans; a no-op when already initialized or
    disabled."""
    global _provider, _initialized
    if _initialized:
        logger.debug("[OtelSdk] Already initialized, skipping")
        return
    if not config.get("enabled"):
        logger.debug("[OtelSdk] OTEL tracing is disabled")
        return

    logger.debug(
        "[OtelSdk] Initializing OpenTelemetry SDK serviceName=%s endpoint=%s localExport=%s",
        config.get("serviceName"),
        config.get("endpoint"),
        config.get("localExport"),
    )
    if config.get("debug"):
        logging.getLogger("promptphoo.tracing").setLevel(logging.DEBUG)

    resource = resource_from_attributes(
        {
            ATTR_SERVICE_NAME: config.get("serviceName"),
            ATTR_SERVICE_VERSION: UPSTREAM_VERSION,
        }
    )

    span_processors: list[SpanProcessor] = []
    if config.get("localExport"):
        from promptphoo.tracing.local_span_exporter import LocalSpanExporter

        span_processors.append(BatchSpanProcessor(LocalSpanExporter()))
        logger.debug("[OtelSdk] Added local span exporter")

    endpoint = config.get("endpoint")
    if endpoint:
        from promptphoo.tracing.otlp_http_exporter import OTLPTraceExporter

        span_processors.append(BatchSpanProcessor(OTLPTraceExporter(endpoint)))
        logger.debug("[OtelSdk] Added OTLP exporter to %s", endpoint)

    _provider = BasicTracerProvider(resource=resource, span_processors=span_processors)
    # Registers the provider globally along with the W3C Trace Context propagator.
    _provider.register()
    _initialized = True
    logger.info("[OtelSdk] OpenTelemetry SDK initialized successfully")


async def shutdown_otel() -> None:
    """Flush pending spans and release the provider."""
    global _provider, _initialized
    if not _initialized or _provider is None:
        return
    logger.debug("[OtelSdk] Shutting down OpenTelemetry SDK")
    provider = _provider
    try:
        await provider.shutdown()
        logger.info("[OtelSdk] OpenTelemetry SDK shut down successfully")
    except Exception as error:
        logger.error("[OtelSdk] Error shutting down OpenTelemetry SDK: %s", error)
    finally:
        _provider = None
        _initialized = False
        if trace.get_tracer_provider() is provider:
            trace.disable()
            propagation.disable()


async def flush_otel() -> None:
    """Force-export any pending spans."""
    if not _initialized or _provider is None:
        return
    logger.debug("[OtelSdk] Flushing pending spans")
    try:
        await _provider.force_flush()
        logger.debug("[OtelSdk] Spans flushed successfully")
    except Exception as error:
        logger.error("[OtelSdk] Error flushing spans: %s", error)


def is_otel_initialized() -> bool:
    return _initialized
