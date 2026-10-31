"""Ports src/tracing/providers/types.ts.

Configs, options and results are camelCase dicts:
- TraceProviderConfig: {id, endpoint?, projectId?, auth?: {token?, username?, password?},
  headers?, timeout?}
- FetchTraceOptions: {earliestStartTime?, maxSpans?, abortSignal?}
- FetchTraceResult: {spans: [SpanData], fetchedAt}
"""

from __future__ import annotations

from typing import Any, Protocol


class TraceProviderError(Exception):
    def __init__(
        self, message: str, status_code: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.name = "TraceProviderError"
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class TraceProvider(Protocol):
    """An external trace backend that returns the spans for a W3C trace ID."""

    id: str

    async def fetch_trace(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any] | None: ...
