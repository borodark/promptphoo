"""Ports src/tracing/providers/braintrust.ts: Braintrust project traces correlated with an
OpenTelemetry trace ID, queried through BTQL."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from promptphoo.fetch.header_parser import date_parse
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, js_string, json_stringify
from promptphoo.tracing.providers.fetch import (
    MAX_TRACE_RESPONSE_BYTES,
    fetch_trace_response,
    read_limited_response,
    release_response,
    validate_trace_provider_endpoint,
)
from promptphoo.tracing.providers.tempo import (
    content_length_exceeds,
    is_positive_safe_integer,
    is_trace_id,
    now_ms,
)
from promptphoo.tracing.providers.types import TraceProviderError
from promptphoo.tracing.tool_attributes import get_normalized_tool_attributes

logger = logging.getLogger(__name__)

MAX_SPANS = 10_000
PROJECT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z", re.IGNORECASE
)


def _js_round(value: float) -> int:
    """Math.round: halves round up."""
    return math.floor(value + 0.5)


def _timestamp_ms(value: Any) -> int | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
        return _js_round(value * 1000) if value < 10_000_000_000 else _js_round(value)
    if isinstance(value, str):
        parsed = date_parse(value)
        return None if parsed is None else _js_round(parsed)
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _transform_span(row: Any, options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    span_id = row.get("span_id") if is_truthy(row.get("span_id")) else row.get("id")
    metrics = row.get("metrics")
    metrics_dict = _dict(metrics)
    start_time = _timestamp_ms(metrics_dict.get("start"))
    if start_time is None:
        start_time = _timestamp_ms(row.get("created"))
    if not is_truthy(span_id) or start_time is None:
        return None
    earliest = (options or {}).get("earliestStartTime")
    if earliest is not None and start_time < earliest:
        return None

    parents = row.get("span_parents")
    parent_span_id = None
    if isinstance(parents, list):
        parent_span_id = next((p for p in reversed(parents) if is_truthy(p) and p != span_id), None)
    span_attributes = _dict(row.get("span_attributes"))
    name = (
        span_attributes["name"]
        if isinstance(span_attributes.get("name"), str)
        else ("braintrust.span")
    )
    is_tool_span = span_attributes.get("type") == "tool"
    attributes: dict[str, Any] = {**_dict(row.get("metadata")), **span_attributes}
    if is_truthy(metrics):
        attributes["braintrust.metrics"] = metrics
    if "input" in row:
        attributes["braintrust.input"] = row["input"]
    if "output" in row:
        attributes["braintrust.output"] = row["output"]
    if isinstance(span_attributes.get("type"), str):
        attributes["braintrust.span.type"] = span_attributes["type"]
    if is_tool_span:
        attributes.update(get_normalized_tool_attributes(name, row.get("input")))
        if "input" in row and row["input"] is None:
            attributes["tool.arguments"] = "null"  # JSON.stringify(null)

    span: dict[str, Any] = {"spanId": span_id}
    if is_truthy(parent_span_id):
        span["parentSpanId"] = parent_span_id
    span["name"] = name
    span["startTime"] = start_time
    end_time = _timestamp_ms(metrics_dict.get("end"))
    if end_time is not None:
        span["endTime"] = end_time
    span["attributes"] = attributes
    error = row.get("error")
    span["statusCode"] = 2 if is_truthy(error) else 1
    if is_truthy(error):
        span["statusMessage"] = js_string(error)
    return span


class BraintrustProvider:
    """Retrieve Braintrust project traces correlated with an OpenTelemetry trace ID."""

    id = "braintrust"

    def __init__(self, config: dict[str, Any]) -> None:
        if not config.get("endpoint"):
            raise ValueError("Braintrust provider requires endpoint configuration")
        project_id = config.get("projectId")
        if (
            not project_id
            or not isinstance(project_id, str)
            or not PROJECT_ID_PATTERN.search(project_id)
        ):
            raise ValueError("Braintrust provider requires a valid projectId")
        auth = config.get("auth") or {}
        if not auth.get("token"):
            raise ValueError("Braintrust provider requires an auth token")
        validate_trace_provider_endpoint(config["endpoint"], "Braintrust")
        if config.get("timeout") is not None and not is_positive_safe_integer(config["timeout"]):
            raise ValueError("Braintrust provider timeout must be a positive integer")

        self.config = config
        self.base_url = re.sub(r"/\Z", "", config["endpoint"])
        self.project_id = project_id
        self.token = auth["token"]

    def _build_headers(self) -> dict[str, str]:
        headers = {
            name: value
            for name, value in (self.config.get("headers") or {}).items()
            if name.lower() not in ("accept", "authorization", "content-type")
        }
        return {
            **headers,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    async def fetch_trace(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if not is_trace_id(trace_id):
            raise TraceProviderError("Trace ID must contain 32 hexadecimal characters")

        options = options or {}
        normalized = trace_id.lower()
        requested = options.get("maxSpans")
        max_spans = min(MAX_SPANS if requested is None else requested, MAX_SPANS)
        # Braintrust native root_span_id values do not necessarily match W3C trace IDs.
        # Customers should log the propagated ID as metadata.trace_id or
        # metadata.promptfoo_trace_id. The traces shape returns every span in a matching trace,
        # including child spans that do not repeat the correlation metadata.
        query = "\n".join(
            [
                "SELECT id, span_id, root_span_id, span_parents, created, input, output,",
                "  error, metadata, metrics, span_attributes",
                f"FROM project_logs('{self.project_id}', shape => 'traces')",
                "WHERE created >= now() - INTERVAL 1 DAY",
                f"  AND (metadata.trace_id = '{normalized}'",
                f"    OR metadata.promptfoo_trace_id = '{normalized}'",
                f"    OR metadata.\"promptfoo.trace_id\" = '{normalized}'",
                f"    OR root_span_id = '{normalized}')",
                f"LIMIT {js_string(max_spans)}",
            ]
        )

        response = await fetch_trace_response(
            f"{self.base_url}/btql",
            {
                "disableTransientRetries": True,
                "method": "POST",
                "headers": self._build_headers(),
                "body": json_stringify({"query": query, "fmt": "json"}),
            },
            timeout_ms=self.config.get("timeout") or 10_000,
            abort_signal=options.get("abortSignal"),
        )

        if response.status == 404:
            await release_response(response, "Braintrust")
            raise TraceProviderError(
                "Braintrust BTQL endpoint returned HTTP 404; check the endpoint and project "
                "configuration",
                status_code=response.status,
            )
        if not response.ok:
            await release_response(response, "Braintrust")
            raise TraceProviderError(
                f"Braintrust returned HTTP {response.status}",
                status_code=response.status,
                retryable=response.status in (408, 429) or response.status >= 500,
            )

        if content_length_exceeds(response.header("content-length"), MAX_TRACE_RESPONSE_BYTES):
            await release_response(response, "Braintrust")
            raise TraceProviderError("Braintrust trace exceeds the maximum response size")
        body = await read_limited_response(response, "Braintrust")

        result = js_json_parse(body)
        rows = None
        if isinstance(result, dict):
            rows = result.get("rows")
            if rows is None:
                rows = result.get("data")
        if not isinstance(rows, list):
            raise TraceProviderError("Braintrust returned an invalid query response")
        if not rows:
            return None

        spans: list[dict[str, Any]] = []
        for row in rows:
            if len(spans) >= max_spans:
                break
            span = _transform_span(row, options)
            if not span:
                logger.warning("[BraintrustProvider] Skipping malformed span")
                continue
            spans.append(span)

        return {"spans": spans, "fetchedAt": now_ms()} if spans else None
