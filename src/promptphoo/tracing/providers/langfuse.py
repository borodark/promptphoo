"""Ports src/tracing/providers/langfuse.ts: trace observations from Langfuse's public v2
Observations API."""

from __future__ import annotations

import base64
import logging
import math
import re
from datetime import UTC, datetime
from typing import Any

from promptphoo.fetch.header_parser import date_parse
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import UNDEFINED
from promptphoo.tracing.providers.fetch import (
    MAX_TRACE_RESPONSE_BYTES,
    fetch_trace_response,
    read_limited_response,
    release_response,
    validate_trace_provider_endpoint,
)
from promptphoo.tracing.providers.tempo import (
    MAX_SAFE_INTEGER,
    content_length_exceeds,
    is_positive_safe_integer,
    is_trace_id,
    now_ms,
)
from promptphoo.tracing.providers.types import TraceProviderError
from promptphoo.tracing.tool_attributes import get_normalized_tool_attributes

logger = logging.getLogger(__name__)

MAX_SPANS = 10_000
MAX_PAGE_SIZE = 1_000
RESERVED_ATTRIBUTE_KEY_SEGMENTS = frozenset({"__proto__", "constructor", "prototype"})
GEN_AI_OPERATION_NAMES = {
    "AGENT": "invoke_agent",
    "CHAIN": "invoke_workflow",
    "EMBEDDING": "embeddings",
    "GENERATION": "chat",
    "RETRIEVER": "retrieval",
    "TOOL": "execute_tool",
}
GEN_AI_MODEL_PARAMETER_NAMES = {
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "frequencyPenalty": "gen_ai.request.frequency_penalty",
    "max_tokens": "gen_ai.request.max_tokens",
    "maxTokens": "gen_ai.request.max_tokens",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "presencePenalty": "gen_ai.request.presence_penalty",
    "seed": "gen_ai.request.seed",
    "temperature": "gen_ai.request.temperature",
    "top_k": "gen_ai.request.top_k",
    "top_p": "gen_ai.request.top_p",
    "topK": "gen_ai.request.top_k",
    "topP": "gen_ai.request.top_p",
}
_TEXT_COMPLETION = re.compile(r"^text[ _-]completion(?:\b|\Z)")
_GENERATE_CONTENT = re.compile(r"^generate[ _-]content(?:\b|\Z)")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value)


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return js_json_parse(value)
    except (ValueError, RecursionError):
        return value


def _safe_attributes(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}
    return {
        key: v
        for key, v in value.items()
        if not any(segment in RESERVED_ATTRIBUTE_KEY_SEGMENTS for segment in key.split("."))
    }


def _first_numeric_value(*values: Any) -> int | float | None:
    return next((v for v in values if _is_finite_number(v)), None)


def _normalized_model_parameters(value: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for name, parameter in _safe_attributes(value).items():
        if name in GEN_AI_MODEL_PARAMETER_NAMES and _is_finite_number(parameter):
            attributes[GEN_AI_MODEL_PARAMETER_NAMES[name]] = parameter
    return attributes


def _observation_operation_name(
    observation: dict[str, Any], attributes: dict[str, Any]
) -> str | None:
    original = attributes.get("gen_ai.operation.name")
    if isinstance(original, str) and original.strip():
        return original
    obs_type = observation.get("type")
    if not isinstance(obs_type, str):
        return None
    name = observation.get("name")
    if obs_type == "GENERATION" and isinstance(name, str):
        normalized = name.strip().lower()
        if _TEXT_COMPLETION.search(normalized):
            return "text_completion"
        if _GENERATE_CONTENT.search(normalized):
            return "generate_content"
    return GEN_AI_OPERATION_NAMES.get(obs_type)


def _nullish(value: Any) -> bool:
    return value is None or value is UNDEFINED


def _coalesce(*values: Any) -> Any:
    for value in values:
        if not _nullish(value):
            return value
    return None


def _observation_attributes(observation: dict[str, Any]) -> dict[str, Any]:
    get = observation.get
    metadata = _safe_attributes(get("metadata"))
    resource_attributes = _safe_attributes(metadata.get("resourceAttributes"))
    telemetry_attributes = _safe_attributes(metadata.get("attributes"))
    usage_details = _safe_attributes(get("usageDetails"))
    cost_details = _safe_attributes(get("costDetails"))
    input_cost = _coalesce(cost_details.get("input"), get("calculatedInputCost"), get("inputCost"))
    output_cost = _coalesce(
        cost_details.get("output"), get("calculatedOutputCost"), get("outputCost")
    )
    total_cost = _coalesce(cost_details.get("total"), get("calculatedTotalCost"), get("totalCost"))
    input_tokens = _first_numeric_value(
        get("inputUsage"),
        usage_details.get("input"),
        usage_details.get("input_tokens"),
        usage_details.get("prompt_tokens"),
    )
    output_tokens = _first_numeric_value(
        get("outputUsage"),
        usage_details.get("output"),
        usage_details.get("output_tokens"),
        usage_details.get("completion_tokens"),
    )
    total_tokens = _first_numeric_value(
        get("totalUsage"), usage_details.get("total"), usage_details.get("total_tokens")
    )
    reasoning_tokens = _first_numeric_value(
        usage_details.get("reasoning"), usage_details.get("reasoning_tokens")
    )
    cache_read_tokens = _first_numeric_value(
        usage_details.get("cache_read_input_tokens"), usage_details.get("cached_tokens")
    )
    cache_creation_tokens = _first_numeric_value(usage_details.get("cache_creation_input_tokens"))
    operation_name = _observation_operation_name(observation, {**metadata, **telemetry_attributes})
    has_input = "input" in observation and get("input") is not UNDEFINED
    has_output = "output" in observation and get("output") is not UNDEFINED
    parsed_input = _parse_json_value(get("input"))
    parsed_output = _parse_json_value(get("output"))
    name = get("name")
    observation_name = name if isinstance(name, str) and name.strip() else None
    obs_type = get("type")

    out: dict[str, Any] = {
        **resource_attributes,
        **metadata,
        **_normalized_model_parameters(get("modelParameters")),
    }
    if operation_name:
        out["gen_ai.operation.name"] = operation_name
    if isinstance(obs_type, str):
        out["langfuse.observation.type"] = obs_type
    if has_input:
        out["langfuse.input"] = parsed_input
    if obs_type == "TOOL" and observation_name:
        out.update(
            get_normalized_tool_attributes(observation_name, parsed_input if has_input else None)
        )
        if has_input and parsed_input is None:
            out["tool.arguments"] = "null"  # JSON.stringify(null); None means undefined above
        if has_input:
            out["gen_ai.tool.call.arguments"] = parsed_input
        if has_output:
            out["gen_ai.tool.call.result"] = parsed_output
    if obs_type == "AGENT" and observation_name:
        out["gen_ai.agent.name"] = observation_name
    if obs_type == "CHAIN" and observation_name:
        out["gen_ai.workflow.name"] = observation_name
    if obs_type == "EVALUATOR" and observation_name:
        out["gen_ai.evaluation.name"] = observation_name
    if obs_type == "GUARDRAIL" and observation_name:
        out["guardrail.name"] = observation_name
    if has_output:
        out["langfuse.output"] = parsed_output
    if isinstance(get("providedModelName"), str):
        out["gen_ai.request.model"] = get("providedModelName")
    if not _nullish(get("modelParameters")):
        out["langfuse.model.parameters"] = get("modelParameters")
    if not _nullish(get("usageDetails")):
        out["langfuse.usage"] = get("usageDetails")
    if input_tokens is not None:
        out["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        out["gen_ai.usage.output_tokens"] = output_tokens
    if total_tokens is not None:
        out["langfuse.usage.total_tokens"] = total_tokens
        out["promptfoo.usage.total_tokens"] = total_tokens
    if reasoning_tokens is not None:
        out["gen_ai.usage.reasoning.output_tokens"] = reasoning_tokens
    if cache_read_tokens is not None:
        out["gen_ai.usage.cache_read.input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None:
        out["gen_ai.usage.cache_creation.input_tokens"] = cache_creation_tokens
    if _is_number(input_cost):
        out["langfuse.cost.input"] = input_cost
    if _is_number(output_cost):
        out["langfuse.cost.output"] = output_cost
    if _is_number(total_cost):
        out["langfuse.cost.total"] = total_cost
    out.update(telemetry_attributes)
    return out


def _js_date_parse(value: str) -> float:
    parsed = date_parse(value)
    return math.nan if parsed is None else float(round(parsed))


def _transform_observation(
    observation: dict[str, Any], trace_id: str, options: dict[str, Any] | None
) -> dict[str, Any] | None:
    obs_id = observation.get("id")
    obs_trace_id = observation.get("traceId")
    if (
        not isinstance(obs_id, str)
        or not obs_id
        or not isinstance(obs_trace_id, str)
        or obs_trace_id.lower() != trace_id
        or not isinstance(observation.get("startTime"), str)
    ):
        return None

    earliest = (options or {}).get("earliestStartTime")
    start_time = _js_date_parse(observation["startTime"])
    if math.isnan(start_time) or (earliest is not None and start_time < earliest):
        return None

    raw_end = observation.get("endTime")
    end_time = _js_date_parse(raw_end) if isinstance(raw_end, str) else math.nan
    if not _nullish(raw_end) and (math.isnan(end_time) or end_time < start_time):
        return None
    parent = observation.get("parentObservationId")
    if not _nullish(parent) and (not isinstance(parent, str) or not parent or parent == obs_id):
        return None

    span: dict[str, Any] = {"spanId": obs_id}
    if isinstance(parent, str):
        span["parentSpanId"] = parent
    name = observation.get("name")
    span["name"] = name if isinstance(name, str) and name.strip() else "langfuse.observation"
    span["startTime"] = int(start_time)
    if math.isfinite(end_time):
        span["endTime"] = int(end_time)
    span["attributes"] = _observation_attributes(observation)
    span["statusCode"] = 2 if observation.get("level") == "ERROR" else 1
    status_message = observation.get("statusMessage")
    if isinstance(status_message, str) and status_message:
        span["statusMessage"] = status_message
    return span


def _add_observations(
    observations: list[Any],
    spans: list[dict[str, Any]],
    seen_span_ids: set[str],
    trace_id: str,
    max_spans: int,
    options: dict[str, Any] | None,
) -> None:
    for observation in observations:
        if not isinstance(observation, dict):
            logger.warning("[LangfuseProvider] Skipping malformed observation")
            continue
        span = _transform_observation(observation, trace_id, options)
        if not span:
            logger.warning("[LangfuseProvider] Skipping malformed or unrelated observation")
            continue
        if span["spanId"] not in seen_span_ids:
            seen_span_ids.add(span["spanId"])
            spans.append(span)
        if len(spans) >= max_spans:
            return


def _meta(response: dict[str, Any]) -> dict[str, Any]:
    meta = response.get("meta")
    return meta if isinstance(meta, dict) else {}


def _get_next_cursor(response: dict[str, Any], seen_cursors: set[str]) -> str | None:
    cursor = _meta(response).get("cursor")
    if _nullish(cursor):
        return None
    if not isinstance(cursor, str) or not cursor:
        raise TraceProviderError("Langfuse returned an invalid pagination cursor")
    if cursor in seen_cursors:
        raise TraceProviderError("Langfuse returned a repeated pagination cursor")
    seen_cursors.add(cursor)
    return cursor


def _is_safe_integer(value: Any) -> bool:
    if not _is_number(value) or not math.isfinite(value):
        return False
    return float(value).is_integer() and abs(value) <= MAX_SAFE_INTEGER


def _get_next_page(response: dict[str, Any]) -> int | None:
    meta = _meta(response)
    page = meta.get("page")
    total_pages = meta.get("totalPages")
    if _nullish(page) and _nullish(total_pages):
        return None
    data = response.get("data")
    if page == 1 and total_pages == 0 and isinstance(data, list) and len(data) == 0:
        return None
    if not _is_safe_integer(page) or not _is_safe_integer(total_pages):
        raise TraceProviderError("Langfuse returned invalid pagination metadata")
    page_number, page_count = int(page), int(total_pages)  # type: ignore[arg-type]
    if page_number < 1 or page_count < page_number:
        raise TraceProviderError("Langfuse returned invalid pagination metadata")
    return page_number + 1 if page_number < page_count else None


def _iso_string(ms: float) -> str:
    """`new Date(ms).toISOString()`."""
    dt = datetime.fromtimestamp(ms / 1000, UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ms) % 1000:03d}Z"


class LangfuseProvider:
    """Retrieve trace observations from Langfuse's public v2 Observations API."""

    id = "langfuse"

    def __init__(self, config: dict[str, Any]) -> None:
        if not config.get("endpoint"):
            raise ValueError("Langfuse provider requires endpoint configuration")
        auth = config.get("auth") or {}
        if not auth.get("username") or not auth.get("password") or auth.get("token"):
            raise ValueError(
                "Langfuse provider requires public and secret keys as basic-auth credentials"
            )
        validate_trace_provider_endpoint(config["endpoint"], "Langfuse")
        if config.get("timeout") is not None and not is_positive_safe_integer(config["timeout"]):
            raise ValueError("Langfuse provider timeout must be a positive integer")

        self.config = config
        self.base_url = re.sub(r"/\Z", "", config["endpoint"])
        token = base64.b64encode(f"{auth['username']}:{auth['password']}".encode()).decode()
        self.authorization = f"Basic {token}"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            name: value
            for name, value in (self.config.get("headers") or {}).items()
            if name.lower() not in ("accept", "authorization")
        }
        return {**headers, "Accept": "application/json", "Authorization": self.authorization}

    async def fetch_trace(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        from promptphoo.whatwg_url import URL

        if not is_trace_id(trace_id):
            raise TraceProviderError("Trace ID must contain 32 hexadecimal characters")

        options = options or {}
        normalized_trace_id = trace_id.lower()
        requested = options.get("maxSpans")
        max_spans = min(max(MAX_SPANS if requested is None else requested, 1), MAX_SPANS)
        page_size = min(max_spans, MAX_PAGE_SIZE)
        spans: list[dict[str, Any]] = []
        seen_span_ids: set[str] = set()
        seen_cursors: set[str] = set()
        page: int | None = None
        cursor: str | None = None
        remaining_bytes = MAX_TRACE_RESPONSE_BYTES

        while True:
            url = URL(f"{self.base_url}/api/public/v2/observations")
            url.search_params_set("traceId", normalized_trace_id)
            url.search_params_set("fields", "core,basic,io,metadata,model,usage")
            url.search_params_set("limit", str(int(page_size)))
            if options.get("earliestStartTime") is not None:
                url.search_params_set("fromStartTime", _iso_string(options["earliestStartTime"]))
            if cursor:
                url.search_params_set("cursor", cursor)
            if page:
                url.search_params_set("page", str(page))

            response = await fetch_trace_response(
                url.href,
                {
                    "disableTransientRetries": True,
                    "method": "GET",
                    "headers": self._build_headers(),
                },
                timeout_ms=self.config.get("timeout") or 10_000,
                abort_signal=options.get("abortSignal"),
            )

            if response.status == 404:
                await release_response(response, "Langfuse")
                return None
            if not response.ok:
                await release_response(response, "Langfuse")
                raise TraceProviderError(
                    f"Langfuse returned HTTP {response.status}",
                    status_code=response.status,
                    retryable=response.status in (408, 429) or response.status >= 500,
                )

            if content_length_exceeds(response.header("content-length"), remaining_bytes):
                await release_response(response, "Langfuse")
                raise TraceProviderError("Langfuse trace exceeds the maximum response size")
            body = await read_limited_response(response, "Langfuse", remaining_bytes)
            remaining_bytes -= len(body.encode("utf-8"))
            result = js_json_parse(body)
            if not isinstance(result, dict) or not isinstance(result.get("data"), list):
                raise TraceProviderError("Langfuse returned an invalid observations response")

            _add_observations(
                result["data"], spans, seen_span_ids, normalized_trace_id, max_spans, options
            )
            page = _get_next_page(result)
            cursor = None if page else _get_next_cursor(result, seen_cursors)
            if not ((page or cursor) and len(spans) < max_spans):
                break

        if not spans:
            return None
        return {"spans": spans, "fetchedAt": now_ms()}
