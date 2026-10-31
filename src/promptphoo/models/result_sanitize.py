"""Persistence / output sanitizing for eval results.

Ports the module-level helpers of src/models/evalResult.ts (provider snapshots, sanitizeForDb,
credential-header redaction, trace/repeat linkage metadata and the PROMPTFOO_STRIP_* output
projections). Blob extraction (`extractAndStoreBinaryData`) lives in blobs/extractor.py and
trace projection (`projectTracesForOutput`) in output/writer.py.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from promptphoo.envars import get_env_bool, parse_env_bool
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string
from promptphoo.providers.base import is_api_provider
from promptphoo.sanitizer import (
    GRADING_PROVIDER_TYPE_KEYS,
    REDACTED,
    is_provider_type_map,
    is_secret_field,
    map_test_provider_refs,
    safe_json_stringify,
    sanitize_object,
    strip_provider_prompt_selectors,
    strip_test_provider_prompt_selectors,
)

logger = logging.getLogger(__name__)

PROMPTFOO_METADATA_KEY = "__promptfoo"
TRACE_LINKAGE_KEY = "traceLinkage"
REPEAT_LINKAGE_KEY = "repeatLinkage"

_VALID_RESULT_FAILURE_REASONS = frozenset([0, 1, 2])  # ResultFailureReason NONE/ASSERT/ERROR


def is_result_failure_reason(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and value in _VALID_RESULT_FAILURE_REASONS
    )


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _as_record(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _absent(value: Any) -> bool:
    return value is None or value is UNDEFINED


def _deep_strict_equal(a: Any, b: Any) -> bool:
    """util.isDeepStrictEqual over JSON-shaped values (bool is not a number)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_deep_strict_equal(x, y) for x, y in zip(a, b, strict=True))
    if type(a) is not type(b):
        return False
    return a == b


def _json_parse_stringified(value: Any) -> Any:
    """JSON.parse(safeJsonStringify(value) as string): raises when it was unserializable."""
    serialized = safe_json_stringify(value)
    if serialized is None:
        raise ValueError('"undefined" is not valid JSON')
    return json.loads(serialized)


_BLOB_SCHEME = "promptfoo://blob/"
_BLOB_HASH_REGEX = re.compile(r"^[a-f0-9]{64}\Z", re.I)
_BLOB_URI_REGEX = re.compile(r"promptfoo://blob/([a-f0-9]{64})", re.I)


def _extract_blob_hashes_from_value(value: Any) -> list[str]:
    def from_string(s: str) -> list[str]:
        if _BLOB_SCHEME not in s:
            return []
        return [m.group(1).lower() for m in _BLOB_URI_REGEX.finditer(s)]

    if isinstance(value, str):
        return from_string(value)
    if not isinstance(value, dict):
        return []
    h = value.get("hash")
    if isinstance(h, str) and _BLOB_HASH_REGEX.match(h):
        return [h.lower()]
    uri = value.get("uri")
    if isinstance(uri, str):
        return from_string(uri)
    return []


def _strip_media_references(value: Any) -> Any:
    if _extract_blob_hashes_from_value(value) or (
        isinstance(value, str) and re.match(r"^data:[^,]*,", value, re.I)
    ):
        return "[output stripped]"
    if isinstance(value, list):
        return [_strip_media_references(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_media_references(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------------------------


def sanitize_provider_config(config: Any) -> Any:
    return sanitize_object(
        _json_parse_stringified(config),
        context="provider config",
        sanitize_urls=True,
        max_depth=math.inf,
    )


def _provider_options(pid: Any, label: Any, config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": pid}
    if label is not UNDEFINED:
        out["label"] = label
    if is_truthy(config):
        out["config"] = sanitize_provider_config(config)
    return out


def sanitize_provider(provider: Any) -> Any:
    """Removes circular references from a provider and returns ProviderOptions-shaped data."""
    try:
        if is_api_provider(provider):
            return _provider_options(
                provider.id(),
                UNDEFINED if provider.label is None else provider.label,
                provider.config,
            )
        if isinstance(provider, dict):
            pid = provider.get("id")
            return _provider_options(
                pid() if callable(pid) else pid,
                provider.get("label", UNDEFINED),
                provider.get("config"),
            )
    except Exception:
        pass
    serialized = safe_json_stringify(provider)
    return None if serialized is None else json.loads(serialized)


def to_serializable_provider_ref(provider: Any) -> Any:
    """Snapshot live provider references for replay without retaining runtime client state."""
    if is_api_provider(provider):
        transform = getattr(provider, "transform", None)
        extras = sanitize_object(
            {
                "transform": transform if isinstance(transform, str) else UNDEFINED,
                "delay": UNDEFINED if provider.delay is None else provider.delay,
                "inputs": UNDEFINED if provider.inputs is None else provider.inputs,
            },
            context="provider options",
            sanitize_urls=True,
            max_depth=math.inf,
        )
        out = {**sanitize_provider(provider), **(extras if isinstance(extras, dict) else {})}
        if provider.prompts:
            out["prompts"] = list(provider.prompts)
        return out
    if isinstance(provider, list):
        return [to_serializable_provider_ref(p) for p in provider]
    if is_provider_type_map(provider):
        serialized: dict[str, Any] | None = None
        for t in GRADING_PROVIDER_TYPE_KEYS:
            if is_api_provider(provider.get(t)):
                if serialized is None:
                    serialized = dict(provider)
                serialized[t] = to_serializable_provider_ref(provider[t])
        return serialized if serialized is not None else provider
    return provider


def map_grading_result_provider_refs(
    grading_result: Any, map_provider: Callable[[Any], Any]
) -> Any:
    """Map only grading assertion providers and component results."""
    visited: dict[int, dict[str, Any]] = {}

    def project(value: Any) -> Any:
        result = _as_record(value)
        if result is None:
            return value
        previous = visited.get(id(result))
        if previous is not None:
            return previous
        projected = dict(result)
        visited[id(result)] = projected
        if is_truthy(result.get("assertion", UNDEFINED)):
            projected["assertion"] = map_test_provider_refs(result["assertion"], map_provider)
        if isinstance(result.get("componentResults"), list):
            projected["componentResults"] = [project(c) for c in result["componentResults"]]
        return projected

    return project(grading_result)


def serialize_result_provider_refs(result: dict[str, Any]) -> dict[str, Any]:
    """Snapshot live grading references before generic result serialization."""

    def serialize_provider(provider: Any) -> Any:
        if not is_api_provider(provider) and not is_provider_type_map(provider):
            return provider
        return sanitize_object(
            to_serializable_provider_ref(provider),
            context="grading provider",
            sanitize_urls=True,
            max_depth=math.inf,
            throw_on_error=True,
        )

    projected = dict(result)
    if is_truthy(result.get("testCase", UNDEFINED)):
        projected["testCase"] = map_test_provider_refs(result["testCase"], serialize_provider)
    prompt = _as_record(result.get("prompt"))
    config = _as_record(prompt.get("config")) if prompt else None
    if prompt is not None and config is not None and "provider" in config:
        projected["prompt"] = {
            **prompt,
            "config": {**config, "provider": serialize_provider(config["provider"])},
        }
    if is_truthy(result.get("gradingResult", UNDEFINED)):
        projected["gradingResult"] = map_grading_result_provider_refs(
            result["gradingResult"], serialize_provider
        )
    return projected


# ---------------------------------------------------------------------------------------------
# DB sanitizing
# ---------------------------------------------------------------------------------------------


def sanitize_for_db(obj: Any) -> Any:
    """Strip circular references and non-serializable values (a JSON round trip)."""
    if obj is None or obj is UNDEFINED:
        return obj
    try:
        serialized = safe_json_stringify(obj)
        if serialized is None:
            logger.debug("sanitizeForDb: Failed to serialize object, using fallback")
            return [] if isinstance(obj, list | tuple) else None
        return json.loads(serialized)
    except Exception as error:
        logger.debug("sanitizeForDb: Parse error, using fallback: %s", error)
        return [] if isinstance(obj, list | tuple) else None


def sanitize_for_db_with_secrets(obj: Any) -> Any:
    """sanitize_for_db plus credential redaction at any depth (provider-config slots)."""
    if obj is None or obj is UNDEFINED:
        return obj
    return sanitize_object(obj, context="evalResult field", max_depth=math.inf)


SENSITIVE_RESPONSE_HEADER_NAMES = frozenset(
    [
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "openai-organization",
        "openai-project",
        "openai-version",
        "x-request-id",
        "x-amzn-requestid",
        "x-amzn-trace-id",
        "x-amz-security-token",
        "x-amz-cf-id",
        "x-azure-ref",
        "x-correlation-id",
        "x-trace-id",
        "cf-ray",
        "cf-cache-status",
        "x-openai-proxy-wasm",
        "via",
    ]
)
SENSITIVE_RESPONSE_HEADER_PREFIXES = ("x-ratelimit-",)


def is_sensitive_response_header(header_name: str) -> bool:
    normalized = header_name.lower()
    if normalized in SENSITIVE_RESPONSE_HEADER_NAMES:
        return True
    return any(normalized.startswith(p) for p in SENSITIVE_RESPONSE_HEADER_PREFIXES)


def is_sensitive_request_header(header_name: str) -> bool:
    return is_sensitive_response_header(header_name) or is_secret_field(header_name)


def _redact_sensitive_headers(
    headers: Mapping[str, Any],
    source_headers: Mapping[str, Any] | None = None,
    is_sensitive_header: Callable[[str], bool] = is_sensitive_response_header,
) -> dict[str, Any] | None:
    if source_headers is None:
        source_headers = headers
    mutated = False
    nxt: dict[str, Any] = {}
    for key, value in headers.items():
        if (
            key in source_headers
            and _deep_strict_equal(source_headers[key], value)
            and is_sensitive_header(key)
        ):
            nxt[key] = REDACTED
            mutated = True
        else:
            nxt[key] = value
    return nxt if mutated else None


def _redact_http_headers_on_metadata(
    metadata: Any,
    *,
    legacy_headers_source: Any = None,
    redact_legacy_headers: bool = False,
) -> Any:
    if not isinstance(metadata, dict):
        return metadata
    m = metadata
    next_metadata: dict[str, Any] | None = None

    legacy_headers = m.get("headers")
    if redact_legacy_headers:
        legacy_source = legacy_headers
    else:
        legacy_source = (
            legacy_headers_source.get("headers")
            if isinstance(legacy_headers_source, dict)
            else None
        )
    if isinstance(legacy_headers, dict) and isinstance(legacy_source, dict):
        redacted = _redact_sensitive_headers(
            legacy_headers, legacy_source, is_sensitive_request_header
        )
        if redacted is not None:
            next_metadata = {**m, "headers": redacted}

    http = m.get("http")
    if not isinstance(http, dict):
        return next_metadata if next_metadata is not None else metadata

    next_http: dict[str, Any] | None = None
    for slot in ("headers", "requestHeaders"):
        slot_value = http.get(slot)
        if isinstance(slot_value, dict):
            redacted = (
                _redact_sensitive_headers(slot_value, slot_value, is_sensitive_request_header)
                if slot == "requestHeaders"
                else _redact_sensitive_headers(slot_value)
            )
            if redacted is not None:
                if next_http is None:
                    next_http = dict(http)
                next_http[slot] = redacted

    if next_http is None:
        return next_metadata if next_metadata is not None else metadata
    if next_metadata is None:
        next_metadata = dict(m)
    next_metadata["http"] = next_http
    return next_metadata


def _redact_http_headers_on_grading_result(grading_result: Any) -> Any:
    if not isinstance(grading_result, dict):
        return grading_result
    gr = grading_result
    mutated = False
    nxt = dict(gr)
    if "metadata" in gr and gr["metadata"] is not UNDEFINED:
        redacted = _redact_http_headers_on_metadata(gr["metadata"])
        if redacted is not gr["metadata"]:
            nxt["metadata"] = redacted
            mutated = True
    if isinstance(gr.get("componentResults"), list):
        component_mutated = False
        components = []
        for component in gr["componentResults"]:
            redacted = _redact_http_headers_on_grading_result(component)
            if redacted is not component:
                component_mutated = True
            components.append(redacted)
        if component_mutated:
            nxt["componentResults"] = components
            mutated = True
    return nxt if mutated else grading_result


def sanitize_response_for_db(response: Any) -> Any:
    if not is_truthy(response):
        return response
    metadata = response.get("metadata", UNDEFINED)
    redacted = _redact_http_headers_on_metadata(metadata, redact_legacy_headers=True)
    if redacted is metadata:
        return response
    return {**response, "metadata": redacted}


def sanitize_metadata_for_db(metadata: Any, response_metadata: Any = UNDEFINED) -> Any:
    return _redact_http_headers_on_metadata(
        metadata, legacy_headers_source=sanitize_for_db(response_metadata)
    )


def _sanitize_grading_result_assertions(grading_result: Any) -> Any:
    if not isinstance(grading_result, dict):
        return grading_result
    gr = grading_result
    assertion = _as_record(gr.get("assertion"))
    sanitized_assertion = dict(assertion) if assertion is not None else None
    if sanitized_assertion is not None:
        for field in ("provider", "config"):
            if field in sanitized_assertion:
                sanitized_assertion[field] = sanitize_object(
                    sanitized_assertion[field],
                    context="grading assertion",
                    max_depth=math.inf,
                    sanitize_urls=True,
                    throw_on_error=True,
                )
    out = dict(gr)
    if sanitized_assertion is not None:
        out["assertion"] = sanitized_assertion
    if isinstance(gr.get("componentResults"), list):
        out["componentResults"] = [
            _sanitize_grading_result_assertions(c) for c in gr["componentResults"]
        ]
    return out


def sanitize_grading_result_for_db(grading_result: Any) -> Any:
    return _redact_http_headers_on_grading_result(
        _sanitize_grading_result_assertions(grading_result)
    )


def redact_sensitive_result_fields_for_db(
    *, response: Any, grading_result: Any, metadata: Any
) -> dict[str, Any]:
    """Credential-header redaction for the sanitize_for_db'd response/gradingResult/metadata.

    Returns {"response", "gradingResult", "metadata"}.
    """
    response_metadata = (
        response.get("metadata", UNDEFINED) if isinstance(response, dict) else UNDEFINED
    )
    return {
        "response": sanitize_response_for_db(response),
        "gradingResult": sanitize_grading_result_for_db(grading_result),
        "metadata": sanitize_metadata_for_db(metadata, response_metadata),
    }


# ---------------------------------------------------------------------------------------------
# trace / repeat linkage
# ---------------------------------------------------------------------------------------------


def persist_trace_metadata(
    metadata: dict[str, Any] | None, trace_id: str | None, evaluation_id: str | None
) -> dict[str, Any] | None:
    if not trace_id and not evaluation_id:
        return strip_trace_linkage_from_metadata(metadata)
    metadata_record = metadata if metadata is not None else {}
    existing = metadata_record.get(PROMPTFOO_METADATA_KEY, UNDEFINED)
    promptfoo_metadata = _as_record(existing)
    if existing is not UNDEFINED and promptfoo_metadata is None:
        logger.warning(
            f"[EvalResult] Overwriting non-object metadata.{PROMPTFOO_METADATA_KEY} with "
            "internal trace linkage; the key is reserved for promptfoo internals."
        )
    if promptfoo_metadata is not None and TRACE_LINKAGE_KEY in promptfoo_metadata:
        logger.warning(
            f"[EvalResult] Overwriting metadata.{PROMPTFOO_METADATA_KEY}.{TRACE_LINKAGE_KEY} "
            "with internal trace linkage; the path is reserved for promptfoo internals."
        )
    linkage: dict[str, Any] = {}
    if not _absent(trace_id):
        linkage["traceId"] = trace_id
    if not _absent(evaluation_id):
        linkage["evaluationId"] = evaluation_id
    return {
        **metadata_record,
        PROMPTFOO_METADATA_KEY: {**(promptfoo_metadata or {}), TRACE_LINKAGE_KEY: linkage},
    }


def _strip_linkage(metadata: Any, linkage_key: str) -> Any:
    metadata_record = _as_record(metadata)
    promptfoo_metadata = (
        _as_record(metadata_record.get(PROMPTFOO_METADATA_KEY)) if metadata_record else None
    )
    if (
        metadata_record is None
        or promptfoo_metadata is None
        or linkage_key not in promptfoo_metadata
    ):
        return metadata
    remaining = {k: v for k, v in promptfoo_metadata.items() if k != linkage_key}
    stripped = {k: v for k, v in metadata_record.items() if k != PROMPTFOO_METADATA_KEY}
    if remaining:
        stripped[PROMPTFOO_METADATA_KEY] = remaining
    return stripped


def strip_trace_linkage_from_metadata(metadata: Any) -> Any:
    return _strip_linkage(metadata, TRACE_LINKAGE_KEY)


def persist_repeat_metadata(
    metadata: dict[str, Any] | None, repeat_index: int | None, repeat_group_id: str | None
) -> dict[str, Any] | None:
    if _absent(repeat_index) or not repeat_group_id:
        return strip_repeat_linkage_from_metadata(metadata)
    metadata_record = metadata if metadata is not None else {}
    existing = metadata_record.get(PROMPTFOO_METADATA_KEY, UNDEFINED)
    promptfoo_metadata = _as_record(existing)
    if existing is not UNDEFINED and promptfoo_metadata is None:
        logger.warning(
            f"[EvalResult] Overwriting non-object metadata.{PROMPTFOO_METADATA_KEY} with "
            "internal repeat linkage; the key is reserved for promptfoo internals."
        )
    if promptfoo_metadata is not None and REPEAT_LINKAGE_KEY in promptfoo_metadata:
        logger.warning(
            f"[EvalResult] Overwriting metadata.{PROMPTFOO_METADATA_KEY}.{REPEAT_LINKAGE_KEY} "
            "with internal repeat linkage; the path is reserved for promptfoo internals."
        )
    return {
        **metadata_record,
        PROMPTFOO_METADATA_KEY: {
            **(promptfoo_metadata or {}),
            REPEAT_LINKAGE_KEY: {"repeatIndex": repeat_index, "repeatGroupId": repeat_group_id},
        },
    }


def strip_repeat_linkage_from_metadata(metadata: Any) -> Any:
    return _strip_linkage(metadata, REPEAT_LINKAGE_KEY)


def surface_repeat_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """{"metadata", "repeatIndex"?, "repeatGroupId"?} with the reserved linkage stripped."""
    metadata_record = metadata if metadata is not None else {}
    promptfoo_metadata = _as_record(metadata_record.get(PROMPTFOO_METADATA_KEY))
    repeat_linkage = (
        _as_record(promptfoo_metadata.get(REPEAT_LINKAGE_KEY)) if promptfoo_metadata else None
    )
    out: dict[str, Any] = {}
    if repeat_linkage is not None:
        idx = repeat_linkage.get("repeatIndex")
        if (
            isinstance(idx, int | float)
            and not isinstance(idx, bool)
            and math.isfinite(idx)
            and float(idx).is_integer()
            and idx >= 0
        ):
            out["repeatIndex"] = idx
        gid = repeat_linkage.get("repeatGroupId")
        if isinstance(gid, str) and gid:
            out["repeatGroupId"] = gid
    has_linkage = promptfoo_metadata is not None and REPEAT_LINKAGE_KEY in promptfoo_metadata
    out["metadata"] = (
        strip_repeat_linkage_from_metadata(metadata_record) if has_linkage else metadata_record
    )
    return out


def surface_trace_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """{"metadata", "traceId"?, "evaluationId"?} with the reserved linkage stripped."""
    metadata_record = metadata if metadata is not None else {}
    promptfoo_metadata = _as_record(metadata_record.get(PROMPTFOO_METADATA_KEY))
    trace_linkage = (
        _as_record(promptfoo_metadata.get(TRACE_LINKAGE_KEY)) if promptfoo_metadata else None
    )
    out: dict[str, Any] = {}
    if trace_linkage is not None:
        if isinstance(trace_linkage.get("traceId"), str):
            out["traceId"] = trace_linkage["traceId"]
        if isinstance(trace_linkage.get("evaluationId"), str):
            out["evaluationId"] = trace_linkage["evaluationId"]
    has_linkage = promptfoo_metadata is not None and TRACE_LINKAGE_KEY in promptfoo_metadata
    out["metadata"] = (
        strip_trace_linkage_from_metadata(metadata_record) if has_linkage else metadata_record
    )
    return out


# ---------------------------------------------------------------------------------------------
# output projections (PROMPTFOO_STRIP_*)
# ---------------------------------------------------------------------------------------------


def get_strip_flags(env: Mapping[str, Any] | None = None) -> dict[str, bool]:
    def get_flag(key: str) -> bool:
        value = env.get(key, UNDEFINED) if env else UNDEFINED
        if value is UNDEFINED:
            return get_env_bool(key, False)
        return parse_env_bool(js_string(value), False)

    return {
        "shouldStripPromptText": get_flag("PROMPTFOO_STRIP_PROMPT_TEXT"),
        "shouldStripResponseOutput": get_flag("PROMPTFOO_STRIP_RESPONSE_OUTPUT"),
        "shouldStripTestVars": get_flag("PROMPTFOO_STRIP_TEST_VARS"),
        "shouldStripGradingResult": get_flag("PROMPTFOO_STRIP_GRADING_RESULT"),
        "shouldStripMetadata": get_flag("PROMPTFOO_STRIP_METADATA"),
    }


def project_output_metadata(
    metadata: Any, strip_output: bool, response_metadata: Any, test_metadata: Any = None
) -> Any:
    if (
        not strip_output
        or not is_truthy(metadata)
        or not is_truthy(response_metadata)
        or not isinstance(metadata, dict)
    ):
        return metadata
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if not (isinstance(response_metadata, dict) and key in response_metadata) or (
            isinstance(test_metadata, dict)
            and _deep_strict_equal(value, test_metadata.get(key, UNDEFINED))
        ):
            out[key] = value
            continue
        if key in ("audio", "blobUris"):
            continue
        out[key] = _strip_media_references(sanitize_for_db(value))
    return out


_STRIPPED_RESPONSE_FIELDS = (
    "raw",
    "providerTransformedOutput",
    "conversationOutput",
    "audio",
    "video",
    "images",
)


def project_provider_response(
    response: dict[str, Any] | None, *, strip_metadata: bool, strip_output: bool
) -> dict[str, Any] | None:
    if not is_truthy(response):
        return response
    assert response is not None
    if not strip_metadata and not strip_output:
        return response
    projected = {k: v for k, v in response.items() if not (strip_metadata and k == "metadata")}
    if strip_output:
        projected["output"] = "[output stripped]"
        for key in _STRIPPED_RESPONSE_FIELDS:
            projected.pop(key, None)
        if is_truthy(projected.get("metadata", UNDEFINED)):
            projected["metadata"] = project_output_metadata(
                projected["metadata"], True, projected["metadata"]
            )
    return projected


def project_prompt(prompt: dict[str, Any], strip_prompt_text: bool) -> dict[str, Any]:
    if not strip_prompt_text:
        return prompt
    out = {**prompt, "raw": "[prompt stripped]"}
    out.pop("template", None)
    config = prompt.get("config")
    if isinstance(config, dict) and is_truthy(config.get("provider", UNDEFINED)):
        out["config"] = {**config, "provider": strip_provider_prompt_selectors(config["provider"])}
    return out


def project_test_case(
    test_case: dict[str, Any],
    *,
    strip_metadata: bool,
    strip_vars: bool,
    strip_output: bool,
    strip_prompt_text: bool,
) -> dict[str, Any]:
    if not strip_metadata and not strip_vars and not strip_output and not strip_prompt_text:
        return test_case
    projected = {k: v for k, v in test_case.items() if not (strip_metadata and k == "metadata")}
    if strip_vars:
        projected.pop("vars", None)
    if strip_output:
        projected.pop("providerOutput", None)
    md = test_case.get("metadata")
    pf = md.get(PROMPTFOO_METADATA_KEY) if isinstance(md, dict) else None
    if strip_metadata and isinstance(pf, dict) and pf.get("remote") is True:
        projected["metadata"] = {PROMPTFOO_METADATA_KEY: {"remote": True}}
    return strip_test_provider_prompt_selectors(projected) if strip_prompt_text else projected


def project_grading_result(grading_result: Any, strip_prompt_text: bool) -> Any:
    return (
        map_grading_result_provider_refs(grading_result, strip_provider_prompt_selectors)
        if strip_prompt_text
        else grading_result
    )


def sanitize_result_for_jsonl_artifact(
    result: dict[str, Any], strip_flags: Mapping[str, bool] | None = None
) -> dict[str, Any]:
    """Sanitize a result row for a JSONL output artifact (header redaction + STRIP_* flags)."""
    flags = strip_flags if strip_flags is not None else get_strip_flags()
    strip_prompt_text = bool(flags.get("shouldStripPromptText"))
    strip_response_output = bool(flags.get("shouldStripResponseOutput"))
    strip_test_vars = bool(flags.get("shouldStripTestVars"))
    strip_grading_result = bool(flags.get("shouldStripGradingResult"))
    strip_metadata = bool(flags.get("shouldStripMetadata"))

    artifact = serialize_result_provider_refs(result)
    redacted = redact_sensitive_result_fields_for_db(
        response=sanitize_for_db(artifact.get("response", UNDEFINED)),
        grading_result=sanitize_for_db(artifact.get("gradingResult", UNDEFINED)),
        metadata=sanitize_for_db(artifact.get("metadata", UNDEFINED)),
    )
    redacted_response = redacted["response"]
    response = project_provider_response(
        redacted_response if is_truthy(redacted_response) else None,
        strip_metadata=strip_metadata,
        strip_output=strip_response_output,
    )

    out: dict[str, Any] = dict(result)
    if is_truthy(artifact.get("testCase", UNDEFINED)):
        out["testCase"] = project_test_case(
            sanitize_for_db_with_secrets(artifact["testCase"]),
            strip_metadata=strip_metadata,
            strip_vars=strip_test_vars,
            strip_output=strip_response_output,
            strip_prompt_text=strip_prompt_text,
        )
    if artifact.get("vars", UNDEFINED) is not UNDEFINED:
        out["vars"] = {} if strip_test_vars else sanitize_for_db_with_secrets(artifact["vars"])
    if is_truthy(artifact.get("prompt", UNDEFINED)):
        out["prompt"] = project_prompt(
            sanitize_for_db_with_secrets(artifact["prompt"]), strip_prompt_text
        )
    if is_truthy(artifact.get("provider", UNDEFINED)):
        out["provider"] = sanitize_provider(artifact["provider"])
    out["response"] = UNDEFINED if response is None else response
    out["gradingResult"] = (
        None
        if strip_grading_result
        else project_grading_result(redacted["gradingResult"], strip_prompt_text)
    )
    out["namedScores"] = sanitize_for_db(artifact.get("namedScores", UNDEFINED))
    if strip_metadata:
        out["metadata"] = {}
    else:
        test_case = artifact.get("testCase")
        out["metadata"] = project_output_metadata(
            redacted["metadata"],
            strip_response_output,
            redacted_response.get("metadata") if isinstance(redacted_response, dict) else None,
            test_case.get("metadata") if isinstance(test_case, dict) else None,
        )
    # JS `undefined` properties vanish when the row is serialized.
    return {k: v for k, v in out.items() if v is not UNDEFINED}
