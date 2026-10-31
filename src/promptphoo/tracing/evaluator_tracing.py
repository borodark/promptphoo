# SHIM: replaced by the trace-store branch at merge
"""Ports the trace-context parts of src/tracing/evaluatorTracing.ts: trace and span ids, the
W3C traceparent and generateTraceContextIfNeeded. The OTLP receiver lifecycle and the
OpenTelemetry root span are left to the trace-store branch."""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from promptphoo import envars
from promptphoo.jsvalues import json_stringify

logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    return secrets.token_hex(16)


def generate_span_id() -> str:
    return secrets.token_hex(8)


def generate_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def is_tracing_enabled(test: dict[str, Any], test_suite: dict[str, Any] | None = None) -> bool:
    metadata_enabled = (test.get("metadata") or {}).get("tracingEnabled") is True
    yaml_config_enabled = ((test_suite or {}).get("tracing") or {}).get("enabled") is True
    env_enabled = envars.get_env_bool("PROMPTFOO_TRACING_ENABLED", False)
    return metadata_enabled or yaml_config_enabled or env_enabled


async def generate_trace_context_if_needed(
    test: dict[str, Any],
    evaluate_options: dict[str, Any] | None,
    test_idx: int,
    prompt_idx: int,
    test_suite: dict[str, Any] | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    from promptphoo.tracing.store import get_trace_store

    execution_metadata = execution_metadata or {}
    if not is_tracing_enabled(test, test_suite):
        return None
    logger.debug("[EvaluatorTracing] Tracing enabled for test case")
    trace_store = get_trace_store()

    metadata = test.get("metadata") or {}
    evaluation_id = metadata.get("evaluationId") or (evaluate_options or {}).get("eventSource")
    if not evaluation_id:
        logger.warning(
            "[EvaluatorTracing] No evaluation ID found in test metadata or evaluateOptions, "
            "trace will not be linked to evaluation"
        )
        evaluation_id = f"eval-{int(time.time() * 1000)}"
    test_case_id = metadata.get("testCaseId") or test.get("id") or f"{test_idx}-{prompt_idx}"

    trace_id = generate_trace_id()
    span_id = generate_span_id()
    traceparent = generate_traceparent(trace_id, span_id, True)
    logger.debug(
        "[EvaluatorTracing] Generated trace context: traceId=%s, spanId=%s", trace_id, span_id
    )

    tracing = (test_suite or {}).get("tracing") or {}
    trace_metadata: dict[str, Any] = {"testIdx": test_idx, "promptIdx": prompt_idx}
    if execution_metadata.get("providerId"):
        trace_metadata["providerId"] = execution_metadata["providerId"]
    if execution_metadata.get("repeatIndex") is not None:
        trace_metadata["repeatIndex"] = execution_metadata["repeatIndex"]
    trace_metadata["vars"] = test.get("vars")
    trace_metadata["commandToolNames"] = tracing.get("commandToolNames")
    trace_metadata["otlpHttpRedactAttributes"] = (
        (tracing.get("otlp") or {}).get("http") or {}
    ).get("redactAttributes")
    try:
        await trace_store.create_trace(
            {
                "traceId": trace_id,
                "evaluationId": evaluation_id or "",
                "testCaseId": test_case_id or "",
                # JSON.stringify drops the undefined keys.
                "metadata": {k: v for k, v in trace_metadata.items() if v is not None},
            }
        )
    except Exception as e:
        logger.error("[EvaluatorTracing] Failed to create trace: %s", e)

    logger.debug(
        "[EvaluatorTracing] Trace context ready: %s for test case %s",
        traceparent,
        json_stringify(test_case_id),
    )
    return {"traceparent": traceparent, "evaluationId": evaluation_id, "testCaseId": test_case_id}
