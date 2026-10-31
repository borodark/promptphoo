# SHIM: replaced by the evaluator_tracing branch at merge
"""The trace-context half of src/tracing/evaluatorTracing.ts that the span wrappers need:
`is_tracing_enabled` and `generate_trace_context_if_needed`, which starts the
`promptfoo.test_case` root span and creates the trace record. The OTLP receiver lifecycle
lives on the other branch.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from promptphoo.envars import get_env_bool
from promptphoo.jsvalues import json_stringify
from promptphoo.tracing.genai_tracer import PromptfooAttributes, get_genai_tracer
from promptphoo.tracing.otel_api import (
    ROOT_CONTEXT,
    SpanKind,
    SpanOptions,
    TraceFlags,
    is_span_context_valid,
)
from promptphoo.tracing.span_roles import SPAN_ROLE_ATTRIBUTE

logger = logging.getLogger("promptphoo.tracing")


def generate_trace_id() -> str:
    return secrets.token_hex(16)


def generate_span_id() -> str:
    return secrets.token_hex(8)


def generate_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def is_tracing_enabled(test: dict[str, Any], test_suite: dict[str, Any] | None = None) -> bool:
    metadata_enabled = (test.get("metadata") or {}).get("tracingEnabled") is True
    yaml_config_enabled = ((test_suite or {}).get("tracing") or {}).get("enabled") is True
    env_enabled = get_env_bool("PROMPTFOO_TRACING_ENABLED", False)
    return metadata_enabled or yaml_config_enabled or env_enabled


async def generate_trace_context_if_needed(
    test: dict[str, Any],
    evaluate_options: dict[str, Any] | None,
    test_idx: int,
    prompt_idx: int,
    test_suite: dict[str, Any] | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not is_tracing_enabled(test, test_suite):
        return None
    execution_metadata = execution_metadata or {}
    from promptphoo.tracing.store import get_trace_store

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

    root_attributes: dict[str, Any] = {
        SPAN_ROLE_ATTRIBUTE: "test_case",
        PromptfooAttributes.EVAL_ID: evaluation_id,
        PromptfooAttributes.TEST_INDEX: test_idx,
        "promptfoo.test_case.id": test_case_id,
        "promptfoo.prompt.index": prompt_idx,
    }
    if execution_metadata.get("providerId"):
        root_attributes[PromptfooAttributes.PROVIDER_ID] = execution_metadata["providerId"]
    if execution_metadata.get("promptLabel"):
        root_attributes[PromptfooAttributes.PROMPT_LABEL] = execution_metadata["promptLabel"]
    if execution_metadata.get("repeatIndex") is not None:
        root_attributes["promptfoo.repeat.index"] = execution_metadata["repeatIndex"]

    candidate = get_genai_tracer().start_span(
        "promptfoo.test_case",
        SpanOptions(kind=SpanKind.INTERNAL, attributes=root_attributes),
        ROOT_CONTEXT,
    )
    root_context = candidate.span_context()
    root_span = candidate if is_span_context_valid(root_context) else None
    trace_id = root_context.trace_id if root_span else generate_trace_id()
    span_id = root_context.span_id if root_span else generate_span_id()
    traceparent = generate_traceparent(
        trace_id,
        span_id,
        (root_context.trace_flags & TraceFlags.SAMPLED) != 0 if root_span else True,
    )

    tracing = (test_suite or {}).get("tracing") or {}
    trace_metadata: dict[str, Any] = {"testIdx": test_idx, "promptIdx": prompt_idx}
    if execution_metadata.get("providerId"):
        trace_metadata["providerId"] = execution_metadata["providerId"]
    if execution_metadata.get("repeatIndex") is not None:
        trace_metadata["repeatIndex"] = execution_metadata["repeatIndex"]
    trace_metadata["vars"] = test.get("vars")
    if tracing.get("commandToolNames") is not None:
        trace_metadata["commandToolNames"] = tracing["commandToolNames"]
    redact = ((tracing.get("otlp") or {}).get("http") or {}).get("redactAttributes")
    if redact is not None:
        trace_metadata["otlpHttpRedactAttributes"] = redact
    try:
        await trace_store.create_trace(
            {
                "traceId": trace_id,
                "evaluationId": evaluation_id or "",
                "testCaseId": test_case_id or "",
                "metadata": trace_metadata,
            }
        )
    except Exception as error:
        logger.error("[EvaluatorTracing] Failed to create trace: %s", error)

    logger.debug(
        "[EvaluatorTracing] Trace context ready: %s for test case %s",
        traceparent,
        json_stringify(test_case_id),
    )
    out: dict[str, Any] = {
        "traceparent": traceparent,
        "evaluationId": evaluation_id,
        "testCaseId": test_case_id,
    }
    if root_span is not None:
        out["rootSpan"] = root_span
    return out
