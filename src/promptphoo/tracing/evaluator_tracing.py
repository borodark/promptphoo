"""Evaluator-side tracing: the shared OTLP receiver's lifecycle and per-test trace context.
Ports src/tracing/evaluatorTracing.ts.

`generate_trace_context_if_needed` returns upstream's EvaluationTraceContext dict,
`{traceparent, evaluationId, testCaseId, rootSpan?}`, or None when tracing is off. Each test
case starts a `promptfoo.test_case` root span through the GenAI tracer (the in-house stand-in
for OpenTelemetry, otel_api.py) and takes the trace and span ids from it while the SDK is
recording; otherwise the ids are random, as upstream's are.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import Any

from promptphoo import cli_state
from promptphoo.envars import get_env_bool
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify
from promptphoo.tracing.genai_tracer import PromptfooAttributes, get_genai_tracer
from promptphoo.tracing.otel_api import (
    ROOT_CONTEXT,
    SpanKind,
    SpanOptions,
    TraceFlags,
    is_span_context_valid,
)
from promptphoo.tracing.span_roles import SPAN_ROLE_ATTRIBUTE

logger = logging.getLogger(__name__)

DEFAULT_OTLP_ACCEPT_FORMATS = ("json", "protobuf")
# The evaluator waits this long before stopping a receiver it acquired (upstream: 3000 ms).
SPAN_EXPORT_GRACE_SECONDS = 3.0

# Shared OTLP receiver lifecycle state: whether it is started, the in-flight start/stop tasks
# that serialize concurrent callers, and a refcount of the evaluations using it.
_otlp_receiver_started = False
_otlp_receiver_start_task: asyncio.Future[None] | None = None
_otlp_receiver_stop_task: asyncio.Future[None] | None = None
_otlp_receiver_users = 0


def reset_tracing_state() -> None:
    """Reset module state (for tests)."""
    global _otlp_receiver_started, _otlp_receiver_start_task, _otlp_receiver_stop_task
    global _otlp_receiver_users
    _otlp_receiver_started = False
    _otlp_receiver_start_task = None
    _otlp_receiver_stop_task = None
    _otlp_receiver_users = 0
    cli_state.set_active_otlp_receiver()
    logger.debug("[EvaluatorTracing] Tracing state reset")


def generate_trace_id() -> str:
    """A 16-byte trace id as 32 hex characters."""
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """An 8-byte span id as 16 hex characters."""
    return secrets.token_hex(8)


def generate_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    """A W3C Trace Context traceparent: version-trace-id-parent-id-trace-flags."""
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def is_otlp_receiver_started() -> bool:
    return _otlp_receiver_started


def _tracing(test_suite: dict[str, Any] | None) -> dict[str, Any]:
    tracing = (test_suite or {}).get("tracing")
    return tracing if isinstance(tracing, dict) else {}


def _http_tracing(test_suite: dict[str, Any] | None) -> dict[str, Any]:
    otlp = _tracing(test_suite).get("otlp")
    http = otlp.get("http") if isinstance(otlp, dict) else None
    return http if isinstance(http, dict) else {}


def _metadata(test: Any) -> dict[str, Any]:
    metadata = test.get("metadata") if isinstance(test, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def is_suite_tracing_enabled(test_suite: dict[str, Any]) -> bool:
    """The suite-wide check used by the evaluator and the receiver startup: the env flag,
    `tracing.enabled`, or `metadata.tracingEnabled` on defaultTest or any test."""
    default_test = test_suite.get("defaultTest")
    return (
        get_env_bool("PROMPTFOO_TRACING_ENABLED", False)
        or _tracing(test_suite).get("enabled") is True
        or (
            isinstance(default_test, dict) and _metadata(default_test).get("tracingEnabled") is True
        )
        or any(_metadata(t).get("tracingEnabled") is True for t in test_suite.get("tests") or [])
    )


async def _prune_trace_store_if_needed(test_suite: dict[str, Any]) -> None:
    # Default retention is only materialized on parsed configs; ad-hoc suites opt in.
    storage = _tracing(test_suite).get("storage")
    retention_days = storage.get("retentionDays") if isinstance(storage, dict) else None
    if (
        not isinstance(retention_days, int | float)
        or isinstance(retention_days, bool)
        or retention_days <= 0
    ):
        return
    try:
        from promptphoo.tracing.store import get_trace_store

        await get_trace_store().delete_old_traces(retention_days)
        logger.debug(
            "[EvaluatorTracing] Pruned trace store entries older than %s days",
            js_string(retention_days),
        )
    except Exception as error:
        logger.warning("[EvaluatorTracing] Failed to prune old traces: %s", error)


def _register_trace_policy_if_available(trace_policy: dict[str, Any] | None) -> None:
    if not trace_policy:
        return
    from promptphoo.tracing.otlp_receiver import register_otlp_receiver_trace_policy

    register_otlp_receiver_trace_policy(trace_policy)


async def start_otlp_receiver_if_needed(
    test_suite: dict[str, Any], evaluation_id: str | None = None
) -> bool:
    """Make sure the shared OTLP receiver is running for this evaluation when tracing is on.

    The first caller starts it; later ones join it through a refcount. Registers this
    evaluation's trace policy, prunes expired traces when retention is configured, and honours
    `tracing.failOnReceiverStartFailure`. Returns whether this caller acquired the receiver
    (the caller then releases it with `stop_otlp_receiver_if_needed`).
    """
    global _otlp_receiver_started, _otlp_receiver_start_task, _otlp_receiver_users
    tracing = _tracing(test_suite)
    logger.debug(
        "[EvaluatorTracing] Checking tracing configuration %s",
        json_stringify({"tracing": _loggable_tracing(tracing), "testSuiteKeys": list(test_suite)}),
    )
    http_tracing = _http_tracing(test_suite)
    command_tool_names = tracing.get("commandToolNames", UNDEFINED)
    tracing_enabled = is_suite_tracing_enabled(test_suite)
    if tracing_enabled:
        await _prune_trace_store_if_needed(test_suite)

    if tracing_enabled and is_truthy(http_tracing.get("enabled")):
        requested = http_tracing.get("acceptFormats")
        normalized = [
            f
            for f in (requested if isinstance(requested, list) else [])
            if f in ("json", "protobuf")
        ]
        accept_formats = normalized or list(DEFAULT_OTLP_ACCEPT_FORMATS)
        redact_attributes = http_tracing.get("redactAttributes")
        trace_policy: dict[str, Any] | None = None
        if evaluation_id:
            trace_policy = {"evaluationId": evaluation_id}
            if command_tool_names is not UNDEFINED:
                trace_policy["commandToolNames"] = command_tool_names
            if redact_attributes is not None:
                trace_policy["redactAttributes"] = redact_attributes

        if _otlp_receiver_stop_task is not None:
            await _otlp_receiver_stop_task

        while True:
            if _otlp_receiver_started:
                _otlp_receiver_users += 1
                logger.debug(
                    "[EvaluatorTracing] OTLP receiver already started; preserving active "
                    "receiver defaults for %d active evaluations",
                    _otlp_receiver_users,
                )
                break
            pending = _otlp_receiver_start_task
            if pending is None:
                break
            try:
                await asyncio.shield(pending)
            except Exception:
                # The initiating evaluation applies its failure policy. A waiter may retry.
                pass
        if _otlp_receiver_started:
            _register_trace_policy_if_available(trace_policy)
            return True

        port = http_tracing.get("port") or 4318
        host = http_tracing.get("host") or "127.0.0.1"

        async def start() -> None:
            global _otlp_receiver_started, _otlp_receiver_users
            logger.debug(
                "[EvaluatorTracing] Tracing configuration detected, starting OTLP receiver"
            )
            from promptphoo.tracing.otlp_receiver import start_otlp_receiver

            logger.debug(
                "[EvaluatorTracing] Starting OTLP receiver on %s:%s%s",
                host,
                port,
                f" (redact: {','.join(map(js_string, redact_attributes))})"
                if isinstance(redact_attributes, list) and redact_attributes
                else "",
            )
            extra: dict[str, Any] = {}
            if command_tool_names is not UNDEFINED:
                extra["commandToolNames"] = command_tool_names
            if redact_attributes is not None:
                extra["redactAttributes"] = redact_attributes
            if trace_policy:
                extra["tracePolicy"] = trace_policy
            await start_otlp_receiver(port, host, accept_formats, extra)
            cli_state.set_active_otlp_receiver(
                {"host": host, "port": port, "acceptFormats": accept_formats}
            )
            _otlp_receiver_started = True
            _otlp_receiver_users += 1
            logger.info(
                "[EvaluatorTracing] OTLP receiver successfully started on port %s for tracing",
                port,
            )

        start_task = asyncio.ensure_future(start())
        _otlp_receiver_start_task = start_task
        try:
            await asyncio.shield(start_task)
        except Exception as error:
            message = str(error)
            if tracing.get("failOnReceiverStartFailure") is True:
                logger.error(
                    "[EvaluatorTracing] Failed to start OTLP receiver and "
                    "tracing.failOnReceiverStartFailure is true: %s",
                    message,
                )
                raise RuntimeError(
                    f"Failed to start OTLP tracing receiver: {message}. Set "
                    "tracing.failOnReceiverStartFailure: false to swallow this error and "
                    "continue without traces."
                ) from error
            logger.error("[EvaluatorTracing] Failed to start OTLP receiver: %s", message)
            return False
        finally:
            _otlp_receiver_start_task = None
        return True

    if _otlp_receiver_started:
        logger.debug("[EvaluatorTracing] OTLP receiver already started, skipping initialization")
    else:
        logger.debug("[EvaluatorTracing] Tracing not enabled or OTLP HTTP receiver not configured")
        logger.debug(
            "[EvaluatorTracing] tracing.enabled: %s", js_string(tracing.get("enabled", UNDEFINED))
        )
        logger.debug(
            "[EvaluatorTracing] tracing.otlp.http.enabled: %s",
            js_string(http_tracing.get("enabled", UNDEFINED)),
        )
    return False


def _loggable_tracing(tracing: dict[str, Any]) -> Any:
    provider = tracing.get("provider")
    if isinstance(provider, dict):
        return {**tracing, "provider": {"id": provider.get("id")}}
    return tracing or UNDEFINED


async def stop_otlp_receiver_if_needed(
    receiver_acquired: bool = True, evaluation_id: str | None = None
) -> None:
    """Release this evaluation's hold on the receiver and stop it when no one else uses it."""
    global _otlp_receiver_started, _otlp_receiver_stop_task, _otlp_receiver_users
    if not receiver_acquired:
        return
    # Drop this evaluation's trace policy so the shared receiver's map does not grow.
    if evaluation_id:
        from promptphoo.tracing.otlp_receiver import deregister_otlp_receiver_trace_policy

        deregister_otlp_receiver_trace_policy(evaluation_id)

    if _otlp_receiver_users > 0:
        _otlp_receiver_users -= 1
    if _otlp_receiver_users > 0:
        logger.debug(
            "[EvaluatorTracing] Preserving OTLP receiver for %d active evaluations",
            _otlp_receiver_users,
        )
        return

    if _otlp_receiver_started and _otlp_receiver_stop_task is None:
        _otlp_receiver_started = False

        async def stop() -> None:
            global _otlp_receiver_started
            try:
                logger.debug("[EvaluatorTracing] Stopping OTLP receiver")
                from promptphoo.tracing.otlp_receiver import stop_otlp_receiver

                await stop_otlp_receiver()
                cli_state.set_active_otlp_receiver()
                logger.info("[EvaluatorTracing] OTLP receiver stopped successfully")
            except Exception as error:
                _otlp_receiver_started = True
                logger.error("[EvaluatorTracing] Failed to stop OTLP receiver: %s", error)

        stop_task = asyncio.ensure_future(stop())
        _otlp_receiver_stop_task = stop_task
        try:
            await stop_task
        finally:
            _otlp_receiver_stop_task = None


def is_tracing_enabled(test: dict[str, Any], test_suite: dict[str, Any] | None = None) -> bool:
    """Tracing is on for a test when its metadata has `tracingEnabled: true`, the suite has
    `tracing.enabled: true`, or PROMPTFOO_TRACING_ENABLED is set."""
    metadata_enabled = _metadata(test).get("tracingEnabled") is True
    yaml_config_enabled = _tracing(test_suite).get("enabled") is True
    env_enabled = get_env_bool("PROMPTFOO_TRACING_ENABLED", False)
    result = metadata_enabled or yaml_config_enabled or env_enabled
    logger.debug(
        "[EvaluatorTracing] isTracingEnabled check: metadata=%s, yamlConfig=%s, env=%s, result=%s",
        js_string(metadata_enabled),
        js_string(yaml_config_enabled),
        js_string(env_enabled),
        js_string(result),
    )
    return result


async def generate_trace_context_if_needed(
    test: dict[str, Any],
    evaluate_options: dict[str, Any] | None,
    test_idx: int,
    prompt_idx: int,
    test_suite: dict[str, Any] | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Create the trace record and traceparent for one test case when tracing is enabled.

    `execution_metadata` takes upstream's keys: `providerId`, `promptLabel`, `repeatIndex`.
    """
    execution_metadata = execution_metadata or {}
    tracing_enabled = is_tracing_enabled(test, test_suite)
    if not tracing_enabled:
        return None
    logger.debug("[EvaluatorTracing] Tracing enabled for test case")
    logger.debug(
        "[EvaluatorTracing] Test metadata: %s", json_stringify(test.get("metadata", UNDEFINED))
    )

    logger.debug("[EvaluatorTracing] Importing trace store")
    from promptphoo.tracing.store import get_trace_store

    trace_store = get_trace_store()
    metadata = _metadata(test)
    evaluation_id = metadata.get("evaluationId") or (evaluate_options or {}).get("eventSource")
    if not evaluation_id:
        logger.warning(
            "[EvaluatorTracing] No evaluation ID found in test metadata or evaluateOptions, "
            "trace will not be linked to evaluation"
        )
        evaluation_id = f"eval-{int(time.time() * 1000)}"
    test_case_id = metadata.get("testCaseId") or test.get("id") or f"{test_idx}-{prompt_idx}"

    provider_id = execution_metadata.get("providerId")
    prompt_label = execution_metadata.get("promptLabel")
    repeat_index = execution_metadata.get("repeatIndex")
    root_attributes: dict[str, Any] = {
        SPAN_ROLE_ATTRIBUTE: "test_case",
        PromptfooAttributes.EVAL_ID: evaluation_id,
        PromptfooAttributes.TEST_INDEX: test_idx,
        "promptfoo.test_case.id": test_case_id,
        "promptfoo.prompt.index": prompt_idx,
    }
    if provider_id:
        root_attributes[PromptfooAttributes.PROVIDER_ID] = provider_id
    if prompt_label:
        root_attributes[PromptfooAttributes.PROMPT_LABEL] = prompt_label
    if repeat_index is not None:
        root_attributes["promptfoo.repeat.index"] = repeat_index

    candidate = get_genai_tracer().start_span(
        "promptfoo.test_case",
        SpanOptions(kind=SpanKind.INTERNAL, attributes=root_attributes),
        ROOT_CONTEXT,
    )
    root_context = candidate.span_context()
    # A non-recording tracer hands back an invalid span context; fall back to random ids.
    root_span = candidate if is_span_context_valid(root_context) else None
    trace_id = root_context.trace_id if root_span else generate_trace_id()
    span_id = root_context.span_id if root_span else generate_span_id()
    traceparent = generate_traceparent(
        trace_id,
        span_id,
        (root_context.trace_flags & TraceFlags.SAMPLED) != 0 if root_span else True,
    )
    logger.debug(
        "[EvaluatorTracing] Generated trace context: traceId=%s, spanId=%s", trace_id, span_id
    )

    try:
        logger.debug("[EvaluatorTracing] Creating trace record for traceId=%s", trace_id)
        trace_metadata: dict[str, Any] = {"testIdx": test_idx, "promptIdx": prompt_idx}
        if provider_id:
            trace_metadata["providerId"] = provider_id
        if repeat_index is not None:
            trace_metadata["repeatIndex"] = repeat_index
        trace_metadata["vars"] = test.get("vars", UNDEFINED)
        trace_metadata["commandToolNames"] = _tracing(test_suite).get("commandToolNames", UNDEFINED)
        trace_metadata["otlpHttpRedactAttributes"] = _http_tracing(test_suite).get(
            "redactAttributes", UNDEFINED
        )
        await trace_store.create_trace(
            {
                "traceId": trace_id,
                "evaluationId": evaluation_id or "",
                "testCaseId": test_case_id or "",
                "metadata": trace_metadata,
            }
        )
        logger.debug("[EvaluatorTracing] Trace record created successfully")
    except Exception as error:
        logger.error("[EvaluatorTracing] Failed to create trace: %s", error)

    logger.debug(
        "[EvaluatorTracing] Trace context ready: %s for test case %s", traceparent, test_case_id
    )
    context: dict[str, Any] = {
        "traceparent": traceparent,
        "evaluationId": evaluation_id,
        "testCaseId": test_case_id,
    }
    if root_span is not None:
        context["rootSpan"] = root_span
    return context
