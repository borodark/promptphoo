"""The eval loop. Ports runEval / Evaluator from src/evaluator.ts (the non-redteam, untraced
path). The select-best / max-score pass is in evaluator/comparison.py and the extension hooks
in evaluator/extensions.py. Derived metrics and tracing are not ported yet; configs that use
them warn explicitly.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from promptphoo import cache, cli_state, envars, python_hooks
from promptphoo.assertions.index import run_assertions
from promptphoo.evaluator.build import (
    build_tests_from_suite,
    generate_var_combinations,
    get_default_test,
)
from promptphoo.evaluator.render import FILE_METADATA_KEY, collect_file_metadata, render_prompt
from promptphoo.filters import filter_by_range
from promptphoo.grading_provider import (
    build_configured_provider_map,
    resolve_runtime_grading_provider_references,
)
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import UNDEFINED, json_stringify
from promptphoo.providers import canonicalize_provider_id, load_api_provider
from promptphoo.providers.base import ApiProvider, is_api_provider
from promptphoo.scheduler import RateLimitRegistry, create_provider_rate_limit_options
from promptphoo.scheduler.cancellation import sleep_with_abort
from promptphoo.scheduler.provider_call_execution_context import (
    with_provider_call_execution_context,
)
from promptphoo.scheduler.rate_limit_registry import sanitize_provider_id_for_log
from promptphoo.templating import template_references_variable
from promptphoo.tokens import (
    accumulate_grading_token_usage,
    accumulate_response_token_usage,
    create_empty_token_usage,
)
from promptphoo.transform import transform

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENCY = 4
CONVERSATION_VAR_NAME = "_conversation"
EVAL_RUNTIME_VAR_KEYS = ("__evalId", "__evalStepId", "__repeatIndex")


class ResultFailureReason:
    NONE = 0
    ASSERT = 1
    ERROR = 2


class EvaluationStore(Protocol):
    """What the evaluator needs from an Eval record (models/eval.py implements it)."""

    id: str
    prompts: list[dict[str, Any]]
    persisted: bool
    results: list[Any]  # in-memory EvalResult records when not persisted

    result_persistence_failed: bool

    async def append_prompts(self, prompts: list[dict[str, Any]]) -> None: ...
    async def append_result(self, row: dict[str, Any]) -> None: ...
    async def read_results_by_test_idx(self, test_idx: int) -> list[Any]: ...
    async def read_completed_index_pairs(self, exclude_errors: bool = False) -> set[str]: ...
    async def read_failed_results_by_test_idx(self, test_idx: int) -> list[Any]: ...
    def record_result_persistence_failure(self, row: dict[str, Any]) -> None: ...
    def has_result_persistence_failure(self, result: Any) -> bool: ...
    async def save_result(self, result: Any) -> None: ...
    def set_vars(self, vars_: list[str]) -> None: ...
    def set_duration_ms(self, ms: int) -> None: ...
    async def save(self) -> None: ...


# --- small helpers --------------------------------------------------------------------------


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_id_from_prompt(prompt: dict[str, Any]) -> str:
    if prompt.get("label"):
        return sha256(prompt["label"])
    if prompt.get("id"):
        return sha256(prompt["id"])
    raw = prompt.get("raw")
    return sha256(raw if isinstance(raw, str) else json_stringify(raw))


def create_default_prompt_metrics() -> dict[str, Any]:
    return {
        "score": 0,
        "testPassCount": 0,
        "testFailCount": 0,
        "testErrorCount": 0,
        "testCacheHitCount": 0,
        "assertPassCount": 0,
        "assertFailCount": 0,
        "totalLatencyMs": 0,
        "tokenUsage": create_empty_token_usage(),
        "namedScores": {},
        "namedScoresCount": {},
        "namedScoreWeights": {},
        "cost": 0,
    }


def get_provider_identifier(provider: ApiProvider) -> str:
    return provider.label or provider.id()


def _ref_matches(ref: str, label: str | None, pid: str | None) -> bool:
    if label == ref or (pid and pid == ref):
        return True
    if ref.endswith("*"):
        prefix = ref[:-1]
        if (label and label.startswith(prefix)) or (pid and pid.startswith(prefix)):
            return True
    return bool((label and label.startswith(f"{ref}:")) or (pid and pid.startswith(f"{ref}:")))


def is_allowed_prompt(prompt: dict[str, Any], allowed: list[str] | None) -> bool:
    if not isinstance(allowed, list):
        return True
    return any(_ref_matches(r, prompt.get("label"), prompt.get("id")) for r in allowed)


def is_provider_allowed(provider: ApiProvider, allowed: list[str] | None) -> bool:
    if not isinstance(allowed, list):
        return True
    pid = provider.id()
    label = provider.label
    canonical = canonicalize_provider_id(pid)
    for ref in allowed:
        if label and label == ref:
            return True
        if pid == ref or canonical == canonicalize_provider_id(ref):
            return True
        if ref.endswith("*") and (
            (label and label.startswith(ref[:-1]))
            or pid.startswith(ref[:-1])
            or canonical.startswith(ref[:-1])
        ):
            return True
        if (
            (label and label.startswith(f"{ref}:"))
            or pid.startswith(f"{ref}:")
            or (canonical.startswith(f"{ref}:"))
        ):
            return True
    return False


_conversation_cache: dict[str, bool] = {}


def prompt_uses_conversation_variable(prompt: dict[str, Any]) -> bool:
    raw = prompt.get("raw") or ""
    if raw not in _conversation_cache:
        try:
            _conversation_cache[raw] = template_references_variable(raw, CONVERSATION_VAR_NAME)
        except Exception:
            return False
    return _conversation_cache[raw]


def get_repeat_cache_namespace(repeat_index: int, options: dict[str, Any]) -> str | None:
    if repeat_index > 0 or (options.get("repeat") or 1) > 1:
        return f"repeat:{repeat_index}"
    return None


def _normalize_repeat(value: Any, fallback: int = 1) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else (fallback)
    )


def _omit_runtime_vars(vars_: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in vars_.items() if k not in EVAL_RUNTIME_VAR_KEYS}


def _error_with_stack(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}" if not str(error) else str(error)
    stack = "".join(traceback.format_exception(error))
    return f"{message}\n\n{stack}".rstrip("\n")


def _session_id(response: dict[str, Any], vars_: dict[str, Any]) -> str | None:
    for value in (response.get("sessionId"), vars_.get("sessionId")):
        if value not in (None, ""):
            return value if isinstance(value, str) else json_stringify(value)
    return None


# --- resume: saved prompt columns -----------------------------------------------------------


def _invariant(condition: Any, message: str) -> None:
    """util/invariant"""
    if not condition:
        raise RuntimeError(f"Invariant failed: {message}")


def _defined(value: Any) -> Any:
    return None if value is UNDEFINED else value


def backfill_named_score_weights(metrics: dict[str, Any]) -> None:
    """backfillNamedScoreWeights: metrics saved before weights existed count each assertion
    once."""
    weights = metrics.get("namedScoreWeights") or {}
    metrics["namedScoreWeights"] = weights
    for name, count in (metrics.get("namedScoresCount") or {}).items():
        if name not in weights:
            weights[name] = count


def matches_saved_prompt_layout(
    prompts: list[dict[str, Any]], saved: list[dict[str, Any]], allow_template_difference: bool
) -> bool:
    from promptphoo.jsonutil import js_deep_equal

    if len(prompts) != len(saved):
        return False

    def same(key: str, a: dict[str, Any], b: dict[str, Any]) -> bool:
        return js_deep_equal(_defined(a.get(key)), _defined(b.get(key)))

    for prompt, old in zip(prompts, saved, strict=True):
        if (
            not same("provider", prompt, old)
            or not same("label", prompt, old)
            or not same("raw", prompt, old)
            or not (same("template", prompt, old) or allow_template_difference)
            or not same("config", prompt, old)
        ):
            return False
    return True


def can_restore_saved_text(prompts: list[dict[str, Any]], saved: list[dict[str, Any]]) -> bool:
    """canRestoreSavedText: saved text snapshots replay unless they overlap a callable prompt."""
    if not any(p.get("function") for p in prompts):
        return True
    text_raw: set[Any] = set()
    callable_raw: set[Any] = set()
    callable_labels: set[Any] = set()
    for prompt in prompts:
        if prompt.get("function"):
            callable_raw.add(prompt.get("raw"))
            callable_labels.add(prompt.get("label"))
        else:
            text_raw.add(prompt.get("raw"))
    return all(
        s.get("raw") not in callable_raw
        and s.get("label") not in callable_labels
        and s.get("raw") in text_raw
        for s in saved
    )


def prompt_identity_candidates(prompts: list[dict[str, Any]]) -> dict[str, dict[Any, None]]:
    """promptIdentityCandidates: generated id + raw + label -> the authored ids sharing it."""
    identities: dict[str, dict[Any, None]] = {}
    for prompt in prompts:
        key = json_stringify([generate_id_from_prompt(prompt), prompt.get("raw"),
                              prompt.get("label")])  # fmt: skip
        identities.setdefault(key, {})[prompt.get("id")] = None
    return identities


async def filter_completed_resume_steps(steps: list[Any], store: EvaluationStore) -> None:
    """filterCompletedResumeSteps: drop the (testIdx, promptIdx) pairs already saved."""
    if not cli_state.resume or not store.persisted:
        return
    try:
        completed = await store.read_completed_index_pairs(exclude_errors=cli_state.retry_mode)
        original = len(steps)
        steps[:] = [s for s in steps if f"{s.test_idx}:{s.prompt_idx}" not in completed]
        skipped = original - len(steps)
        if skipped > 0:
            logger.info("Resuming: skipping %d previously completed cases", skipped)
    except Exception as err:
        logger.warning("Resume: failed to load completed results. Running full evaluation. %s", err)


def is_grading_abort(error: BaseException, signal: Any) -> bool:
    from promptphoo.fetch.errors import is_abort_error

    return bool(
        signal is not None
        and signal.aborted
        and (is_abort_error(error) or error is signal.reason
             or isinstance(error, asyncio.CancelledError))
    )  # fmt: skip


async def await_with_abort(awaitable: Any, signal: Any) -> Any:
    """Run `awaitable` as a task that is cancelled when `signal` aborts, raising the abort
    reason. Stands in for upstream's AbortSignal plumbing through provider calls."""
    if signal is None:
        return await awaitable
    if signal.aborted:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()  # never started
        signal.throw_if_aborted()
    task = asyncio.ensure_future(awaitable)

    def on_abort() -> None:
        task.cancel()

    signal.add_listener(on_abort)
    try:
        return await task
    except asyncio.CancelledError:
        if signal.aborted and task.cancelled():
            raise signal.reason from None
        raise
    finally:
        signal.remove_listener(on_abort)


def combine_signals(*signals: Any) -> Any:
    """AbortSignal.any"""
    from promptphoo.scheduler.cancellation import AbortController

    present = [s for s in signals if s is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    controller = AbortController()
    for s in present:
        if s.aborted:
            controller.abort(s.reason)
            return controller.signal
    for s in present:
        s.add_listener(lambda s=s: controller.abort(s.reason))
    return controller.signal


# --- run options ----------------------------------------------------------------------------


@dataclass
class RunEvalOptions:
    provider: ApiProvider
    prompt: dict[str, Any]
    test: dict[str, Any]
    test_suite: dict[str, Any]
    test_idx: int
    prompt_idx: int
    repeat_index: int
    evaluate_options: dict[str, Any]
    conversations: dict[str, list[Any]]
    registers: dict[str, Any]
    eval_id: str
    delay: float = 0
    filters: dict[str, Any] | None = None
    repeat_group_id: str | None = None
    rate_limit_registry: RateLimitRegistry | None = None


def _provider_setup(provider: ApiProvider) -> dict[str, Any]:
    out: dict[str, Any] = {"id": provider.id()}
    if provider.label is not None:
        out["label"] = provider.label
    if provider.config is not None:
        out["config"] = provider.config
    return out


def _merge_prompt_config(prompt_config: Any, test_options: Any) -> dict[str, Any]:
    options = {k: v for k, v in (test_options or {}).items() if k != "repeat"}
    return {**(prompt_config or {}), **options}


# --- single step ----------------------------------------------------------------------------


async def run_eval(step: RunEvalOptions) -> list[dict[str, Any]]:
    """runEval: one provider call plus grading, returned as EvaluateResult rows."""
    enabled = False if step.evaluate_options.get("cache") is False else cache.is_cache_enabled()
    with (
        cache.with_cache_enabled(enabled),
        cache.with_cache_namespace(
            get_repeat_cache_namespace(step.repeat_index, step.evaluate_options)
        ),
    ):
        return await _run_eval_internal(step)


async def _run_eval_internal(step: RunEvalOptions) -> list[dict[str, Any]]:
    provider, prompt, test = step.provider, step.prompt, step.test
    if provider.delay is None:
        provider.delay = step.delay or envars.get_env_int("PROMPTFOO_DELAY_MS", 0)

    vars_ = copy.deepcopy(test.get("vars") or {})
    file_metadata = collect_file_metadata(vars_)
    conversation_key = json_stringify(
        [get_provider_identifier(provider), prompt.get("id"), step.prompt_idx,
         (test.get("metadata") or {}).get("conversationId")]
    )  # fmt: skip
    prompt_config = _merge_prompt_config(prompt.get("config"), test.get("options"))
    setup: dict[str, Any] = {
        "provider": _provider_setup(provider),
        "prompt": {"raw": "", "label": prompt.get("label"), "config": prompt_config},
        "vars": vars_,
    }
    if (
        not envars.get_env_bool("PROMPTFOO_DISABLE_CONVERSATION_VAR")
        and not (test.get("options") or {}).get("disableConversationVar")
        and prompt_uses_conversation_variable(prompt)
    ):
        vars_[CONVERSATION_VAR_NAME] = step.conversations.get(conversation_key) or []
    vars_.update(step.registers)
    if step.eval_id:
        vars_["__evalId"] = step.eval_id
    vars_["__evalStepId"] = (
        f"test-{step.test_idx}-prompt-{step.prompt_idx}-repeat-{step.repeat_index}"
    )
    vars_["__repeatIndex"] = step.repeat_index

    latency_ms = 0.0
    prompt_for_render = dict(prompt)
    try:
        rendered_prompt = await render_prompt(prompt_for_render, vars_, step.filters, provider)
        prompt_config = _merge_prompt_config(prompt_for_render.get("config"), test.get("options"))
        setup = {
            "provider": _provider_setup(provider),
            "prompt": {"raw": rendered_prompt, "label": prompt_for_render.get("label"),
                       "config": prompt_config},
            "vars": vars_,
        }  # fmt: skip
        try:
            rendered_json = js_json_parse(rendered_prompt)
        except ValueError:
            rendered_json = None

        trace_context = (
            None
            if test.get("providerOutput") is not None
            else await _generate_trace_context(step, prompt_for_render)
        )

        async def run_execution() -> list[dict[str, Any]]:
            nonlocal latency_ms
            start = time.monotonic()
            if test.get("providerOutput") is not None:
                response: dict[str, Any] = {
                    "output": test["providerOutput"],
                    "tokenUsage": create_empty_token_usage(),
                    "cost": 0,
                    "cached": False,
                }
            else:
                active = test["provider"] if is_api_provider(test.get("provider")) else provider
                context = {
                    "vars": vars_,
                    "prompt": _context_prompt(prompt_for_render, prompt_config),
                    "filters": step.filters,
                    "originalProvider": provider,
                    "test": test,
                    "getCache": cache.get_cache,
                    "repeatIndex": step.repeat_index,
                    "testIdx": step.test_idx,
                }
                if not cache.is_cache_enabled():
                    context["bustCache"] = True
                if step.eval_id:
                    context["evaluationId"] = step.eval_id
                if trace_context:
                    context["traceparent"] = trace_context.get("traceparent")
                    context["evaluationId"] = trace_context.get("evaluationId")
                    context["testCaseId"] = trace_context.get("testCaseId")
                response = await _call_active_provider(
                    active,
                    rendered_prompt,
                    context,
                    step.rate_limit_registry,
                    trace_context=trace_context,
                    prompt_label=prompt_for_render.get("label"),
                    test_idx=step.test_idx,
                )
                if not isinstance(response, dict):
                    raise TypeError(f"Provider returned {type(response).__name__}, expected a dict")
            latency_ms = round((time.monotonic() - start) * 1000)
            if response.get("metadata"):
                response["metadata"] = js_json_parse(json_stringify(response["metadata"]))
            response = _normalize_cached_target_response(response)

            if not response.get("error"):
                history = step.conversations.setdefault(conversation_key, [])
                last_input = None
                if isinstance(rendered_json, list) and rendered_json:
                    last = rendered_json[-1]
                    last_input = (last.get("content") if isinstance(last, dict) else None) or last
                conversation_output = response.get("conversationOutput")
                history.append({
                    "prompt": rendered_json or rendered_prompt,
                    "input": last_input or rendered_json or rendered_prompt,
                    "output": conversation_output if conversation_output is not None
                    else (response.get("output") or ""),
                    "metadata": response.get("metadata"),
                })  # fmt: skip

            if (
                not response.get("cached")
                and not provider.handles_own_delay
                and provider.delay
                and provider.delay > 0
            ):
                await sleep_with_abort(provider.delay)

            persisted_vars = _omit_runtime_vars(vars_)
            ret: dict[str, Any] = {
                **setup,
                "vars": persisted_vars,
                "prompt": {**setup["prompt"], "raw": rendered_prompt},
                "response": response,
                "success": False,
                "failureReason": ResultFailureReason.NONE,
                "score": 0,
                "namedScores": {},
                "latencyMs": response.get("latencyMs") if response.get("latencyMs") is not None
                else latency_ms,
                "cost": response.get("cost"),
            }  # fmt: skip
            if response.get("incurredCost") is not None:
                ret["incurredCost"] = response["incurredCost"]
            ret["metadata"] = {
                **(test.get("metadata") or {}),
                **(response.get("metadata") or {}),
                FILE_METADATA_KEY: file_metadata,
            }
            ret["promptIdx"] = step.prompt_idx
            if step.repeat_group_id is not None:
                ret["repeatGroupId"] = step.repeat_group_id
                ret["repeatIndex"] = step.repeat_index
            ret["testIdx"] = step.test_idx
            ret["testCase"] = test
            ret["promptId"] = prompt.get("id") or ""
            ret["tokenUsage"] = create_empty_token_usage()
            if not ret["metadata"].get("sessionIds") and not ret["metadata"].get("sessionId"):
                session = _session_id(response, persisted_vars)
                if session is not None:
                    ret["metadata"]["sessionId"] = session

            await _apply_response_outcome(step, ret, response, rendered_prompt, persisted_vars,
                                          latency_ms, trace_context)  # fmt: skip
            if response.get("tokenUsage"):
                accumulate_response_token_usage(ret["tokenUsage"], response)
            store_as = (test.get("options") or {}).get("storeOutputAs")
            if store_as and (ret.get("response") or {}).get("output"):
                step.registers[store_as] = ret["response"]["output"]
            return [ret]

        return await _run_with_test_case_span(trace_context, step.test_idx, run_execution)
    except asyncio.CancelledError:
        raise
    except Exception as err:
        logger.error("Provider call failed during eval: %s", err)
        logger.debug("%s", traceback.format_exc())
        row = {
            **setup,
            "vars": _omit_runtime_vars(setup["vars"]),
            "error": _error_with_stack(err),
            "success": False,
            "failureReason": ResultFailureReason.ERROR,
            "score": 0,
            "namedScores": {},
            "latencyMs": latency_ms,
            "promptIdx": step.prompt_idx,
        }
        if step.repeat_group_id is not None:
            row["repeatGroupId"] = step.repeat_group_id
            row["repeatIndex"] = step.repeat_index
        row.update(
            {
                "testIdx": step.test_idx,
                "testCase": test,
                "promptId": prompt.get("id") or "",
                "metadata": {
                    **(test.get("metadata") or {}),
                    "errorContext": {
                        "providerId": provider.id(),
                        **({"providerLabel": provider.label} if provider.label else {}),
                    },
                },
            }
        )
        return [row]


async def _call_active_provider(
    active_provider: ApiProvider,
    rendered_prompt: str,
    call_api_context: dict[str, Any],
    rate_limit_registry: RateLimitRegistry | None,
    abort_signal: Any = None,
    *,
    trace_context: dict[str, Any] | None = None,
    prompt_label: str | None = None,
    test_idx: int | None = None,
) -> dict[str, Any]:
    """callActiveProvider: route the target call through the eval's rate-limit registry, which
    queues it per rate-limit key, learns quota from response headers, adapts concurrency and
    retries rate-limited or transient failures."""
    logger.debug("Provider type: %s", sanitize_provider_id_for_log(active_provider.id()))

    async def call_api(on_response_headers: Any = None) -> dict[str, Any]:
        call_api_options = (
            {
                **({"abortSignal": abort_signal} if abort_signal else {}),
                **({"onResponseHeaders": on_response_headers} if on_response_headers else {}),
            }
            if abort_signal or on_response_headers
            else None
        )
        return await with_provider_call_execution_context(
            {
                "abortSignal": abort_signal,
                "rateLimitRegistry": rate_limit_registry,
                "rateLimitProvider": active_provider,
            },
            lambda: _invoke_target(
                active_provider, rendered_prompt, call_api_context, call_api_options,
                trace_context, prompt_label, test_idx,
            ),
        )  # fmt: skip

    if rate_limit_registry:
        response = await rate_limit_registry.execute(
            active_provider, call_api, create_provider_rate_limit_options(abort_signal)
        )
    else:
        response = await call_api()
    if isinstance(response, dict):
        logger.debug("Provider response properties: %s", ", ".join(response))
        logger.debug("Provider response cached property explicitly: %s", response.get("cached"))
    return response


async def _invoke_target(
    active_provider: ApiProvider,
    rendered_prompt: str,
    call_api_context: dict[str, Any],
    call_api_options: dict[str, Any] | None,
    trace_context: dict[str, Any] | None,
    prompt_label: str | None,
    test_idx: int | None,
) -> dict[str, Any]:
    """The target call, inside a target span when the step is traced (withTracedProviderCall)."""
    if not (trace_context or {}).get("traceparent"):
        return await active_provider.call_api(rendered_prompt, call_api_context, call_api_options)
    from promptphoo.tracing.target_tracer import with_traced_provider_call

    return await with_traced_provider_call(
        {
            "provider": active_provider,
            "callContext": call_api_context,
            "promptLabel": prompt_label,
            "evalId": call_api_context.get("evaluationId"),
            "testIndex": test_idx,
        },
        lambda context: active_provider.call_api(rendered_prompt, context, call_api_options),
    )


async def _generate_trace_context(
    step: RunEvalOptions, prompt_for_render: dict[str, Any]
) -> dict[str, Any] | None:
    """generateTraceContextIfNeeded for one step: the test-case root span and trace record."""
    from promptphoo.tracing.evaluator_tracing import generate_trace_context_if_needed

    test = step.test
    provider = test["provider"] if is_api_provider(test.get("provider")) else step.provider
    return await generate_trace_context_if_needed(
        test,
        step.evaluate_options,
        step.test_idx,
        step.prompt_idx,
        step.test_suite,
        {
            "providerId": provider.id(),
            "promptLabel": prompt_for_render.get("label"),
            "repeatIndex": step.repeat_index,
        },
    )


async def _run_with_test_case_span(
    trace_context: dict[str, Any] | None,
    test_idx: int,
    run_execution: Any,
) -> list[dict[str, Any]]:
    """withTestCaseSpan around the step, with the tracing hooks graders read from context."""
    if not trace_context:
        return await run_execution()
    from promptphoo.scheduler.provider_call_execution_context import (
        with_provider_call_tracing_context,
    )
    from promptphoo.tracing.span_roles import get_active_traceparent
    from promptphoo.tracing.target_tracer import (
        with_grader_span,
        with_test_case_span,
        with_traced_provider_call,
    )

    return await with_provider_call_tracing_context(
        {
            "getActiveTraceparent": get_active_traceparent,
            "testIndex": test_idx,
            "withGraderSpan": with_grader_span,
            "withProviderSpan": with_traced_provider_call,
        },
        lambda: with_test_case_span(trace_context.get("rootSpan"), run_execution),
    )


def _trace_id(trace_context: dict[str, Any] | None) -> str | None:
    """getTraceId: the trace id of a well-formed version-00 traceparent."""
    traceparent = (trace_context or {}).get("traceparent")
    if not traceparent:
        return None
    parts = traceparent.split("-")
    if len(parts) == 4 and len(parts[1]) == 32:
        return parts[1]
    return None


def _is_tracing_enabled_for_test(test_suite: dict[str, Any], test_case: dict[str, Any]) -> bool:
    return (
        envars.get_env_bool("PROMPTFOO_TRACING_ENABLED", False)
        or (test_case.get("metadata") or {}).get("tracingEnabled") is True
        or (test_suite.get("tracing") or {}).get("enabled") is True
    )


def _create_run_eval_test(
    test_suite: dict[str, Any], test_case: dict[str, Any], vars_: Any, eval_id: str
) -> dict[str, Any]:
    """createRunEvalTest: traced tests carry tracingEnabled and the eval id in metadata."""
    base_test = {**test_case, "vars": vars_}
    if not _is_tracing_enabled_for_test(test_suite, test_case):
        return base_test
    return {
        **base_test,
        "metadata": {
            **(test_case.get("metadata") or {}),
            "tracingEnabled": True,
            "evaluationId": eval_id,
        },
    }


def _suite_tracing_enabled(test_suite: dict[str, Any]) -> bool:
    """The evaluator's check for initializing OTEL: env, suite, defaultTest or any test."""
    default_test = test_suite.get("defaultTest")
    return (
        envars.get_env_bool("PROMPTFOO_TRACING_ENABLED", False)
        or (test_suite.get("tracing") or {}).get("enabled") is True
        or (
            isinstance(default_test, dict)
            and (default_test.get("metadata") or {}).get("tracingEnabled") is True
        )
        or any(
            ((t or {}).get("metadata") or {}).get("tracingEnabled") is True
            for t in (test_suite.get("tests") or [])
            if isinstance(t, dict)
        )
    )


def _context_prompt(prompt: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The prompt as Node's call context shows it: `config` sits before `template`."""
    out: dict[str, Any] = {}
    for key, value in prompt.items():
        if key == "template" and "config" not in out:
            out["config"] = config
        if key != "config":
            out[key] = value
    out.setdefault("config", config)
    return out


def _normalize_cached_target_response(response: dict[str, Any]) -> dict[str, Any]:
    if not response.get("cached") and not response.get("tokenUsage"):
        return response
    usage = create_empty_token_usage()
    accumulate_response_token_usage(usage, response)
    out = {**response, "tokenUsage": usage}
    if response.get("cost") is not None and (
        response.get("cached") or response.get("incurredCost") is not None
    ):
        incurred = response.get("incurredCost")
        out["incurredCost"] = (
            incurred
            if incurred is not None
            else (0 if response.get("cached") else response["cost"])
        )
    return out


async def _apply_response_outcome(
    step: RunEvalOptions,
    ret: dict[str, Any],
    response: dict[str, Any],
    rendered_prompt: str,
    vars_: dict[str, Any],
    latency_ms: float,
    trace_context: dict[str, Any] | None = None,
) -> None:
    if response.get("error"):
        ret["error"] = response["error"]
        ret["failureReason"] = ResultFailureReason.ERROR
        ret["success"] = False
        return
    if response.get("output") is None:
        ret.update(success=False, score=0, error="No output",
                   failureReason=ResultFailureReason.ERROR)  # fmt: skip
        return

    provider, test = step.provider, step.test
    processed = dict(response)
    if provider.transform:
        processed["output"] = await transform(
            provider.transform, processed["output"], {"vars": vars_, "prompt": step.prompt}
        )
    provider_transformed_output = processed["output"]
    test_transform = (test.get("options") or {}).get("transform") or (
        test.get("options") or {}
    ).get("postprocess")
    if test_transform:
        processed["output"] = await transform(
            test_transform,
            processed["output"],
            {"vars": vars_, "prompt": step.prompt,
             **({"metadata": response["metadata"]} if response.get("metadata") else {})},
        )  # fmt: skip
    if processed.get("output") is None:
        raise ValueError("Response output should not be null")

    assertion_response = {**processed, "providerTransformedOutput": provider_transformed_output}
    check = await with_provider_call_execution_context(
        {"abortSignal": None, "rateLimitRegistry": step.rate_limit_registry},
        lambda: run_assertions(
            prompt=rendered_prompt,
            provider=provider,
            provider_response=assertion_response,
            test=test,
            vars=vars_,
            latency_ms=response.get("latencyMs") if response.get("latencyMs") is not None
            else latency_ms,
            assert_scoring_function=test.get("assertScoringFunction"),
            trace_id=_trace_id(trace_context),
        ),
    )  # fmt: skip
    if not check["pass"]:
        ret["error"] = check.get("reason")
        ret["failureReason"] = ResultFailureReason.ASSERT
    ret["success"] = check["pass"]
    ret["score"] = check["score"]
    ret["namedScores"] = check.get("namedScores") or {}
    if not ret.get("tokenUsage"):
        ret["tokenUsage"] = create_empty_token_usage()
    accumulate_grading_token_usage(
        ret["tokenUsage"], check.get("tokensUsed"),
        cached=(check.get("metadata") or {}).get("cachedResponse"),
    )  # fmt: skip
    ret["gradingResult"] = check
    ret["response"] = processed


# --- the evaluator --------------------------------------------------------------------------


@dataclass
class EvalStats:
    successes: int = 0
    failures: int = 0
    errors: int = 0
    token_usage: dict[str, Any] = field(default_factory=create_empty_token_usage)


async def _apply_input_transform(test: dict[str, Any], default_transform: Any) -> None:
    input_transform = (test.get("options") or {}).get("transformVars") or default_transform
    if not input_transform:
        return
    transformed = await transform(
        input_transform, test["vars"], {"prompt": {}, "uuid": str(uuid.uuid4())}, True, "vars"
    )
    if not isinstance(transformed, dict):
        raise ValueError("Transform function did not return a valid object")
    test["vars"] = {**test["vars"], **transformed}


async def _prepare_test_case(test_suite: dict[str, Any], test: dict[str, Any], index: int) -> None:
    default = get_default_test(test_suite) or {}
    if not isinstance(default.get("assert") or [], list):
        raise ValueError(f"defaultTest.assert is not an array in test case #{index + 1}")
    if not isinstance(test.get("assert") or [], list):
        raise ValueError(f"testCase.assert is not an array in test case #{index + 1}")
    disable_defaults = (test.get("options") or {}).get("disableDefaultAsserts") is True
    test["assert"] = [*([] if disable_defaults else default.get("assert") or []),
                      *(test.get("assert") or [])]  # fmt: skip
    if test.get("threshold") is None and default.get("threshold") is not None:
        test["threshold"] = default["threshold"]
    test["options"] = {**(default.get("options") or {}), **(test.get("options") or {})}
    test["metadata"] = {**(default.get("metadata") or {}), **(test.get("metadata") or {})}
    if test.get("prompts") is None and default.get("prompts") is not None:
        test["prompts"] = default["prompts"]
    if not test.get("provider") and default.get("provider"):
        dp = default["provider"]
        if isinstance(dp, dict) and dp.get("id"):
            test["provider"] = await load_api_provider(dp["id"], options=dp)
        else:
            test["provider"] = dp
    if not test.get("assertScoringFunction") and default.get("assertScoringFunction"):
        test["assertScoringFunction"] = default["assertScoringFunction"]
    if test.get("providers") is None and default.get("providers") is not None:
        test["providers"] = default["providers"]
    scoring = test.get("assertScoringFunction")
    if isinstance(scoring, str):
        path, fn = python_hooks.split_path_and_function(scoring)
        if not path.endswith(".py"):
            raise ValueError(
                f"assertScoringFunction must be a Python file in promptphoo, got {scoring!r}"
            )
        test["assertScoringFunction"] = python_hooks.load_function(path, fn or "get_score")


class Evaluator:
    def __init__(
        self, test_suite: dict[str, Any], store: EvaluationStore, options: dict[str, Any]
    ) -> None:
        self.test_suite = test_suite
        self.store = store
        self.options = options
        self.stats = EvalStats()
        self.conversations: dict[str, list[Any]] = {}
        self.registers: dict[str, Any] = {}
        self.retry_error_result_ids: set[str] = set(
            (cli_state.retry_error_result_ids or []) if cli_state.retry_mode else []
        )
        from promptphoo.evaluator.comparison import ComparisonPass

        self.comparison = ComparisonPass(self)

        # Create rate limit registry for adaptive concurrency control
        self.rate_limit_registry: RateLimitRegistry | None = RateLimitRegistry(
            {"maxConcurrency": options.get("maxConcurrency") or DEFAULT_MAX_CONCURRENCY}
        )

        # Add debug logging for rate limit events
        self.rate_limit_registry.on(
            "ratelimit:hit",
            lambda data: logger.debug(
                "[Scheduler] Rate limit hit for %s %s",
                data["rateLimitKey"],
                json_stringify(
                    {
                        "retryAfterMs": data.get("retryAfterMs"),
                        "resetAt": data.get("resetAt"),
                        "concurrencyChange": data.get("concurrencyChange"),
                    }
                ),
            ),
        )
        self.rate_limit_registry.on(
            "ratelimit:learned",
            lambda data: logger.debug(
                "[Scheduler] Learned rate limits for %s %s",
                data["rateLimitKey"],
                json_stringify(
                    {"requestLimit": data.get("requestLimit"), "tokenLimit": data.get("tokenLimit")}
                ),
            ),
        )
        self.rate_limit_registry.on(
            "concurrency:decreased",
            lambda data: logger.debug(
                "[Scheduler] Concurrency decreased for %s %s",
                data["rateLimitKey"],
                json_stringify({"previous": data["previous"], "current": data["current"]}),
            ),
        )
        self.rate_limit_registry.on("concurrency:increased", lambda data: logger.debug(
            "[Scheduler] Concurrency increased for %s %s", data["rateLimitKey"],
            json_stringify({"previous": data["previous"], "current": data["current"]}),
        ))  # fmt: skip

    def _allowed_prompts(self, provider: ApiProvider) -> Any:
        """getProviderPromptOverride(...) ?? provider.prompts"""
        prompt_map = self.test_suite.get("providerPromptMap") or {}
        allowed = None
        if provider.label and provider.label in prompt_map:
            allowed = prompt_map[provider.label]
        elif provider.id() in prompt_map:
            allowed = prompt_map[provider.id()]
        return provider.prompts if allowed is None else allowed

    def _build_prompt_columns(self) -> tuple[list[dict], list[tuple[ApiProvider, list]]]:
        """buildLivePromptColumns"""
        prompts: list[dict[str, Any]] = []
        columns_by_provider: list[tuple[ApiProvider, list[tuple[int, dict]]]] = []
        for provider in self.test_suite["providers"]:
            key = get_provider_identifier(provider)
            allowed = self._allowed_prompts(provider)
            columns: list[tuple[int, dict[str, Any]]] = []
            for prompt in self.test_suite["prompts"]:
                if not is_allowed_prompt(prompt, allowed):
                    continue
                columns.append((len(prompts), prompt))
                # Node prompt objects carry `id` first (often undefined), so it serializes first.
                completed = {"id": None, **{k: v for k, v in prompt.items() if k != "function"}}
                completed.setdefault("config", UNDEFINED)  # Node prompts carry config: undefined
                completed.update(
                    id=generate_id_from_prompt(prompt), provider=key, label=prompt.get("label"),
                    metrics=create_default_prompt_metrics(),
                )  # fmt: skip
                prompts.append(completed)
            columns_by_provider.append((provider, columns))
        return prompts, columns_by_provider

    def _build_completed_prompts(
        self, restore_prompt_columns: bool, tests: list[dict[str, Any]]
    ) -> tuple[list[dict], list[tuple[ApiProvider, list]]]:
        """buildCompletedPrompts: on resume, restore the saved columns (and their metrics) so
        result indices keep pointing at the same column. The extension-hook recovery path is
        not ported (extensions are not supported)."""
        suite, store = self.test_suite, self.store
        providers: list[ApiProvider] = suite["providers"]
        provider_indices: dict[str, int] = {}
        for index, provider in enumerate(providers):
            provider_indices[get_provider_identifier(provider)] = index
        can_restore_text = not restore_prompt_columns or can_restore_saved_text(
            suite["prompts"], store.prompts
        )
        prompts, columns_by_provider = self._build_prompt_columns()
        matches_saved_layout = matches_saved_prompt_layout(
            prompts, store.prompts, restore_prompt_columns and can_restore_text
        )
        restored_prompts: list[dict[str, Any]] = []
        if restore_prompt_columns:
            for saved in store.prompts:
                metrics = (
                    copy.deepcopy(saved["metrics"])
                    if saved.get("metrics")
                    else create_default_prompt_metrics()
                )
                backfill_named_score_weights(metrics)
                restored_prompts.append({**saved, "metrics": metrics})
        if restore_prompt_columns and matches_saved_layout:
            # Keep positional ids and each saved column's own metrics, even when provider or
            # prompt identifiers are duplicated.
            columns_by_provider = [
                (provider, [(idx, {**prompt, "template": store.prompts[idx].get("template")})
                            for idx, prompt in columns])
                for provider, columns in columns_by_provider
            ]  # fmt: skip
            return restored_prompts, columns_by_provider

        if restore_prompt_columns and can_restore_text and len(provider_indices) == len(providers):
            return restored_prompts, self._restore_columns_by_identity(
                restored_prompts, provider_indices, tests
            )

        if cli_state.resume and store.persisted and store.prompts:
            # Resume completion uses column indices. Compare definitions, not generated ids.
            _invariant(
                matches_saved_layout,
                "Cannot resume evaluation because the saved provider/prompt columns differ. "
                "Start a new evaluation instead.",
            )
            # Explicit retry configs keep their current definitions; metrics belong to the
            # saved column positions.
            for index, prompt in enumerate(prompts):
                metrics = store.prompts[index].get("metrics")
                if metrics:
                    prompt["metrics"] = copy.deepcopy(metrics)
                    backfill_named_score_weights(prompt["metrics"])
        return prompts, columns_by_provider

    def _restore_columns_by_identity(
        self,
        restored_prompts: list[dict[str, Any]],
        provider_indices: dict[str, int],
        tests: list[dict[str, Any]],
    ) -> list[tuple[ApiProvider, list]]:
        """Recover only the saved columns with an unambiguous match in the current config."""
        suite = self.test_suite
        providers: list[ApiProvider] = suite["providers"]
        identities = prompt_identity_candidates(suite["prompts"])
        all_ids: dict[Any, None] = {}
        for ids in identities.values():
            all_ids.update(ids)
        default = get_default_test(suite) or {}
        filters = [
            {
                "prompts": t.get("prompts") if t.get("prompts") is not None
                else default.get("prompts"),
                "providers": t.get("providers") if t.get("providers") is not None
                else default.get("providers"),
            }
            for t in tests
        ]  # fmt: skip

        def identity_filters(ids: dict[Any, None]) -> list[dict[str, Any]]:
            return [
                f for f in filters
                if any(i and is_allowed_prompt({"raw": "", "label": i, "id": i}, f["prompts"])
                       for i in ids)
            ]  # fmt: skip

        ambiguous = {
            key: identity_filters(ids) if len(ids) > 1 else [] for key, ids in identities.items()
        }
        unknown = identity_filters(all_ids)
        restored: list[tuple[ApiProvider, list]] = [(p, []) for p in providers]
        reordered = (
            "Cannot resume evaluation because saved providers are missing or reordered. "
            "Start a new evaluation instead."
        )
        previous = -1
        for prompt_idx, prompt in enumerate(restored_prompts):
            provider_index = provider_indices.get(prompt.get("provider") or "", -1)
            _invariant(
                provider_index >= 0
                and provider_index >= previous
                and get_provider_identifier(providers[provider_index]) == prompt.get("provider"),
                reordered,
            )
            previous = provider_index
            provider = providers[provider_index]
            key = json_stringify([prompt.get("id"), prompt.get("raw"), prompt.get("label")])
            ids = identities.get(key)
            runtime_prompt = {**prompt, "id": next(iter(ids)) if ids and len(ids) == 1 else None}
            _invariant(
                is_allowed_prompt(runtime_prompt, self._allowed_prompts(provider)),
                "Cannot resume evaluation because provider prompt selectors exclude saved "
                "columns. Start a new evaluation instead.",
            )
            _invariant(
                all(
                    not is_provider_allowed(provider, f["providers"])
                    or is_allowed_prompt(runtime_prompt, f["prompts"])
                    for f in ambiguous.get(key, unknown)
                ),
                "Cannot resume evaluation because saved prompt IDs cannot be matched to test "
                "filters. Start a new evaluation instead.",
            )
            restored[provider_index][1].append((prompt_idx, runtime_prompt))
        return restored

    async def _build_run_options(
        self, tests: list[dict[str, Any]], columns_by_provider: list
    ) -> list[RunEvalOptions]:
        suite, options = self.test_suite, self.options
        default = get_default_test(suite) or {}
        steps: list[RunEvalOptions] = []
        test_idx = 0
        configured_provider_map = build_configured_provider_map(suite.get("providers") or [])
        for index, test in enumerate(tests):
            await _prepare_test_case(suite, test, index)
            resolve_runtime_grading_provider_references(test, configured_provider_map)
            test_options = test.get("options") or {}
            prefix = (
                test_options.get("prefix") or (default.get("options") or {}).get("prefix") or ""
            )
            suffix = (
                test_options.get("suffix") or (default.get("options") or {}).get("suffix") or ""
            )
            if envars.get_env_bool("PROMPTFOO_DISABLE_VAR_EXPANSION") or test_options.get(
                "disableVarExpansion"
            ):
                combinations = [test.get("vars")]
            else:
                combinations = generate_var_combinations(test.get("vars") or {})
            global_repeat = _normalize_repeat(options.get("repeat"))
            repeat = _normalize_repeat(test_options.get("repeat"), global_repeat)
            effective = {**options, "repeat": repeat}
            for repeat_index in range(repeat):
                for combo_index, vars_ in enumerate(combinations):
                    group = f"test-{index}-vars-{combo_index}" if repeat > 1 else None
                    for provider, columns in columns_by_provider:
                        if not is_provider_allowed(provider, test.get("providers")):
                            continue
                        for prompt_idx, prompt in columns:
                            if not is_allowed_prompt(prompt, test.get("prompts")):
                                continue
                            steps.append(
                                RunEvalOptions(
                                    provider=provider,
                                    prompt={**prompt, "raw": prefix + prompt["raw"] + suffix,
                                            "template": prompt.get("template", prompt["raw"])},
                                    test=_create_run_eval_test(suite, test, vars_,
                                                               self.store.id),
                                    test_suite=suite,
                                    test_idx=test_idx,
                                    prompt_idx=prompt_idx,
                                    repeat_index=repeat_index,
                                    evaluate_options=effective,
                                    conversations=self.conversations,
                                    registers=self.registers,
                                    eval_id=self.store.id,
                                    delay=options.get("delay") or 0,
                                    filters=suite.get("nunjucksFilters"),
                                    repeat_group_id=group,
                                    rate_limit_registry=self.rate_limit_registry,
                                )
                            )  # fmt: skip
                    test_idx += 1
        return steps

    def _update_metrics(self, metrics: dict[str, Any], row: dict[str, Any]) -> None:
        from promptphoo.assertions.index import render_metric_name

        metrics["score"] += row["score"]
        grading = row.get("gradingResult") or {}
        test_vars = (row.get("testCase") or {}).get("vars") or {}
        for name, value in (row.get("namedScores") or {}).items():
            count = (
                sum(
                    1
                    for c in grading.get("componentResults") or []
                    if render_metric_name((c.get("assertion") or {}).get("metric"), test_vars)
                    == name
                )
                or 1
            )
            weights = grading.get("namedScoreWeights") or {}
            if name in weights:
                weight_total = weights[name] if weights[name] is not None else 0
                weighted = value * weight_total
            else:
                weight_total = count
                weighted = value
            metrics["namedScores"][name] = metrics["namedScores"].get(name, 0) + weighted
            metrics["namedScoresCount"][name] = metrics["namedScoresCount"].get(name, 0) + count
            metrics["namedScoreWeights"][name] = (
                metrics["namedScoreWeights"].get(name, 0) + weight_total
            )
        if row.get("success"):
            metrics["testPassCount"] += 1
        elif row.get("failureReason") == ResultFailureReason.ERROR:
            metrics["testErrorCount"] += 1
        else:
            metrics["testFailCount"] += 1
        if (row.get("response") or {}).get("cached"):
            metrics["testCacheHitCount"] = (metrics.get("testCacheHitCount") or 0) + 1
        components = grading.get("componentResults") or []
        metrics["assertPassCount"] += sum(1 for c in components if c.get("pass"))
        metrics["assertFailCount"] += sum(1 for c in components if not c.get("pass"))
        metrics["totalLatencyMs"] += row.get("latencyMs") or 0
        accumulate_response_token_usage(metrics["tokenUsage"], row.get("response"))
        if grading.get("tokensUsed"):
            accumulate_grading_token_usage(
                metrics["tokenUsage"], grading["tokensUsed"],
                cached=(grading.get("metadata") or {}).get("cachedResponse"),
            )  # fmt: skip
        if row.get("incurredCost") is not None or metrics.get("incurredCost") is not None:
            base = metrics.get("incurredCost")
            base = base if base is not None else metrics["cost"]
            extra = row.get("incurredCost")
            extra = extra if extra is not None else (row.get("cost") or 0)
            metrics["incurredCost"] = base + extra
        metrics["cost"] += row.get("cost") or 0

    def _track_row_stats(self, row: dict[str, Any]) -> None:
        if row.get("success"):
            self.stats.successes += 1
        elif row.get("failureReason") == ResultFailureReason.ERROR:
            self.stats.errors += 1
        else:
            self.stats.failures += 1
        if row.get("tokenUsage"):
            accumulate_response_token_usage(
                self.stats.token_usage, {"tokenUsage": row["tokenUsage"]}
            )

    async def evaluate(self) -> EvaluationStore:
        suite = self.test_suite
        with cli_state.with_base_path(suite.get("basePath") or cli_state.base_path()), \
                cli_state.with_env(suite.get("env") or cli_state.env()):  # fmt: skip
            enabled = False if self.options.get("cache") is False else cache.is_cache_enabled()
            with cache.with_cache_enabled(enabled):
                max_ms = self.options.get("maxEvalTimeMs") or envars.get_max_eval_time_ms()
                otel_initialized = False
                try:
                    if _suite_tracing_enabled(suite):
                        from promptphoo.tracing.otel_config import get_default_otel_config
                        from promptphoo.tracing.otel_sdk import initialize_otel

                        logger.debug("[Evaluator] Initializing OTEL SDK for tracing")
                        initialize_otel(get_default_otel_config())
                        otel_initialized = True
                    if max_ms and max_ms > 0:
                        return await asyncio.wait_for(self._run(time.time()), max_ms / 1000)
                    return await self._run(time.time())
                finally:
                    if otel_initialized:
                        from promptphoo.tracing.otel_sdk import flush_otel, shutdown_otel

                        logger.debug("[Evaluator] Flushing OTEL spans...")
                        await flush_otel()
                        await shutdown_otel()
                    self._dispose_rate_limit_registry()
                    for provider in suite["providers"]:
                        try:
                            await provider.cleanup()
                        except Exception as e:
                            logger.warning("Provider cleanup failed: %s", e)

    def _dispose_rate_limit_registry(self) -> None:
        # Log rate limit metrics for debugging before cleanup
        if self.rate_limit_registry:
            for key, m in self.rate_limit_registry.get_metrics().items():
                if m["totalRequests"] > 0:
                    logger.debug(
                        "[Scheduler] Final metrics for %s %s",
                        sanitize_provider_id_for_log(key),
                        json_stringify({
                            "totalRequests": m["totalRequests"],
                            "completedRequests": m["completedRequests"],
                            "failedRequests": m["failedRequests"],
                            "rateLimitHits": m["rateLimitHits"],
                            "retriedRequests": m["retriedRequests"],
                            "avgLatencyMs": round(m["avgLatencyMs"]),
                            "p50LatencyMs": round(m["p50LatencyMs"]),
                            "p99LatencyMs": round(m["p99LatencyMs"]),
                        }),
                    )  # fmt: skip
        # Clean up rate limit registry resources
        if self.rate_limit_registry:
            self.rate_limit_registry.dispose()

    async def _run(self, start: float) -> EvaluationStore:
        suite, options = self.test_suite, self.options
        if not options.get("silent"):
            logger.info("Starting evaluation %s", self.store.id)
        from promptphoo.evaluator.extensions import (
            check_extensions_supported,
            run_before_all_extensions,
        )
        from promptphoo.suggestions import maybe_add_generated_prompts

        check_extensions_supported(suite.get("extensions"))
        suite = self.test_suite = await run_before_all_extensions(suite)
        await maybe_add_generated_prompts(suite, options)
        if suite.get("derivedMetrics"):
            logger.warning("derivedMetrics are not supported yet; they will be skipped")

        # The caller's cancellation and the CLI pause (Ctrl+C) both stop the run.
        abort_signal = combine_signals(options.get("abortSignal"), options.get("pauseSignal"))
        restore_prompt_columns = bool(
            options.get("restorePromptColumns")
            and cli_state.resume
            and self.store.persisted
            and self.store.prompts
        )

        tests = build_tests_from_suite(suite)
        tests = filter_by_range(tests, options.get("filterRange"))
        prompts, columns_by_provider = self._build_completed_prompts(restore_prompt_columns, tests)
        await self.store.append_prompts(prompts)
        from promptphoo.providers.azure.warnings import maybe_emit_azure_openai_warning

        maybe_emit_azure_openai_warning(suite, tests)

        var_names: dict[str, None] = {}  # insertion-ordered like a JS Set
        default = get_default_test(suite) or {}
        for test in tests:
            test["vars"] = {**(default.get("vars") or {}), **(test.get("vars") or {})}
            await _apply_input_transform(test, (default.get("options") or {}).get("transformVars"))
            var_names.update(dict.fromkeys(test["vars"]))

        concurrency = options.get("maxConcurrency") or DEFAULT_MAX_CONCURRENCY
        steps = await self._build_run_options(tests, columns_by_provider)
        from promptphoo.evaluator.comparison import (
            build_repeat_cache_context_by_test_idx,
            get_comparison_providers,
            get_result_index_key,
            mark_comparison_rows,
        )

        rows_with_select_best, rows_with_max_score = mark_comparison_rows(steps)
        if cli_state.resume and self.store.persisted:
            for step in steps:
                if step.test_idx in rows_with_select_best:
                    self.comparison.comparison_providers[get_result_index_key(step)] = (
                        get_comparison_providers(step.test)
                    )
        repeat_cache_context_by_test_idx = build_repeat_cache_context_by_test_idx(steps)
        await filter_completed_resume_steps(steps, self.store)

        uses_conversation = any(prompt_uses_conversation_variable(p) for p in prompts)
        if concurrency > 1:
            if uses_conversation:
                logger.info(
                    "Setting concurrency to 1 because the %s variable is used.",
                    CONVERSATION_VAR_NAME,
                )
                concurrency = 1
            elif any((t.get("options") or {}).get("storeOutputAs") for t in tests):
                logger.info("Setting concurrency to 1 because storeOutputAs is used.")
                concurrency = 1

        serial = [s for s in steps if (s.test.get("options") or {}).get("runSerially")]
        concurrent = [s for s in steps if not (s.test.get("options") or {}).get("runSerially")]
        index_of = {id(s): i for i, s in enumerate(steps)}
        all_vars: dict[str, None] = dict(var_names)
        processed: set[int] = set()
        progress = options.get("progressCallback")
        state = {"complete": 0}
        timeout_ms = options.get("timeoutMs") or envars.get_eval_timeout_ms()

        if not options.get("silent"):
            if serial:
                logger.info("Running %d test cases serially...", len(serial))
            if concurrent:
                logger.info(
                    "Running %d test cases (up to %d at a time)...", len(concurrent), concurrency
                )

        async def process(step: RunEvalOptions) -> None:
            idx = index_of[id(step)]
            try:
                if timeout_ms and timeout_ms > 0:
                    call = asyncio.wait_for(
                        self._run_eval_step_after_before_each(step), timeout_ms / 1000
                    )
                else:
                    call = self._run_eval_step_after_before_each(step)
                # A paused or cancelled step is not saved, so resume runs it again.
                rows = await await_with_abort(call, abort_signal)
            except TimeoutError as e:
                rows = [self._timeout_row(step, timeout_ms, e)]
            processed.add(idx)
            for row in rows:
                all_vars.update(dict.fromkeys(row.get("vars") or {}))
                self._track_row_stats(row)
                state["complete"] += 1
                if suite.get("extensions"):
                    await self._apply_after_each_extensions(step, row)
                self.comparison.prepare_row(row)
                try:
                    await self.store.append_result(row)
                except Exception as e:
                    self.store.record_result_persistence_failure(row)
                    logger.error("[Evaluator] Error saving result: %s", e)
                metrics = prompts[row["promptIdx"]]["metrics"]
                self._update_metrics(metrics, row)
                if progress:
                    progress(state["complete"], len(steps), idx, step, metrics)

        try:
            for step in serial:
                await process(step)
            semaphore = asyncio.Semaphore(concurrency)

            async def limited(step: RunEvalOptions) -> None:
                async with semaphore:
                    await process(step)

            outcomes = await asyncio.gather(
                *(limited(s) for s in concurrent), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            if abort_signal is not None:
                abort_signal.throw_if_aborted()
        except BaseException:
            if abort_signal is None or not abort_signal.aborted:
                raise
            return await self._save_interrupted_eval(prompts, all_vars)

        if not await self.comparison.process(
            prompts,
            repeat_cache_context_by_test_idx,
            rows_with_select_best,
            rows_with_max_score,
            abort_signal,
        ):
            return await self._save_interrupted_eval(prompts, all_vars)

        await self.store.append_prompts(prompts)
        self.store.set_vars(list(all_vars))
        await self._run_after_all_extensions(suite)
        self.store.set_duration_ms(round((time.time() - start) * 1000))
        if self.store.persisted:
            await self.store.save()
        return self.store

    async def _save_interrupted_eval(
        self, prompts: list[dict[str, Any]], all_vars: dict[str, None]
    ) -> EvaluationStore:
        """saveInterruptedEval: keep the rows and metrics so far; the eval stays resumable."""
        logger.info("Evaluation interrupted, saving progress...")
        self.store.set_vars(list(all_vars))
        await self.store.append_prompts(prompts)
        return self.store

    async def _run_eval_step_after_before_each(self, step: RunEvalOptions) -> list[dict]:
        from promptphoo.evaluator.extensions import run_extension_hook

        out = await run_extension_hook(
            self.test_suite.get("extensions"), "beforeEach", {"test": step.test}
        )
        step.test = out["test"]
        return await run_eval(step)

    async def _apply_after_each_extensions(self, step: RunEvalOptions, row: dict) -> None:
        """Merge afterEach namedScores / metadata / response.metadata into the row before it
        is persisted. The hook gets copies, so a failing hook leaves the row untouched."""
        from promptphoo.evaluator.extensions import (
            run_extension_hook,
            synchronize_legacy_transport_headers,
        )
        from promptphoo.numeric import filter_finite_scores

        try:
            original_metadata = row.get("metadata")
            response = row.get("response")
            original_response_metadata = response.get("metadata") if response else None
            out = await run_extension_hook(
                self.test_suite.get("extensions"),
                "afterEach",
                {
                    "test": step.test,
                    "result": {
                        **row,
                        "namedScores": {**(row.get("namedScores") or {})},
                        "metadata": {**(row.get("metadata") or {})},
                        "response": {**response, "metadata": {**(response.get("metadata") or {})}}
                        if response
                        else response,
                    },
                },
            )
            result = out["result"]
            row["namedScores"] = filter_finite_scores(result.get("namedScores") or {})
            hook_response = result.get("response")
            row["metadata"] = synchronize_legacy_transport_headers(
                original_metadata,
                original_response_metadata,
                result.get("metadata"),
                hook_response.get("metadata") if isinstance(hook_response, dict) else None,
            )
            if response and hook_response:
                response["metadata"] = hook_response.get("metadata")
        except Exception as error:
            logger.error(
                "afterEach extension hook failed, persisting row without hook modifications: %s",
                error,
            )

    async def _run_after_all_extensions(self, suite: dict[str, Any]) -> None:
        from promptphoo.evaluator.extensions import run_extension_hook

        if not suite.get("extensions"):
            return
        load_results = getattr(self.store, "load_results", None)
        results = load_results() if load_results else []
        await run_extension_hook(
            suite["extensions"],
            "afterAll",
            {
                "prompts": getattr(self.store, "prompts", []),
                "results": [r.to_evaluate_result() for r in results],
                "suite": suite,
                "evalId": self.store.id,
                "config": getattr(self.store, "config", {}),
            },
        )

    def _timeout_row(self, step: RunEvalOptions, timeout_ms: int, error: Exception) -> dict:
        row: dict[str, Any] = {
            "provider": _provider_setup(step.provider),
            "prompt": {"raw": step.prompt["raw"], "label": step.prompt.get("label"),
                       "config": step.prompt.get("config")},
            "vars": step.test.get("vars") or {},
            "error": f"Evaluation timed out after {timeout_ms}ms: Error: Evaluation timed out "
            f"after {timeout_ms}ms",
            "success": False,
            "failureReason": ResultFailureReason.ERROR,
            "score": 0,
            "namedScores": {},
            "latencyMs": timeout_ms,
            "promptIdx": step.prompt_idx,
        }  # fmt: skip
        if step.repeat_group_id is not None:
            row["repeatGroupId"] = step.repeat_group_id
            row["repeatIndex"] = step.repeat_index
        row.update(testIdx=step.test_idx, testCase=step.test, promptId=step.prompt.get("id") or "")
        return row


async def evaluate(
    test_suite: dict[str, Any], store: EvaluationStore, options: dict[str, Any]
) -> EvaluationStore:
    return await Evaluator(test_suite, store, options).evaluate()
