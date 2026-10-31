"""Assertion dispatcher. Ports runAssertion / runAssertions from src/assertions/index.ts."""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import logging
import os
from typing import Any

from promptphoo import cli_state, envars, python_hooks
from promptphoo.assertions.base import AssertionParams, GradingResult, Handler, coerce_string
from promptphoo.assertions.result import AssertionsResult
from promptphoo.files import is_javascript_file
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ApiProvider
from promptphoo.scheduler.provider_call_execution_context import get_provider_call_tracing_context
from promptphoo.templating import get_engine
from promptphoo.transform import transform
from promptphoo.yamlload import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_ASSERTIONS_MAX_CONCURRENCY = 3
SCRIPT_RESULT_ASSERTIONS = {"javascript", "python", "ruby"}

MODEL_GRADED_ASSERTION_TYPES = {
    "agent-rubric", "answer-relevance", "context-faithfulness", "context-recall",
    "context-relevance", "factuality", "llm-rubric", "model-graded-closedqa",
    "model-graded-factuality", "search-rubric", "trajectory:goal-success",
}  # fmt: skip

# Handler modules, merged in order. Missing optional modules are skipped so a partial port
# still loads; unknown types then fail with Node's "Unknown assertion type" error.
_HANDLER_MODULES = (
    "basic", "script", "markup", "sql", "refusal", "tools", "trajectory", "textmetrics",
    "modelgraded", "modelgraded_rag", "deepeval",
)  # fmt: skip
_handlers: dict[str, Handler] | None = None


def get_handlers() -> dict[str, Handler]:
    global _handlers
    if _handlers is None:
        merged: dict[str, Handler] = {}
        for name in _HANDLER_MODULES:
            try:
                module = importlib.import_module(f"promptphoo.assertions.{name}")
            except ModuleNotFoundError as e:
                if e.name != f"promptphoo.assertions.{name}":
                    raise
                continue
            merged.update(module.HANDLERS)
        _handlers = merged
    return _handlers


def render_metric_name(metric: str | None, vars_: dict[str, Any]) -> str | None:
    if not metric:
        return metric
    try:
        return get_engine().render_string(metric, vars_)
    except Exception as e:
        logger.warning('Failed to render metric template "%s": %s', metric, e)
        return metric


def is_assertion_inverse(assertion: dict[str, Any]) -> bool:
    return assertion["type"].startswith("not-")


def get_assertion_base_type(assertion: dict[str, Any]) -> str:
    t = assertion["type"]
    return t[4:] if t.startswith("not-") else t


def process_file_reference(file_ref: str) -> Any:
    base = cli_state.base_path() or ""
    path = os.path.abspath(os.path.join(base, file_ref[len("file://") :]))
    with open(path, encoding="utf-8") as f:
        content = f.read()
    ext = os.path.splitext(path)[1]
    if ext in (".json", ".yaml", ".yml"):
        return load_yaml(content)
    if ext == ".txt":
        return content.strip()
    raise ValueError(f"Unsupported file type: {path}")


def _clone_test_value(value: Any) -> Any:
    """Deep-copy plain data, but share loaded providers (graders resolved to configured
    provider instances may sit in `assert` entries; they hold clients and locks)."""
    if isinstance(value, dict):
        return {k: _clone_test_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_test_value(v) for v in value]
    if isinstance(value, ApiProvider) or callable(value):
        return value
    return copy.deepcopy(value)


def get_final_test(test: dict[str, Any], assertion: dict[str, Any]) -> dict[str, Any]:
    provider = test.get("provider")
    options = test.get("options") or {}
    base = {k: v for k, v in test.items() if k != "provider"}
    if options.get("provider") is not None:
        base["options"] = {**options, "provider": None}
    ret = _clone_test_value(base)
    ret["options"] = ret.get("options") or {}
    if provider is not None:
        ret["provider"] = provider
    ret["options"]["provider"] = (
        assertion.get("provider")
        if is_truthy(assertion.get("provider"))
        else options.get("provider")
    )
    rubric_prompt = assertion.get("rubricPrompt")
    ret["options"]["rubricPrompt"] = (
        rubric_prompt if is_truthy(rubric_prompt) else ret["options"].get("rubricPrompt")
    )
    return ret


def _split_file_ref(file_ref: str) -> tuple[str, str | None]:
    if ":" in file_ref:
        idx = file_ref.index(":")
        return file_ref[:idx], file_ref[idx + 1 :]
    return file_ref, None


async def run_assertion(
    *,
    assertion: dict[str, Any],
    test: dict[str, Any],
    provider_response: dict[str, Any],
    prompt: str | None = None,
    provider: Any = None,
    vars: dict[str, Any] | None = None,  # noqa: A002 - Node's parameter name
    latency_ms: float | None = None,
    trace_id: str | None = None,
) -> GradingResult:
    """runAssertion: a traced assertion runs inside a `grader <type>` span."""
    kwargs: dict[str, Any] = {
        "assertion": assertion, "test": test, "provider_response": provider_response,
        "prompt": prompt, "provider": provider, "vars": vars, "latency_ms": latency_ms,
        "trace_id": trace_id,
    }  # fmt: skip
    if not trace_id:
        return await _run_assertion_internal(**kwargs)
    tracing_context = get_provider_call_tracing_context()
    with_grader_span = tracing_context.get("withGraderSpan") if tracing_context else None
    if not tracing_context or with_grader_span is None:
        return await _run_assertion_internal(**kwargs)
    return await with_grader_span(
        {
            "graderId": assertion.get("type"),
            "evalId": (test.get("metadata") or {}).get("evaluationId"),
            "testIndex": tracing_context.get("testIndex"),
        },
        lambda: _run_assertion_internal(**kwargs),
    )


async def _run_assertion_internal(
    *,
    assertion: dict[str, Any],
    test: dict[str, Any],
    provider_response: dict[str, Any],
    prompt: str | None = None,
    provider: Any = None,
    vars: dict[str, Any] | None = None,  # noqa: A002 - Node's parameter name
    latency_ms: float | None = None,
    trace_id: str | None = None,
) -> GradingResult:
    resolved_vars = vars if vars is not None else (test.get("vars") or {})
    output = provider_response.get("output")
    if not assertion.get("type"):
        raise ValueError(f"Assertion must have a type: {json_stringify(assertion)}")

    if assertion.get("transform"):
        output = await transform(
            assertion["transform"],
            output,
            {
                "vars": resolved_vars,
                "prompt": {"label": prompt},
                **(
                    {"metadata": provider_response["metadata"]}
                    if provider_response.get("metadata")
                    else {}
                ),
            },
        )

    context: dict[str, Any] = {
        "prompt": prompt,
        "vars": resolved_vars,
        "test": test,
        "logProbs": provider_response.get("logProbs"),
        "provider": provider,
        "providerResponse": provider_response,
    }
    if assertion.get("config"):
        context["config"] = copy.deepcopy(assertion["config"])
    if provider_response.get("metadata"):
        context["metadata"] = provider_response["metadata"]

    rendered: Any = assertion.get("value")
    value_from_script: Any = None
    engine = get_engine()
    if isinstance(rendered, str):
        if rendered.startswith("file://"):
            file_path, fn_name = _split_file_ref(rendered[len("file://") :])
            file_path = os.path.abspath(os.path.join(cli_state.base_path() or "", file_path))
            if is_javascript_file(file_path):
                raise ValueError(
                    f"JavaScript assertion files are not supported by promptphoo: {rendered}. "
                    "Use a Python file (file://assert.py) instead."
                )
            if file_path.endswith(".py"):
                try:
                    value_from_script = await python_hooks.run_python(
                        file_path, fn_name or "get_assert", [output, context]
                    )
                except Exception as e:
                    return {"pass": False, "score": 0, "reason": str(e), "assertion": assertion}
            elif file_path.endswith(".rb"):
                raise ValueError("Ruby assertion files are not supported by promptphoo")
            else:
                rendered = process_file_reference(rendered)
        elif rendered.startswith("package:"):
            raise ValueError("package: assertion values are not supported by promptphoo")
        else:
            rendered = engine.render_string(rendered, resolved_vars)
    elif isinstance(rendered, list):
        rendered = [
            (process_file_reference(v) if v.startswith("file://")
             else engine.render_string(v, resolved_vars))
            if isinstance(v, str) else v
            for v in rendered
        ]  # fmt: skip

    base_type = get_assertion_base_type(assertion)
    if value_from_script is not None and base_type not in SCRIPT_RESULT_ASSERTIONS:
        if callable(value_from_script):
            kind = "a function"
        elif isinstance(value_from_script, bool):
            kind = "a boolean"
        elif isinstance(value_from_script, dict) and "pass" in value_from_script:
            kind = "a GradingResult"
        else:
            kind = None
        if kind:
            noun = {"a function": "functions", "a boolean": "boolean values",
                    "a GradingResult": "GradingResult objects"}[kind]  # fmt: skip
            raise ValueError(
                f'Script for "{assertion["type"]}" assertion returned {kind}. '
                f"Only javascript/python/ruby assertion types can return {noun}. "
                "For other assertion types, return the expected value (string, number, array, "
                "or object)."
            )
        rendered = value_from_script

    # Generate a traceparent for grader calls to link them to the main trace.
    tracing_context = get_provider_call_tracing_context()
    get_traceparent = tracing_context.get("getActiveTraceparent") if tracing_context else None
    active_traceparent = get_traceparent() if get_traceparent else None
    grader_traceparent = None
    if trace_id:
        if active_traceparent and active_traceparent.split("-")[1] == trace_id:
            grader_traceparent = active_traceparent
        else:
            from promptphoo.tracing.otel_trace_sdk import generate_span_id

            grader_traceparent = f"00-{trace_id}-{generate_span_id()}-01"
    provider_call_context = (
        {"originalProvider": provider, "prompt": {"raw": prompt or "", "label": ""},
         "vars": resolved_vars,
         **({"traceparent": grader_traceparent} if grader_traceparent else {})}
        if provider is not None else None
    )  # fmt: skip
    final_test = get_final_test(
        test if vars is None else {**test, "vars": resolved_vars}, assertion
    )
    params = AssertionParams(
        assertion=assertion,
        base_type=base_type,
        inverse=is_assertion_inverse(assertion),
        output=output,
        output_string=coerce_string(output),
        rendered_value=rendered,
        value_from_script=value_from_script,
        prompt=prompt,
        provider=provider,
        provider_response=provider_response,
        test=final_test,
        latency_ms=latency_ms,
        cost=provider_response.get("cost"),
        log_probs=provider_response.get("logProbs"),
        assertion_value_context=context,
        provider_call_context=provider_call_context,
    )

    if base_type.startswith("promptfoo:redteam:"):
        raise ValueError("Red team assertions are not supported by promptphoo")
    handler = get_handlers().get(base_type)
    if handler is None:
        raise ValueError(f"Unknown assertion type: {assertion['type']}")
    res = handler(params)
    if inspect.isawaitable(res):
        res = await res
    if isinstance(rendered, str) and rendered != assertion.get("value"):
        res["metadata"] = {**(res.get("metadata") or {}), "renderedAssertionValue": rendered}
    if assertion.get("weight") == 0:
        return {**res, "pass": True}
    return res


async def run_assertions(
    *,
    test: dict[str, Any],
    provider_response: dict[str, Any],
    prompt: str | None = None,
    provider: Any = None,
    vars: dict[str, Any] | None = None,  # noqa: A002
    latency_ms: float | None = None,
    assert_scoring_function: Any = None,
    trace_id: str | None = None,
) -> GradingResult:
    asserts_cfg = test.get("assert") or []
    if not asserts_cfg:
        return AssertionsResult.no_asserts_result()

    main = AssertionsResult(threshold=test.get("threshold"))
    sub_results: list[AssertionsResult] = []
    entries: list[tuple[dict[str, Any], AssertionsResult, int]] = []
    for i, assertion in enumerate(asserts_cfg):
        if assertion.get("type") == "assert-set":
            sub = AssertionsResult(
                threshold=assertion.get("threshold"),
                parent_assertion_set={"assertionSet": assertion, "index": i},
            )
            sub_results.append(sub)
            entries.extend((a, sub, j) for j, a in enumerate(assertion.get("assert") or []))
        else:
            entries.append((assertion, main, i))

    metric_vars = vars if vars is not None else (test.get("vars") or {})
    concurrency = max(
        1,
        envars.get_env_int(
            "PROMPTFOO_ASSERTIONS_MAX_CONCURRENCY", DEFAULT_ASSERTIONS_MAX_CONCURRENCY
        ),
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(assertion: dict[str, Any], target: AssertionsResult, index: int) -> None:
        if assertion["type"].startswith("select-") or assertion["type"] == "max-score":
            return
        async with semaphore:
            res = await run_assertion(
                assertion=assertion, test=test, provider_response=provider_response,
                prompt=prompt, provider=provider, vars=vars, latency_ms=latency_ms,
                trace_id=trace_id,
            )  # fmt: skip
        target.add_result(
            index=index,
            result=res,
            metric=render_metric_name(assertion.get("metric"), metric_vars),
            weight=assertion.get("weight"),
        )

    tasks = [asyncio.create_task(run_one(*e)) for e in entries]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    for sub in sub_results:
        res = await sub.test_result()
        parent = sub.parent_assertion_set or {}
        assertion_set = parent.get("assertionSet") or {}
        main.add_result(
            index=parent["index"],
            result=res,
            metric=render_metric_name(assertion_set.get("metric"), metric_vars),
            weight=assertion_set.get("weight"),
        )
    return await main.test_result(assert_scoring_function)


def read_assertions(file_path: str) -> list[dict[str, Any]]:
    try:
        with open(file_path, encoding="utf-8") as f:
            assertions = load_yaml(f.read())
        if not isinstance(assertions, list) or not assertions or "type" not in assertions[0]:
            raise ValueError("Assertions file must be an array of assertion objects")
        return assertions
    except Exception as e:
        raise ValueError(f"Failed to read assertions from {file_path}:\n{e}") from e


async def run_compare_assertion(
    test: dict[str, Any],
    assertion: dict[str, Any],
    outputs: list[Any],
    context: dict[str, Any] | None = None,
) -> list[GradingResult]:
    """runCompareAssertion: grade every output of one test together (select-best)."""
    from promptphoo.matchers.comparison import matches_select_best

    if not isinstance(assertion.get("value"), str):
        raise ValueError("select-best must have a string value")
    # The matcher needs options and vars, not the assertion list. A runtime assertion can
    # contain a provider with a live client, which get_final_test cannot deep-copy.
    final = get_final_test({**test, "assert": None}, assertion)
    comparison_results = await matches_select_best(
        assertion["value"], outputs, final.get("options"), final.get("vars"), context
    )
    # The runtime assertion may contain a live grader and secrets. Results only need
    # the comparison criteria and scoring labels, so keep provider config out of memory.
    safe_assertion: dict[str, Any] = {"type": assertion["type"], "value": assertion["value"]}
    for key in ("metric", "weight"):
        if assertion.get(key) is not None:
            safe_assertion[key] = assertion[key]
    return [{**result, "assertion": safe_assertion} for result in comparison_results]
