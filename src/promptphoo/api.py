"""Library entry point: run an evaluation from Python. Ports src/evaluate.ts and
src/node/evaluate.ts (named `api`, not `evaluate`, so the function and the module cannot
shadow each other).

    from promptphoo import evaluate

    record = evaluate({
        "prompts": ["{{question}}"],
        "providers": ["echo"],
        "tests": [{"vars": {"question": "2+2?"}, "assert": [{"type": "contains", "value": "2"}]}],
    })
    summary = record.to_evaluate_summary()

The test suite is the same mapping a `promptfooconfig.yaml` holds (prompts, providers, tests,
defaultTest, scenarios, extensions, env, basePath, outputPath, ...), plus `writeLatestResults`
to persist the eval in the shared database. Options mirror upstream's EvaluateOptions:
`cache`, `maxConcurrency`, `repeat`, `delay`, `timeoutMs`, `maxEvalTimeMs`, `progressCallback`,
`abortSignal`, `silent`; a suite's own `evaluateOptions` apply under them. Everything the CLI
would read from a config file works here too:
`file://provider.py`, `file://tests.py:generate_tests`, `python` assertions and Python extension
hooks run in this process.

State is scoped to the call (`cli_state` context variables, the cache override), nothing is read
from `.env`, nothing is written to the process environment, and no exit codes or signal handlers
are involved. Errors are exceptions. Cancelling through `abortSignal` (`promptphoo.AbortController`)
ends the run early and returns the partial record, as upstream does; check `signal.aborted`.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any

from promptphoo import cache, cli_state, db
from promptphoo.evaluator.run import evaluate as do_evaluate
from promptphoo.files import maybe_load_from_external_file
from promptphoo.models.eval import Eval, get_author
from promptphoo.models.result_sanitize import to_serializable_provider_ref
from promptphoo.prompts import read_prompts
from promptphoo.providers import load_api_providers
from promptphoo.scheduler.cancellation import throw_if_aborted
from promptphoo.templating import read_filters
from promptphoo.testcases import read_tests

ProgressCallback = Callable[[int, int, int, Any, Any], None]

# Keys of the suite that describe the run rather than the config stored with the eval.
_SUITE_ONLY_KEYS = ("writeLatestResults", "author")


def _serializable_config(suite: Mapping[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    """createSerializableUnifiedConfig: the suite as it is stored with the eval. Provider
    objects become references; parsed prompts replace the raw prompt list."""
    config = {k: v for k, v in suite.items() if k not in _SUITE_ONLY_KEYS}
    config["basePath"] = cli_state.base_path()
    config["providers"] = to_serializable_provider_ref(suite.get("providers"))
    config["prompts"] = prompts
    return config


async def _runtime_test_suite(
    suite: Mapping[str, Any], providers: list[Any], base_path: str, env: dict[str, Any] | None
) -> dict[str, Any]:
    """createRuntimeTestSuite: prompts, tests and filters read the way the CLI reads them."""
    default_test = suite.get("defaultTest")
    if isinstance(default_test, str) and default_test.startswith("file://"):
        default_test = maybe_load_from_external_file(default_test)
    return {
        **suite,
        "defaultTest": default_test,
        "providers": providers,
        "tests": await read_tests(suite.get("tests"), base_path, env),
        "nunjucksFilters": read_filters(suite.get("nunjucksFilters") or {}, base_path),
        "prompts": await read_prompts(suite.get("prompts") or [], base_path),
    }


async def _evaluate_with_env(suite: Mapping[str, Any], options: dict[str, Any]) -> Eval:
    """evaluateWithEnv: load, run, write outputs. Grading providers named in tests and
    assertions are resolved by the evaluator when they run, as on the CLI path."""
    base_path = cli_state.base_path() or ""
    env = suite.get("env") or {}
    write = bool(suite.get("writeLatestResults"))
    if write:
        db.run_db_migrations()

    providers = await load_api_providers(suite.get("providers"), base_path=base_path, env=env)
    throw_if_aborted(options.get("abortSignal"))
    runtime_suite = await _runtime_test_suite(suite, providers, base_path, env)

    config = _serializable_config(suite, runtime_suite["prompts"])
    author = get_author(suite.get("author"))
    record = (
        Eval.create(config, runtime_suite["prompts"], author=author)
        if write
        else Eval(config, author=author)
    )

    # A config-file-shaped suite may carry evaluateOptions; explicit options win, as on the CLI.
    run_options = {
        "isRedteam": bool(suite.get("redteam")),
        "showProgressBar": False,
        **(suite.get("evaluateOptions") or {}),
        **options,
    }
    cache_enabled = False if run_options.get("cache") is False else None
    with cache.with_cache_enabled(cache_enabled):
        await do_evaluate(runtime_suite, record, run_options)

    output_path = suite.get("outputPath") or options.get("outputPath")
    paths = [output_path] if isinstance(output_path, str) else list(output_path or [])
    if paths:
        from promptphoo.output.writer import write_multiple_outputs

        if record.persisted:
            record.clear_results()
        write_multiple_outputs(paths, record, None)
    return record


async def evaluate_with_source(
    test_suite: Mapping[str, Any], options: Mapping[str, Any] | None = None
) -> Eval:
    """evaluateWithSource: run with the caller's `eventSource`. State (config, base path, env
    overrides) is scoped to this call and restored afterwards."""
    if not isinstance(test_suite, Mapping):
        raise TypeError("test_suite must be a mapping shaped like a promptfooconfig")
    suite = dict(test_suite)
    options = dict(options or {})
    config = {k: v for k, v in suite.items() if k not in ("prompts", "providers")}
    base_path = os.path.abspath(suite.get("basePath") or "")
    with (
        cli_state.with_config(config),
        cli_state.with_base_path(base_path),
        cli_state.with_env(suite.get("env") or {}),
    ):
        return await _evaluate_with_env(suite, options)


async def evaluate_async(
    test_suite: Mapping[str, Any], options: Mapping[str, Any] | None = None
) -> Eval:
    """Upstream's `evaluate(testSuite, options)`: run a test suite and return its Eval record.
    Results are persisted only when `test_suite["writeLatestResults"]` is true."""
    return await evaluate_with_source(test_suite, {**(options or {}), "eventSource": "library"})


def evaluate(test_suite: Mapping[str, Any], options: Mapping[str, Any] | None = None) -> Eval:
    """`evaluate_async` for synchronous callers (scripts, Flask views, worker threads).

    Runs its own event loop. Inside a running loop (an async framework) await
    `evaluate_async` instead; calling this there raises rather than deadlocking (D79)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(evaluate_async(test_suite, options))
    raise RuntimeError(
        "promptphoo.evaluate() was called from a running event loop; await "
        "promptphoo.evaluate_async() instead, or call evaluate() from a worker thread."
    )


def evaluate_in_thread(
    test_suite: Mapping[str, Any], options: Mapping[str, Any] | None = None
) -> Eval:
    """`evaluate` on a fresh thread with its own event loop, for callers that are already inside
    one (an async web framework) and want to block anyway."""
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["value"] = asyncio.run(evaluate_async(test_suite, options))
        except BaseException as exc:  # re-raised on the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=run, name="promptphoo-evaluate", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


__all__ = [
    "ProgressCallback",
    "evaluate",
    "evaluate_async",
    "evaluate_in_thread",
    "evaluate_with_source",
]
