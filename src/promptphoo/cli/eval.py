"""`promptfoo eval`. Ports src/commands/eval.ts and src/node/doEval.ts.

--share is accepted for compatibility and only warns. --resume / --retry-errors replay a
saved eval (see D36); Ctrl+C pauses a run that writes to the database, so it can be
resumed. --watch polls files (cli/watch.py); -x runs Python extension hooks (D38).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import signal
import time
from typing import Any

import click

from promptphoo import cache, cli_state, db, envars
from promptphoo.cli.options import VariadicOption, collect_key_value
from promptphoo.cli.retry import (
    delete_error_results,
    get_error_result_ids,
    get_persisted_provider_filter_options,
    get_provider_filter_regex_error,
    recalculate_prompt_metrics,
)
from promptphoo.config.load import maybe_read_config, resolve_configs
from promptphoo.envfile import setup_env
from promptphoo.evaluator.run import DEFAULT_MAX_CONCURRENCY, evaluate
from promptphoo.filters import filter_tests
from promptphoo.models.eval import Eval, get_author
from promptphoo.providers import filter_providers

logger = logging.getLogger("promptphoo")

DEFAULT_CONFIG_EXTENSIONS = ("yaml", "yml", "json", "cjs", "cts", "js", "mjs", "mts", "ts")


class EvalRunError(RuntimeError):
    pass


def load_default_config(
    directory: str | None = None, config_name: str = "promptfooconfig"
) -> tuple[dict[str, Any], str | None]:
    directory = directory or os.getcwd()
    for ext in DEFAULT_CONFIG_EXTENSIONS:
        path = os.path.join(directory, f"{config_name}.{ext}")
        config = maybe_read_config(path)
        if config:
            return config, path
    return {}, None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_config_paths(
    config_paths: list[str], default_config: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    hint = (
        f"Looked for promptfooconfig.{{{','.join(DEFAULT_CONFIG_EXTENSIONS)}}}. Run "
        '"promptphoo init" or pass --config path/to/promptfooconfig.yaml.'
    )
    resolved: list[str] = []
    configless: list[str] = []
    for path in config_paths:
        if not os.path.isdir(path):
            resolved.append(path)
            continue
        dir_config, dir_config_path = load_default_config(path)
        if dir_config_path:
            resolved.append(dir_config_path)
            default_config = {**default_config, **dir_config}
        else:
            configless.append(path)
            logger.warning("No configuration file found in directory: %s. %s", path, hint)
    if configless and not resolved:
        raise EvalRunError(f"No configuration file found in {', '.join(configless)}. {hint}")
    return resolved, default_config


async def do_eval(
    cmd: dict[str, Any],
    default_config: dict[str, Any],
    default_config_path: str | None,
    evaluate_options: dict[str, Any],
    watch_state: dict[str, Any] | None = None,
) -> tuple[Eval, int]:
    """Run one eval. Returns (eval record, process exit code).

    `watch_state`, when given, receives the resolved config, config paths and test sources
    that `--watch` needs to build its watch list."""
    start = time.time()
    setup_env(cmd.get("envPath"))
    if cmd.get("write"):
        db.run_db_migrations()
    if default_config_path:
        default_config, _ = load_default_config(
            os.path.dirname(default_config_path),
            os.path.splitext(os.path.basename(default_config_path))[0],
        )
    if cmd.get("config") is not None:
        cmd["config"], default_config = _resolve_config_paths(cmd["config"], default_config)

    # Check for conflicting options
    resume_raw = cmd.get("resume")
    retry_errors = cmd.get("retryErrors")
    if resume_raw and retry_errors:
        raise EvalRunError(
            "Cannot use --resume and --retry-errors together. Please use one or the other."
        )
    if resume_raw and cmd.get("tags"):
        raise EvalRunError(
            "Cannot use --tag with --resume. Resumed evaluations keep their original tags."
        )
    if retry_errors and cmd.get("tags"):
        raise EvalRunError(
            "Cannot use --tag with --retry-errors. Retried evaluations keep their original tags."
        )

    # When resuming, load the config from the saved eval and skip CLI filters that would
    # change test indices.
    resume_eval: Eval | None = None
    retry_error_result_ids: list[str] | None = None
    if resume_raw:
        if cmd.get("write") is False:
            raise EvalRunError(
                "Cannot use --resume with --no-write. Resume functionality requires database "
                "persistence."
            )
        resume_id = "latest" if resume_raw is True else resume_raw
        resume_eval = Eval.latest() if resume_id == "latest" else Eval.find_by_id(resume_id)
        if resume_eval is None:
            raise EvalRunError(f"Could not find evaluation to resume: {resume_id}")
        logger.info("Resuming evaluation %s...", resume_eval.id)
        resolved = await _resolve_replay_configs(resume_eval, "resuming")
    elif retry_errors:
        if cmd.get("write") is False:
            raise EvalRunError(
                "Cannot use --retry-errors with --no-write. Retry functionality requires "
                "database persistence."
            )
        logger.info("🔄 Retrying ERROR results from latest evaluation...")
        latest = Eval.latest()
        if latest is None:
            raise EvalRunError("No previous evaluation found to retry errors from")
        # Capture the ERROR ids before the retry; they are deleted only once it succeeds.
        retry_error_result_ids = get_error_result_ids(latest.id)
        if not retry_error_result_ids:
            logger.info("✅ No ERROR results found in the latest evaluation")
            return latest, 0
        logger.info("Found %d ERROR results to retry", len(retry_error_result_ids))
        logger.info(
            "🔄 Running evaluation with resume mode to retry %d test cases...",
            len(retry_error_result_ids),
        )
        resume_eval = latest
        resolved = await _resolve_replay_configs(resume_eval, "retrying errors for")
    else:
        resolved = await resolve_configs(cmd, default_config)
    config: dict[str, Any] = resolved["config"]
    test_suite: dict[str, Any] = resolved["test_suite"]
    command_line_options: dict[str, Any] = resolved.get("command_line_options") or {}
    if watch_state is not None:
        watch_state.update(
            config=config,
            config_paths=[p for p in (cmd.get("config") or [default_config_path]) if p],
            test_sources=resolved.get("test_sources"),
        )
    cli_state.set_base_path(resolved["base_path"])
    if command_line_options.get("safeMode"):
        cli_state.set_safe_mode(True)

    action = "retrying errors for" if retry_errors else "resuming"
    persisted_provider_filter = (
        get_persisted_provider_filter_options(
            (resume_eval.runtime_options or {}).get("providerFilter")
        ).get("filterProviders")
        if resume_eval
        else None
    )
    cli_provider_filter = cmd.get("filterProviders") or cmd.get("filterTargets")
    if resume_eval and cli_provider_filter and cli_provider_filter != persisted_provider_filter:
        stored = f'"{persisted_provider_filter}"' if persisted_provider_filter else "(none)"
        logger.warning(
            'Ignoring --filter-providers/--filter-targets "%s": %s evaluation %s with stored '
            "provider filter %s to preserve test indices.",
            cli_provider_filter, action, resume_eval.id, stored,
        )  # fmt: skip
    if resume_eval and persisted_provider_filter and not test_suite["providers"]:
        raise EvalRunError(
            f'Stored provider filter "{persisted_provider_filter}" matched no providers while '
            f"{action} evaluation {resume_eval.id}. The evaluation was not changed."
        )
    if resume_eval:
        cli_state.resume = True
        if retry_error_result_ids:
            cli_state.retry_mode = True
            cli_state.retry_error_result_ids = retry_error_result_ids

    if config.get("evaluateOptions"):
        evaluate_options = {
            **evaluate_options,
            **config["evaluateOptions"],
            "eventSource": evaluate_options.get("eventSource"),
        }

    # Resume and error retries keep the original expansion and cache settings.
    resume_runtime = (resume_eval.runtime_options if resume_eval else None) or {}
    if resume_eval:
        persisted = resume_eval.runtime_options or config.get("evaluateOptions") or {}
        persisted_repeat = persisted.get("repeat")
        repeat = (
            persisted_repeat
            if isinstance(persisted_repeat, int)
            and not isinstance(persisted_repeat, bool)
            and persisted_repeat > 0
            else 1
        )
        cache_enabled = _first_defined(persisted.get("cache"), True)
        max_concurrency = _to_int(persisted.get("maxConcurrency")) or DEFAULT_MAX_CONCURRENCY
        delay = _to_int(persisted.get("delay")) or 0
        explicit_max = _first_defined(
            resume_runtime.get("maxConcurrency"),
            cmd.get("maxConcurrency"),
            command_line_options.get("maxConcurrency"),
            evaluate_options.get("maxConcurrency"),
        )
    else:
        iterations = _first_defined(
            cmd.get("repeat"), command_line_options.get("repeat"), evaluate_options.get("repeat")
        )
        iterations = _to_int(iterations)
        repeat = iterations if iterations and iterations > 0 else 1
        cache_enabled = _first_defined(
            cmd.get("cache"), command_line_options.get("cache"), evaluate_options.get("cache"),
            True,
        )  # fmt: skip
        explicit_max = _first_defined(
            cmd.get("maxConcurrency"),
            command_line_options.get("maxConcurrency"),
            evaluate_options.get("maxConcurrency"),
        )
        max_concurrency = _to_int(explicit_max) or DEFAULT_MAX_CONCURRENCY
        delay = (
            _to_int(
                _first_defined(
                    cmd.get("delay"),
                    command_line_options.get("delay"),
                    evaluate_options.get("delay"),
                    0,
                )
            )
            or 0
        )
    if cache_enabled is False:
        logger.info("Cache is disabled.")
        cache.disable_cache()
    if delay > 0:
        max_concurrency = 1
        cli_state.set_max_concurrency(1)
        logger.info(
            "Running at concurrency=1 because %sms delay was requested between API calls", delay
        )
    elif explicit_max is not None:
        cli_state.set_max_concurrency(_to_int(explicit_max))

    has_scenarios = bool(test_suite.get("scenarios"))
    can_synthesize_default = test_suite.get("scenarios") is None
    explicit_test_count = len(test_suite["tests"]) if test_suite.get("tests") is not None else None
    persisted_filter_range = resume_runtime.get("filterRange")
    persisted_filter_range = (
        persisted_filter_range if isinstance(persisted_filter_range, str) else None
    )
    resume_filter_range = _first_defined(
        persisted_filter_range,
        command_line_options.get("filterRange"),
        evaluate_options.get("filterRange"),
    )
    if resume_eval and cmd.get("filterRange") and cmd["filterRange"] != resume_filter_range:
        logger.warning(
            "Ignoring --filter-range %s: resuming %s with stored range %s to preserve test "
            "indices.",
            cmd["filterRange"], resume_eval.id, resume_filter_range or "(none)",
        )  # fmt: skip
    filter_range = (
        resume_filter_range
        if resume_eval
        else _first_defined(
            cmd.get("filterRange"),
            command_line_options.get("filterRange"),
            evaluate_options.get("filterRange"),
        )
    )
    filter_sample = _first_defined(
        cmd.get("filterSample"), command_line_options.get("filterSample")
    )
    filter_seed = _first_defined(
        cmd.get("filterSampleSeed"), command_line_options.get("filterSampleSeed")
    )
    has_filter = any(
        v is not None
        for v in (filter_range, cmd.get("filterFailing"), cmd.get("filterFailingOnly"),
                  cmd.get("filterErrorsOnly"), cmd.get("filterFirstN"),
                  cmd.get("filterMetadata"), cmd.get("filterPattern"), filter_sample)
    )  # fmt: skip
    apply_to_implicit_default = (
        has_filter and can_synthesize_default and not test_suite.get("tests")
    )
    # Apply filtering only when not resuming, to preserve test indices.
    if not resume_eval:
        if apply_to_implicit_default:
            default_test = test_suite.get("defaultTest")
            metadata = default_test.get("metadata") if isinstance(default_test, dict) else None
            test_suite["tests"] = [{"metadata": metadata}] if metadata else [{}]
        test_suite["tests"] = await filter_tests(
            test_suite,
            {
                "failing": cmd.get("filterFailing"),
                "failingOnly": cmd.get("filterFailingOnly"),
                "errorsOnly": cmd.get("filterErrorsOnly"),
                "firstN": cmd.get("filterFirstN"),
                "metadata": cmd.get("filterMetadata"),
                "pattern": cmd.get("filterPattern"),
                "range": None if has_scenarios else filter_range,
                "sample": filter_sample,
                "sampleSeed": filter_seed,
            },
        )
        if (
            not test_suite["tests"]
            and not has_scenarios
            and ((explicit_test_count or 0) > 0 or apply_to_implicit_default)
        ):
            test_suite["scenarios"] = []
        test_suite["providers"] = filter_providers(test_suite["providers"], cli_provider_filter)

    provider_filter = persisted_provider_filter if resume_eval else cli_provider_filter

    options: dict[str, Any] = {
        # Orchestration keys are not persisted with the eval.
        **{
            k: v
            for k, v in evaluate_options.items()
            if k not in ("providerFilter", "restorePromptColumns", "abortSignal", "pauseSignal")
        },
        "showProgressBar": False,
        "repeat": repeat,
        "filterRange": filter_range,
        "maxConcurrency": max_concurrency,
        "cache": cache_enabled,
    }
    if delay > 0:
        options["delay"] = delay
    options = {k: v for k, v in options.items() if v is not None}

    if not resume_eval and cmd.get("grader"):
        default_test = test_suite.get("defaultTest")
        default_test = default_test if isinstance(default_test, dict) else {}
        default_test["options"] = {**(default_test.get("options") or {}), "provider": cmd["grader"]}
        test_suite["defaultTest"] = default_test
    if not resume_eval and cmd.get("var"):
        default_test = test_suite.get("defaultTest")
        default_test = default_test if isinstance(default_test, dict) else {}
        default_test["vars"] = {**(default_test.get("vars") or {}), **cmd["var"]}
        test_suite["defaultTest"] = default_test
    runtime_tags = (
        None
        if resume_eval
        else {**(command_line_options.get("tags") or {}), **(cmd.get("tags") or {})}
    )
    if runtime_tags:
        config["tags"] = {**(config.get("tags") or {}), **runtime_tags}
    options.update(resolve_suggestion_options(cmd, command_line_options, options))

    runtime_options = {
        **options,
        **({"providerFilter": provider_filter} if provider_filter else {}),
    }
    author = get_author()
    if resume_eval:
        record = resume_eval
    elif cmd.get("write"):
        record = Eval.create(
            config, test_suite["prompts"], author=author, runtime_options=runtime_options
        )
    else:
        record = Eval(config, author=author, runtime_options=runtime_options)

    if resume_raw and not record.use_old_results():
        _backfill_cache_hit_counts(record)

    # Graceful pause via Ctrl+C, only when writing to the database. Pause/resume SIGINT
    # behavior is CLI policy; library callers own their cancellation.
    is_cli = evaluate_options.get("eventSource") == "cli"
    pause = _PauseHandler() if is_cli and cmd.get("write") is not False else None
    try:
        if pause:
            pause.install()
        await evaluate(
            test_suite,
            record,
            {
                **options,
                "restorePromptColumns": bool(resume_eval),
                "filterRange": filter_range if has_scenarios or resume_eval else None,
                **({"abortSignal": evaluate_options["abortSignal"]}
                   if evaluate_options.get("abortSignal") else {}),
                **({"pauseSignal": pause.controller.signal} if pause else {}),
            },
        )  # fmt: skip
        # Retry-errors cleanup, now that the retry succeeded: drop the old ERROR rows and
        # recalculate metrics. Skipped when paused.
        if retry_errors and cli_state.retry_error_result_ids and not (pause and pause.paused):
            error_ids = cli_state.retry_error_result_ids
            try:
                delete_error_results(error_ids)
                await recalculate_prompt_metrics(record)
                logger.debug(
                    "Cleaned up %d old ERROR results after successful retry", len(error_ids)
                )
            except Exception as e:  # noqa: BLE001 - cleanup failure is non-fatal
                logger.warning("Post-retry cleanup had issues. Retry results are saved. %s", e)
            finally:
                cli_state.retry_error_result_ids = None
                cli_state.retry_mode = False
    finally:
        if pause:
            pause.uninstall()
        cli_state.resume = False
        cli_state.retry_mode = False
        cli_state.retry_error_result_ids = None

    if pause and pause.paused:
        from promptphoo.output.table import terminal_max_width

        border = "=" * terminal_max_width()
        logger.info("%s", border)
        logger.info("⏸ Evaluation paused. ID: %s", record.id)
        logger.info("» Resume with: promptphoo eval --resume %s", record.id)
        logger.info("%s", border)
        return record, 0

    # Persisted evals reload their results for the table and outputs.
    if record.persisted:
        record.clear_results()
    return record, _report(cmd, config, command_line_options, record, start, max_concurrency)


async def _resolve_replay_configs(eval_record: Eval, action: str) -> dict[str, Any]:
    """resolveReplayConfigs: resolve the saved config with the saved provider filter."""
    from promptphoo.config.load import ConfigResolutionError, render_config_env_templates

    options = get_persisted_provider_filter_options(
        (eval_record.runtime_options or {}).get("providerFilter")
    )
    provider_filter = options.get("filterProviders")
    regex_error = get_provider_filter_regex_error(provider_filter) if provider_filter else None
    if provider_filter and regex_error:
        raise ConfigResolutionError(
            f'Could not apply stored provider filter "{provider_filter}" while {action} '
            f"evaluation {eval_record.id}: {regex_error}. The evaluation was not changed."
        )
    replay_config = eval_record.config
    if (replay_config.get("tracing") or {}).get("provider"):
        rendered = render_config_env_templates(replay_config)
        replay_config = {
            **replay_config,
            **({"env": rendered["env"]} if rendered.get("env") else {}),
            "tracing": rendered.get("tracing"),
        }
    configs = await resolve_configs(dict(options), replay_config)
    configs["test_suite"]["providers"] = filter_providers(
        configs["test_suite"]["providers"], provider_filter
    )
    return configs


def _backfill_cache_hit_counts(record: Eval) -> None:
    """Older evals lack testCacheHitCount. Recover it without recalculating other metrics."""
    pending: dict[int, int] = {
        idx: 0
        for idx, prompt in enumerate(record.prompts)
        if prompt.get("metrics") and "testCacheHitCount" not in prompt["metrics"]
    }
    if not pending:
        return
    for batch in record.fetch_results_batched():
        for result in batch:
            if result.prompt_idx in pending and (result.response or {}).get("cached"):
                pending[result.prompt_idx] += 1
    # Publish only after every batch succeeded.
    for idx, count in pending.items():
        record.prompts[idx]["metrics"]["testCacheHitCount"] = count


class _PauseHandler:
    """The CLI's SIGINT policy: the first Ctrl+C pauses (aborts the pause signal so running
    steps stop and are left for --resume), a second one force-exits with 130."""

    def __init__(self) -> None:
        from promptphoo.scheduler.cancellation import AbortController

        self.controller = AbortController()
        self.paused = False
        self._installed = False
        self._force_exit: asyncio.TimerHandle | None = None

    def install(self) -> None:
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT, self._on_sigint)
            self._installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            pass  # no signal support (non-main thread or platform)

    def uninstall(self) -> None:
        if self._force_exit:
            self._force_exit.cancel()
            self._force_exit = None
        if self._installed:
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)
            self._installed = False

    def _on_sigint(self) -> None:
        was_paused, self.paused = self.paused, True
        if was_paused:
            # Skip closing the DB: a WAL checkpoint could block the escape hatch.
            logger.warning("Force exiting...")
            os._exit(130)
        logger.info("Pausing evaluation... Press Ctrl+C again to force exit.")
        self.controller.abort()
        self._force_exit = asyncio.get_running_loop().call_later(10, self._timed_out)

    @staticmethod
    def _timed_out() -> None:
        logger.warning("Evaluation shutdown timed out, force exiting...")
        os._exit(130)


def _report(
    cmd: dict[str, Any],
    config: dict[str, Any],
    command_line_options: dict[str, Any],
    record: Eval,
    start: float,
    max_concurrency: int,
) -> int:
    from promptphoo.output import convert_results_to_table, write_multiple_outputs
    from promptphoo.output.summary import generate_eval_summary
    from promptphoo.output.table import generate_table

    stats = record.get_stats()
    successes, failures, errors = stats["successes"], stats["failures"], stats["errors"]
    total = successes + failures + errors
    pass_rate = (successes / total) * 100 if total else math.nan

    results_file = record.to_results_file()
    if cmd.get("table") and logger.getEffectiveLevel() > logging.DEBUG and total < 500:
        table = convert_results_to_table(results_file)
        logger.info(
            "\n%s",
            generate_table(
                table,
                cmd.get("tableCellMaxLength")
                or command_line_options.get("tableCellMaxLength")
                or 250,
            ),
        )
        if len(table["body"]) > 25:
            rows_left = len(table["body"]) - 25
            logger.info(
                "... %d more row%s not shown ...\n", rows_left, "" if rows_left == 1 else "s"
            )
    if total >= 500:
        logger.info("Skipping table output because there are more than 500 tests.")

    output_path = config.get("outputPath")
    paths = [p for p in (output_path if isinstance(output_path, list) else [output_path])
             if isinstance(p, str) and p]  # fmt: skip
    for line in generate_eval_summary(
        eval_id=record.id,
        is_redteam=bool(config.get("redteam")),
        write_to_database=cmd.get("write") is not False,
        token_usage=stats["tokenUsage"],
        successes=successes,
        failures=failures,
        errors=errors,
        cache_hits=stats.get("cacheHits"),
        duration=round(time.time() - start),
        max_concurrency=max_concurrency,
    ):
        logger.info("%s", line)
    if paths:
        write_multiple_outputs(paths, record, None)
        logger.info("Writing output to %s", ", ".join(paths))

    threshold = envars.get_env_float("PROMPTFOO_PASS_RATE_THRESHOLD", 100)
    failed_exit = envars.get_env_int("PROMPTFOO_FAILED_TEST_EXIT_CODE", 100)
    below = pass_rate < (threshold if math.isfinite(threshold) else 100)
    if below:
        if envars.get_env_float("PROMPTFOO_PASS_RATE_THRESHOLD") is not None:
            logger.info(
                "Pass rate %.2f%% is below the threshold of %s%%",
                pass_rate,
                _js_number(threshold),
            )
        return failed_exit
    return 0


def resolve_suggestion_options(
    cmd: dict[str, Any], command_line_options: dict[str, Any], evaluate_options: dict[str, Any]
) -> dict[str, Any]:
    """resolveSuggestionOptions: --suggest-prompts N enables suggestions on its own."""
    if cmd.get("suggest_prompts") is not None:
        return {"generateSuggestions": True, "suggestionsCount": cmd["suggest_prompts"]}
    explicit = _first_defined(
        cmd.get("generateSuggestions"),
        command_line_options.get("generateSuggestions"),
        evaluate_options.get("generateSuggestions"),
    )
    if explicit is not True:
        return {"generateSuggestions": False} if explicit is False else {}
    count = _first_defined(
        cmd.get("suggestionsCount"),
        command_line_options.get("suggestionsCount"),
        evaluate_options.get("suggestionsCount"),
    )
    return {"generateSuggestions": True, **({} if count is None else {"suggestionsCount": count})}


def _js_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _first_defined(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


# --- click wiring ---------------------------------------------------------------------------


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    evaluate_options: dict[str, Any] = {}
    if default_config.get("evaluateOptions"):
        eo = default_config["evaluateOptions"]
        for key in ("generateSuggestions", "suggestionsCount", "maxConcurrency", "showProgressBar"):
            if key in eo:
                evaluate_options[key] = eo[key]
    clo = default_config.get("commandLineOptions") or {}
    default_test = default_config.get("defaultTest")
    default_test_options = (
        (default_test or {}).get("options") or {} if isinstance(default_test, dict) else {}
    )

    @cli.command("eval", help="Evaluate prompts")
    @click.option(
        "-c",
        "--config",
        cls=VariadicOption,
        help="Path to configuration file. Automatically loads promptfooconfig.yaml",
    )
    @click.option("-a", "--assertions", help="Path to assertions file")
    @click.option("-p", "--prompts", cls=VariadicOption, help="Paths to prompt files (.txt)")
    @click.option(
        "-r",
        "--providers",
        cls=VariadicOption,
        help="One of: openai:chat, openai:completion, openai:<model name>, or path to "
        "custom API caller module",
    )
    @click.option("-t", "--tests", help="Path to CSV with test cases")
    @click.option(
        "-v",
        "--vars",
        "vars_",
        default=clo.get("vars"),
        help="Path to CSV with test cases (alias for --tests)",
    )
    @click.option("--model-outputs", help="Path to JSON containing list of LLM output strings")
    @click.option(
        "--prompt-prefix",
        default=default_test_options.get("prefix"),
        help="This prefix is prepended to every prompt",
    )
    @click.option(
        "--prompt-suffix",
        default=default_test_options.get("suffix"),
        help="This suffix is appended to every prompt.",
    )
    @click.option("--var", "var", multiple=True, help="Set a variable in key=value format")
    @click.option("--tag", multiple=True, help="Set an eval tag in key=value format.")
    @click.option(
        "-j",
        "--max-concurrency",
        type=int,
        help=f"Maximum number of concurrent API calls (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    @click.option("--repeat", type=int, help="Number of times to run each test (default: 1)")
    @click.option(
        "--delay", type=int, help="Delay between each test (in milliseconds) (default: 0)"
    )
    @click.option(
        "--no-cache",
        "no_cache",
        is_flag=True,
        default=False,
        help="Do not read or write results to disk cache",
    )
    @click.option("--remote", is_flag=True, default=False, hidden=True)
    @click.option(
        "--safe-mode",
        is_flag=True,
        default=bool(clo.get("safeMode")),
        help="Disable dynamic inline code execution for transforms and assertions",
    )
    @click.option("-n", "--filter-first-n", type=int, help="Only run the first N tests")
    @click.option(
        "--filter-pattern",
        help="Only run tests whose description matches the regular expression pattern",
    )
    @click.option("--filter-range", help="Only run tests whose zero-based index is in the range")
    @click.option(
        "--filter-prompts",
        help="Only run tests with prompts whose id or label matches the regex pattern",
    )
    @click.option(
        "--filter-providers",
        "--filter-targets",
        "filter_providers_",
        help="Only run tests with these providers (regex match)",
    )
    @click.option("--filter-sample", type=int, help="Only run a random sample of N tests")
    @click.option(
        "--filter-sample-seed",
        type=int,
        help="Numeric seed used to make --filter-sample repeatable",
    )
    @click.option(
        "--filter-failing",
        metavar="<path or id>",
        help="Path to json output file or eval ID to filter non-passing tests from "
        "(failures + errors)",
    )
    @click.option(
        "--filter-failing-only",
        metavar="<path or id>",
        help="Path to json output file or eval ID to filter assertion failures from "
        "(excludes errors)",
    )
    @click.option(
        "--filter-errors-only",
        metavar="<path or id>",
        help="Path to json output file or eval ID to filter error tests from",
    )
    @click.option(
        "--filter-metadata", multiple=True, help="Only run tests whose metadata matches key=value"
    )
    @click.option("-o", "--output", cls=VariadicOption, help="Path to output file")
    @click.option("--table/--no-table", default=clo.get("table", True), help="Output table in CLI")
    @click.option(
        "--table-cell-max-length", type=int, help="Truncate console table cells to this length"
    )
    @click.option("--share/--no-share", default=None, help="Create a shareable URL")
    @click.option(
        "--resume",
        is_flag=False,
        flag_value="latest",
        default=None,
        metavar="[evalId]",
        help="Resume a paused/incomplete evaluation. Defaults to latest when omitted",
    )
    @click.option(
        "--retry-errors",
        is_flag=True,
        default=False,
        help="Retry all ERROR results from the latest evaluation",
    )
    @click.option(
        "--write/--no-write",
        default=clo.get("write", True),
        help="Write results to promptfoo directory",
    )
    @click.option("--grader", default=clo.get("grader"), help="Model that will grade outputs")
    @click.option(
        "--suggest-prompts",
        type=click.IntRange(1, 50),
        help="Generate N new prompts (1-50) and append them to the prompt list",
    )
    @click.option(
        "-w", "--watch", is_flag=True, default=False, help="Watch for changes in config and re-run"
    )
    @click.option(
        "-x",
        "--extension",
        cls=VariadicOption,
        help="Extension hooks to run (e.g., file://handler.py:afterAll)",
    )
    @click.option("--description", help="Description of the eval run")
    @click.option("--progress-bar/--no-progress-bar", default=None)
    @click.option(
        "--env-file", "--env-path", "env_file", multiple=True, help="Path(s) to .env file(s)."
    )
    @click.option("--verbose", is_flag=True, default=False, help="Show debug logs")
    @click.argument("extra", nargs=-1)
    def eval_cmd(**opts: Any) -> None:  # fmt: skip
        from promptphoo.output import SUPPORTED_OUTPUT_FILE_FORMATS, get_output_file_format

        if opts["extra"]:
            raise click.UsageError(
                f"Unknown command: {opts['extra'][0]}. Did you mean -c {opts['extra'][0]}?"
            )
        for path in opts["output"] or ():
            if not get_output_file_format(path):
                raise click.UsageError(
                    f"Unsupported output file format: {path}. Please use one of: "
                    f"{', '.join(SUPPORTED_OUTPUT_FILE_FORMATS)}."
                )
        cmd = _to_command_options(opts)
        if opts["safe_mode"]:
            cli_state.set_safe_mode(True)
        try:
            exit_code = run_eval_command(cmd, default_config, default_config_path, evaluate_options)
        finally:
            db.close_db()
        if exit_code:
            raise SystemExit(exit_code)

    return eval_cmd


def _recoverable_errors() -> tuple[type[BaseException], ...]:
    """Errors reported as one log line (handleRecoverableWatchError) instead of a traceback."""
    from promptphoo.config.load import ConfigResolutionError
    from promptphoo.evaluator.extensions import ExtensionHookError
    from promptphoo.suggestions import PromptSuggestionsRejectedError

    return EvalRunError, ExtensionHookError, PromptSuggestionsRejectedError, ConfigResolutionError


def _log_error(error: BaseException) -> None:
    """logConfigResolutionError for config errors (at their own level), else logger.error."""
    cli_message = getattr(error, "cli_message", None)
    if cli_message is not None:
        log = logger.warning if getattr(error, "log_level", "error") == "warn" else logger.error
        log("%s", cli_message)
    else:
        logger.error("%s", error)


def run_eval_command(
    cmd: dict[str, Any],
    default_config: dict[str, Any],
    default_config_path: str | None,
    evaluate_options: dict[str, Any],
    *,
    poll_interval: float | None = None,
    should_stop: Any = None,
) -> int:
    """Run `eval` once, or keep re-running it on file changes with --watch. Returns the exit
    code. In watch mode the pass-rate exit code is not applied, as upstream."""
    from promptphoo.cli import watch

    def run_once(watch_state: dict[str, Any] | None = None) -> int:
        _, code = asyncio.run(
            do_eval(
                dict(cmd),
                default_config,
                default_config_path,
                {**evaluate_options, "eventSource": "cli"},
                watch_state,
            )
        )
        return code

    watching = bool(cmd.get("watch")) and not cmd.get("resume")
    state: dict[str, Any] = {}
    try:
        exit_code = run_once(state if watching else None)
    except _recoverable_errors() as e:
        _log_error(e)
        return 1
    if not watching:
        return exit_code

    config_paths = state.get("config_paths") or []
    if not config_paths:
        logger.error(
            "Could not locate config file(s) to watch. Pass --config path/to/promptfooconfig.yaml "
            "or run from a directory containing promptfooconfig.{%s}.",
            ",".join(DEFAULT_CONFIG_EXTENSIONS),
        )
        return 1
    paths = watch.get_watch_paths(cmd, state["config"], config_paths, state.get("test_sources"))

    def rerun() -> None:
        try:
            run_once()
        except _recoverable_errors() as e:
            _log_error(e)

    watch.watch_and_rerun(
        paths,
        rerun,
        interval=watch.DEFAULT_POLL_INTERVAL if poll_interval is None else poll_interval,
        should_stop=should_stop or (lambda: False),
    )
    return 0


def _to_command_options(opts: dict[str, Any]) -> dict[str, Any]:
    def listed(value: Any) -> list[str] | None:
        return list(value) if value else None

    cmd: dict[str, Any] = {
        "config": listed(opts["config"]),
        "assertions": opts["assertions"],
        "prompts": listed(opts["prompts"]),
        "providers": listed(opts["providers"]),
        "tests": opts["tests"],
        "vars": opts["vars_"],
        "modelOutputs": opts["model_outputs"],
        "promptPrefix": opts["prompt_prefix"],
        "promptSuffix": opts["prompt_suffix"],
        "var": collect_key_value("--var", opts["var"]),
        "tags": collect_key_value("--tag", opts["tag"]),
        "maxConcurrency": opts["max_concurrency"],
        "repeat": opts["repeat"],
        "delay": opts["delay"],
        "filterFirstN": opts["filter_first_n"],
        "filterPattern": opts["filter_pattern"],
        "filterRange": opts["filter_range"],
        "filterPrompts": opts["filter_prompts"],
        "filterProviders": opts["filter_providers_"],
        "filterSample": opts["filter_sample"],
        "filterSampleSeed": opts["filter_sample_seed"],
        "filterMetadata": listed(opts["filter_metadata"]),
        "output": listed(opts["output"]),
        "table": opts["table"],
        "tableCellMaxLength": opts["table_cell_max_length"],
        "write": opts["write"],
        "grader": opts["grader"],
        "description": opts["description"],
        "progressBar": opts["progress_bar"],
        "envPath": listed(opts["env_file"]),
        "resume": opts["resume"],
        "retryErrors": opts["retry_errors"] or None,
        "watch": opts["watch"],
        "suggest_prompts": opts["suggest_prompts"],
        "extension": listed(opts["extension"]),
        "filterFailing": opts["filter_failing"],
        "filterFailingOnly": opts["filter_failing_only"],
        "filterErrorsOnly": opts["filter_errors_only"],
    }
    if opts["no_cache"]:
        cmd["cache"] = False
    if opts["share"]:
        logger.warning("--share is not supported by promptphoo; results stay local.")
    return {k: v for k, v in cmd.items() if v is not None}
