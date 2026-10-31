"""`promptfoo eval`. Ports src/commands/eval.ts and src/node/doEval.ts.

--resume, --retry-errors, --watch, --share, --suggest-prompts, -x extensions and the
--filter-failing family are accepted for compatibility but are Phase 5 work and fail clearly.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Any

import click

from promptphoo import cache, cli_state, db, envars
from promptphoo.cli.options import VariadicOption, collect_key_value
from promptphoo.config.load import maybe_read_config, resolve_configs
from promptphoo.envfile import setup_env
from promptphoo.evaluator.run import DEFAULT_MAX_CONCURRENCY, evaluate
from promptphoo.filters import filter_tests
from promptphoo.models.eval import Eval, get_author
from promptphoo.providers import filter_providers

logger = logging.getLogger("promptphoo")

DEFAULT_CONFIG_EXTENSIONS = ("yaml", "yml", "json", "cjs", "cts", "js", "mjs", "mts", "ts")
_UNSUPPORTED_OPTIONS = {
    "resume": "--resume",
    "retry_errors": "--retry-errors",
    "watch": "--watch",
    "suggest_prompts": "--suggest-prompts",
    "extension": "--extension",
    "filter_failing": "--filter-failing",
    "filter_failing_only": "--filter-failing-only",
    "filter_errors_only": "--filter-errors-only",
}


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
        '"promptfoo init" or pass --config path/to/promptfooconfig.yaml.'
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
) -> tuple[Eval, int]:
    """Run one eval. Returns (eval record, process exit code)."""
    start = time.time()
    setup_env(cmd.get("envPath"))
    for key, flag in _UNSUPPORTED_OPTIONS.items():
        if cmd.get(key):
            raise EvalRunError(f"{flag} is not supported by promptphoo yet")

    if cmd.get("write"):
        db.run_db_migrations()
    if default_config_path:
        default_config, _ = load_default_config(
            os.path.dirname(default_config_path),
            os.path.splitext(os.path.basename(default_config_path))[0],
        )
    if cmd.get("config") is not None:
        cmd["config"], default_config = _resolve_config_paths(cmd["config"], default_config)

    resolved = await resolve_configs(cmd, default_config)
    config: dict[str, Any] = resolved["config"]
    test_suite: dict[str, Any] = resolved["test_suite"]
    command_line_options: dict[str, Any] = resolved.get("command_line_options") or {}
    cli_state.set_base_path(resolved["base_path"])
    if command_line_options.get("safeMode"):
        cli_state.set_safe_mode(True)

    if config.get("evaluateOptions"):
        evaluate_options = {
            **evaluate_options,
            **config["evaluateOptions"],
            "eventSource": evaluate_options.get("eventSource"),
        }

    iterations = _first_defined(
        cmd.get("repeat"), command_line_options.get("repeat"), evaluate_options.get("repeat")
    )
    iterations = _to_int(iterations)
    repeat = iterations if iterations and iterations > 0 else 1
    cache_enabled = _first_defined(
        cmd.get("cache"), command_line_options.get("cache"), evaluate_options.get("cache"), True
    )
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
    filter_range = _first_defined(
        cmd.get("filterRange"),
        command_line_options.get("filterRange"),
        evaluate_options.get("filterRange"),
    )
    filter_sample = _first_defined(
        cmd.get("filterSample"), command_line_options.get("filterSample")
    )
    filter_seed = _first_defined(
        cmd.get("filterSampleSeed"), command_line_options.get("filterSampleSeed")
    )
    has_filter = any(
        v is not None
        for v in (filter_range, cmd.get("filterFirstN"), cmd.get("filterMetadata"),
                  cmd.get("filterPattern"), filter_sample)
    )  # fmt: skip
    apply_to_implicit_default = (
        has_filter and can_synthesize_default and not test_suite.get("tests")
    )
    if apply_to_implicit_default:
        default_test = test_suite.get("defaultTest")
        metadata = default_test.get("metadata") if isinstance(default_test, dict) else None
        test_suite["tests"] = [{"metadata": metadata}] if metadata else [{}]
    test_suite["tests"] = await filter_tests(
        test_suite,
        {
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

    provider_filter = cmd.get("filterProviders") or cmd.get("filterTargets")
    test_suite["providers"] = filter_providers(test_suite["providers"], provider_filter)

    options: dict[str, Any] = {
        **{k: v for k, v in evaluate_options.items() if k not in ("providerFilter",)},
        "showProgressBar": False,
        "repeat": repeat,
        "filterRange": filter_range,
        "maxConcurrency": max_concurrency,
        "cache": cache_enabled,
    }
    if delay > 0:
        options["delay"] = delay
    options = {k: v for k, v in options.items() if v is not None}

    if cmd.get("grader"):
        default_test = test_suite.get("defaultTest")
        default_test = default_test if isinstance(default_test, dict) else {}
        default_test["options"] = {**(default_test.get("options") or {}), "provider": cmd["grader"]}
        test_suite["defaultTest"] = default_test
    if cmd.get("var"):
        default_test = test_suite.get("defaultTest")
        default_test = default_test if isinstance(default_test, dict) else {}
        default_test["vars"] = {**(default_test.get("vars") or {}), **cmd["var"]}
        test_suite["defaultTest"] = default_test
    runtime_tags = cmd.get("tags") or command_line_options.get("tags")
    if runtime_tags:
        config["tags"] = {**(config.get("tags") or {}), **runtime_tags}

    runtime_options = {
        **options,
        **({"providerFilter": provider_filter} if provider_filter else {}),
    }
    author = get_author()
    if cmd.get("write"):
        record = Eval.create(
            config, test_suite["prompts"], author=author, runtime_options=runtime_options
        )
    else:
        record = Eval(config, author=author, runtime_options=runtime_options)

    await evaluate(
        test_suite,
        record,
        {
            **options,
            "filterRange": filter_range if has_scenarios else None,
        },
    )
    return record, _report(cmd, config, command_line_options, record, start, max_concurrency)


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
    @click.option("--filter-failing")
    @click.option("--filter-failing-only")
    @click.option("--filter-errors-only")
    @click.option(
        "--filter-metadata", multiple=True, help="Only run tests whose metadata matches key=value"
    )
    @click.option("-o", "--output", cls=VariadicOption, help="Path to output file")
    @click.option("--table/--no-table", default=clo.get("table", True), help="Output table in CLI")
    @click.option(
        "--table-cell-max-length", type=int, help="Truncate console table cells to this length"
    )
    @click.option("--share/--no-share", default=None, help="Create a shareable URL")
    @click.option("--resume", is_flag=False, flag_value="latest", default=None)
    @click.option("--retry-errors", is_flag=True, default=False)
    @click.option(
        "--write/--no-write",
        default=clo.get("write", True),
        help="Write results to promptfoo directory",
    )
    @click.option("--grader", default=clo.get("grader"), help="Model that will grade outputs")
    @click.option("--suggest-prompts", type=int)
    @click.option("-w", "--watch", is_flag=True, default=False)
    @click.option("-x", "--extension", cls=VariadicOption)
    @click.option("--description", help="Description of the eval run")
    @click.option("--progress-bar/--no-progress-bar", default=None)
    @click.option(
        "--env-file", "--env-path", "env_file", multiple=True, help="Path(s) to .env file(s)."
    )
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
            _, exit_code = asyncio.run(
                do_eval(
                    cmd,
                    default_config,
                    default_config_path,
                    {**evaluate_options, "eventSource": "cli"},
                )
            )
        except EvalRunError as e:
            logger.error("%s", e)
            raise SystemExit(1) from e
        finally:
            db.close_db()
        if exit_code:
            raise SystemExit(exit_code)

    return eval_cmd


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
        "retry_errors": opts["retry_errors"],
        "watch": opts["watch"],
        "suggest_prompts": opts["suggest_prompts"],
        "extension": listed(opts["extension"]),
        "filter_failing": opts["filter_failing"],
        "filter_failing_only": opts["filter_failing_only"],
        "filter_errors_only": opts["filter_errors_only"],
    }
    if opts["no_cache"]:
        cmd["cache"] = False
    if opts["share"]:
        logger.warning("--share is not supported by promptphoo; results stay local.")
    return {k: v for k, v in cmd.items() if v is not None}
