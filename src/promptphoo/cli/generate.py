"""`promptfoo generate dataset|assertions`. Ports src/commands/generate/dataset.ts and
src/commands/generate/assertions.ts (`generate redteam` is out of scope)."""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Any

import click

from promptphoo import cache
from promptphoo.envfile import setup_env
from promptphoo.synthesis import parse_int

logger = logging.getLogger("promptphoo")

ASSERTION_TYPES = ("pi", "g-eval", "llm-rubric")


class GenerateError(RuntimeError):
    pass


def serialize_object_array_as_csv(rows: list[dict[str, Any]]) -> str:
    """serializeObjectArrayAsCSV from src/csv.ts (deliberately not formula-escaped)."""
    from promptphoo.jsvalues import js_string
    from promptphoo.synthesis import invariant

    invariant(len(rows) > 0, "No variables to serialize")
    column_names = ",".join(rows[0].keys())
    body = "\n".join(
        '"' + '","'.join(js_string(v).replace('"', '""') for v in row.values()) + '"'
        for row in rows
    )
    return "\n".join([column_names, body]) + "\n"


async def _resolve_test_suite(
    config_path: str | None, default_config: dict[str, Any], config_type: str
) -> dict[str, Any]:
    from promptphoo.config.load import resolve_configs

    if not config_path:
        raise GenerateError(
            "Could not find a config file. Pass --config path/to/promptfooconfig.yaml or run "
            '"promptphoo init" to create one.'
        )
    resolved = await resolve_configs({"config": [config_path]}, default_config, config_type)
    return resolved["test_suite"]


# Upstream declares `.option('--no-cache', ..., false)`, which makes commander default `cache`
# to false, so generate commands always run with the cache disabled. Kept as is.


def _setup(env_file: Any, use_cache: bool) -> None:
    setup_env(list(env_file) if env_file else None)
    if not use_cache:
        logger.info("Cache is disabled.")
        cache.disable_cache()


def _int_option(value: Any) -> Any:
    parsed = parse_int(value)
    return None if isinstance(parsed, float) and math.isnan(parsed) else parsed


async def do_generate_dataset(options: dict[str, Any]) -> None:
    from promptphoo.cli.watch import print_border
    from promptphoo.output import jsyaml
    from promptphoo.synthesis.dataset import synthesize_from_test_suite
    from promptphoo.yamlload import load_yaml

    _setup(options.get("env_file"), options.get("cache", False))
    config_path = options.get("config") or options.get("default_config_path")
    test_suite = await _resolve_test_suite(
        config_path, options.get("default_config") or {}, "DatasetGeneration"
    )
    results = await synthesize_from_test_suite(
        test_suite,
        {
            "instructions": options.get("instructions"),
            # Number.parseInt: NaN falls back to the defaults inside synthesize.
            "num_personas": _int_option(options.get("num_personas")),
            "num_test_cases_per_persona": _int_option(options.get("num_test_cases_per_persona")),
            "provider": options.get("provider"),
        },
    )
    config_addition = {"tests": [{"vars": result} for result in results]}
    yaml_string = jsyaml.dump(config_addition)
    output = options.get("output")
    if output:
        if output.endswith(".csv"):
            Path(output).write_text(serialize_object_array_as_csv(results), encoding="utf-8")
        elif output.endswith(".yaml"):
            Path(output).write_text(yaml_string, encoding="utf-8")
        else:
            raise GenerateError(f"Unsupported output file type: {output}")
        print_border()
        logger.info("Wrote %d new test cases to %s", len(results), output)
        print_border()
    else:
        print_border()
        logger.info("New test Cases")
        print_border()
        logger.info("%s", yaml_string)

    print_border()
    if options.get("write") and config_path:
        existing = load_yaml(Path(config_path).read_text(encoding="utf-8")) or {}
        existing_tests = existing.get("tests")
        if isinstance(existing_tests, list):
            tests_array = existing_tests
        elif existing_tests:
            tests_array = [existing_tests]
        else:
            tests_array = []
        existing["tests"] = [*tests_array, *config_addition["tests"]]
        Path(config_path).write_text(jsyaml.dump(existing), encoding="utf-8")
        logger.info("Wrote %d new test cases to %s", len(results), config_path)
        logger.info("Run promptphoo eval to run the generated tests")
    else:
        logger.info(
            "Copy the above test cases or run promptphoo generate dataset --write to write "
            "directly to the config"
        )


async def do_generate_assertions(options: dict[str, Any]) -> None:
    from promptphoo.cli.watch import print_border
    from promptphoo.output import jsyaml
    from promptphoo.synthesis.assertions import synthesize_from_test_suite
    from promptphoo.yamlload import load_yaml

    _setup(options.get("env_file"), options.get("cache", False))
    config_path = options.get("config") or options.get("default_config_path")
    test_suite = await _resolve_test_suite(
        config_path, options.get("default_config") or {}, "AssertionGeneration"
    )
    results = await synthesize_from_test_suite(
        test_suite,
        {
            "instructions": options.get("instructions"),
            "num_questions": parse_int(options.get("num_assertions") or "5"),
            "provider": options.get("provider"),
            "type": options.get("type") or "pi",
        },
    )
    config_addition = {"assert": results}
    yaml_string = jsyaml.dump(config_addition)
    output = options.get("output")
    if output:
        if output.endswith(".yaml"):
            Path(output).write_text(yaml_string, encoding="utf-8")
        else:
            raise GenerateError(f"Unsupported output file type: {output}")
        print_border()
        logger.info("Wrote %d new assertions to %s", len(results), output)
        print_border()
    else:
        print_border()
        logger.info("New test Cases")
        print_border()
        logger.info("%s", yaml_string)

    print_border()
    if options.get("write") and config_path:
        existing = load_yaml(Path(config_path).read_text(encoding="utf-8")) or {}
        default_test = existing.get("defaultTest")
        default_test = default_test if isinstance(default_test, dict) else {}
        existing["defaultTest"] = {
            **default_test,
            "assert": [*(default_test.get("assert") or []), *config_addition["assert"]],
        }
        Path(config_path).write_text(jsyaml.dump(existing), encoding="utf-8")
        logger.info("Wrote %d new test cases to %s", len(results), config_path)
        logger.info("Run promptphoo eval to run the generated assertions")
    else:
        logger.info(
            "Copy the above test cases or run promptphoo generate assertions --write to write "
            "directly to the config"
        )


def _validate_assertion_type(ctx: click.Context, param: click.Parameter, value: str) -> str:
    if value not in ASSERTION_TYPES:
        raise click.BadParameter(f"Option --type must be one of: {', '.join(ASSERTION_TYPES)}.")
    return value


def _run(coro: Any) -> None:
    from promptphoo import db
    from promptphoo.config.load import ConfigResolutionError

    try:
        asyncio.run(coro)
    except (GenerateError, ConfigResolutionError, ValueError, RuntimeError) as e:
        logger.error("%s", getattr(e, "cli_message", None) or e)
        raise SystemExit(1) from e
    finally:
        db.close_db()


def _common(fn: Any) -> Any:
    fn = click.option(
        "--env-file", "--env-path", "env_file", multiple=True, help="Path to .env file"
    )(fn)
    return click.option("--verbose", is_flag=True, default=False, help="Show debug logs")(fn)


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.group("generate", help="Generate synthetic data")
    def generate() -> None:
        pass

    @generate.command("dataset", help="Generate test cases")
    @click.option(
        "-i", "--instructions", help="Additional instructions to follow while generating test cases"
    )
    @click.option(
        "-c", "--config", help="Path to configuration file. Defaults to promptfooconfig.yaml"
    )
    @click.option("-o", "--output", help="Path to output file. Supports CSV and YAML output.")
    @click.option(
        "-w", "--write", is_flag=True, help="Write results to promptfoo configuration file"
    )
    @click.option(
        "--provider",
        help="Provider to use for generating adversarial tests. Defaults to the default grading "
        "provider.",
    )
    @click.option(
        "--numPersonas", "num_personas", default="5", help="Number of personas to generate"
    )
    @click.option(
        "--numTestCasesPerPersona",
        "num_test_cases_per_persona",
        default="3",
        help="Number of test cases per persona",
    )
    @click.option(
        "--no-cache",
        "no_cache",
        is_flag=True,
        default=True,
        help="Do not read or write results to disk cache",
    )
    @_common
    def dataset(**opts: Any) -> None:
        _run(
            do_generate_dataset(
                {
                    **opts,
                    "cache": not opts.pop("no_cache"),
                    "default_config": default_config,
                    "default_config_path": default_config_path,
                }
            )
        )

    @generate.command("assertions", help="Generate additional subjective/objective assertions")
    @click.option(
        "-t",
        "--type",
        "type_",
        default="pi",
        callback=_validate_assertion_type,
        help="The type of natural language assertion to generate (pi, g-eval, or llm-rubric)",
    )
    @click.option(
        "-c",
        "--config",
        help="Path to configuration file. Defaults to promptfooconfig.yaml. Requires at least 1 "
        "prompt to be defined.",
    )
    @click.option("-o", "--output", help="Path to output file. Supports YAML output")
    @click.option(
        "-w", "--write", is_flag=True, help="Write results to promptfoo configuration file"
    )
    @click.option("--numAssertions", "num_assertions", help="Number of assertions to generate")
    @click.option(
        "--provider",
        help="Provider to use for generating assertions. Defaults to the default grading provider.",
    )
    @click.option(
        "-i", "--instructions", help="Additional instructions to follow while generating assertions"
    )
    @click.option(
        "--no-cache",
        "no_cache",
        is_flag=True,
        default=True,
        help="Do not read or write results to disk cache",
    )
    @_common
    def assertions(**opts: Any) -> None:
        _run(
            do_generate_assertions(
                {
                    **opts,
                    "type": opts.pop("type_"),
                    "cache": not opts.pop("no_cache"),
                    "default_config": default_config,
                    "default_config_path": default_config_path,
                }
            )
        )

    return generate
