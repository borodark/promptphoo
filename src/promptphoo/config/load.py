"""Config file loading and resolution. Ports src/util/config/load.ts (declarative configs only).

Runtime config is plain dicts, like the Node implementation; the published JSON Schema is used
only to warn about invalid configs, mirroring Node's `safeParse` + `logger.warn`.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from promptphoo import cli_state, envars
from promptphoo.files import glob_sync, maybe_load_from_external_file
from promptphoo.jsvalues import is_truthy, js_string, json_stringify
from promptphoo.promptfoo_command import promptfoo_command
from promptphoo.refs import dereference
from promptphoo.templating import read_filters, render_env_only_in_object
from promptphoo.testcases import (
    is_remote_tests_reference,
    read_test,
    read_test_config,
    read_test_configs,
    read_tests,
)
from promptphoo.yamlload import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_EXTENSIONS = ("yaml", "yml", "json", "cjs", "cts", "js", "mjs", "mts", "ts")


class ConfigResolutionError(Exception):
    def __init__(self, message: str, cli_message: str | None = None, log_level: str = "error"):
        super().__init__(message)
        self.cli_message = cli_message or message
        self.log_level = "warn" if log_level == "warn" else "error"


# --- validation -------------------------------------------------------------------------------


@cache
def _config_schema() -> dict[str, Any]:
    return json.loads(
        (resources.files("promptphoo") / "_data" / "config-schema.json").read_text("utf-8")
    )


@cache
def _config_validator() -> Any:
    import jsonschema

    return jsonschema.Draft7Validator(_config_schema())


CONFIG_PROVIDER_INPUT_ERROR = (
    "Exactly one of 'targets' or 'providers' must be provided, but not both"
)


def config_schema_issues(config: Any, allow_missing_providers: bool = False) -> list[Any]:
    """zod-style issues for `config` against the vendored UnifiedConfig schema.

    The root `not: {required: [providers, targets]}` is reported as upstream's refine message,
    which zod only checks once the object itself parsed without aborting issues.
    """
    from promptphoo.config.zod_errors import Issue, collect_issues

    errors = [
        e
        for e in _config_validator().iter_errors(config)
        if not (str(e.validator) == "not" and not list(e.absolute_path))
    ]
    issues = collect_issues(errors, _config_schema())
    if isinstance(config, dict) and not any(i.aborts for i in issues):
        count = sum(config.get(k) is not None for k in ("providers", "targets"))
        if count > 1 or (count == 0 and not allow_missing_providers):
            issues.append(Issue([], CONFIG_PROVIDER_INPUT_ERROR, False))
    return issues


def validate_config(config: dict[str, Any], config_path: str) -> list[str]:
    """Log schema errors for a declarative config as a warning, formatted like z.prettifyError."""
    from promptphoo.config.zod_errors import prettify_issues

    issues = config_schema_issues(config, allow_missing_providers=bool(config.get("redteam")))
    if not issues:
        return []
    text = prettify_issues(issues)
    logger.warning("Invalid configuration file %s:\n%s", config_path, text)
    return text.split("\n")


# --- reading ----------------------------------------------------------------------------------


def dereference_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref, except inside provider function/tool `parameters` (JSON schemas)."""
    if envars.get_env_bool("PROMPTFOO_DISABLE_REF_PARSER"):
        return raw
    saved: dict[int, tuple[list[Any], list[Any]]] = {}
    providers = raw.get("providers")
    if isinstance(providers, list):
        for index, provider in enumerate(providers):
            if not isinstance(provider, dict):
                continue
            if "config" not in provider and provider:
                provider = next(iter(provider.values()))
            config = provider.get("config") if isinstance(provider, dict) else None
            if not isinstance(config, dict):
                continue
            functions = config.get("functions")
            tools = config.get("tools")
            fn_params = (
                [f.pop("parameters", None) if isinstance(f, dict) else None for f in functions]
                if isinstance(functions, list)
                else []
            )
            tool_params = (
                [
                    t["function"].pop("parameters", None)
                    if isinstance(t, dict) and isinstance(t.get("function"), dict)
                    else None
                    for t in tools
                ]
                if isinstance(tools, list)
                else []
            )
            saved[index] = (fn_params, tool_params)

    config = dereference(raw)

    if saved and isinstance(config.get("providers"), list):
        for index, (fn_params, tool_params) in saved.items():
            provider = config["providers"][index]
            if "config" not in provider:
                provider = next(iter(provider.values()))
            pconf = provider["config"]
            for func, params in zip(pconf.get("functions") or [], fn_params, strict=False):
                if params is not None:
                    func["parameters"] = params
            for tool, params in zip(pconf.get("tools") or [], tool_params, strict=False):
                if params is not None:
                    tool.setdefault("function", {})["parameters"] = params
    return config


def render_config_env_templates(config: dict[str, Any]) -> dict[str, Any]:
    """Two-pass env rendering: config.env from process/file env, then the config with it."""
    with cli_state.with_env(config.get("env")):
        base = (
            {}
            if envars.is_template_process_env_disabled()
            else {**os.environ, **(cli_state.env_file_overrides() or {})}
        )
        raw_env = config.get("env")
        rendered_env = render_env_only_in_object(raw_env, base, True) if raw_env else None
        filtered = (
            {k: v for k, v in rendered_env.items() if v is not None}
            if isinstance(rendered_env, dict)
            else rendered_env
        )
        with cli_state.with_env(filtered):
            return render_env_only_in_object(config, filtered)


def read_config(config_path: str) -> dict[str, Any]:
    with cli_state.with_env(None):
        return _read_config_in_scope(config_path)


def _read_config_in_scope(config_path: str) -> dict[str, Any]:
    ext = os.path.splitext(config_path)[1]
    if ext not in (".json", ".yaml", ".yml"):
        raise ValueError(
            f"Unsupported configuration file format: {ext}. promptphoo reads YAML and JSON "
            "configs; JavaScript/TypeScript configs are not supported."
        )
    raw = load_yaml(Path(config_path).read_text(encoding="utf-8")) or {}
    ret = dereference_config(raw)
    ret = render_config_env_templates(ret)
    validate_config(ret, config_path)

    if "targets" in ret:
        ret["providers"] = ret.pop("targets")
    for key in ("plugins", "strategies"):
        if key in ret:
            ret.setdefault("redteam", {})[key] = ret.pop(key)
    if not ret.get("prompts"):
        tests = ret.get("tests")
        has_any_prompt = (
            not tests
            or isinstance(tests, str)
            or (
                isinstance(tests, list)
                and any(isinstance(t, dict) and "prompt" in (t.get("vars") or {}) for t in tests)
            )
        )
        if not has_any_prompt and not _first_target_has_inputs(ret.get("providers")):
            logger.warning(
                'Warning: Expected top-level "prompts" property in config or a test variable '
                'named "prompt"'
            )
        ret["prompts"] = ["{{prompt}}"]
    if ret.get("basePath") is not None:
        ret["basePath"] = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(config_path)), ret["basePath"])
        )
    return ret


def _first_target_has_inputs(providers: Any) -> bool:
    if not isinstance(providers, list) or not providers or not isinstance(providers[0], dict):
        return False
    inputs = providers[0].get("inputs")
    return isinstance(inputs, dict) and bool(inputs)


def maybe_read_config(config_path: str) -> dict[str, Any] | None:
    if not os.path.exists(config_path):
        return None
    try:
        return read_config(config_path)
    except FileNotFoundError:
        return None


# --- combining --------------------------------------------------------------------------------


def _dedupe_key(provider: Any) -> Any:
    if isinstance(provider, str):
        return provider
    try:
        return json_stringify(provider)
    except (TypeError, ValueError):
        return id(provider)


def prepare_combined_config(config_paths: list[str]) -> dict[str, Any]:
    """Combine configs; returns {config, test_sources, prompt_base_paths}."""
    sources: list[tuple[dict[str, Any], str]] = []
    for config_path in config_paths:
        resolved = os.path.abspath(config_path)
        matches = glob_sync(resolved)
        if not matches:
            raise FileNotFoundError(
                f'No configuration file found at {config_path}. Run "promptphoo init" to create '
                "one or pass --config path/to/promptfooconfig.yaml."
            )
        for match in matches:
            config = read_config(match)
            sources.append((config, config.get("basePath") or os.path.dirname(match)))

    configs = [c for c, _ in sources]
    combined_env: dict[str, Any] = {}
    for c in configs:
        combined_env.update(c.get("env") or {})

    providers: list[Any] = []
    seen: set[Any] = set()
    for c in configs:
        p = c.get("providers")
        for provider in [p] if isinstance(p, str) else p if isinstance(p, list) else []:
            key = _dedupe_key(provider)
            if key not in seen:
                providers.append(provider)
                seen.add(key)

    extensions: list[str] = []
    for c in configs:
        if isinstance(c.get("extensions"), list):
            extensions.extend(c["extensions"])
    if len(extensions) > 1 and len(configs) > 1:
        logger.warning(
            "Warning: Multiple configurations and extensions detected. Currently, all extensions "
            "are run across all configs and do not respect their original promptfooconfig."
        )

    redteam: dict[str, Any] | None = None
    for c in configs:
        if c.get("redteam"):
            redteam = redteam or {"plugins": [], "strategies": []}
            for key, value in c["redteam"].items():
                if key in ("entities", "plugins", "strategies"):
                    if isinstance(value, list):
                        items = list(redteam.get(key) or [])
                        for v in value:
                            if v not in items:
                                items.append(v)
                        redteam[key] = sorted(items, key=json_stringify)
                else:
                    redteam[key] = value

    def resolve_config_path(base: str, reference: str) -> str:
        if "{{" in reference:
            with cli_state.with_env(combined_env):
                reference = render_env_only_in_object(reference)
        if "{{" in reference or is_remote_tests_reference(reference):
            return reference
        prefix = "file://" if reference.startswith("file://") else ""
        return prefix + os.path.abspath(os.path.join(base, reference[len(prefix) :]))

    def resolve_nested(base: str, value: Any) -> Any:
        if isinstance(value, str):
            return resolve_config_path(base, value) if value.startswith("file://") else value
        if isinstance(value, list):
            return [resolve_nested(base, v) for v in value]
        if isinstance(value, dict):
            return {k: resolve_nested(base, v) for k, v in value.items()}
        return value

    def make_absolute(base: str, prompt: Any) -> Any:
        if isinstance(prompt, str):
            return resolve_config_path(base, prompt) if prompt.startswith("file://") else prompt
        if prompt.get("id"):
            out = {**prompt}
            if prompt["id"].startswith("file://"):
                out["raw"] = prompt.get("raw") or resolve_config_path(base, prompt["id"])
            return out
        if isinstance(prompt.get("raw"), str):
            return prompt
        raise ValueError(f"Invalid prompt object: {json_stringify(prompt)}")

    def make_test_absolute(base: str, test: Any) -> Any:
        if isinstance(test, str):
            return resolve_config_path(base, test)
        if not isinstance(test, dict):
            return test
        if isinstance(test.get("path"), str):
            out = {**test, "path": resolve_config_path(base, test["path"])}
            if "config" in test:
                out["config"] = resolve_nested(base, test["config"])
            return out
        out = {**test}
        vars_ = test.get("vars")
        if vars_:
            if isinstance(vars_, str):
                out["vars"] = resolve_config_path(base, vars_)
            elif isinstance(vars_, list):
                out["vars"] = [resolve_config_path(base, v) for v in vars_]
            else:
                out["vars"] = resolve_nested(base, vars_)
        provider = test.get("provider")
        if isinstance(provider, str) and provider.startswith("file://"):
            out["provider"] = resolve_config_path(base, provider)
        elif (
            isinstance(provider, dict)
            and isinstance(provider.get("id"), str)
            and provider["id"].startswith("file://")
        ):
            out["provider"] = {**provider, "id": resolve_config_path(base, provider["id"])}
        return out

    are_lists = all(isinstance(c.get("prompts"), str | list) for c in configs)
    prompts: list[Any] | dict[str, Any] = [] if are_lists else {}
    prompt_base_paths: list[tuple[Any, str]] = []
    for config, base in sources:
        cp = config.get("prompts")
        if isinstance(cp, str):
            if not isinstance(prompts, list):
                raise ValueError("Cannot mix string and map-type prompts")
            p = make_absolute(base, cp)
            prompts.append(p)
            prompt_base_paths.append((p, base))
        elif isinstance(cp, list):
            if not isinstance(prompts, list):
                raise ValueError("Cannot mix configs with map and array-type prompts")
            for prompt in cp:
                if not (
                    isinstance(prompt, str)
                    or (
                        isinstance(prompt, dict)
                        and (
                            isinstance(prompt.get("raw"), str)
                            or isinstance(prompt.get("label"), str)
                        )
                    )
                ):
                    raise ValueError(
                        f"Invalid prompt: {json_stringify(prompt)}. Prompts must be either a "
                        "string or an object with a 'raw' or 'label' string property."
                    )
                prompt_base_paths.append((make_absolute(base, prompt), base))
        else:
            if not isinstance(prompts, dict):
                raise ValueError("Cannot mix configs with map and array-type prompts")
            for raw, label in (cp or {}).items():
                key = resolve_config_path(base, raw) if raw.startswith("file://") else raw
                prompts[key] = label
                prompt_base_paths.append((key, base))
    if isinstance(prompts, list):
        # Strings dedupe (a JS Set of primitives); prompt objects are always distinct.
        seen_strings: set[str] = set()
        ordered: list[tuple[Any, str]] = []
        for p, base in prompt_base_paths:
            if isinstance(p, str):
                if p in seen_strings:
                    continue
                seen_strings.add(p)
            ordered.append((p, base))
        prompt_base_paths = ordered
        prompts = [p for p, _ in ordered]

    scenarios: list[Any] | None = None
    for config, base in sources:
        if config.get("scenarios") is None:
            continue
        scenarios = scenarios or []
        raw_scenarios = config["scenarios"]
        for source in raw_scenarios if isinstance(raw_scenarios, list) else [raw_scenarios]:
            if isinstance(source, str) and source.startswith("file://"):
                with cli_state.with_base_path(base), cli_state.with_env(combined_env):
                    loaded = maybe_load_from_external_file(source)
            else:
                loaded = source
            for scenario in loaded if isinstance(loaded, list) else [loaded]:
                if isinstance(scenario, dict) and scenario.get("tests"):
                    out = {**scenario}
                    if isinstance(scenario.get("config"), list):
                        out["config"] = [make_test_absolute(base, t) for t in scenario["config"]]
                    tests = scenario["tests"]
                    out["tests"] = [
                        make_test_absolute(base, t)
                        for t in (tests if isinstance(tests, list) else [tests])
                    ]
                    scenarios.append(out)
                else:
                    scenarios.append(scenario)

    default_test: Any = None
    for config, base in sources:
        curr = config.get("defaultTest")
        if isinstance(curr, str):
            default_test = make_test_absolute(base, curr)
            continue
        if isinstance(default_test, str):
            continue
        if not default_test and not curr:
            default_test = None
            continue
        c = curr if isinstance(curr, dict) else {}
        p = default_test if isinstance(default_test, dict) else {}
        default_test = {
            **p,
            **c,
            "vars": {**(p.get("vars") or {}), **(c.get("vars") or {})},
            "assert": [*(p.get("assert") or []), *(c.get("assert") or [])],
            "options": {**(p.get("options") or {}), **(c.get("options") or {})},
            "metadata": {**(p.get("metadata") or {}), **(c.get("metadata") or {})},
        }

    derived: list[Any] | None = None
    for c in configs:
        if c.get("derivedMetrics"):
            derived = [*(derived or []), *c["derivedMetrics"]]

    def merged(key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in configs:
            out.update(c.get(key) or {})
        return out

    output_path: list[str] = []
    for c in configs:
        op = c.get("outputPath")
        output_path.extend([op] if isinstance(op, str) else op if isinstance(op, list) else [])

    if any(c.get("sharing") is False for c in configs):
        sharing: Any = False
    else:
        sharing = next((c["sharing"] for c in configs if isinstance(c.get("sharing"), dict)), None)

    combined: dict[str, Any] = {
        "tags": merged("tags"),
        "basePath": sources[0][1] if sources else None,
        "description": ", ".join(
            "" if c.get("description") is None else js_string(c["description"]) for c in configs
        ),
        "providers": providers,
        "prompts": prompts,
        "tests": [],
        "scenarios": scenarios,
        "defaultTest": default_test,
        "derivedMetrics": derived,
        "nunjucksFilters": merged("nunjucksFilters"),
        "env": combined_env,
        "evaluateOptions": merged("evaluateOptions"),
        "outputPath": output_path,
        "commandLineOptions": merged("commandLineOptions"),
        "extensions": extensions,
        "redteam": redteam,
        "metadata": merged("metadata"),
        "sharing": sharing,
        "tracing": next((c["tracing"] for c in configs if c.get("tracing")), None),
    }
    return {
        "config": {k: v for k, v in combined.items() if v is not None},
        "prompt_base_paths": [b for _, b in prompt_base_paths],
        "test_sources": [{"tests": c.get("tests"), "base_path": b} for c, b in sources],
    }


async def _read_test_sources(
    sources: list[dict[str, Any]], env: Any, load_providers: bool = True
) -> list[dict[str, Any]]:
    read = read_tests if load_providers else read_test_configs
    tests: list[dict[str, Any]] = []
    for source in sources:
        try:
            tests.extend(await read(source["tests"], source["base_path"], env))
        except Exception as e:
            raise ConfigResolutionError(
                f"Failed to load tests from {source['base_path'] or os.getcwd()}: {e}"
            ) from e
    return tests


async def combine_configs(config_paths: list[str]) -> dict[str, Any]:
    prepared = prepare_combined_config(config_paths)
    config = prepared["config"]
    return {
        **config,
        "tests": await _read_test_sources(prepared["test_sources"], config.get("env")),
    }


def stringify_env_values(env: dict[str, Any] | None) -> dict[str, Any] | None:
    if not env:
        return env
    return {k: js_string(v) if isinstance(v, int | float | bool) else v for k, v in env.items()}


# --- resolution -------------------------------------------------------------------------------


async def resolve_configs(
    cmd: dict[str, Any],
    default_config: dict[str, Any] | None = None,
    config_type: str | None = None,
) -> dict[str, Any]:
    """Ports resolveConfigs: returns {test_suite, config, base_path, command_line_options}.

    `config_type` is upstream's `type`: "DatasetGeneration" and "AssertionGeneration" configs
    need no providers."""
    default_config = default_config or {}
    file_config: dict[str, Any] = {}
    test_sources: list[dict[str, Any]] | None = None
    prompt_base_paths: list[str] | None = None
    if cmd.get("config"):
        prepared = prepare_combined_config(cmd["config"])
        file_config = prepared["config"]
        prompt_base_paths = prepared["prompt_base_paths"]
        test_sources = (
            [] if cmd.get("tests") or cmd.get("vars") or cmd.get("assertions")
            else prepared["test_sources"]
        )  # fmt: skip
        default_config = {}

    with (
        cli_state.with_base_path(None),
        cli_state.with_env(file_config.get("env") or default_config.get("env") or {}),
        cli_state.with_config({**default_config, **file_config}),
    ):
        resolved = await _resolve_loaded_config(
            cmd, file_config, default_config, test_sources, prompt_base_paths, config_type
        )
    cli_state.set_base_path(resolved["base_path"])
    cli_state.set_config(resolved["config"])
    return {
        **resolved,
        "test_sources": test_sources
        if test_sources is not None
        else [{"tests": default_config.get("tests"), "base_path": resolved["base_path"]}],
    }


async def _resolve_loaded_config(
    cmd: dict[str, Any],
    file_config: dict[str, Any],
    default_config: dict[str, Any],
    test_sources: list[dict[str, Any]] | None,
    prompt_base_paths: list[str] | None,
    config_type: str | None = None,
) -> dict[str, Any]:
    from promptphoo import providers as providers_mod
    from promptphoo.prompts import read_prompts

    config_paths = cmd.get("config")
    if cmd.get("assertions"):
        if not cmd.get("modelOutputs"):
            raise ConfigResolutionError("You must provide --model-outputs when using --assertions")
        model_outputs = json.loads(
            Path(os.getcwd(), cmd["modelOutputs"]).read_text(encoding="utf-8")
        )
        from promptphoo.assertions.readers import read_assertions

        assertions = read_assertions(cmd["assertions"])
        file_config["prompts"] = ["{{output}}"]
        file_config["providers"] = ["echo"]
        file_config["tests"] = [
            {"vars": {"output": o}, "assert": assertions}
            if isinstance(o, str)
            else {
                "vars": {
                    "output": o["output"],
                    **({"tags": ", ".join(o["tags"])} if o.get("tags") is not None else {}),
                },
                "assert": assertions,
            }
            for o in model_outputs
        ]

    def pick(key: str) -> Any:
        value = file_config.get(key)
        return value if is_truthy(value) else default_config.get(key)

    base_path = os.path.abspath(
        file_config.get("basePath")
        or default_config.get("basePath")
        or (os.path.dirname(config_paths[0]) if config_paths else "")
    )
    command_line_options = pick("commandLineOptions")
    cli_state.set_base_path(base_path)

    default_test_raw = pick("defaultTest")
    processed_default_test: Any = None
    if isinstance(default_test_raw, str) and default_test_raw.startswith("file://"):
        processed_default_test = maybe_load_from_external_file(default_test_raw)
    elif default_test_raw:
        processed_default_test = default_test_raw

    if envars.get_env_bool("PROMPTFOO_DISABLE_SHARING"):
        sharing: Any = False
    elif file_config.get("sharing") is not None:
        sharing = file_config["sharing"]
    else:
        sharing = default_config.get("sharing")
    config: dict[str, Any] = {
        "basePath": base_path,
        "tags": pick("tags"),
        "description": cmd.get("description") or pick("description"),
        "prompts": cmd.get("prompts") or pick("prompts") or [],
        "providers": pick("providers") or [],
        "tests": cmd.get("tests") or cmd.get("vars") or pick("tests") or [],
        "scenarios": pick("scenarios"),
        "env": pick("env"),
        "sharing": sharing,
        "derivedMetrics": pick("derivedMetrics"),
        "outputPath": cmd.get("output") or pick("outputPath"),
        "extensions": [*(cmd.get("extension") or []), *(pick("extensions") or [])],
        "metadata": pick("metadata"),
        "redteam": pick("redteam"),
        "tracing": pick("tracing"),
        "evaluateOptions": pick("evaluateOptions"),
    }
    config = {k: v for k, v in config.items() if v is not None}
    cli_state.set_config(config)

    prompts_list = config["prompts"] if isinstance(config["prompts"], list) else [config["prompts"]]
    has_prompts = any(prompts_list)
    providers_list = (
        config["providers"] if isinstance(config["providers"], list) else [config["providers"]]
    )
    has_providers = bool(cmd.get("providers")) or any(providers_list)
    if not config_paths and not has_prompts and not has_providers and not envars.is_ci():
        raise ConfigResolutionError(
            "No promptfooconfig found",
            cli_message=(
                "⚠️  No promptfooconfig found\n\n"
                f"Searched in {os.getcwd()} for promptfooconfig.{{"
                + ", ".join(DEFAULT_CONFIG_EXTENSIONS)
                + "}\n\nTry running with:\n\n"
                f"{promptfoo_command('')} eval -c path/to/promptfooconfig.yaml\n\n"
                f"Or create a config with:\n\n{promptfoo_command('init')}"
            ),
            log_level="warn",
        )
    if not has_prompts:
        raise ConfigResolutionError("You must provide at least 1 prompt")
    if config_type not in ("DatasetGeneration", "AssertionGeneration") and not has_providers:
        raise ConfigResolutionError(
            "You must specify at least 1 provider (for example, openai:gpt-4.1)"
        )
    if not isinstance(config["providers"], list):
        raise ConfigResolutionError("providers must be an array")

    runtime_env = stringify_env_values(config.get("env"))

    if processed_default_test:
        config["defaultTest"] = await read_test_config(
            processed_default_test, base_path, True, runtime_env
        )
    parsed_default_test = (
        await read_test(copy.deepcopy(config["defaultTest"]), base_path, True, runtime_env)
        if config.get("defaultTest")
        else None
    )

    resolved_provider_configs = providers_mod.resolve_provider_configs(
        config["providers"], base_path=base_path
    )
    cli_filtered = (
        providers_mod.resolve_cli_providers_with_config(cmd["providers"], resolved_provider_configs)
        if cmd.get("providers")
        else resolved_provider_configs
    ) or []
    filter_option = cmd.get("filterProviders") or cmd.get("filterTargets")
    filtered_provider_configs = providers_mod.filter_provider_configs(cli_filtered, filter_option)
    if (
        filter_option
        and isinstance(filtered_provider_configs, list)
        and not filtered_provider_configs
    ):
        logger.warning(
            'No providers matched the filter "%s". Check your --filter-providers/--filter-targets '
            "value.",
            filter_option,
        )

    if cmd.get("prompts"):
        parsed_prompts = await read_prompts(config["prompts"])
    else:
        parsed_prompts = await read_prompts(
            config["prompts"],
            base_path,
            os.path.relpath(base_path, os.getcwd()),
            None if cmd.get("assertions") else prompt_base_paths,
        )
    if cmd.get("filterPrompts"):
        from promptphoo.filters import filter_prompts

        parsed_prompts = filter_prompts(parsed_prompts, cmd["filterPrompts"])
        if not parsed_prompts:
            logger.warning(
                'No prompts matched the filter "%s". Check your --filter-prompts value.',
                cmd["filterPrompts"],
            )

    parsed_providers = await providers_mod.load_api_providers(
        filtered_provider_configs, env=runtime_env, base_path=base_path
    )
    test_configs = await _read_test_sources(
        test_sources
        if test_sources
        else [
            {"tests": config.get("tests") or [], "base_path": "" if cmd.get("tests") else base_path}
        ],
        runtime_env,
        False,
    )
    config["tests"] = [copy.deepcopy(t) for t in test_configs]
    parsed_tests = [await read_test(t, base_path, False, runtime_env) for t in test_configs]

    parsed_scenarios = config.get("scenarios")
    if parsed_scenarios:
        parsed_scenarios = maybe_load_from_external_file(parsed_scenarios)
        flat: list[Any] = []
        for s in parsed_scenarios:
            flat.extend(s if isinstance(s, list) else [s])
        parsed_scenarios = [
            {**s, "tests": list(s["tests"]) if isinstance(s.get("tests"), list) else s.get("tests")}
            if isinstance(s, dict)
            else s
            for s in flat
        ]
    if isinstance(parsed_scenarios, list):
        from promptphoo.filters import derive_scenario_sample_seed, filter_tests

        config["scenarios"] = []
        sample = cmd.get("filterSample", (command_line_options or {}).get("filterSample"))
        sample_seed = cmd.get(
            "filterSampleSeed", (command_line_options or {}).get("filterSampleSeed")
        )
        for index, scenario in enumerate(parsed_scenarios):
            if not isinstance(scenario, dict):
                raise ConfigResolutionError("scenario must be an object")
            if isinstance(scenario.get("tests"), str):
                scenario["tests"] = maybe_load_from_external_file(scenario["tests"])
            if isinstance(scenario.get("tests"), list):
                scenario["tests"] = await _read_test_sources(
                    [{"tests": scenario["tests"], "base_path": base_path}], runtime_env, False
                )
            filtered = await filter_tests(
                {**scenario, "providers": parsed_providers, "prompts": parsed_prompts},
                {
                    "firstN": cmd.get("filterFirstN"),
                    "pattern": cmd.get("filterPattern"),
                    "failing": cmd.get("filterFailing"),
                    "sample": sample,
                    "sampleSeed": None
                    if sample_seed is None
                    else derive_scenario_sample_seed(sample_seed, index),
                },
            )
            config["scenarios"].append(copy.deepcopy({**scenario, "tests": filtered}))
            scenario["tests"] = [
                await read_test(t, base_path, False, runtime_env) for t in filtered
            ]

    if not parsed_prompts:
        raise ConfigResolutionError(
            "No prompts found. Add a `prompts:` entry to your config or pass "
            "--prompts path/to/prompt.txt."
        )

    default_options = {
        k: v
        for k, v in {
            "prefix": cmd.get("promptPrefix"),
            "suffix": cmd.get("promptSuffix"),
            "provider": cmd.get("grader"),
        }.items()
        if v is not None
    }
    default_test: dict[str, Any] = {
        **({"metadata": config["metadata"]} if config.get("metadata") is not None else {}),
        **(parsed_default_test or {}),
        "options": {**default_options, **((parsed_default_test or {}).get("options") or {})},
    }

    test_suite: dict[str, Any] = {
        "basePath": base_path,
        "description": config.get("description"),
        "tags": config.get("tags"),
        "prompts": parsed_prompts,
        "providers": parsed_providers,
        "tests": parsed_tests,
        "scenarios": parsed_scenarios,
        "defaultTest": default_test,
        "derivedMetrics": config.get("derivedMetrics"),
        "nunjucksFilters": read_filters(pick("nunjucksFilters") or {}, base_path),
        "redteam": config.get("redteam"),
        "extensions": config.get("extensions"),
        "tracing": config.get("tracing"),
        "env": runtime_env,
    }
    test_suite = {k: v for k, v in test_suite.items() if v is not None}

    from promptphoo.assertions.validate import validate_assertions

    # defaultTest can still be a file:// string here; only an object is checked.
    validate_assertions(
        test_suite.get("tests") or [],
        test_suite["defaultTest"] if isinstance(test_suite.get("defaultTest"), dict) else None,
    )

    if command_line_options and command_line_options.get("envPath") and base_path:
        env_paths = command_line_options["envPath"]
        env_paths = env_paths if isinstance(env_paths, list) else [env_paths]
        resolved_paths = [p if os.path.isabs(p) else os.path.join(base_path, p) for p in env_paths]
        command_line_options = {
            **command_line_options,
            "envPath": resolved_paths[0] if len(resolved_paths) == 1 else resolved_paths,
        }

    return {
        "config": config,
        "test_suite": test_suite,
        "base_path": base_path,
        "command_line_options": command_line_options,
        "selected_provider_configs": filtered_provider_configs,
    }
