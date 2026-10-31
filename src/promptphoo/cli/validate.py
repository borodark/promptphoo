"""`promptfoo validate [config]` and `promptfoo validate target`. Ports src/commands/validate.ts
and the connectivity/session checks of src/node/testProvider.ts.

Differences from upstream (see D45):
- The config is checked against the vendored `_data/config-schema.json` and reported in
  `z.prettifyError` form (`config/zod_errors.py`). Upstream's second pass, TestSuiteSchema over
  the loaded suite, is not ported: the suite holds Python provider objects, and resolve_configs
  already rejects the shapes that pass would catch.
- HTTP targets are tested as upstream does with PROMPTFOO_DISABLE_REMOTE_GENERATION set: the
  promptfoo-cloud response analyzer and the remote session judge are never called, so a session
  test always ends in "remote grading is disabled" for manual review. Cloud UUID targets fail.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any

import click

from promptphoo import cache, cli_state
from promptphoo.cli.common import common_options, exit_with
from promptphoo.cli.options import VariadicOption
from promptphoo.envfile import setup_env
from promptphoo.jsvalues import json_stringify

logger = logging.getLogger("promptphoo")

VALIDATE_FAILURE_PREFIX = "Failed to validate configuration: "
LOAD_FAILURE_PREFIX = "Failed to load configuration: "
TEST_RESULT_INDENT = "    "
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    r"|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff",
    re.I,
)


def js_dedent(text: str) -> str:
    """The `dedent` package on an already-interpolated string: strip the smallest indent of
    the indented lines after the first, then trim."""
    indents = [len(m) for m in re.findall(r"\n([ \t]+)(?=\S)", text)]
    if indents:
        text = re.sub(r"\n[ \t]{" + str(min(indents)) + "}", "\n", text)
    return text.strip()


def is_uuid(value: str) -> bool:
    return bool(_UUID_RE.fullmatch(value))


def _provider_id(provider: Any) -> str:
    pid = provider.id() if callable(getattr(provider, "id", None)) else None
    if pid is None and isinstance(provider, dict):
        pid = provider.get("id")
    return pid or ""


def is_http_provider(provider: Any) -> bool:
    """src/util/httpProvider.ts isHttpProvider, for loaded providers and provider options."""
    provider_id = _provider_id(provider)
    if provider_id.startswith(("http:", "https:")) or provider_id in ("http", "https"):
        return True
    config = (
        provider.get("config") if isinstance(provider, dict) else getattr(provider, "config", None)
    )
    url = config.get("url") if isinstance(config, dict) else None
    if isinstance(url, str):
        return url.startswith(("http:", "https:"))
    return False


def patch_http_config_for_validation(provider_options: dict[str, Any]) -> dict[str, Any]:
    config = provider_options.get("config") or {}
    return {
        **provider_options,
        "config": {
            **config,
            "maxRetries": 1,
            "headers": {**(config.get("headers") or {}), "x-promptfoo-silent": "true"},
        },
    }


def log_config_resolution_error(error: Any, prefix: str | None = None) -> None:
    message = f"{prefix}{error.cli_message}" if prefix else error.cli_message
    (logger.warning if error.log_level == "warn" else logger.error)(message)


# --- provider tests (src/node/testProvider.ts) -------------------------------------------------


def _get_session_id(provider: Any) -> Any:
    getter = getattr(provider, "get_session_id", None)
    return getter() if callable(getter) else None


async def check_provider_connectivity(provider: Any, prompt: str = "Hello World!") -> dict:
    from promptphoo.evaluator.run import evaluate
    from promptphoo.models.eval import Eval

    variables: dict[str, str] = {}
    if not (provider.config or {}).get("sessionParser"):
        variables["sessionId"] = str(uuid.uuid4())
    test_suite = {
        "providers": [provider],
        "prompts": [{"raw": prompt, "label": "Connectivity Test"}],
        "providerPromptMap": {(provider.label or provider.id()): ["Connectivity Test"]},
        "tests": [{"vars": variables}],
    }
    try:
        eval_record = Eval({})
        await evaluate(
            test_suite, eval_record, {"maxConcurrency": 1, "showProgressBar": False, "silent": True}
        )
        result = eval_record.to_evaluate_summary()["results"][0]
        response = result.get("response") or {}
        session_id = (
            _get_session_id(provider) or response.get("sessionId") or variables.get("sessionId")
        )
        error = result.get("error")
        out: dict[str, Any] = {
            "success": not error,
            "message": f"Provider returned error: {error}"
            if error
            else "Provider test completed. Remote grading disabled - please review the response "
            "manually.",
            "providerResponse": result.get("response"),
            "transformedRequest": (response.get("metadata") or {}).get("transformedRequest"),
            "sessionId": session_id,
        }
        if error:
            out["error"] = error
        return out
    except Exception as err:  # noqa: BLE001 - reported as a failed test
        return {
            "success": False,
            "message": "Error evaluating the provider. Please review the error details.",
            "error": str(err),
        }


def determine_effective_session_source(provider: Any) -> str:
    config = provider.config or {}
    return config.get("sessionSource") or ("server" if config.get("sessionParser") else "client")


async def check_provider_session(provider: Any) -> dict:
    """testProviderSession with `skipConfigValidation` and remote grading disabled."""
    try:
        source = determine_effective_session_source(provider)
        initial_session_id = None if source == "server" else str(uuid.uuid4())
        first_prompt = "What can you help me with?"
        second_prompt = "What was the last thing I asked you?"
        first = await provider.call_api(
            first_prompt,
            {
                "vars": {"sessionId": initial_session_id} if initial_session_id else {},
                "prompt": {"raw": first_prompt, "label": "Session Test - Request 1"},
            },
        )
        first = first or {}
        if first.get("error"):
            return {
                "success": False,
                "message": f"First request failed: {first['error']}",
                "error": first["error"],
            }
        extracted = _get_session_id(provider) or first.get("sessionId") or initial_session_id
        session_id = extracted or ""
        if source == "server" and not session_id.strip():
            return {
                "success": False,
                "message": "Session extraction failed: The session parser did not extract a "
                "session ID from the server response",
                "reason": "The session parser expression did not return a valid session ID. "
                "Check that the parser matches your server's response format.",
            }
        second = await provider.call_api(
            second_prompt,
            {
                "vars": {"sessionId": extracted} if extracted else {},
                "prompt": {"raw": second_prompt, "label": "Session Test - Request 2"},
            },
        )
        second = second or {}
        if second.get("error"):
            return {
                "success": False,
                "message": f"Second request failed: {second['error']}",
                "error": second["error"],
            }
        return {
            "success": False,
            "message": "Session test completed. Remote grading is disabled - please examine the "
            "results yourself.",
            "reason": "Manual review required - remote grading is disabled",
        }
    except Exception as err:  # noqa: BLE001
        return {
            "success": False,
            "message": f"Failed to test session: {err}",
            "error": str(err),
        }


# --- display -----------------------------------------------------------------------------------


def _header(provider_id: str) -> None:
    logger.info("")
    logger.info(f"Provider: {provider_id}")
    logger.info("─" * 50)


def _preview(output: Any) -> str:
    try:
        full = json_stringify(output)
    except Exception:  # noqa: BLE001
        full = str(output)
    return full[:100] + ("..." if len(full) > 100 else "")


async def check_basic_connectivity(provider: Any) -> dict[str, Any]:
    _header(provider.id())
    try:
        result = await provider.call_api(
            "Hello, world!",
            {
                "debug": True,
                "prompt": {"raw": "Hello, world!", "label": "Hello, world!"},
                "vars": {},
            },
        )
        result = result or {}
        if result.get("error"):
            logger.error("  ✗ Connectivity test")
            logger.error(f"    {result['error']}")
            return {"success": False, "error": result["error"]}
        if result.get("output") is not None:
            logger.info("  ✓ Connectivity test")
            logger.info(f"    Response: {_preview(result['output'])}")
            return {"success": True}
        logger.warning("  ✗ Connectivity test")
        logger.info("    No output received from provider")
        return {"success": False, "error": "No output received"}
    except Exception as err:  # noqa: BLE001
        logger.error("  ✗ Connectivity test")
        logger.error(f"    {err}")
        return {"success": False, "error": str(err)}


def display_test_result(result: dict[str, Any], test_name: str) -> None:
    analysis = result.get("analysis")
    session_id = result.get("sessionId")
    reason = result.get("reason")
    message = result.get("message")
    if result.get("success"):
        logger.info(f"  ✓ {test_name}")
        if message and message != "Test completed":
            logger.info(f"{TEST_RESULT_INDENT}{message}")
        if session_id:
            logger.info(f"{TEST_RESULT_INDENT}Session ID: {session_id}")
    else:
        has_suggestions = bool((analysis or {}).get("changes_needed"))
        error = result.get("error")
        if error and not has_suggestions:
            logger.error(f"  ✗ {test_name}")
            if message:
                logger.error(f"{TEST_RESULT_INDENT}{message}")
            if error != message:
                logger.error(f"{TEST_RESULT_INDENT}{error}")
        else:
            logger.warning(f"  {'⚠' if has_suggestions else '✗'} {test_name}")
            if message:
                logger.info(f"{TEST_RESULT_INDENT}{message}")
        if reason:
            logger.info(f"{TEST_RESULT_INDENT}Reason: {reason}")
    if analysis and analysis.get("changes_needed"):
        logger.info("")
        logger.info(f"{TEST_RESULT_INDENT}Suggestions:")
        if analysis.get("changes_needed_reason"):
            logger.info(f"{TEST_RESULT_INDENT}{analysis['changes_needed_reason']}")
        suggestions = analysis.get("changes_needed_suggestions")
        if isinstance(suggestions, list):
            logger.info("")
            for i, suggestion in enumerate(suggestions, 1):
                logger.info(f"{TEST_RESULT_INDENT}{i}. {suggestion}")
    transformed = result.get("transformedRequest")
    if isinstance(transformed, dict):
        logger.debug("")
        logger.debug(f"{TEST_RESULT_INDENT}Request details:")
        if transformed.get("url"):
            logger.debug(f"{TEST_RESULT_INDENT}  URL: {transformed['url']}")
        if transformed.get("method"):
            logger.debug(f"{TEST_RESULT_INDENT}  Method: {transformed['method']}")


async def check_http_provider(provider: Any) -> dict[str, Any]:
    provider_id = provider.id()
    _header(provider_id)
    summary: dict[str, Any] = {
        "providerId": provider_id,
        "connectivityPassed": False,
        "sessionPassed": None,
        "hasSuggestions": False,
        "sessionSkipped": False,
    }
    logger.info("  ◌ Connectivity test (running...)")
    connectivity = await check_provider_connectivity(provider)
    display_test_result(connectivity, "Connectivity test")
    summary["hasSuggestions"] = bool((connectivity.get("analysis") or {}).get("changes_needed"))
    worked = connectivity["success"] or (
        summary["hasSuggestions"] and not connectivity.get("error")
    )
    summary["connectivityPassed"] = worked
    if worked:
        if (provider.config or {}).get("stateful") is False:
            logger.info("  ○ Session test (skipped - target is stateless)")
            summary["sessionSkipped"] = True
        else:
            logger.info("  ◌ Session test (running...)")
            session = await check_provider_session(provider)
            display_test_result(session, "Session test")
            summary["sessionPassed"] = session["success"]
    else:
        logger.info("  ○ Session test (skipped - connectivity failed)")
        summary["sessionSkipped"] = True
    return summary


async def load_providers_for_testing(target: str | None, config: dict[str, Any]) -> list[Any]:
    from promptphoo.providers import load_api_provider, load_api_providers

    if target:
        if is_uuid(target):
            raise RuntimeError(
                f"Cloud target {target} cannot be loaded: promptfoo cloud is not supported by "
                "promptphoo"
            )
        if target.startswith(("http:", "https:")):
            provider = await load_api_provider(
                target,
                options={"config": {"maxRetries": 1, "headers": {"x-promptfoo-silent": "true"}}},
                base_path=cli_state.base_path(),
            )
        else:
            provider = await load_api_provider(target, base_path=cli_state.base_path())
        return [provider]
    providers = config.get("providers")
    if not providers or (isinstance(providers, list) and not providers):
        logger.info(
            "No providers found in configuration to test. Add providers to your config or run "
            "`promptphoo validate -t <provider-id>` to test a specific provider."
        )
        return []
    if isinstance(providers, list):
        providers = [
            patch_http_config_for_validation(p)
            if isinstance(p, dict) and is_http_provider(p)
            else p
            for p in providers
        ]
    return await load_api_providers(
        providers, env=config.get("env"), base_path=cli_state.base_path()
    )


def display_test_summary(summaries: list[dict[str, Any]]) -> int:
    logger.info("")
    logger.info("Summary")
    logger.info("═" * 50)
    passed = failed = suggestions = skipped = 0
    for s in summaries:
        parts = []
        if s["connectivityPassed"]:
            parts.append("connectivity: passed")
            passed += 1
        else:
            parts.append("connectivity: failed")
            failed += 1
        if s["sessionSkipped"]:
            parts.append("session: skipped")
            skipped += 1
        elif s["sessionPassed"] is True:
            parts.append("session: passed")
            passed += 1
        elif s["sessionPassed"] is False:
            parts.append("session: failed")
            failed += 1
        if s["hasSuggestions"]:
            suggestions += 1
        logger.info(f"  {s['providerId']}")
        logger.info(f"    {', '.join(parts)}")
    logger.info("")
    status = []
    if passed:
        status.append(f"{passed} passed")
    if failed:
        status.append(f"{failed} failed")
    if skipped:
        status.append(f"{skipped} skipped")
    logger.info(f"Tests: {', '.join(status)}")
    if suggestions:
        logger.info(
            f"\n{suggestions} provider(s) have configuration suggestions - see above for details"
        )
    return 1 if failed else 0


async def run_provider_tests(target: str | None, config: dict[str, Any]) -> int:
    summaries: list[dict[str, Any]] = []
    try:
        providers = await load_providers_for_testing(target, config)
        if not providers:
            return 0
        for provider in providers:
            try:
                if is_http_provider(provider):
                    summaries.append(await check_http_provider(provider))
                else:
                    result = await check_basic_connectivity(provider)
                    summaries.append(
                        {
                            "providerId": provider.id(),
                            "connectivityPassed": result["success"],
                            "sessionPassed": None,
                            "hasSuggestions": False,
                            "sessionSkipped": True,
                        }
                    )
            except Exception as err:  # noqa: BLE001
                logger.error("")
                logger.error(f"Provider: {provider.id()}")
                logger.error("─" * 50)
                logger.error("  ✗ Test execution failed")
                logger.error(f"    {err}")
                summaries.append(
                    {
                        "providerId": provider.id(),
                        "connectivityPassed": False,
                        "sessionPassed": None,
                        "hasSuggestions": False,
                        "sessionSkipped": True,
                    }
                )
        if summaries:
            return display_test_summary(summaries)
        return 0
    except Exception as err:  # noqa: BLE001
        logger.error(f"Provider tests failed: {err}")
        return 1


# --- commands ----------------------------------------------------------------------------------


async def do_validate(
    config_paths: list[str] | None,
    env_path: list[str] | None,
    default_config: dict[str, Any],
    default_config_path: str | None,
) -> int:
    from promptphoo.config.load import ConfigResolutionError, config_schema_issues, resolve_configs
    from promptphoo.config.zod_errors import prettify_issues

    setup_env(env_path)
    paths = config_paths or ([default_config_path] if default_config_path else None)
    try:
        resolved = await resolve_configs(
            {"config": paths, **({"envPath": env_path} if env_path else {})}, default_config
        )
        issues = config_schema_issues(resolved["config"])
        if issues:
            logger.error(
                js_dedent(
                    "Configuration validation error:\n"
                    f"Config file path(s): {', '.join(paths) if paths else 'N/A'}\n"
                    f"{prettify_issues(issues)}"
                )
            )
            return 1
        logger.info("Configuration is valid.")
        return 0
    except ConfigResolutionError as err:
        log_config_resolution_error(err, VALIDATE_FAILURE_PREFIX)
        return 1
    except Exception as err:  # noqa: BLE001
        logger.error(f"{VALIDATE_FAILURE_PREFIX}{err}")
        return 1


async def do_validate_target(
    target: str | None,
    config: str | None,
    env_path: list[str] | None,
    default_config: dict[str, Any],
) -> int:
    from promptphoo.config.load import ConfigResolutionError, resolve_configs

    setup_env(env_path)
    cache.disable_cache()
    if not target and not config:
        logger.error("Please specify either -t <provider-id> or -c <config-path>")
        return 1
    logger.info("")
    logger.info("Validating Target")
    logger.info("═" * 50)
    if config:
        logger.info(f"Configuration: {config}")
        try:
            resolved = await resolve_configs(
                {"config": [config], **({"envPath": env_path} if env_path else {})},
                default_config,
            )
            return await run_provider_tests(
                None, {**resolved["config"], "env": resolved["test_suite"].get("env")}
            )
        except ConfigResolutionError as err:
            log_config_resolution_error(err, LOAD_FAILURE_PREFIX)
            return 1
        except Exception as err:  # noqa: BLE001
            logger.error(f"{LOAD_FAILURE_PREFIX}{err}")
            return 1
    logger.info(f"Target: {target}")
    try:
        return await run_provider_tests(target, {})
    except Exception as err:  # noqa: BLE001
        logger.error(f"Failed to test provider: {err}")
        return 1


class _DefaultConfigGroup(click.Group):
    """commander's `command('config', { isDefault: true })`: no subcommand runs `config`."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args or args[0] not in self.commands and args[0] not in ("-h", "--help"):
            args = ["config", *args]
        return super().parse_args(ctx, args)


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.group(
        "validate", cls=_DefaultConfigGroup, help="Validate configuration files and test providers"
    )
    def validate_group() -> None:
        pass

    @validate_group.command("config", help="Validate a promptfoo configuration file")
    @click.option(
        "-c",
        "--config",
        cls=VariadicOption,
        help="Path to configuration file. Automatically loads promptfooconfig.yaml",
    )
    @common_options
    def validate_config_cmd(config: tuple[str, ...], env_file: tuple[str, ...], **_: Any) -> None:
        exit_with(
            asyncio.run(
                do_validate(
                    list(config) or None,
                    list(env_file) or None,
                    default_config,
                    default_config_path,
                )
            )
        )

    @validate_group.command(
        "target", help="Test providers from a config file or a specific provider"
    )
    @click.option("-t", "--target", help="Provider ID or cloud UUID to test")
    @click.option("-c", "--config", help="Path to configuration file to test all providers")
    @common_options
    def validate_target_cmd(
        target: str | None, config: str | None, env_file: tuple[str, ...], **_: Any
    ) -> None:
        exit_with(
            asyncio.run(do_validate_target(target, config, list(env_file) or None, default_config))
        )

    return validate_group
