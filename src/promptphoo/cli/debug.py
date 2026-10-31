"""`promptfoo debug`. Ports src/commands/debug.ts.

The report keeps upstream's shape. `version` is promptphoo's own with the upstream version it
tracks, `platform.nodeVersion` becomes `platform.pythonVersion`, and loaded providers in
`testSuite` serialize as their id, label and config rather than every instance field.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
from typing import Any

import click

from promptphoo import UPSTREAM_VERSION, __version__, envars
from promptphoo.cli.common import common_options
from promptphoo.jsvalues import UNDEFINED, json_stringify

logger = logging.getLogger("promptphoo")

ISSUES_URL = "https://github.com/promptfoo/promptfoo/issues"
_ARCH = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "i386": "ia32", "i686": "ia32"}


def print_border() -> None:
    from promptphoo.output.table import terminal_max_width

    logger.info("=" * terminal_max_width())


def _platform() -> dict[str, Any]:
    machine = platform.machine()
    return {
        "os": sys.platform,
        "release": platform.release(),
        "arch": _ARCH.get(machine.lower(), machine.lower()),
        "pythonVersion": platform.python_version(),
    }


def _env() -> dict[str, Any]:
    def either(upper: str) -> str | None:
        return envars.get_env_string(upper) or envars.get_env_string(upper.lower()) or None

    out = {
        "NODE_ENV": envars.get_env_string("NODE_ENV"),
        "httpProxy": either("HTTP_PROXY"),
        "httpsProxy": either("HTTPS_PROXY"),
        "allProxy": either("ALL_PROXY"),
        "noProxy": either("NO_PROXY"),
        "nodeExtra": envars.get_env_string("NODE_EXTRA_CA_CERTS"),
        "nodeTls": envars.get_env_string("NODE_TLS_REJECT_UNAUTHORIZED"),
        "telemetryDisabled": envars.get_env_bool("PROMPTFOO_DISABLE_TELEMETRY"),
        "telemetryDebug": envars.get_env_bool("PROMPTFOO_TELEMETRY_DEBUG"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _jsonable(value: Any) -> Any:
    from promptphoo.providers.base import ApiProvider

    if isinstance(value, ApiProvider):
        out: dict[str, Any] = {"id": value.id()}
        if value.label is not None:
            out["label"] = value.label
        out["config"] = _jsonable(value.config)
        return out
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


async def do_debug(
    config: str | None, default_config: dict[str, Any], default_config_path: str | None
) -> None:
    from promptphoo.config.load import resolve_configs

    config_info: dict[str, Any] = {
        "defaultConfigPath": UNDEFINED if default_config_path is None else default_config_path,
        "specifiedConfigPath": UNDEFINED if config is None else config,
        "configExists": False,
        "configContent": None,
    }
    debug_info = {
        "version": f"{__version__} (promptfoo {UPSTREAM_VERSION})",
        "platform": _platform(),
        "env": _env(),
        "configInfo": config_info,
    }
    config_path = config or default_config_path
    if config_path and os.path.exists(config_path):
        config_info["configExists"] = True
        try:
            resolved = await resolve_configs({"config": [config_path]}, default_config)
            config_info["configContent"] = _jsonable(
                {
                    "config": resolved["config"],
                    "testSuite": resolved["test_suite"],
                    "basePath": resolved["base_path"],
                    "commandLineOptions": resolved.get("command_line_options", UNDEFINED),
                    "selectedProviderConfigs": resolved.get("selected_provider_configs", UNDEFINED),
                    "testSources": [
                        {"tests": s.get("tests", UNDEFINED), "basePath": s.get("base_path")}
                        for s in resolved.get("test_sources") or []
                    ],
                }
            )
        except Exception as err:  # noqa: BLE001 - reported inside the debug output
            config_info["configContent"] = f"Error loading config: Error: {err}"

    print_border()
    logger.info("Promptfoo Debug Information")
    print_border()
    logger.info(json_stringify(debug_info, 2))
    print_border()
    logger.info(f"Please include this output when reporting issues on GitHub: {ISSUES_URL}")


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.command("debug", help="Display debug information for troubleshooting")
    @click.option(
        "-c",
        "--config",
        help="Path to configuration file. Defaults to promptfooconfig.yaml",
    )
    @common_options
    def debug_cmd(config: str | None, **_: Any) -> None:
        asyncio.run(do_debug(config, default_config, default_config_path))

    return debug_cmd
