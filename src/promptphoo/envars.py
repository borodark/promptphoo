"""Environment variable access with config-layer overrides. Ports src/envars.ts.

Lookup order for `get_env_string`: provider layer, suite layer (config `env`), file layer
(--env-file), then os.environ.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any, Literal, overload

from promptphoo import cli_state

Layer = Literal["suite", "file", "provider"]

_TRUTHY = {"1", "true", "yes", "yup", "yeppers"}


def get_env_overrides(layer: Layer = "suite") -> Mapping[str, Any] | None:
    """Read one config layer without mixing in os.environ."""
    if layer == "provider":
        return cli_state.provider_env()
    if layer == "file":
        return cli_state.env_file_overrides()
    return cli_state.env()


def get_process_env() -> dict[str, str]:
    """Environment inherited by child processes, including invocation-local file values."""
    env = dict(os.environ)
    file_env = get_env_overrides("file")
    if file_env:
        env.update({k: str(v) for k, v in file_env.items() if v is not None})
    return env


@overload
def get_env_string(key: str) -> str | None: ...
@overload
def get_env_string(key: str, default: str) -> str: ...
def get_env_string(key: str, default: str | None = None) -> str | None:
    for layer in ("provider", "suite"):
        overrides = get_env_overrides(layer)
        if overrides is not None and overrides.get(key) is not None:
            return _js_string(overrides[key])
    file_env = get_env_overrides("file")
    value = file_env.get(key) if file_env else None
    if value is None:
        value = os.environ.get(key)
    if value is None:
        return default
    return _js_string(value)


def get_provider_env_string(env: Mapping[str, Any] | None, key: str) -> str | None:
    """Read registered provider-layer then explicit provider values, never os.environ."""
    for layer in (env, get_env_overrides("provider")):
        if layer and layer.get(key) is not None:
            return _js_string(layer[key])
    return None


def parse_env_bool(value: str | None, default: bool | None = None) -> bool:
    # JS `input || defaultValue`: an empty string falls through to the default.
    if value:
        return value.lower() in _TRUTHY
    return bool(default)


def get_env_bool(key: str, default: bool | None = None) -> bool:
    return parse_env_bool(get_env_string(key), default)


@overload
def get_env_int(key: str) -> int | None: ...
@overload
def get_env_int(key: str, default: int) -> int: ...
def get_env_int(key: str, default: int | None = None) -> int | None:
    value = get_env_string(key)
    if not value:
        return default
    parsed = _parse_int_prefix(value)
    return default if parsed is None else parsed


@overload
def get_env_float(key: str) -> float | None: ...
@overload
def get_env_float(key: str, default: float) -> float: ...
def get_env_float(key: str, default: float | None = None) -> float | None:
    value = get_env_string(key)
    if not value:
        return default
    parsed = parse_env_float(value)
    return default if parsed is None else parsed


def parse_env_float(value: str | None) -> float | None:
    """JS Number.parseFloat: parse the longest numeric prefix, else None."""
    if value is None:
        return None
    import re

    m = re.match(r"\s*([+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))", value)
    if not m:
        return None
    return float(m.group(1).replace("Infinity", "inf"))


def is_template_process_env_disabled() -> bool:
    """Suite flags can restrict template access to os.environ but cannot lift operator limits."""

    def disabled(env: Mapping[str, Any]) -> bool:
        return parse_env_bool(
            _opt_str(env.get("PROMPTFOO_DISABLE_TEMPLATE_ENV_VARS")),
            parse_env_bool(_opt_str(env.get("PROMPTFOO_SELF_HOSTED"))),
        )

    return (
        disabled(os.environ)
        or disabled(get_env_overrides("file") or {})
        or disabled(get_env_overrides() or {})
    )


def get_eval_timeout_ms(default: int = 0) -> int:
    return get_env_int("PROMPTFOO_EVAL_TIMEOUT_MS", default)


def get_max_eval_time_ms(default: int = 0) -> int:
    return get_env_int("PROMPTFOO_MAX_EVAL_TIME_MS", default)


def is_ci() -> bool:
    def has_identifier(key: str) -> bool:
        value = get_env_string(key)
        return bool(value and value.lower() not in ("false", "0"))

    return any(
        get_env_bool(k)
        for k in (
            "CI",
            "GITHUB_ACTIONS",
            "TRAVIS",
            "CIRCLECI",
            "JENKINS",
            "GITLAB_CI",
            "APPVEYOR",
            "TF_BUILD",
            "BUDDY",
            "BUILDKITE",
        )
    ) or any(
        has_identifier(k)
        for k in ("JENKINS_URL", "CODEBUILD_BUILD_ID", "BITBUCKET_COMMIT", "TEAMCITY_VERSION")
    )


def is_non_interactive() -> bool:
    return is_ci() or not sys.stdin.isatty() or not sys.stdout.isatty()


def _parse_int_prefix(value: str) -> int | None:
    """JS Number.parseInt(str, 10)."""
    import re

    m = re.match(r"\s*([+-]?\d+)", value)
    return int(m.group(1)) if m else None


def _js_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None else _js_string(value)
