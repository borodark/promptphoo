"""Process-wide CLI state. Ports src/cliState.ts.

Node uses AsyncLocalStorage for scoped values; here each scoped field is a ContextVar so asyncio
tasks spawned inside a scope inherit it, and `with_*` context managers restore the outer value.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

EnvOverrides = Mapping[str, Any]

_base_path: ContextVar[str | None] = ContextVar("base_path", default=None)
_config: ContextVar[dict[str, Any] | None] = ContextVar("config", default=None)
_env: ContextVar[EnvOverrides | None] = ContextVar("env", default=None)
_env_file_overrides: ContextVar[EnvOverrides | None] = ContextVar("env_file", default=None)
_provider_env: ContextVar[EnvOverrides | None] = ContextVar("provider_env", default=None)
_safe_mode: ContextVar[bool] = ContextVar("safe_mode", default=False)
_max_concurrency: ContextVar[int | None] = ContextVar("max_concurrency", default=None)

# Unscoped flags, mirroring the plain properties on the Node cliState object.
remote: bool = False
resume: bool = False
retry_mode: bool = False


def base_path() -> str | None:
    return _base_path.get()


def set_base_path(value: str | None) -> None:
    _base_path.set(value)


def config() -> dict[str, Any] | None:
    return _config.get()


def set_config(value: dict[str, Any] | None) -> None:
    _config.set(value)


def env() -> EnvOverrides | None:
    """Suite layer: the innermost config `env`."""
    return _env.get()


def set_env(value: EnvOverrides | None) -> None:
    _env.set(value)


def env_file_overrides() -> EnvOverrides | None:
    """File layer: values from --env-file, acting as process defaults."""
    return _env_file_overrides.get()


def set_env_file_overrides(value: EnvOverrides | None) -> None:
    _env_file_overrides.set(value)


def provider_env() -> EnvOverrides | None:
    """Provider layer: a provider's own `env`, active only while it is being called."""
    return _provider_env.get()


def safe_mode() -> bool:
    return _safe_mode.get()


def max_concurrency() -> int | None:
    return _max_concurrency.get()


@contextmanager
def _scoped(var: ContextVar, value: Any) -> Iterator[None]:
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def with_base_path(value: str | None):
    return _scoped(_base_path, value)


def with_config(value: dict[str, Any] | None):
    return _scoped(_config, value)


def with_env(value: EnvOverrides | None):
    return _scoped(_env, value)


def with_env_file_overrides(value: EnvOverrides | None):
    return _scoped(_env_file_overrides, value)


def with_provider_env(value: EnvOverrides | None):
    return _scoped(_provider_env, value)


def with_safe_mode(enabled: bool):
    return _scoped(_safe_mode, enabled)


def with_max_concurrency(value: int | None):
    return _scoped(_max_concurrency, value)
