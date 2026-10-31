"""Provider credential and endpoint env resolution. Ports src/providers/credentials.ts and
src/providers/env.ts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from promptphoo import envars


def resolve_provider_env(
    env: Mapping[str, Any] | None,
    names: Sequence[str],
    ambient_names: Sequence[str] | None = None,
) -> dict[str, str] | None:
    """Resolve aliases within each scope; an empty value masks only that variable.

    Scopes, highest first: the provider's explicit env, the registered provider layer, the
    provider env, the suite `env`, the --env-file layer, then the process environment.
    Returns `{"name", "value"}` like upstream.
    """
    ambient_names = names if ambient_names is None else ambient_names
    masked: set[str] = set()
    ambient = {name: envars.get_env_string(name) for name in names}
    layers: list[tuple[Mapping[str, Any] | None, Sequence[str]]] = [
        (env, names),
        (envars.get_env_overrides("provider"), names),
        (env, names),
        (envars.get_env_overrides("suite"), names),
        (envars.get_env_overrides("file"), names),
        (ambient, ambient_names),
    ]
    for layer, layer_names in layers:
        for name in layer_names:
            value = layer.get(name) if layer else None
            if name in masked or value is None:
                continue
            masked.add(name)
            if value:
                return {"name": name, "value": str(value)}
    return None


def resolve_provider_api_key(
    config: Mapping[str, Any] | None,
    env: Mapping[str, Any] | None,
    default_envars: Sequence[str],
    ambient_envars: Sequence[str] | None = None,
) -> str | None:
    """A named credential selects its own namespace, rather than falling back to another
    vendor."""
    ambient_envars = default_envars if ambient_envars is None else ambient_envars
    config = config or {}
    if config.get("apiKey"):
        return config["apiKey"]
    api_key_envar = config.get("apiKeyEnvar")
    names = [api_key_envar] if api_key_envar else list(default_envars)
    resolved = resolve_provider_env(env, names, names if api_key_envar else ambient_envars)
    return resolved["value"] if resolved else None


def resolve_configured_api_key(provider: Any) -> str | None:
    config = getattr(provider, "config", None) or {}
    if config.get("apiKey") is not None:
        return config["apiKey"]
    api_key_envar = config.get("apiKeyEnvar")
    if not api_key_envar:
        return None
    value = envars.get_provider_env_string(getattr(provider, "env", None), api_key_envar)
    return value if value is not None else envars.get_env_string(api_key_envar)
