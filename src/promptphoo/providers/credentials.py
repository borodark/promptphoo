"""Provider credential lookup. Ports src/providers/credentials.ts and src/providers/env.ts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from promptphoo import envars


def resolve_provider_env(
    env: Mapping[str, Any] | None,
    names: Sequence[str],
    ambient_names: Sequence[str] | None = None,
) -> tuple[str, str] | None:
    """First non-empty value across the provider, suite, file and process layers.

    A layer that defines a name (even as an empty string) masks it in later layers.
    """
    ambient_names = names if ambient_names is None else ambient_names
    ambient = {name: envars.get_env_string(name) for name in names}
    masked: set[str] = set()
    layers: list[Mapping[str, Any] | None] = [
        env,
        envars.get_env_overrides("provider"),
        env,
        envars.get_env_overrides("suite"),
        envars.get_env_overrides("file"),
        ambient,
    ]
    for layer in layers:
        for name in ambient_names if layer is ambient else names:
            value = layer.get(name) if layer else None
            if name in masked or value is None:
                continue
            masked.add(name)
            if value:
                return name, str(value)
    return None


def resolve_provider_api_key(
    config: Mapping[str, Any] | None,
    env: Mapping[str, Any] | None,
    default_envars: Sequence[str],
    ambient_envars: Sequence[str] | None = None,
) -> str | None:
    """A named credential (`apiKeyEnvar`) selects its own namespace."""
    if config and config.get("apiKey"):
        return config["apiKey"]
    api_key_envar = (config or {}).get("apiKeyEnvar")
    names = [api_key_envar] if api_key_envar else list(default_envars)
    ambient = names if api_key_envar else (ambient_envars or default_envars)
    found = resolve_provider_env(env, names, ambient)
    return found[1] if found else None
