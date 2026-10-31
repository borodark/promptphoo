"""Provider loading. Ports src/providers/index.ts, src/util/providerRef.ts and
src/util/eval/filterProviders.ts.

Provider refs come in five shapes: a string id, a `file://*.yaml|json` config file, ProviderOptions
(`{id, config, ...}`), a ProviderOptionsMap (`{"openai:gpt-4o": {config}}`), or a callable.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from promptphoo import cli_state
from promptphoo.files import maybe_load_config_from_external_file, node_join
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider, ProviderResponse, is_api_provider
from promptphoo.providers.registry import find_factory
from promptphoo.templating import render_env_only_in_object
from promptphoo.yamlload import load_yaml

__all__ = [
    "ApiProvider",
    "ProviderResponse",
    "filter_provider_configs",
    "filter_providers",
    "get_provider_ids",
    "is_api_provider",
    "load_api_provider",
    "load_api_providers",
    "merge_provider_env",
    "normalize_provider_ref",
    "resolve_cli_providers_with_config",
    "resolve_provider",
    "resolve_provider_configs",
]

PROVIDER_OPTION_KEYS = {"id", "label", "config", "prompts", "transform", "delay", "env", "inputs"}
EnvOverrides = Mapping[str, Any] | None


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def is_provider_config_file_reference(path: str) -> bool:
    return path.startswith("file://") and path.endswith((".yaml", ".yml", ".json"))


def canonicalize_provider_id(provider_id: str) -> str:
    """Resolve relative file/exec/python provider paths so refs compare equal (providerRef.ts)."""
    if provider_id.startswith("file://"):
        file_path = provider_id[len("file://") :]
        return provider_id if os.path.isabs(file_path) else f"file://{os.path.abspath(file_path)}"
    for prefix in ("exec:", "python:", "golang:"):
        if provider_id.startswith(prefix):
            file_path = provider_id[len(prefix) :]
            if "/" in file_path or "\\" in file_path:
                return f"{prefix}{os.path.abspath(file_path)}"
            return provider_id
    if provider_id.endswith((".js", ".ts", ".mjs")) and ("/" in provider_id or "\\" in provider_id):
        return f"file://{os.path.abspath(provider_id)}"
    return provider_id


def normalize_provider_ref(provider: Any, index: int | None = None) -> dict[str, Any]:
    """Classify a provider reference without loading anything. Returns a descriptor dict with
    `kind` in named/file/function/options/map/unknown, `id`, and for loadable kinds
    `load_path` / `load_options`."""
    unknown_id = "unknown" if index is None else f"unknown-{index}"
    if isinstance(provider, str):
        if not _valid_id(provider):
            return {"kind": "unknown", "id": unknown_id}
        kind = "file" if is_provider_config_file_reference(provider) else "named"
        return {"kind": kind, "id": provider, "load_path": provider}
    if callable(provider) and not isinstance(provider, ApiProvider | dict):
        label = getattr(provider, "label", None)
        label = label if isinstance(label, str) else None
        fallback = "custom-function" if index is None else f"custom-function-{index}"
        return {"kind": "function", "id": label or fallback, "label": label}
    if isinstance(provider, dict):
        label = provider.get("label") if isinstance(provider.get("label"), str) else None
        if _valid_id(provider.get("id")):
            return {"kind": "options", "id": provider["id"], "label": label,
                    "load_options": provider, "load_path": provider["id"]}  # fmt: skip
        keys = list(provider)
        if len(keys) == 1 and keys[0] not in PROVIDER_OPTION_KEYS:
            original = keys[0]
            body = provider[original]
            if isinstance(body, dict) and _valid_id(original):
                pid = body["id"] if _valid_id(body.get("id")) else original
                body_label = body.get("label") if isinstance(body.get("label"), str) else None
                return {"kind": "map", "id": pid, "label": body_label,
                        "load_options": {**body, "id": pid}, "load_path": original}  # fmt: skip
        if _valid_id(label):
            return {"kind": "unknown", "id": label, "label": label}
    return {"kind": "unknown", "id": unknown_id}


def read_provider_config_file(path: str, base_path: str | None = None) -> tuple[list[dict], bool]:
    """Read a provider config file. Returns (configs, was_array)."""
    relative = path[len("file://") :]
    resolved = (
        relative if os.path.isabs(relative) else node_join(base_path or os.getcwd(), relative)
    )
    try:
        with open(resolved, encoding="utf-8") as f:
            raw = load_yaml(f.read())
    except Exception as e:
        raise ValueError(f"Failed to load provider config {relative}: {e}") from e
    content = maybe_load_config_from_external_file(raw, None, base_path)
    if not content:
        raise ValueError(f"Provider config {relative} is undefined")
    return (list(content) if isinstance(content, list) else [content]), isinstance(content, list)


def merge_provider_env(*layers: EnvOverrides) -> dict[str, Any] | None:
    """Merge env layers low-to-high priority, skipping None values.

    Node additionally drops conflicting aliases of credential env vars per provider family
    (getProviderEnvAliasGroups / getProviderEndpointAliases), so a key set in a higher layer
    clears its aliases from the lower ones. That is not ported: every layer is merged key by key.
    """
    merged: dict[str, Any] | None = None
    for layer in layers:
        if not layer:
            if layer is not None and merged is None:
                merged = {}
            continue
        merged = merged or {}
        merged.update({k: v for k, v in layer.items() if v is not None})
    return merged


def _describe_invalid(provider: Any) -> str:
    try:
        return json_stringify(provider)[:200]
    except Exception:
        return repr(provider)[:200]


class FunctionProvider(ApiProvider):
    """A Python callable used as a provider (programmatic API)."""

    def __init__(self, fn: Callable[..., Any], pid: str, env: EnvOverrides) -> None:
        super().__init__({"env": env})
        self._fn = fn
        self._pid = pid
        for key in ("label", "prompts", "transform", "delay", "inputs", "config"):
            value = getattr(fn, key, None)
            if value is not None:
                setattr(self, key, value)

    def default_id(self) -> str:
        return self.label or self._pid

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        with cli_state.with_env(self.env):
            result = self._fn(prompt, context, options)
            if inspect.isawaitable(result):
                result = await result
        return result


async def load_api_provider(
    provider_path: str,
    *,
    options: dict[str, Any] | None = None,
    base_path: str | None = None,
    env: EnvOverrides = None,
) -> ApiProvider:
    suite_env = env if env is not None else cli_state.env()
    env_copy = dict(suite_env) if suite_env is not None else None
    base_path = base_path if base_path is not None else cli_state.base_path()
    with cli_state.with_base_path(base_path), cli_state.with_env(env_copy):
        return await _create_api_provider(provider_path, options or {}, base_path, env_copy)


async def _create_api_provider(
    provider_path: str,
    options: dict[str, Any],
    base_path: str | None,
    env: EnvOverrides,
    template_env: EnvOverrides = None,
) -> ApiProvider:
    template_env = template_env or merge_provider_env(env, options.get("env"))

    def render(value: Any) -> Any:
        with cli_state.with_env(template_env):
            return render_env_only_in_object(value, template_env)

    rendered_path = render(provider_path)
    rendered_config = render(options["config"]) if options.get("config") else None
    rendered_id = render(options["id"]) if options.get("id") else None
    merged_env = merge_provider_env(env, options.get("env"))

    api_key_envar = (rendered_config or {}).get("apiKeyEnvar")
    if merged_env is not None and template_env and isinstance(api_key_envar, str):
        if api_key_envar in template_env:
            merged_env[api_key_envar] = template_env[api_key_envar]

    provider_options: dict[str, Any] = {
        "config": {**({"basePath": base_path} if base_path is not None else {}),
                   **(rendered_config or {})},
        "env": merged_env,
    }  # fmt: skip
    if rendered_id is not None:
        provider_options["id"] = rendered_id
    if options.get("label") is not None:
        provider_options["label"] = options["label"]
    if options.get("delay") is not None:
        provider_options["delay"] = options["delay"]

    if rendered_path.startswith("promptfoo://provider/"):
        raise ValueError("Cloud providers (promptfoo://provider/...) are not supported")

    if is_provider_config_file_reference(rendered_path):
        configs, was_array = read_provider_config_file(rendered_path, base_path)
        relative = rendered_path[len("file://") :]
        if not configs:
            raise ValueError(f"Provider config file {relative} contains no providers")
        if was_array:
            raise ValueError(
                f"Multiple providers found in {relative}. "
                "Use loadApiProviders instead of loadApiProvider."
            )
        content = configs[0]
        if not content.get("id"):
            raise ValueError(f"Provider config {relative} must have an id")
        file_env = merge_provider_env(env, content.get("env"), options.get("env"))
        with cli_state.with_env(file_env):
            return await _create_api_provider(
                content["id"],
                {**content, "prompts": options.get("prompts") or content.get("prompts"),
                 "env": file_env},
                base_path,
                file_env,
            )  # fmt: skip

    factory = find_factory(rendered_path)
    if factory is None:
        raise ValueError(
            f"Could not identify provider: {provider_path}.\n\n"
            "Please check your configuration and ensure the provider is correctly specified.\n\n"
            "For more information on supported providers, visit: "
            "https://promptfoo.dev/docs/providers/"
        )
    with cli_state.with_env(merged_env):
        provider = factory(rendered_path, provider_options)
        if inspect.isawaitable(provider):
            provider = await provider
    provider.transform = options.get("transform")
    provider.delay = options.get("delay")
    provider.inputs = options.get("inputs")
    provider.label = provider.label or render(options.get("label") or "")
    provider.prompts = options.get("prompts")
    return provider


async def resolve_provider(
    provider: Any,
    resolved: dict[str, ApiProvider],
    *,
    env: EnvOverrides = None,
    base_path: str | None = None,
) -> ApiProvider:
    if provider is None:
        raise ValueError("Provider cannot be null or undefined")
    if is_api_provider(provider):
        return provider
    if isinstance(provider, str):
        if provider in resolved:
            return resolved[provider]
        return await load_api_provider(provider, base_path=base_path, env=env)
    if isinstance(provider, dict):
        d = normalize_provider_ref(provider)
        if d["kind"] not in ("options", "map"):
            raise ValueError(
                "Provider object must have an 'id' field or be a ProviderOptionsMap "
                '(e.g. { "openai:responses:gpt-5.4": { config: ... } }). '
                f"Got: {_describe_invalid(provider)}"
            )
        return await load_api_provider(
            d["load_path"], options=d["load_options"], base_path=base_path, env=env
        )
    if callable(provider):
        d = normalize_provider_ref(provider)
        return FunctionProvider(provider, d["id"], env if env is not None else cli_state.env())
    raise ValueError(
        "Invalid provider type. Expected a string (provider id), an object with an 'id' field, "
        f"a ProviderOptionsMap, or a function. Got: {type(provider).__name__}"
    )


async def _load_from_file(path: str, base_path: str | None, env: EnvOverrides) -> list:
    configs, _ = read_provider_config_file(path, base_path)
    relative = path[len("file://") :]
    for c in configs:
        if not c.get("id"):
            raise ValueError(f"Provider config in {relative} must have an id")
    return list(
        await asyncio.gather(
            *(load_api_provider(c["id"], options=c, base_path=base_path, env=env) for c in configs)
        )
    )


async def load_api_providers(
    provider_paths: Any, *, base_path: str | None = None, env: EnvOverrides = None
) -> list[ApiProvider]:
    env = env if env is not None else cli_state.env()
    base_path = base_path if base_path is not None else cli_state.base_path()
    with cli_state.with_base_path(base_path), cli_state.with_env(env):
        if isinstance(provider_paths, str):
            if is_provider_config_file_reference(provider_paths):
                return await _load_from_file(provider_paths, base_path, env)
            return [await load_api_provider(provider_paths, base_path=base_path, env=env)]
        if is_api_provider(provider_paths):
            return [provider_paths]
        if callable(provider_paths) and not isinstance(provider_paths, dict):
            d = normalize_provider_ref(provider_paths)
            return [FunctionProvider(provider_paths, d["id"], env)]
        if isinstance(provider_paths, list):

            async def one(idx: int, provider: Any) -> list[ApiProvider]:
                if is_api_provider(provider):
                    return [provider]
                d = normalize_provider_ref(provider, idx)
                kind = d["kind"]
                if kind == "file":
                    return await _load_from_file(d["load_path"], base_path, env)
                if kind == "named":
                    return [await load_api_provider(d["load_path"], base_path=base_path, env=env)]
                if kind == "function":
                    return [FunctionProvider(provider, d["id"], env)]
                if kind in ("options", "map"):
                    return [await load_api_provider(d["load_path"], options=d["load_options"],
                                                    base_path=base_path, env=env)]  # fmt: skip
                raise ValueError(
                    f"Invalid provider at index {idx}: expected a provider id string, "
                    "ProviderOptions with an 'id' field, or a ProviderOptionsMap "
                    '(e.g. { "openai:responses:gpt-5.4": { config: ... } }). '
                    f"Got: {_describe_invalid(provider)}"
                )

            nested = await asyncio.gather(*(one(i, p) for i, p in enumerate(provider_paths)))
            return [p for group in nested for p in group]
    raise ValueError("Invalid providers list")


def get_provider_ids(provider_paths: Any) -> list[str]:
    def from_file(path: str) -> list[str]:
        configs, _ = read_provider_config_file(path, cli_state.base_path() or os.getcwd())
        relative = path[len("file://") :]
        for c in configs:
            if not c.get("id"):
                raise ValueError(f"Provider config in {relative} must have an id")
        return [c["id"] for c in configs]

    if isinstance(provider_paths, str):
        if is_provider_config_file_reference(provider_paths):
            return from_file(provider_paths)
        return [provider_paths]
    if is_api_provider(provider_paths):
        return [provider_paths.id()]
    if callable(provider_paths) and not isinstance(provider_paths, dict):
        return [normalize_provider_ref(provider_paths)["id"]]
    if isinstance(provider_paths, list):
        out: list[str] = []
        for idx, p in enumerate(provider_paths):
            if is_api_provider(p):
                out.append(p.id())
                continue
            d = normalize_provider_ref(p, idx)
            if d["kind"] == "file":
                out.extend(from_file(d["load_path"]))
            elif d["kind"] == "unknown":
                raise ValueError(
                    f"Invalid provider at index {idx}: expected a provider id string, "
                    "ProviderOptions with an 'id' field, or a ProviderOptionsMap. "
                    f"Got: {_describe_invalid(p)}"
                )
            else:
                out.append(d["id"])
        return out
    raise ValueError("Invalid providers list")


def resolve_provider_configs(provider_paths: Any, *, base_path: str | None = None) -> Any:
    """Expand file:// provider config references; leave everything else for load_api_providers."""
    if isinstance(provider_paths, str):
        if is_provider_config_file_reference(provider_paths):
            return read_provider_config_file(provider_paths, base_path)[0]
        return provider_paths
    if not isinstance(provider_paths, list):
        return provider_paths
    results: list[Any] = []
    for provider in provider_paths:
        d = normalize_provider_ref(provider)
        if d["kind"] == "file":
            results.extend(read_provider_config_file(d["load_path"], base_path)[0])
        elif d["kind"] == "named":
            results.append(d["load_path"])
        else:
            results.append(provider)
    return results


def resolve_cli_providers_with_config(cli_providers: list[str], config_providers: Any) -> Any:
    """Map each --providers token to a matching config provider (id, then label, then
    `:token` suffix), keeping unmatched tokens as bare ids."""
    if not isinstance(config_providers, list):
        return cli_providers
    indexed = [(p, normalize_provider_ref(p, i)) for i, p in enumerate(config_providers)]
    out: list[Any] = []
    for token in cli_providers:
        match = (
            next((p for p, d in indexed if d["id"] == token), None)
            or next((p for p, d in indexed if d.get("label") == token), None)
            or next((p for p, d in indexed if d["id"].endswith(":" + token)), None)
        )
        out.append(match if match is not None else token)
    return out


def _js_regex(pattern: str) -> re.Pattern[str]:
    from promptphoo.jsregex import compile_js_regex

    return compile_js_regex(pattern)


def filter_provider_configs(providers: Any, filter_option: str | None) -> Any:
    if not filter_option:
        return providers
    rx = _js_regex(filter_option)
    if isinstance(providers, str):
        return providers if rx.search(providers) else []
    if callable(providers) and not isinstance(providers, dict | list):
        label = getattr(providers, "label", None)
        pid = label or "custom-function"
        return providers if rx.search(pid) or (label and rx.search(label)) else []
    if not isinstance(providers, list):
        return providers
    kept = []
    for i, p in enumerate(providers):
        d = normalize_provider_ref(p, i)
        if rx.search(d["id"]) or (d.get("label") and rx.search(d["label"])):
            kept.append(p)
    return kept


def filter_providers(providers: list[ApiProvider], filter_option: str | None) -> list[ApiProvider]:
    if not filter_option:
        return providers
    rx = _js_regex(filter_option)
    return [p for p in providers if rx.search(p.id()) or (p.label and rx.search(p.label))]
