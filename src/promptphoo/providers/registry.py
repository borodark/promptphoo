"""Provider factory table. Ports the local-provider entries of src/providers/registry.ts.

Each factory is (test, create): `test(path)` decides whether it handles a provider id and
`create(path, options)` builds the provider (sync or async).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable
from typing import Any

from promptphoo.files import is_javascript_file
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.python import PythonProvider
from promptphoo.providers.simple import EchoProvider, ScriptCompletionProvider, SequenceProvider

Factory = Callable[[str, dict[str, Any]], Any]


def _script_path(provider_path: str, options: dict[str, Any]) -> str:
    if provider_path.startswith("file://"):
        path = provider_path[len("file://") :]
    else:
        path = provider_path.split(":", 1)[1]
    if (options.get("config") or {}).get("isCloudConfig") is True and not os.path.isabs(path):
        return os.path.join(os.getcwd(), path)
    return path


def _script_factory(prefix: str, ext: str | None, cls: type) -> tuple[Callable, Factory]:
    def test(path: str) -> bool:
        if path.startswith(f"{prefix}:"):
            return True
        if ext and path.startswith("file://"):
            return path.endswith(f".{ext}") or f".{ext}:" in path
        return False

    def create(path: str, options: dict[str, Any]) -> ApiProvider:
        return cls(_script_path(path, options), options)

    return test, create


def _unsupported(kind: str) -> Factory:
    def create(path: str, options: dict[str, Any]) -> ApiProvider:
        raise ValueError(f"{kind} providers are not supported by promptphoo ({path}). {_HINT}")

    return create


_HINT = "Use a Python provider (file://provider.py) instead."


def _is_js_provider(path: str) -> bool:
    target = path[len("file://") :] if path.startswith("file://") else path
    target = target.split(":", 1)[0] if not os.path.splitdrive(target)[0] else target
    return is_javascript_file(target)


PROVIDER_FACTORIES: list[tuple[Callable[[str], bool], Factory]] = [
    _script_factory("exec", None, ScriptCompletionProvider),
    _script_factory("python", "py", PythonProvider),
    (lambda p: p == "echo", lambda p, o: EchoProvider(o)),
    (lambda p: p == "sequence", lambda p, o: SequenceProvider(o)),
    (_is_js_provider, _unsupported("JavaScript/TypeScript")),
    (lambda p: p.startswith(("golang:", "ruby:")) or p.endswith((".go", ".rb")),
     _unsupported("Go and Ruby")),
]  # fmt: skip


def _lazy(target: str) -> Factory:
    """A factory living in `module:function`, imported on first use (network families)."""

    def create(path: str, options: dict[str, Any]) -> Any:
        module_name, attr = target.split(":")
        return getattr(importlib.import_module(module_name), attr)(path, options)

    return create


def _prefixed(*prefixes: str) -> Callable[[str], bool]:
    return lambda p: p.startswith(prefixes)


_COMPAT = "promptphoo.providers.compatible"

# Upstream's `providerFamilies` (registry.ts) are consulted before every providerMap entry, so a
# family id whose model ends in `.js` is not taken for a JavaScript file provider.
FAMILY_FACTORIES: list[tuple[Callable[[str], bool], str]] = [
    (_prefixed("cerebras:"), f"{_COMPAT}.cerebras:create_provider"),
    (_prefixed("envoy:"), f"{_COMPAT}.simple:create_envoy"),
    (_prefixed("litellm:"), f"{_COMPAT}.litellm:create_provider"),
    (_prefixed("novita:"), f"{_COMPAT}.novita:create_provider"),
    (_prefixed("nscale:", "togetherai:"), f"{_COMPAT}.table:create_provider"),
]

# Network provider families. Each module exposes `create_provider(path, options)`. The
# OpenAI-compatible entries follow upstream providerMap order (see D31).
NETWORK_FACTORIES: list[tuple[Callable[[str], bool], str]] = [
    (_prefixed("abliteration:"), f"{_COMPAT}.abliteration:create_provider"),
    (_prefixed("ai21:"), f"{_COMPAT}.skipped:create_provider"),
    (_prefixed("alibaba:", "alicloud:", "aliyun:", "dashscope:"),
     f"{_COMPAT}.alibaba:create_provider"),
    (_prefixed("anthropic:"), "promptphoo.providers.anthropic:create_provider"),
    (_prefixed("atlascloud:"), f"{_COMPAT}.simple:create_atlascloud"),
    (_prefixed("azure:", "azureopenai:"), "promptphoo.providers.azure:create_provider"),
    (_prefixed("bedrock:"), "promptphoo.providers.bedrock:create_provider"),
    (_prefixed("cloudera:"), f"{_COMPAT}.simple:create_cloudera"),
    (_prefixed("cloudflare-ai:"), f"{_COMPAT}.cloudflare_ai:create_provider"),
    (_prefixed("cloudflare-gateway:"), f"{_COMPAT}.cloudflare_gateway:create_provider"),
    (_prefixed("databricks:"), f"{_COMPAT}.databricks:create_provider"),
    (_prefixed("mlflow-gateway:"), f"{_COMPAT}.mlflow_gateway:create_provider"),
    (_prefixed("deepseek:"), f"{_COMPAT}.deepseek:create_provider"),
    (_prefixed("fireworks:"), f"{_COMPAT}.fireworks:create_provider"),
    (_prefixed("github:"), f"{_COMPAT}.ghmodels:create_provider"),
    (_prefixed("groq:"), f"{_COMPAT}.groq:create_provider"),
    (_prefixed("helicone:"), f"{_COMPAT}.simple:create_helicone"),
    (_prefixed("hyperbolic:"), f"{_COMPAT}.hyperbolic:create_provider"),
    (_prefixed("localai:"), f"{_COMPAT}.localai:create_provider"),
    (_prefixed("meta:"), f"{_COMPAT}.meta:create_provider"),
    (_prefixed("minimax:"), f"{_COMPAT}.minimax:create_provider"),
    (_prefixed("mistral:"), f"{_COMPAT}.skipped:create_provider"),
    (_prefixed("moonshot:"), f"{_COMPAT}.moonshot:create_provider"),
    (_prefixed("nvidia:"), f"{_COMPAT}.nvidia:create_provider"),
    (_prefixed("vertex:", "google:", "palm:"), "promptphoo.providers.google:create_provider"),
    (_prefixed("ollama:"), "promptphoo.providers.ollama:create_provider"),
    (_prefixed("openai:"), "promptphoo.providers.openai:create_provider"),
    (_prefixed("openrouter:"), f"{_COMPAT}.openrouter:create_provider"),
    (_prefixed("orcarouter:"), f"{_COMPAT}.orcarouter:create_provider"),
    (_prefixed("perplexity:"), f"{_COMPAT}.perplexity:create_provider"),
    (_prefixed("portkey:"), f"{_COMPAT}.portkey:create_provider"),
    (_prefixed("truefoundry:"), f"{_COMPAT}.truefoundry:create_provider"),
    (_prefixed("llamaapi:"), f"{_COMPAT}.simple:create_llamaapi"),
    (_prefixed("aimlapi:", "cometapi:"), f"{_COMPAT}.table:create_provider"),
    (_prefixed("docker:"), f"{_COMPAT}.docker:create_provider"),
    (_prefixed("vercel:"), f"{_COMPAT}.skipped:create_provider"),
    (_prefixed("webhook:"), "promptphoo.providers.webhook:create_provider"),
    (_prefixed("xai:"), f"{_COMPAT}.xai:create_provider"),
    (_prefixed("http:", "https:"), "promptphoo.providers.http:create_provider"),
    (_prefixed("jfrog:", "qwak:"), f"{_COMPAT}.simple:create_jfrog"),
    (lambda p: p == "llama" or p.startswith("llama:"), f"{_COMPAT}.llama:create_provider"),
    (_prefixed("snowflake:"), f"{_COMPAT}.snowflake:create_provider"),
]  # fmt: skip


def _find_lazy(
    table: list[tuple[Callable[[str], bool], str]], provider_path: str
) -> Factory | None:
    for test, target in table:
        if test(provider_path) and importlib.util.find_spec(target.split(":")[0]) is not None:
            return _lazy(target)
    return None


def find_factory(provider_path: str) -> Factory | None:
    family = _find_lazy(FAMILY_FACTORIES, provider_path)
    if family is not None:
        return family
    for test, create in PROVIDER_FACTORIES:
        if test(provider_path):
            return create
    return _find_lazy(NETWORK_FACTORIES, provider_path)
