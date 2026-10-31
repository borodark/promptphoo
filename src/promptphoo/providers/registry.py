"""Provider factory table. Ports the local-provider entries of src/providers/registry.ts.

Each factory is (test, create): `test(path)` decides whether it handles a provider id and
`create(path, options)` builds the provider (sync or async).
"""

from __future__ import annotations

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


def find_factory(provider_path: str) -> Factory | None:
    for test, create in PROVIDER_FACTORIES:
        if test(provider_path):
            return create
    return None
