"""Load and call user Python hooks (file://x.py:fn). Ports the in-process part of src/python/.

Hooks run in-process by default. A subprocess runner using the vendored persistent_wrapper.py
protocol is used when a different interpreter is requested (added with the Python provider).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

_modules: dict[tuple[str, float], ModuleType] = {}


def split_path_and_function(ref: str, default_fn: str | None = None) -> tuple[str, str | None]:
    """Split 'path/to/x.py:fn' into ('path/to/x.py', 'fn'). Windows drive colons are kept."""
    ref = ref.removeprefix("file://")
    head, sep, tail = ref.rpartition(":")
    if sep and head and not (len(head) == 1 and head.isalpha()) and "/" not in tail:
        if tail and not tail.endswith(".py"):
            return head, tail
    return ref, default_fn


def load_module(path: str | Path) -> ModuleType:
    """Import a .py file by path. Re-imports when the file changes (watch mode)."""
    resolved = Path(path).resolve()
    key = (str(resolved), resolved.stat().st_mtime)
    if key in _modules:
        return _modules[key]
    name = "promptphoo_user_" + hashlib.sha1(str(resolved).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python file {resolved}")
    module = importlib.util.module_from_spec(spec)
    # Let user modules import siblings, as the Node wrapper does by chdir-ing / sys.path.
    parent = str(resolved.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _modules[key] = module
    return module


def load_function(path: str | Path, fn_name: str) -> Callable[..., Any]:
    """Resolve `fn_name` (may be 'Class.method') in the module at `path`."""
    module = load_module(path)
    target: Any = module
    for part in fn_name.split("."):
        if not hasattr(target, part):
            available = [
                n for n, v in vars(module).items() if callable(v) and not n.startswith("_")
            ]
            raise AttributeError(
                f"Function '{fn_name}' not found in {path}. "
                f"Available functions: {', '.join(available) or '(none)'}"
            )
        target = getattr(target, part)
    if inspect.isclass(target):
        target = target()
    if not callable(target):
        raise TypeError(f"'{fn_name}' in {path} is not callable")
    return target


async def call_async(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a hook, awaiting it if it is async; sync hooks run in a worker thread."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    result = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(result):
        return await result
    return result


def call_sync(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a hook from synchronous code (e.g. template filters)."""
    result = fn(*args)
    if inspect.isawaitable(result):
        return asyncio.run(_await(result))
    return result


async def _await(awaitable: Any) -> Any:
    return await awaitable


class PythonScriptError(RuntimeError):
    pass


def _json_roundtrip(value: Any) -> Any:
    import json

    from promptphoo.jsvalues import json_stringify

    return json.loads(json_stringify(value))


async def run_python(path: str | Path, fn_name: str, args: list[Any]) -> Any:
    """Call `fn_name` in the file at `path` the way Node's runPython does: arguments and result
    pass through JSON, and failures are reported as 'Error running Python script: ...'."""
    import traceback

    try:
        fn = load_function(path, fn_name)
        return _json_roundtrip(await call_async(fn, *_json_roundtrip(args)))
    except Exception as e:
        raise PythonScriptError(
            f"Error running Python script: {e}\nStack Trace: Python Traceback: "
            f"{traceback.format_exc()}"
        ) from e


async def run_python_code(code: str, fn_name: str, args: list[Any]) -> Any:
    """Run inline Python source (e.g. a `python` assertion body) and call `fn_name`."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="promptfoo-python-code-") as tmp:
        path = Path(tmp) / "script.py"
        path.write_text(code, encoding="utf-8")
        return await run_python(path, fn_name, args)
