"""`functionToolCallbacks` loading and execution. Ports executeProviderFunctionCallback and
loadProviderCallbackFromFileUrl (src/providers/functionCallbackUtils.ts) and the callback path
guard from src/util/functions/loadFunction.ts.

Callbacks are Python: a `file://callbacks.py:fn` reference (loaded in-process through
python_hooks) or a callable passed programmatically. JavaScript files and inline JS function
strings are rejected with a clear error instead of being evaluated.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from promptphoo import cli_state, python_hooks
from promptphoo.files import is_javascript_file, parse_file_url
from promptphoo.jsvalues import js_string, json_stringify

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "yup", "yeppers")


class CallbackPathTraversalError(ValueError):
    def __init__(self, file_path: str, base_path: str) -> None:
        super().__init__(
            f"Path traversal rejected: '{file_path}' resolves outside the configured base "
            "directory. Place the callback file inside the base directory, or set "
            "PROMPTFOO_DISABLE_CALLBACK_PATH_GUARD=true to opt out (NOT recommended)."
        )
        self.file_path = file_path
        self.base_path = base_path


class UnsupportedCallbackError(ValueError):
    """A JavaScript callback, which promptphoo cannot run."""


def _is_callback_path_guard_disabled() -> bool:
    value = os.environ.get("PROMPTFOO_DISABLE_CALLBACK_PATH_GUARD")
    return value is not None and value.lower() in _TRUTHY


def _assert_within_base(file_path: str, resolved: str, base: str) -> None:
    relative = os.path.relpath(resolved, base)
    if relative == ".." or relative.startswith(f"..{os.sep}") or relative.startswith("../"):
        raise CallbackPathTraversalError(file_path, base)
    if os.path.isabs(relative):
        raise CallbackPathTraversalError(file_path, base)


def resolve_callback_path(file_path: str, base_path: str | None = None) -> str:
    """Resolve against basePath and assert the result stays inside it (lexically, then via
    realpath when the file exists)."""
    base = os.path.abspath(base_path or cli_state.base_path() or os.getcwd())
    resolved = os.path.abspath(os.path.join(base, file_path))
    if _is_callback_path_guard_disabled():
        return resolved
    _assert_within_base(file_path, resolved, base)
    try:
        real_base, real_target = (
            os.path.realpath(base, strict=True),
            os.path.realpath(resolved, strict=True),
        )
    except OSError:
        return resolved
    _assert_within_base(file_path, real_target, real_base)
    return resolved


def load_callback_from_file_url(file_ref: str, base_path: str | None = None) -> Callable[..., Any]:
    file_path, function_name = parse_file_url(file_ref)
    if is_javascript_file(file_path):
        raise UnsupportedCallbackError(
            f"JavaScript function callbacks are not supported by promptphoo ({file_ref}). "
            "Use a Python file instead: file://callbacks.py:function_name"
        )
    resolved = resolve_callback_path(file_path, base_path)
    if not function_name:
        raise ValueError(
            f"Function callback malformed: {file_path} must export a function or have a "
            "default export as a function"
        )
    try:
        return python_hooks.load_function(resolved, function_name)
    except AttributeError as err:
        raise ValueError(
            f"Function callback malformed: {file_path} must export a named function "
            f"'{function_name}'"
        ) from err


def load_provider_callback_from_file_url(
    file_ref: str, log_prefix: str | None = None
) -> Callable[..., Any]:
    """Resolve a `file://` callback reference through the path-traversal guard."""
    try:
        return load_callback_from_file_url(file_ref)
    except (CallbackPathTraversalError, UnsupportedCallbackError):
        raise
    except Exception as err:
        raise RuntimeError(f"Error loading function from {file_ref}: {err}") from err


async def execute_provider_function_callback(
    function_name: str,
    args: Any,
    callbacks: dict[str, Any] | None,
    cache: dict[str, Callable[..., Any]],
    call_id: str | None = None,
    log_prefix: str | None = None,
) -> str:
    """Load, cache and invoke one `functionToolCallbacks` entry, returning the string a
    tool-result message expects. Errors are re-raised for the caller's fallback."""
    prefix = f"{log_prefix} " if log_prefix else ""
    try:
        callback = cache.get(function_name)
        if callback is None:
            ref = (callbacks or {}).get(function_name)
            if ref and isinstance(ref, str):
                if ref.startswith("file://"):
                    callback = load_provider_callback_from_file_url(ref, log_prefix)
                else:
                    raise UnsupportedCallbackError(
                        f"Inline JavaScript function callbacks are not supported by promptphoo "
                        f"(function '{function_name}'). Use a Python file reference instead: "
                        "file://callbacks.py:function_name"
                    )
                cache[function_name] = callback
            elif callable(ref):
                callback = ref
                cache[function_name] = callback
        if callback is None:
            raise ValueError(f"No callback found for function '{function_name}'")

        logger.debug("%sExecuting function '%s' with args: %s", prefix, function_name, args)
        result = await python_hooks.call_async(callback, args)
        if result is None:
            return ""
        if isinstance(result, (dict, list)):
            try:
                return json_stringify(result)
            except Exception as error:
                logger.warning(
                    "Error stringifying result from function '%s': %s", function_name, error
                )
                return str(result)
        return js_string(result)
    except Exception as error:
        logger.error("%sError executing function '%s': %s", prefix, function_name, error)
        raise


class FunctionCallbackHandler:
    """Ports FunctionCallbackHandler: run `functionToolCallbacks` for function calls in the
    various provider formats, falling back to the JSON-stringified call. The MCP client path is
    not ported (promptphoo has no MCP client)."""

    def __init__(self) -> None:
        self.loaded_callbacks: dict[str, Callable[..., Any]] = {}

    @staticmethod
    def _extract_function_info(call: Any) -> dict[str, Any] | None:
        if not isinstance(call, dict):
            return None
        if call.get("name") and isinstance(call["name"], str):
            return {"name": call["name"], "arguments": call.get("arguments")}
        function = call.get("function")
        if call.get("type") == "function" and isinstance(function, dict) and function.get("name"):
            return {"name": function["name"], "arguments": function.get("arguments")}
        return None

    def _load_external_function(self, file_ref: str) -> Callable[..., Any]:
        try:
            return load_callback_from_file_url(file_ref)
        except Exception as error:
            raise RuntimeError(f"Failed to load function from {file_ref}: {error}") from error

    async def _execute_callback(
        self, function_name: str, args: Any, callbacks: dict[str, Any], context: Any = None
    ) -> str:
        callback = self.loaded_callbacks.get(function_name)
        if callback is None:
            callback_config = callbacks.get(function_name)
            if isinstance(callback_config, str):
                if callback_config.startswith("file://"):
                    callback = self._load_external_function(callback_config)
                else:
                    raise UnsupportedCallbackError(
                        "Inline JavaScript function callbacks are not supported by promptphoo "
                        f"(function '{function_name}'). Use a Python file reference instead: "
                        "file://callbacks.py:function_name"
                    )
            elif callable(callback_config):
                callback = callback_config
            else:
                raise ValueError(f"Invalid callback configuration for {function_name}")
            self.loaded_callbacks[function_name] = callback
        # JS ignores surplus arguments; Python callbacks only get `context` when there is one.
        result = await (
            python_hooks.call_async(callback, args)
            if context is None
            else python_hooks.call_async(callback, args, context)
        )
        return result if isinstance(result, str) else json_stringify(result)

    async def process_call(
        self, call: Any, callbacks: dict[str, Any] | None = None, context: Any = None
    ) -> dict[str, Any]:
        """`{output, isError}` for one call."""
        info = self._extract_function_info(call)
        original = call if isinstance(call, str) else json_stringify(call)
        if not info or not callbacks or not callbacks.get(info["name"]):
            return {"output": original, "isError": False}
        try:
            arguments = info["arguments"] or "{}"
            output = await self._execute_callback(info["name"], arguments, callbacks, context)
            return {"output": output, "isError": False}
        except Exception as error:
            if isinstance(error, CallbackPathTraversalError) or isinstance(
                error.__cause__, CallbackPathTraversalError
            ):
                logger.warning("[FunctionCallback] Rejected callback '%s': %s", info["name"], error)
            else:
                logger.debug("Function callback failed for %s: %s", info["name"], error)
            return {"output": original, "isError": True}

    async def process_calls(
        self, calls: Any, callbacks: dict[str, Any] | None = None, context: Any = None
    ) -> Any:
        """Callback output for a call (or the `\\n`-joined outputs for a list). Without a
        successful callback: the stringified call (single) or the original list."""
        if not calls:
            return calls
        is_list = isinstance(calls, list)
        calls_list = calls if is_list else [calls]
        results = [await self.process_call(c, callbacks, context) for c in calls_list]
        has_success = any(
            not r["isError"] and r["output"] != json_stringify(calls_list[i])
            for i, r in enumerate(results)
        )
        if has_success:
            outputs = [r["output"] for r in results]
            if not is_list and len(outputs) == 1:
                return outputs[0]
            return "\n".join(outputs) if all(isinstance(o, str) for o in outputs) else outputs
        if not is_list and len(results) == 1:
            return results[0]["output"]
        return calls

    def clear_cache(self) -> None:
        self.loaded_callbacks = {}
