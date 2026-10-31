"""Base class for the Google AI Studio and Vertex AI providers. Ports
src/providers/google/base.ts.

MCP integration and streamed function-call argument assembly are not ported. Function
callbacks must be Python (`file://callbacks.py:fn`) or callables; JavaScript callback files and
inline JS function strings fail with a clear error.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.google.auth import GoogleAuthManager
from promptphoo.providers.google.models import get_vertex_model_default_region
from promptphoo.providers.google.util import (
    load_google_tools,
    resolve_google_tool_config,
    validate_function_call,
)
from promptphoo.providers.shared import get_request_timeout_ms

logger = logging.getLogger(__name__)


class GoogleGenericProvider(ApiProvider):
    """Shared functionality for Google AI Studio and Vertex AI."""

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.model_name = model_name
        self.is_vertex_mode = bool(GoogleAuthManager.determine_vertex_mode(self.config, self.env))
        GoogleAuthManager.validate_and_warn(
            {
                "apiKey": self.config.get("apiKey"),
                "credentials": self.config.get("credentials"),
                "projectId": self.config.get("projectId"),
                "region": self.config.get("region"),
                "vertexai": self.is_vertex_mode,
                "googleAuthOptions": self.config.get("googleAuthOptions"),
                "keyFilename": self.config.get("keyFilename"),
                "scopes": self.config.get("scopes"),
                "strictMutualExclusivity": self.config.get("strictMutualExclusivity"),
            },
            self.env,
        )
        self._loaded_function_callbacks: dict[str, Any] = {}
        self._loaded_function_callback_refs: dict[str, Any] = {}
        self._loaded_function_callback_base_paths: dict[str, Any] = {}
        if (self.config.get("mcp") or {}).get("enabled"):
            logger.warning("[Google] MCP integration is not supported by promptphoo; ignoring.")

    def validate_function_tool_call(self, output: Any, vars: Any = None) -> None:
        validate_function_call(output, self.config.get("tools"), vars)

    def get_audio_input_format(self) -> str | None:
        return "google" if self.model_name.startswith("gemini") else None

    def default_id(self) -> str:
        return f"vertex:{self.model_name}" if self.is_vertex_mode else f"google:{self.model_name}"

    def __str__(self) -> str:
        service = "Vertex AI" if self.is_vertex_mode else "Google AI Studio"
        return f"[Google {service} Provider {self.model_name}]"

    def get_api_endpoint(self, action: str | None = None) -> str:
        raise NotImplementedError

    async def get_auth_headers(self) -> dict[str, str]:
        raise NotImplementedError

    def requires_api_key(self) -> bool:
        # Vertex AI supports OAuth/ADC authentication without an API key.
        return not self.is_vertex_mode

    def get_api_key(self) -> str | None:
        api_key = GoogleAuthManager.get_api_key(self.config, self.env, self.is_vertex_mode)[
            "apiKey"
        ]
        if api_key:
            from promptphoo.templating import render_string

            return render_string(api_key, {})
        return None

    def get_region(self) -> str:
        has_api_key = bool(self.get_api_key())
        model_default_region = (
            get_vertex_model_default_region(self.model_name) if self.is_vertex_mode else None
        )
        return GoogleAuthManager.resolve_region(
            self.config, self.env, has_api_key, model_default_region
        )

    async def get_project_id(self) -> str:
        return await GoogleAuthManager.resolve_project_id(
            {
                "projectId": self.config.get("projectId"),
                "credentials": self.config.get("credentials"),
                "googleAuthOptions": self.config.get("googleAuthOptions"),
                "keyFilename": self.config.get("keyFilename"),
                "scopes": self.config.get("scopes"),
            },
            self.env,
        )

    async def get_all_tools(
        self, context: dict[str, Any] | None = None, skip_executable_tool_files: bool = False
    ) -> list[Any]:
        """Tools from the prompt-level config first, falling back to the provider config."""
        context = context or {}
        prompt_config = (context.get("prompt") or {}).get("config") or {}
        config_tools = prompt_config.get("tools")
        if config_tools is None:
            config_tools = self.config.get("tools")
        return await load_google_tools(
            config_tools, context.get("vars"), skip_executable_tool_files
        )

    async def _load_external_function(self, file_ref: str, base_path: str | None) -> Any:
        from promptphoo.files import is_javascript_file
        from promptphoo.python_hooks import load_function, split_path_and_function

        file_path, function_name = split_path_and_function(file_ref[len("file://") :])
        try:
            if is_javascript_file(file_path):
                raise ValueError(
                    "JavaScript function callbacks are not supported by promptphoo; "
                    "use a Python file (file://callbacks.py:function_name) instead"
                )
            if not function_name:
                raise ValueError(f"Function name is required in {file_ref}")
            resolved = (
                file_path if os.path.isabs(file_path) else os.path.join(base_path or "", file_path)
            )
            return load_function(os.path.abspath(resolved), function_name)
        except Exception as error:
            raise ValueError(f"Error loading function from {file_ref}: {error}") from error

    async def execute_function_callback(
        self, function_name: str, args: str, config: dict[str, Any], call_id: str | None = None
    ) -> Any:
        from promptphoo import cli_state
        from promptphoo.python_hooks import call_async

        try:
            callbacks = config.get("functionToolCallbacks") or {}
            callback_ref = callbacks.get(function_name)
            callback_base_path = (
                os.path.abspath(config.get("basePath") or cli_state.base_path() or os.getcwd())
                if isinstance(callback_ref, str) and callback_ref.startswith("file://")
                else None
            )
            callback = self._loaded_function_callbacks.get(function_name)
            if (
                self._loaded_function_callback_refs.get(function_name) is not callback_ref
                and self._loaded_function_callback_refs.get(function_name) != callback_ref
            ) or self._loaded_function_callback_base_paths.get(function_name) != callback_base_path:
                callback = None

            if callback is None:
                if callback_ref and isinstance(callback_ref, str):
                    if callback_ref.startswith("file://"):
                        callback = await self._load_external_function(
                            callback_ref, callback_base_path
                        )
                    else:
                        raise ValueError(
                            f"Failed to parse inline function for '{function_name}': inline "
                            "JavaScript functions are not supported by promptphoo. "
                            "Consider using 'file://' prefix to reference an external file."
                        )
                elif callable(callback_ref):
                    callback = callback_ref
                if callback is not None:
                    self._loaded_function_callbacks[function_name] = callback
                    self._loaded_function_callback_refs[function_name] = callback_ref
                    self._loaded_function_callback_base_paths[function_name] = callback_base_path

            if callback is None:
                raise ValueError(f"No callback found for function '{function_name}'")
            logger.debug("Executing function '%s' with args: %s", function_name, args)
            return await call_async(callback, args)
        except Exception as error:
            logger.error("Error executing function '%s': %s", function_name, error)
            raise

    async def execute_function_tool_callbacks(
        self, output: Any, config: dict[str, Any], tools_disabled: bool
    ) -> Any:
        """Execute explicitly configured callbacks from native parts or JSON envelopes."""
        if tools_disabled:
            return output
        parsed_output = output
        if isinstance(output, str):
            try:
                parsed_output = js_json_parse(output)
            except ValueError:
                return output
        parts = parsed_output if isinstance(parsed_output, list) else [parsed_output]
        callbacks = config.get("functionToolCallbacks")

        def call_of(part: Any) -> Any:
            return part.get("functionCall") if isinstance(part, dict) else None

        if isinstance(output, str) and not any(
            isinstance((call_of(p) or {}).get("name"), str)
            and callbacks
            and call_of(p)["name"] in callbacks
            for p in parts
        ):
            return output
        tool_config = resolve_google_tool_config(config).get("toolConfig") or {}
        streams_args = (
            config.get("streaming") is True
            and (tool_config.get("functionCallingConfig") or {}).get("streamFunctionCallArguments")
            is True
        )
        native_calls = [call_of(p) for p in parts if call_of(p)]
        if streams_args:
            raise ValueError(
                "Streamed function-call arguments (streamFunctionCallArguments) are not "
                "supported by promptphoo."
            )
        if any(
            c.get("willContinue") is True or len(c.get("partialArgs") or []) > 0
            for c in native_calls
        ):
            raise ValueError(
                "Streamed function-call arguments require streaming: true and "
                "toolConfig.functionCallingConfig.streamFunctionCallArguments: true."
            )
        if not native_calls:
            return output
        if not callbacks:
            return output

        prepared: list[tuple[str, str, Any]] = []
        for call in native_calls:
            name = call.get("name")
            if not isinstance(name, str) or name not in callbacks:
                return output
            try:
                raw_args = call.get("args")
                args = js_json_parse(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                prepared.append((name, json_stringify(args), call.get("id")))
            except ValueError:
                return output

        results: list[Any] = []
        for name, args, call_id in prepared:
            try:
                results.append(await self.execute_function_callback(name, args, config, call_id))
            except Exception as error:
                raise ValueError(
                    f"Function callback '{name}' failed after {len(results)} completed "
                    f"callback(s). Check for side effects before retrying: Error: {error}"
                ) from error
        if len(results) == 1:
            return "" if results[0] is None else results[0]
        return "\n".join(
            r if isinstance(r, str) else (json_stringify(r) if r is not None else js_string(r))
            for r in results
        )

    def get_timeout(self) -> int:
        return self.config.get("timeoutMs") or get_request_timeout_ms()
