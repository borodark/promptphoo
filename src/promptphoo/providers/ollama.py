"""Ollama completion, chat and embedding providers. Ports src/providers/ollama.ts and the
`ollama:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo import envars
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.shared import (
    get_request_timeout_ms,
    js_error_string,
    normalize_finish_reason,
    parse_chat_prompt,
    resolve_provider_env,
    should_bust_provider_cache,
    transform_tools,
    with_response_cache_metadata,
)
from promptphoo.tracing.genai_tracer import extract_genai_response, get_test_index, with_genai_span

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

OLLAMA_COMPLETION_OPTION_KEYS = frozenset([
    "num_predict", "num_keep", "seed", "top_k", "top_p", "min_p", "typical_p", "repeat_last_n",
    "temperature", "repeat_penalty", "presence_penalty", "frequency_penalty", "stop", "num_ctx",
    "num_batch", "num_gpu", "main_gpu", "use_mmap", "num_thread", "draft_num_predict", "tfs_z",
    "num_gqa", "f16_kv", "logits_all", "vocab_only", "low_vram", "use_mlock", "embedding_only",
    "rope_frequency_base", "rope_frequency_scale", "penalize_newline", "mirostat", "mirostat_tau",
    "mirostat_eta", "tools", "think", "keep_alive", "truncate", "dimensions", "format", "suffix",
    "system", "template", "raw", "passthrough",
])  # fmt: skip

# Keys in the config block that are NOT members of Ollama's nested `options` object.
OLLAMA_NON_NESTED_OPTION_KEYS = frozenset([
    "tools", "think", "passthrough", "keep_alive", "truncate", "dimensions", "format", "suffix",
    "system", "template", "raw",
])  # fmt: skip

# Which top-level (non-`options`) keys each endpoint actually accepts.
OLLAMA_ENDPOINT_TOP_LEVEL_KEYS: dict[str, frozenset[str]] = {
    "completion": frozenset(
        ["think", "keep_alive", "format", "truncate", "suffix", "system", "template", "raw"]
    ),
    "chat": frozenset(["think", "keep_alive", "format", "truncate", "tools"]),
    "embedding": frozenset(["keep_alive", "truncate", "dimensions"]),
}

# Never user-supplied Ollama options; reporting them as dropped would be noise.
OLLAMA_INTERNAL_CONFIG_KEYS = frozenset(["showThinking", "basePath"])

OLLAMA_DEPRECATED_OPTION_KEYS = frozenset([
    "tfs_z", "num_gqa", "f16_kv", "logits_all", "vocab_only", "low_vram", "use_mlock",
    "embedding_only", "rope_frequency_base", "rope_frequency_scale", "penalize_newline",
    "mirostat", "mirostat_tau", "mirostat_eta",
])  # fmt: skip


def build_ollama_options(config: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """Builds the nested `options` object Ollama expects, dropping top-level keys."""
    dropped: list[str] = []
    wrong_endpoint: list[str] = []
    deprecated: list[str] = []
    supported_top_level = OLLAMA_ENDPOINT_TOP_LEVEL_KEYS[endpoint]
    options: dict[str, Any] = {}
    for key, value in config.items():
        if key in OLLAMA_COMPLETION_OPTION_KEYS:
            if key in OLLAMA_NON_NESTED_OPTION_KEYS:
                if key != "passthrough" and key not in supported_top_level:
                    wrong_endpoint.append(key)
            else:
                options[key] = value
                if key in OLLAMA_DEPRECATED_OPTION_KEYS:
                    deprecated.append(key)
        elif key not in OLLAMA_INTERNAL_CONFIG_KEYS:
            dropped.append(key)

    if wrong_endpoint:
        logger.warning(
            "[Ollama] Ignoring config keys that the %s endpoint does not accept: %s",
            endpoint,
            ", ".join(wrong_endpoint),
        )
    if dropped:
        logger.debug(
            "[Ollama] Ignoring unsupported config keys %s (generation options belong under "
            "`passthrough.options`; other top-level API fields go directly under `passthrough`.)",
            dropped,
        )
    if deprecated:
        logger.debug(
            "[Ollama] Forwarding options that current Ollama releases ignore %s", deprecated
        )
    return options


def split_ollama_passthrough(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Splits `passthrough` so a user-provided `options` object merges with the computed one."""
    passthrough = dict(config.get("passthrough") or {})
    passthrough_options = passthrough.pop("options", None)
    return passthrough_options or {}, passthrough


def is_ollama_http_failure(status: int) -> bool:
    return status < 200 or status >= 300


def stringify_ollama_error_detail(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()[:500]
    if data is None:
        return ""
    try:
        return json_stringify(data)[:500]
    except Exception:
        return ""


def extract_ollama_error_message(data: Any) -> str | None:
    if isinstance(data, dict) and "error" in data:
        error = data["error"]
        if isinstance(error, str):
            return error
        return None if error is None else json_stringify(error)
    if isinstance(data, str):
        for line in data.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("{") or '"error"' not in trimmed:
                continue
            try:
                message = extract_ollama_error_message(js_json_parse(trimmed))
            except ValueError:
                continue
            if message is not None:
                return message
    return None


def get_ollama_response_error(data: Any, status: int, status_text: str) -> str | None:
    message = extract_ollama_error_message(data)
    if is_ollama_http_failure(status):
        detail = message if message is not None else stringify_ollama_error_detail(data)
        return f"Ollama API error: {status} {status_text}{f': {detail}' if detail else ''}"
    if message is not None:
        return f"Ollama error: {message}"
    return None


def normalize_ollama_request_messages(messages: Any) -> Any:
    """Converts `function.arguments` JSON strings back to objects on outgoing messages."""
    if not isinstance(messages, list):
        return messages
    out = []
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list):
            out.append(message)
            continue
        calls = []
        for call in tool_calls:
            fn: Any = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if not isinstance(args, str):
                calls.append(call)
                continue
            try:
                parsed = js_json_parse(args)
            except ValueError:
                calls.append(call)
                continue
            if not isinstance(parsed, dict):
                calls.append(call)
                continue
            calls.append({**call, "function": {**fn, "arguments": parsed}})
        out.append({**message, "tool_calls": calls})
    return out


def collect_ollama_tool_calls(lines: list[Any]) -> list[dict[str, Any]]:
    """Collects tool calls from every chunk, normalized to the OpenAI shape."""
    result = []
    for chunk in lines:
        message = chunk.get("message") if isinstance(chunk, dict) else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list):
            continue
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                continue
            args = fn.get("arguments")
            result.append({
                "function": {
                    "name": fn["name"],
                    "arguments": args
                    if isinstance(args, str)
                    else json_stringify({} if args is None else args),
                }
            })  # fmt: skip
    return result


def extract_ollama_token_usage(final_chunk: dict[str, Any]) -> dict[str, Any] | None:
    if final_chunk.get("prompt_eval_count") is None and final_chunk.get("eval_count") is None:
        return None
    prompt = final_chunk.get("prompt_eval_count") or 0
    completion = final_chunk.get("eval_count") or 0
    total = prompt + completion
    cache_read = final_chunk.get("prompt_eval_cached_count")
    if cache_read is not None:
        return {
            "prompt": prompt,
            "completion": completion,
            "total": total,
            "completionDetails": {"cacheReadInputTokens": cache_read},
            "numRequests": 1,
        }
    return {"prompt": prompt, "completion": completion, "total": total, "numRequests": 1}


def apply_ollama_thinking(output: Any, thinking: str, show_thinking: bool | None = None) -> Any:
    if (
        not thinking
        or not isinstance(output, str)
        or not (True if show_thinking is None else show_thinking)
    ):
        return output
    return f"Thinking: {thinking}\n\n{output}" if output else f"Thinking: {thinking}"


def _top_level(config: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: config[key] for key in keys if key in config}


def _parse_ndjson(data: str) -> list[Any]:
    return [js_json_parse(line) for line in data.split("\n") if line.strip() != ""]


def _finish_and_usage(lines: list[Any]) -> tuple[str | None, dict[str, Any] | None]:
    final_chunk = next((c for c in lines if isinstance(c, dict) and c.get("done")), None)
    finish_reason = normalize_finish_reason((final_chunk or {}).get("done_reason"))
    token_usage = extract_ollama_token_usage(final_chunk) if final_chunk else None
    return finish_reason, token_usage


def _result(output: Any, finish_reason: str | None, token_usage: dict | None) -> dict[str, Any]:
    response: dict[str, Any] = {"output": output}
    if finish_reason:
        response["finishReason"] = finish_reason
    if token_usage:
        response["tokenUsage"] = token_usage
    return response


class _OllamaBase(ApiProvider):
    kind = "completion"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.model_name = model_name

    def default_id(self) -> str:
        return f"ollama:{self.kind}:{self.model_name}"

    def _base_url(self) -> str:
        resolved = resolve_provider_env(self.env, ["OLLAMA_BASE_URL"])
        return (resolved or {}).get("value") or DEFAULT_OLLAMA_BASE_URL

    def _headers(self) -> dict[str, str]:
        api_key = envars.get_provider_env_string(self.env, "OLLAMA_API_KEY")
        if api_key is None:
            api_key = envars.get_env_string("OLLAMA_API_KEY")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        span_context = {
            "system": "ollama",
            "operationName": self.kind,
            "model": self.model_name,
            "providerId": self.id(),
            "temperature": self.config.get("temperature"),
            "topP": self.config.get("top_p"),
            "maxTokens": self.config.get("num_predict"),
            "stopSequences": self.config.get("stop"),
            "testIndex": get_test_index(context),
            "promptLabel": ((context or {}).get("prompt") or {}).get("label"),
            "traceparent": (context or {}).get("traceparent"),
        }
        return await with_genai_span(
            span_context,
            lambda _span: self._call_api_internal(prompt, context, options),
            lambda response: extract_genai_response(response, True),
        )

    async def _call_api_internal(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError

    async def _post(self, path: str, params: dict[str, Any], fmt: Any, context: Any) -> Any:
        from promptphoo.fetch import fetch_with_cache

        return await fetch_with_cache(
            f"{self._base_url()}{path}",
            {"method": "POST", "headers": self._headers(), "body": json_stringify(params)},
            get_request_timeout_ms(),
            fmt,
            should_bust_provider_cache(context),
        )


class OllamaCompletionProvider(_OllamaBase):
    kind = "completion"

    def __str__(self) -> str:
        return f"[Ollama Completion Provider {self.model_name}]"

    async def _call_api_internal(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        passthrough_options, passthrough_rest = split_ollama_passthrough(self.config)
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {**build_ollama_options(self.config, "completion"), **passthrough_options},
            **_top_level(
                self.config,
                ("think", "keep_alive", "format", "truncate", "suffix", "system", "template",
                 "raw"),
            ),
            **passthrough_rest,
        }  # fmt: skip
        logger.debug("Calling Ollama API %s", params)
        try:
            response = await self._post("/api/generate", params, "text", context)
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}. Output:\nundefined"}
        logger.debug("\tOllama generate API response: %s", response.data)
        response_error = get_ollama_response_error(
            response.data, response.status, response.status_text
        )
        if response_error:
            if response.delete_from_cache:
                await response.delete_from_cache()
            return {"error": response_error}

        try:
            lines = _parse_ndjson(response.data)
            output = "".join(
                r for r in (p.get("response") for p in lines) if isinstance(r, str) and r
            )
            thinking = "".join(t for t in (p.get("thinking") for p in lines) if isinstance(t, str))
            output = apply_ollama_thinking(output, thinking, self.config.get("showThinking"))
            finish_reason, token_usage = _finish_and_usage(lines)
            return with_response_cache_metadata(
                _result(output, finish_reason, token_usage), response.cached
            )
        except Exception as err:
            return {
                "error": f"Ollama API response error: {js_error_string(err)}: "
                f"{json_stringify(response.data)}"
            }


class OllamaChatProvider(_OllamaBase):
    kind = "chat"

    def __str__(self) -> str:
        return f"[Ollama Chat Provider {self.model_name}]"

    async def _call_api_internal(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        messages = normalize_ollama_request_messages(
            parse_chat_prompt(prompt, [{"role": "user", "content": prompt}])
        )
        passthrough_options, passthrough_rest = split_ollama_passthrough(self.config)
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": {**build_ollama_options(self.config, "chat"), **passthrough_options},
            **_top_level(self.config, ("think", "keep_alive", "format", "truncate")),
            **passthrough_rest,
        }
        if is_truthy(self.config.get("tools")):
            from promptphoo.assertions.tools import maybe_load_tools_from_external_file

            loaded = await maybe_load_tools_from_external_file(
                self.config["tools"], (context or {}).get("vars")
            )
            if loaded is not None:
                params["tools"] = transform_tools(loaded, "openai")

        logger.debug("[Ollama Chat] Calling Ollama API %s", params)
        try:
            response = await self._post("/api/chat", params, "text", context)
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}. Output:\nundefined"}

        response_error = get_ollama_response_error(
            response.data, response.status, response.status_text
        )
        if response_error:
            if response.delete_from_cache:
                await response.delete_from_cache()
            return {"error": response_error}

        try:
            lines = _parse_ndjson(response.data)
            finish_reason, token_usage = _finish_and_usage(lines)
            content = "".join(
                c
                for c in ((p.get("message") or {}).get("content") for p in lines)
                if isinstance(c, str) and c
            )
            thinking = "".join(
                t
                for t in ((p.get("message") or {}).get("thinking") for p in lines)
                if isinstance(t, str)
            )
            tool_calls = collect_ollama_tool_calls(lines)
            output: Any
            if tool_calls:
                output = (
                    {"content": content, "tool_calls": tool_calls}
                    if content.strip()
                    else (tool_calls)
                )
            else:
                output = content
            output = apply_ollama_thinking(output, thinking, self.config.get("showThinking"))
            return with_response_cache_metadata(
                _result(output, finish_reason, token_usage), response.cached
            )
        except Exception as err:
            return {
                "error": f"Ollama API response error: {js_error_string(err)}: "
                f"{json_stringify(response.data)}"
            }


class OllamaEmbeddingProvider(OllamaCompletionProvider):
    supports_embedding_cancellation = True

    async def call_embedding_api(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        passthrough_options, passthrough_rest = split_ollama_passthrough(self.config)
        truncate = self.config.get("truncate")
        params = {
            "model": self.model_name,
            "input": text,
            # Ollama defaults this to true, which silently embeds only the first num_ctx tokens.
            "truncate": False if truncate is None else truncate,
            **_top_level(self.config, ("dimensions", "keep_alive")),
            "options": {**build_ollama_options(self.config, "embedding"), **passthrough_options},
            **passthrough_rest,
        }
        logger.debug("Calling Ollama embeddings API %s", params)
        try:
            response = await self._post("/api/embed", params, "json", context)
        except Exception as err:
            return {"error": f"API call error: {js_error_string(err)}"}

        response_error = get_ollama_response_error(
            response.data, response.status, response.status_text
        )
        if response_error:
            if response.delete_from_cache:
                await response.delete_from_cache()
            if "input length exceeds the context length" in response_error:
                return {
                    "error": f"{response_error}. Raise `config.num_ctx` (up to the model's own "
                    f"maximum, shown by `ollama show {self.model_name}`), or set "
                    "`config.truncate: true` to embed only the first num_ctx tokens -- note "
                    "that truncating silently changes similarity scores."
                }
            return {"error": response_error}

        try:
            data = response.data
            embeddings = data.get("embeddings") if isinstance(data, dict) else None
            embedding = embeddings[0] if embeddings else None
            if not embedding:
                raise ValueError("No embedding found in Ollama embeddings API response")
            prompt_tokens = data.get("prompt_eval_count")
            result: dict[str, Any] = {"embedding": embedding}
            if prompt_tokens is not None:
                result["tokenUsage"] = {
                    "prompt": prompt_tokens,
                    "total": prompt_tokens,
                    "numRequests": 1,
                }
            return with_response_cache_metadata(result, response.cached)
        except Exception as err:
            return {
                "error": f"API response error: {js_error_string(err)}: "
                f"{json_stringify(response.data)}"
            }


def create_provider(provider_path: str, options: dict[str, Any]) -> ApiProvider:
    splits = provider_path.split(":")
    first_part = splits[1] if len(splits) > 1 else None
    if first_part == "chat":
        return OllamaChatProvider(":".join(splits[2:]), options)
    if first_part == "completion":
        return OllamaCompletionProvider(":".join(splits[2:]), options)
    if first_part in ("embedding", "embeddings"):
        return OllamaEmbeddingProvider(":".join(splits[2:]), options)
    # Default to completion provider
    return OllamaCompletionProvider(":".join(splits[1:]), options)
