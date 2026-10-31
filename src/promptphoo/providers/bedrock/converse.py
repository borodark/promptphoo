"""Bedrock Converse API provider (`bedrock:converse:<model>`). Ports
src/providers/bedrock/converse.ts.

Calls run inside a GenAI tracing span (see D57). Not ported: MCP tool dispatch, ConverseStream
(`callApiStreaming`) and telemetry. `functionToolCallbacks` accepts Python callables or
`file://path.py:function` references (inline JavaScript function strings are rejected).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any

from promptphoo import cache as cache_mod
from promptphoo import envars
from promptphoo.jsvalues import is_truthy, js_string, json_stringify
from promptphoo.providers.anthropic.util import (
    get_claude_model_warning_name,
    is_always_on_adaptive_thinking_claude_model,
    is_claude_thinking_enabled,
    is_forced_tool_choice_unsupported_claude_model,
    is_sampling_params_deprecated_claude_model,
    normalize_claude_thinking_config,
    resolve_claude_sampling_params,
)
from promptphoo.providers.bedrock.base import (
    AwsBedrockGenericProvider,
    aws_error_string,
    create_bedrock_cache_key_hash,
    to_js_json,
)
from promptphoo.providers.bedrock.pricing import calculate_bedrock_cost
from promptphoo.providers.shared import (
    is_openai_tool_array,
    is_openai_tool_choice,
    openai_tool_choice_to_bedrock,
    openai_tools_to_bedrock,
)
from promptphoo.tracing.genai_tracer import get_test_index, with_genai_span

logger = logging.getLogger(__name__)


def _one_hour_cache_write_tokens(cache_details: Any) -> int:
    if not isinstance(cache_details, list):
        return 0
    return sum(
        (d.get("inputTokens") or 0) if d.get("ttl") == "1h" else 0
        for d in cache_details
        if isinstance(d, dict)
    )


def convert_tools_to_converse_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI, Anthropic, shorthand or native tool definitions to Converse `Tool`s."""
    if is_openai_tool_array(tools):
        return openai_tools_to_bedrock(tools)
    out: list[dict[str, Any]] = []
    for tool in tools:
        if is_truthy(tool.get("toolSpec")):
            out.append({"toolSpec": tool["toolSpec"]})
        elif tool.get("type") == "function" and is_truthy(tool.get("function")):
            fn = tool["function"]
            spec: dict[str, Any] = {"name": fn.get("name")}
            if "description" in fn:
                spec["description"] = fn["description"]
            spec["inputSchema"] = {"json": fn.get("parameters") or {}}
            out.append({"toolSpec": spec})
        elif is_truthy(tool.get("name")) and "parameters" in tool and "input_schema" not in tool:
            spec = {"name": tool["name"]}
            if "description" in tool:
                spec["description"] = tool["description"]
            spec["inputSchema"] = {
                "json": tool.get("parameters") or {"type": "object", "properties": {}}
            }
            out.append({"toolSpec": spec})
        elif is_truthy(tool.get("name")):
            spec = {"name": tool["name"]}
            if "description" in tool:
                spec["description"] = tool["description"]
            spec["inputSchema"] = {"json": tool.get("input_schema") or {}}
            out.append({"toolSpec": spec})
        else:
            raise ValueError(f"Invalid tool configuration: {json_stringify(tool)}")
    return out


def _is_named_converse_tool_choice(tool_choice: Any) -> bool:
    if not isinstance(tool_choice, dict) or "tool" not in tool_choice:
        return False
    tool = tool_choice["tool"]
    return isinstance(tool, dict) and isinstance(tool.get("name"), str)


def convert_tool_choice_to_converse_format(tool_choice: Any) -> dict[str, Any] | None:
    if is_openai_tool_choice(tool_choice):
        return openai_tool_choice_to_bedrock(tool_choice)
    if tool_choice == "any":
        return {"any": {}}
    if _is_named_converse_tool_choice(tool_choice):
        return {"tool": {"name": tool_choice["tool"]["name"]}}
    return {"auto": {}}


def _is_disabled_tool_choice(tool_choice: Any) -> bool:
    return tool_choice == "none"


_IMAGE_DATA_URL = re.compile(r"^data:image/([^;]+);base64,(.+)$", re.S)


def _b64(value: str) -> bytes:
    """Node's lenient Buffer.from(value, 'base64')."""
    cleaned = re.sub(r"[^A-Za-z0-9+/\-_]", "", value).replace("-", "+").replace("_", "/")
    cleaned = cleaned[: len(cleaned) - len(cleaned) % 4 if len(cleaned) % 4 == 1 else None]
    return base64.b64decode(cleaned + "=" * (-len(cleaned) % 4))


def _image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    image_data = block["image"] if is_truthy(block.get("image")) else block
    source = image_data.get("source") or {}
    fmt = "png"
    if is_truthy(image_data.get("format")):
        fmt = image_data["format"]
    elif is_truthy(source.get("media_type")):
        fmt = source["media_type"].split("/")[1] if "/" in source["media_type"] else "png"
        fmt = fmt or "png"
    data: bytes | None = None
    if is_truthy(source.get("bytes")):
        raw = source["bytes"]
        if isinstance(raw, str):
            if raw.startswith("data:"):
                m = _IMAGE_DATA_URL.match(raw)
                if m:
                    fmt = "jpeg" if m.group(1) == "jpg" else m.group(1)
                    data = _b64(m.group(2))
            else:
                data = _b64(raw)
        elif isinstance(raw, bytes | bytearray):
            data = bytes(raw)
    elif is_truthy(source.get("data")):
        data = _b64(source["data"])
    if data is None:
        logger.warning("Could not parse image content block")
        return None
    if fmt == "jpg":
        fmt = "jpeg"
    return {"image": {"format": fmt, "source": {"bytes": data}}}


def _document_block(block: dict[str, Any]) -> dict[str, Any] | None:
    doc = block["document"] if is_truthy(block.get("document")) else block
    fmt = doc.get("format") or "txt"
    name = doc.get("name") or "document"
    source = doc.get("source") or {}
    data: bytes | None = None
    raw = source.get("bytes")
    if is_truthy(raw):
        if isinstance(raw, str):
            if raw.startswith("data:"):
                m = re.match(r"^data:[^;]+;base64,(.+)$", raw, re.S)
                if m:
                    data = _b64(m.group(1))
            else:
                data = _b64(raw)
        elif isinstance(raw, bytes | bytearray):
            data = bytes(raw)
    if data is None:
        logger.warning("Could not parse document content block")
        return None
    return {"document": {"format": fmt, "name": name, "source": {"bytes": data}}}


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": json_stringify(content)}]
    for block in content:
        if isinstance(block, str):
            blocks.append({"text": block})
        elif block.get("type") == "text":
            blocks.append({"text": block.get("text")})
        elif block.get("type") == "image" or is_truthy(block.get("image")):
            converted = _image_block(block)
            if converted:
                blocks.append(converted)
        elif block.get("type") == "image_url" or is_truthy(block.get("image_url")):
            image_url = (block.get("image_url") or {}).get("url") or block.get("url")
            if isinstance(image_url, str) and image_url.startswith("data:"):
                m = _IMAGE_DATA_URL.match(image_url)
                if m:
                    fmt = "jpeg" if m.group(1) == "jpg" else m.group(1)
                    blocks.append({"image": {"format": fmt, "source": {"bytes": _b64(m.group(2))}}})
            else:
                logger.warning("Unsupported image_url format (only data URLs supported)")
        elif block.get("type") == "document" or is_truthy(block.get("document")):
            converted = _document_block(block)
            if converted:
                blocks.append(converted)
        elif block.get("type") == "tool_use" or is_truthy(block.get("toolUse")):
            data = block["toolUse"] if is_truthy(block.get("toolUse")) else block
            tool_use: dict[str, Any] = {}
            tool_use_id = data.get("toolUseId") or data.get("id")
            if tool_use_id is not None:
                tool_use["toolUseId"] = tool_use_id
            for key in ("name", "input"):
                if key in data:
                    tool_use[key] = data[key]
            blocks.append({"toolUse": tool_use})
        elif block.get("type") == "tool_result" or is_truthy(block.get("toolResult")):
            data = block["toolResult"] if is_truthy(block.get("toolResult")) else block
            result: dict[str, Any] = {}
            tool_use_id = data.get("toolUseId") or data.get("tool_use_id")
            if tool_use_id is not None:
                result["toolUseId"] = tool_use_id
            raw_content = data.get("content", None)
            result["content"] = (
                [{"text": c} if isinstance(c, str) else c for c in raw_content]
                if isinstance(raw_content, list)
                else [{"text": js_string(raw_content) if "content" in data else "undefined"}]
            )
            if "status" in data:
                result["status"] = data["status"]
            blocks.append({"toolResult": result})
        else:
            blocks.append({"text": json_stringify(block)})
    return blocks


def _converse_span_result(response: dict[str, Any]) -> dict[str, Any]:
    """Converse's span result: the three usage totals and the stop reason from metadata."""
    result: dict[str, Any] = {}
    usage = response.get("tokenUsage")
    if usage:
        result["tokenUsage"] = {
            "prompt": usage.get("prompt"),
            "completion": usage.get("completion"),
            "total": usage.get("total"),
        }
    stop_reason = (response.get("metadata") or {}).get("stopReason")
    if stop_reason:
        result["finishReasons"] = [stop_reason]
    return result


def parse_converse_messages(prompt: str) -> dict[str, Any]:
    """Parse a prompt into {messages, system?} in Converse format."""
    try:
        parsed = json.loads(prompt)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        try:
            system_messages: list[dict[str, Any]] = []
            messages: list[dict[str, Any]] = []
            for msg in parsed:
                role = msg.get("role")
                if role == "system":
                    content = msg.get("content")
                    system_messages.append(
                        {"text": content if isinstance(content, str) else json_stringify(content)}
                    )
                elif role in ("user", "assistant"):
                    messages.append({"role": role, "content": _content_blocks(msg.get("content"))})
            out: dict[str, Any] = {"messages": messages}
            if system_messages:
                out["system"] = system_messages
            return out
        except (AttributeError, TypeError, ValueError):
            pass

    messages = []
    system: list[dict[str, Any]] | None = None
    current_role: str | None = None
    current_content: list[str] = []

    def push_message() -> None:
        nonlocal current_content
        if current_role and current_content:
            messages.append(
                {"role": current_role, "content": [{"text": "\n".join(current_content)}]}
            )
            current_content = []

    for line in prompt.split("\n"):
        trimmed = line.strip()
        lower = trimmed.lower()
        if lower.startswith("system:"):
            push_message()
            system = [{"text": trimmed[7:].strip()}]
            current_role = None
        elif lower.startswith("user:"):
            push_message()
            current_role = "user"
            content = trimmed[5:].strip()
            if content:
                current_content.append(content)
        elif lower.startswith("assistant:"):
            push_message()
            current_role = "assistant"
            content = trimmed[10:].strip()
            if content:
                current_content.append(content)
        elif current_role:
            current_content.append(line)
        else:
            current_role = "user"
            current_content.append(line)
    push_message()
    if not messages:
        messages.append({"role": "user", "content": [{"text": prompt}]})
    result: dict[str, Any] = {"messages": messages}
    if system is not None:
        result["system"] = system
    return result


def extract_text_from_content_blocks(
    content: list[dict[str, Any]], show_thinking: bool = True
) -> str:
    parts: list[str] = []
    for block in content:
        if "text" in block and block["text"]:
            parts.append(block["text"])
        elif "reasoningContent" in block and is_truthy(block["reasoningContent"]):
            reasoning = block["reasoningContent"]
            if show_thinking:
                if "reasoningText" in reasoning and is_truthy(reasoning["reasoningText"]):
                    thinking_text = reasoning["reasoningText"].get("text") or ""
                    signature = reasoning["reasoningText"].get("signature") or ""
                    if thinking_text.strip() != "":
                        parts.append(f"<thinking>\n{thinking_text}\n</thinking>")
                        if signature:
                            parts.append(f"Signature: {signature}")
                elif "redactedContent" in reasoning and is_truthy(reasoning["redactedContent"]):
                    parts.append("<thinking>[Redacted]</thinking>")
        elif "toolUse" in block and is_truthy(block["toolUse"]):
            tool_use = block["toolUse"]
            entry: dict[str, Any] = {"type": "tool_use"}
            for out_key, in_key in (("id", "toolUseId"), ("name", "name"), ("input", "input")):
                if in_key in tool_use:
                    entry[out_key] = tool_use[in_key]
            parts.append(json_stringify(entry))
    return "\n\n".join(parts)


class AwsBedrockConverseProvider(AwsBedrockGenericProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, options)
        self._loaded_function_callbacks: dict[str, Any] = {}
        self._forced_tool_choice_removal_warned = False
        mcp = self.config.get("mcp")
        if isinstance(mcp, dict) and mcp.get("enabled"):
            logger.warning(
                "[Bedrock Converse] MCP servers are not supported by promptphoo; `mcp` is ignored."
            )

    def default_id(self) -> str:
        return f"bedrock:converse:{self.model_name}"

    def __str__(self) -> str:
        return f"[AWS Bedrock Converse Provider {self.model_name}]"

    async def _execute_function_callback(self, function_name: str, args: str) -> str:
        from promptphoo.python_hooks import call_async, load_function, split_path_and_function

        callbacks = self.config.get("functionToolCallbacks") or {}
        try:
            callback = self._loaded_function_callbacks.get(function_name)
            if callback is None:
                ref = callbacks.get(function_name)
                if callable(ref):
                    callback = ref
                elif isinstance(ref, str) and ref.startswith("file://"):
                    path, fn_name = split_path_and_function(ref[len("file://") :])
                    if not fn_name:
                        raise ValueError(f"Function name is required in file reference: {ref}")
                    try:
                        callback = load_function(path, fn_name)
                    except Exception as err:
                        raise RuntimeError(f"Error loading function from {ref}: {err}") from err
                elif isinstance(ref, str):
                    raise ValueError(
                        "Inline JavaScript functionToolCallbacks are not supported by promptphoo; "
                        "use a file://path.py:function reference."
                    )
                if callback is None:
                    raise ValueError(f"No callback found for function '{function_name}'")
                self._loaded_function_callbacks[function_name] = callback
            result = await call_async(callback, args)
            if result is None:
                return ""
            if isinstance(result, dict | list):
                return json_stringify(result)
            return js_string(result)
        except Exception as err:
            logger.debug("[Bedrock Converse] Error executing function '%s': %s", function_name, err)
            raise

    def build_inference_config(self) -> dict[str, Any] | None:
        config = self.config
        reasoning = config.get("reasoningConfig") or {}
        reasoning_enabled = reasoning.get("type") == "enabled"
        high_effort = reasoning.get("maxReasoningEffort") == "high"

        def first(*values: Any) -> Any:
            return next((v for v in values if v is not None), None)

        max_tokens_value = first(
            config.get("maxTokens"),
            config.get("max_tokens"),
            envars.get_env_int("AWS_BEDROCK_MAX_TOKENS"),
        )
        temperature_value = first(
            config.get("temperature"), envars.get_env_float("AWS_BEDROCK_TEMPERATURE")
        )
        top_p_value = first(
            config.get("topP"), config.get("top_p"), envars.get_env_float("AWS_BEDROCK_TOP_P")
        )
        stop_sequences = config.get("stopSequences") or config.get("stop")
        if not stop_sequences:
            env_stop = envars.get_env_string("AWS_BEDROCK_STOP")
            if env_stop:
                try:
                    stop_sequences = json.loads(env_stop)
                except ValueError:
                    pass

        max_tokens = None if reasoning_enabled and high_effort else max_tokens_value
        deprecated = is_sampling_params_deprecated_claude_model(self.model_name)
        temperature = None if reasoning_enabled or deprecated else temperature_value
        top_p = None if reasoning_enabled or deprecated else top_p_value
        if "anthropic.claude" in self.model_name:
            raw_thinking = (config.get("additionalModelRequestFields") or {}).get("thinking")
            thinking = config.get("thinking")
            sampling, warnings = resolve_claude_sampling_params(
                {
                    **({} if temperature is None else {"temperature": temperature}),
                    **({} if top_p is None else {"top_p": top_p}),
                },
                thinking_enabled=is_claude_thinking_enabled(
                    raw_thinking if thinking is None else thinking
                ),
                sampling_params_deprecated=deprecated,
                default_temperature=None,
            )
            for warning in warnings:
                logger.warning(warning)
            temperature = sampling.get("temperature")
            top_p = sampling.get("top_p")

        if max_tokens is not None or temperature is not None or top_p is not None or stop_sequences:
            out: dict[str, Any] = {}
            if max_tokens is not None:
                out["maxTokens"] = max_tokens
            if temperature is not None:
                out["temperature"] = temperature
            if top_p is not None:
                out["topP"] = top_p
            if stop_sequences:
                out["stopSequences"] = stop_sequences
            return out
        return None

    def _effective_tool_choice(self, prompt_config: dict[str, Any] | None) -> Any:
        prompt_config = prompt_config or {}
        for value in (
            prompt_config.get("tool_choice"),
            prompt_config.get("toolChoice"),
            self.config.get("tool_choice"),
            self.config.get("toolChoice"),
        ):
            if value is not None:
                return value
        return None

    async def build_tool_config(
        self, vars: dict[str, Any] | None = None, prompt_config: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        from promptphoo.assertions.tools import maybe_load_tools_from_external_file

        config_tool_choice = self._effective_tool_choice(prompt_config)
        if _is_disabled_tool_choice(config_tool_choice):
            return None
        config_tools = (prompt_config or {}).get("tools")
        if config_tools is None:
            config_tools = self.config.get("tools")
        if not config_tools:
            return None
        tools = await maybe_load_tools_from_external_file(config_tools, vars)
        if not tools:
            return None
        converse_tools = convert_tools_to_converse_format(list(tools))
        requested = (
            convert_tool_choice_to_converse_format(config_tool_choice)
            if is_truthy(config_tool_choice)
            else None
        )
        rejects_forced = is_forced_tool_choice_unsupported_claude_model(self.model_name)
        drop_forced = (
            (rejects_forced or is_always_on_adaptive_thinking_claude_model(self.model_name))
            and requested is not None
            and ("any" in requested or "tool" in requested)
        )
        if drop_forced and not self._forced_tool_choice_removal_warned:
            name = get_claude_model_warning_name(self.model_name) or "this Claude model"
            logger.warning(
                f"Forced tool choice (any/tool) is not supported on {name} and has been omitted. "
                "The model decides when to call tools; remove toolChoice to silence this warning."
                if rejects_forced
                else "Forced tool choice (any/tool) is incompatible with the always-on adaptive "
                f"thinking of {name} and has been omitted. The model decides when to call tools; "
                "remove toolChoice to silence this warning."
            )
            self._forced_tool_choice_removal_warned = True
        tool_choice = None if drop_forced else requested
        out: dict[str, Any] = {"tools": converse_tools}
        if tool_choice:
            out["toolChoice"] = tool_choice
        return out

    def build_guardrail_config(self) -> dict[str, Any] | None:
        if not is_truthy(self.config.get("guardrailIdentifier")):
            return None
        out: dict[str, Any] = {
            "guardrailIdentifier": js_string(self.config["guardrailIdentifier"]),
            "guardrailVersion": js_string(self.config.get("guardrailVersion") or "DRAFT"),
        }
        if is_truthy(self.config.get("trace")):
            out["trace"] = self.config["trace"]
        return out

    def build_additional_model_request_fields(self) -> dict[str, Any] | None:
        fields: dict[str, Any] = dict(self.config.get("additionalModelRequestFields") or {})
        output_config = fields.get("output_config")
        effort = output_config.get("effort") if isinstance(output_config, dict) else None
        if is_sampling_params_deprecated_claude_model(self.model_name):
            for key in ("temperature", "top_p", "top_k"):
                fields.pop(key, None)
            normalized = normalize_claude_thinking_config(
                self.model_name, fields.get("thinking"), effort
            )
            if normalized is None:
                fields.pop("thinking", None)
            else:
                fields["thinking"] = normalized
        if is_truthy(self.config.get("thinking")):
            normalized = normalize_claude_thinking_config(
                self.model_name, self.config["thinking"], effort
            )
            if normalized is not None:
                fields["thinking"] = normalized
        if is_truthy(self.config.get("reasoningConfig")):
            fields["reasoningConfig"] = self.config["reasoningConfig"]
        return fields or None

    def build_performance_config(self) -> dict[str, Any] | None:
        perf: Any = self.config.get("performanceConfig")
        if not is_truthy(perf):
            return None
        return {"latency": perf.get("latency")}

    def build_service_tier(self) -> dict[str, Any] | None:
        tier: Any = self.config.get("serviceTier")
        if not is_truthy(tier):
            return None
        return {"type": tier.get("type")}

    def _is_request_tools_disabled(self, context: dict[str, Any] | None) -> bool:
        prompt_config = ((context or {}).get("prompt") or {}).get("config")
        return _is_disabled_tool_choice(self._effective_tool_choice(prompt_config))

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        inference_config = self.build_inference_config() or {}
        span_context = {
            "system": "bedrock",
            "operationName": "chat",
            "model": self.model_name,
            "providerId": self.id(),
            "maxTokens": inference_config.get("maxTokens"),
            "temperature": inference_config.get("temperature"),
            "topP": inference_config.get("topP"),
            "stopSequences": inference_config.get("stopSequences"),
            "testIndex": get_test_index(context),
            "promptLabel": ((context or {}).get("prompt") or {}).get("label"),
            "traceparent": (context or {}).get("traceparent"),
        }
        return await with_genai_span(
            span_context,
            lambda _span: self._call_api_internal(prompt, context, options),
            _converse_span_result,
        )

    async def _call_api_internal(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        parsed = parse_converse_messages(prompt)
        messages, system = parsed["messages"], parsed.get("system")
        inference_config = self.build_inference_config()
        tool_config = await self.build_tool_config(
            context.get("vars"), (context.get("prompt") or {}).get("config")
        )
        tools_disabled = self._is_request_tools_disabled(context)
        guardrail_config = self.build_guardrail_config()
        additional_fields = self.build_additional_model_request_fields()
        between_tools = ((additional_fields or {}).get("thinking") or {}).get(
            "type"
        ) == "between_tools"
        performance_config = self.build_performance_config()
        service_tier = self.build_service_tier()

        converse_input: dict[str, Any] = {"modelId": self.model_name, "messages": messages}
        if system:
            converse_input["system"] = system
        if inference_config:
            converse_input["inferenceConfig"] = inference_config
        if tool_config:
            converse_input["toolConfig"] = tool_config
        if guardrail_config:
            converse_input["guardrailConfig"] = guardrail_config
        if additional_fields:
            converse_input["additionalModelRequestFields"] = additional_fields
        if is_truthy(self.config.get("additionalModelResponseFieldPaths")):
            converse_input["additionalModelResponseFieldPaths"] = self.config[
                "additionalModelResponseFieldPaths"
            ]
        if performance_config:
            converse_input["performanceConfig"] = performance_config
        if service_tier:
            converse_input["serviceTier"] = service_tier

        cache = cache_mod.get_cache()
        region = self.get_region()
        cache_key = (
            f"bedrock:converse:{self.model_name}:{region}:"
            f"{create_bedrock_cache_key_hash(self.config, converse_input, region)}"
        )
        if cache_mod.is_cache_enabled():
            cached = await cache.get(cache_key)
            if cached:
                result = await self.parse_response(
                    json.loads(cached), tools_disabled, between_tools
                )
                return {**result, "cached": True}

        try:
            client = self.get_bedrock_instance()
            response = await asyncio.to_thread(client.converse, **converse_input)
        except Exception as err:
            message = str(err) or aws_error_string(err)
            logger.error("Bedrock Converse API error: %s", message)
            if "ValidationException" in message:
                return {
                    "error": f"Bedrock Converse API validation error: {message}. Check that your "
                    "model supports the Converse API and all parameters are valid."
                }
            if "AccessDeniedException" in message:
                return {
                    "error": f"Bedrock access denied: {message}. Ensure you have "
                    "bedrock:InvokeModel permission and model access is enabled."
                }
            return {"error": f"Bedrock Converse API error: {message}"}

        response = {k: v for k, v in response.items() if k != "ResponseMetadata"}
        if cache_mod.is_cache_enabled():
            try:
                await cache.set(cache_key, json_stringify(to_js_json(response)))
            except Exception as err:
                logger.error("Failed to cache response: %s", err)
        return await self.parse_response(response, tools_disabled, between_tools)

    async def parse_response(
        self,
        response: dict[str, Any],
        tools_disabled: bool = False,
        between_tools_thinking: bool = False,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = ((response.get("output") or {}).get("message") or {}).get(
            "content"
        ) or []
        show_thinking = self.config.get("showThinking") is not False
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("inputTokens")
        completion_tokens = usage.get("outputTokens")
        total_tokens = usage.get("totalTokens")
        cache_read = usage.get("cacheReadInputTokens")
        cache_write = usage.get("cacheWriteInputTokens")
        token_usage: dict[str, Any] = {}
        if prompt_tokens is not None:
            token_usage["prompt"] = prompt_tokens
        if completion_tokens is not None:
            token_usage["completion"] = completion_tokens
        token_usage["total"] = total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)
        token_usage["numRequests"] = 1
        cost = calculate_bedrock_cost(
            self.model_name,
            prompt_tokens,
            completion_tokens,
            cache_read,
            cache_write,
            self.get_region(),
            self.config.get("serviceTier"),
            _one_hour_cache_write_tokens(usage.get("cacheDetails")),
        )

        metadata: dict[str, Any] = {}
        latency = (response.get("metrics") or {}).get("latencyMs")
        if latency:
            metadata["latencyMs"] = latency
        if response.get("stopReason"):
            metadata["stopReason"] = response["stopReason"]
        if cache_read is not None or cache_write is not None:
            metadata["cacheTokens"] = {
                **({} if cache_read is None else {"read": cache_read}),
                **({} if cache_write is None else {"write": cache_write}),
            }
        for key in ("performanceConfig", "serviceTier", "additionalModelResponseFields", "trace"):
            if is_truthy(response.get(key)):
                metadata[key] = response[key]

        guardrails = (
            {"flagged": True, "reason": "guardrail_intervened"}
            if response.get("stopReason") == "guardrail_intervened"
            else None
        )
        malformed_error: str | None = None
        if response.get("stopReason") == "malformed_model_output":
            malformed_error = (
                "Model produced invalid output. The response could not be parsed correctly."
            )
            metadata["isModelError"] = True
        elif response.get("stopReason") == "malformed_tool_use":
            malformed_error = (
                "Model produced a malformed tool use request. Check tool configuration and input "
                "schema."
            )
            metadata["isModelError"] = True

        def finish(output: str, error: str | None) -> dict[str, Any]:
            result: dict[str, Any] = {"output": output, "tokenUsage": token_usage}
            if cost is not None:
                result["cost"] = cost
            if metadata:
                result["metadata"] = metadata
            if guardrails:
                result["guardrails"] = guardrails
            if error:
                result["error"] = error
            return result

        tool_use_blocks = [b for b in content if b.get("toolUse") is not None]
        dispatch_results: list[str] = []
        handled: set[int] = set()
        callbacks = self.config.get("functionToolCallbacks")
        if not tools_disabled and tool_use_blocks:
            if callbacks:
                for idx, block in enumerate(tool_use_blocks):
                    name = block["toolUse"].get("name")
                    if not name or not callbacks.get(name):
                        continue
                    try:
                        raw_input = block["toolUse"].get("input")
                        args = (
                            raw_input
                            if isinstance(raw_input, str)
                            else json_stringify(raw_input or {})
                        )
                        dispatch_results.append(await self._execute_function_callback(name, args))
                        handled.add(idx)
                    except Exception as err:
                        logger.warning(
                            f"[Bedrock Converse] Function callback failed for {name}: {err}; "
                            "falling back to tool_use output"
                        )
            if len(handled) < len(tool_use_blocks):
                fallback = extract_text_from_content_blocks(
                    [
                        {"toolUse": b["toolUse"]}
                        for i, b in enumerate(tool_use_blocks)
                        if i not in handled
                    ],
                    show_thinking,
                )
                if fallback:
                    dispatch_results.append(fallback)

        if dispatch_results:
            if between_tools_thinking:
                progress = extract_text_from_content_blocks(
                    [b for b in content if is_truthy(b.get("reasoningContent"))], show_thinking
                )
                if progress:
                    dispatch_results.insert(0, progress)
            return finish("\n".join(dispatch_results), malformed_error)
        return finish(extract_text_from_content_blocks(content, show_thinking), malformed_error)
