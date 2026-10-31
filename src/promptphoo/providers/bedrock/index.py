"""Bedrock InvokeModel providers (`bedrock:<model>`) and embeddings.
Ports src/providers/bedrock/index.ts.

Model families ported: Anthropic Claude (Messages), Amazon Nova and Nova 2, Amazon Titan Text,
Meta Llama 2/3/3.1/3.2/3.3/4, Mistral (text and chat), AI21 Jamba, DeepSeek V3 chat, and the
OpenAI chat-compatible families (GLM, MiniMax, Kimi, Nemotron, Gemma, Palmyra, Grok 4.6).
The legacy Claude text-completion, Cohere, DeepSeek R1, gpt-oss and Qwen handlers are not ported;
their models raise a clear error that points at `bedrock:converse:<model>`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from promptphoo import cache as cache_mod
from promptphoo import envars
from promptphoo.jsvalues import is_truthy, js_string, json_stringify
from promptphoo.providers.anthropic.util import (
    clamp_max_tokens_for_thinking_budget,
    get_token_usage,
    is_always_on_adaptive_thinking_claude_model,
    is_claude_thinking_enabled,
    is_forced_tool_choice_unsupported_claude_model,
    is_sampling_params_deprecated_claude_model,
    is_thinking_on_by_default_claude_model,
    normalize_claude_thinking_config,
    output_from_message,
    parse_messages,
    resolve_claude_sampling_params,
)
from promptphoo.providers.bedrock.base import (
    INFERENCE_PROFILE_PREFIX,
    AwsBedrockGenericProvider,
    aws_error_string,
    create_bedrock_cache_key_hash,
    nova_output_from_message,
    nova_parse_messages,
)
from promptphoo.providers.bedrock.pricing import (
    calculate_bedrock_invoke_model_cost,
    is_bedrock_grok46_profile,
)
from promptphoo.providers.bedrock.routing import requires_bedrock_anthropic_messages_model
from promptphoo.providers.shared import parse_chat_prompt
from promptphoo.tokens import create_empty_token_usage

logger = logging.getLogger(__name__)


def coerce_str_to_num(value: Any) -> Any:
    """JS Number() for strings; other values pass through (None stays None)."""
    if value is None or not isinstance(value, str):
        return value
    s = value.strip()
    if s == "":
        return 0
    try:
        n = float(s)
    except ValueError:
        return math.nan
    return int(n) if n.is_integer() else n


def _missing_token_usage() -> dict[str, Any]:
    return {"prompt": None, "completion": None, "total": None, "numRequests": 1}


def _usage(prompt: Any, completion: Any, total: Any) -> dict[str, Any]:
    return {"prompt": prompt, "completion": completion, "total": total, "numRequests": 1}


def _nova_token_usage(response: Any) -> dict[str, Any]:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not is_truthy(usage):
        return _missing_token_usage()
    assert isinstance(usage, dict)
    return _usage(
        coerce_str_to_num(usage.get("inputTokens")),
        coerce_str_to_num(usage.get("outputTokens")),
        coerce_str_to_num(usage.get("totalTokens")),
    )


def _openai_compatible_token_usage(response: Any, _prompt: str = "") -> dict[str, Any]:
    usage = (response.get("usage") if isinstance(response, dict) else None) or {}
    return _usage(
        coerce_str_to_num(usage.get("prompt_tokens")),
        coerce_str_to_num(usage.get("completion_tokens")),
        coerce_str_to_num(usage.get("total_tokens")),
    )


def parse_value(value: Any, default_value: Any) -> Any:
    if isinstance(default_value, int | float) and not isinstance(default_value, bool):
        if isinstance(value, str):
            parsed = envars.parse_env_float(value)
            return default_value if parsed is None else parsed
        return value
    return value


def add_config_param(
    params: dict[str, Any],
    key: str,
    config_value: Any,
    env_value: Any = None,
    default_value: Any = None,
) -> None:
    if config_value is not None or env_value is not None or default_value is not None:
        params[key] = (
            config_value
            if config_value is not None
            else (default_value if env_value is None else parse_value(env_value, default_value))
        )


def _env_int(key: str) -> int | None:
    return envars.get_env_int(key)


def _env_float(key: str) -> float | None:
    return envars.get_env_float(key)


def _chat_messages(prompt: str) -> Any:
    return parse_chat_prompt(prompt, [{"role": "user", "content": prompt}])


@dataclass
class BedrockModelHandler:
    """IBedrockModel: request params, output extraction and token usage for one family."""

    name: str
    params: Callable[..., Awaitable[dict[str, Any]]]
    output: Callable[[dict[str, Any], Any], Any]
    token_usage: Callable[[Any, str], dict[str, Any]] | None = None


# --- Mistral / DeepSeek / AI21 / OpenAI-compatible chat ----------------------------------------


async def _mistral_chat_params(config, prompt, stop=None, model_name=None, vars=None):
    params: dict[str, Any] = {"messages": _chat_messages(prompt)}
    add_config_param(
        params, "max_tokens", config.get("max_tokens"), _env_int("MISTRAL_MAX_TOKENS"), 1024
    )
    add_config_param(
        params, "temperature", config.get("temperature"), _env_float("MISTRAL_TEMPERATURE"), 0
    )
    add_config_param(params, "top_p", config.get("top_p"), _env_float("MISTRAL_TOP_P"), 1)
    if stop:
        add_config_param(params, "stop", stop)
    return params


def _mistral_chat_output(config, response):
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list):
        first = choices[0] if choices else None
        return ((first or {}).get("message") or {}).get("content")
    return None


def _mistral_chat_token_usage(response, _prompt=""):
    if response.get("prompt_tokens") is not None and response.get("completion_tokens") is not None:
        p = coerce_str_to_num(response["prompt_tokens"])
        c = coerce_str_to_num(response["completion_tokens"])
        return _usage(p, c, (p or 0) + (c or 0))
    usage = response.get("usage") or {}
    if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
        p = coerce_str_to_num(usage["prompt_tokens"])
        c = coerce_str_to_num(usage["completion_tokens"])
        total = coerce_str_to_num(usage.get("total_tokens"))
        if total is None:
            total = p + c if p is not None and c is not None else None
        return _usage(p, c, total)
    return _missing_token_usage()


async def _deepseek_chat_params(config, prompt, stop=None, model_name=None, vars=None):
    params: dict[str, Any] = {"messages": _chat_messages(prompt)}
    add_config_param(
        params, "max_tokens", config.get("max_tokens"), _env_int("AWS_BEDROCK_MAX_TOKENS")
    )
    add_config_param(
        params, "temperature", config.get("temperature"), _env_float("AWS_BEDROCK_TEMPERATURE"), 0
    )
    add_config_param(params, "top_p", config.get("top_p"), _env_float("AWS_BEDROCK_TOP_P"), 1.0)
    if stop:
        add_config_param(params, "stop", stop)
    return params


def _deepseek_chat_output(config, response):
    if is_truthy(response.get("error")):
        raise ValueError(f"DeepSeek API error: {js_string(response['error'])}")
    return _first_choice_content(response)


def _first_choice_content(response: dict[str, Any]) -> Any:
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    return ((first or {}).get("message") or {}).get("content") if isinstance(first, dict) else None


async def _ai21_params(config, prompt, stop=None, model_name=None, vars=None):
    params: dict[str, Any] = {"messages": _chat_messages(prompt)}
    add_config_param(
        params, "max_tokens", config.get("max_tokens"), _env_int("AWS_BEDROCK_MAX_TOKENS")
    )
    add_config_param(
        params, "temperature", config.get("temperature"), _env_float("AWS_BEDROCK_TEMPERATURE"), 0
    )
    add_config_param(params, "top_p", config.get("top_p"), _env_float("AWS_BEDROCK_TOP_P"), 1.0)
    add_config_param(params, "stop", config.get("stop"), envars.get_env_string("AWS_BEDROCK_STOP"))
    add_config_param(
        params,
        "frequency_penalty",
        config.get("frequency_penalty"),
        _env_float("AWS_BEDROCK_FREQUENCY_PENALTY"),
    )
    add_config_param(
        params,
        "presence_penalty",
        config.get("presence_penalty"),
        _env_float("AWS_BEDROCK_PRESENCE_PENALTY"),
    )
    return params


def _ai21_output(config, response):
    if is_truthy(response.get("error")):
        raise ValueError(f"AI21 API error: {js_string(response['error'])}")
    return _first_choice_content(response)


async def _openai_compat_params(config, prompt, stop=None, model_name=None, vars=None):
    from promptphoo.assertions.tools import maybe_load_tools_from_external_file

    params: dict[str, Any] = {"messages": _chat_messages(prompt)}
    add_config_param(
        params, "max_tokens", config.get("max_tokens"), _env_int("AWS_BEDROCK_MAX_TOKENS")
    )
    add_config_param(
        params, "temperature", config.get("temperature"), _env_float("AWS_BEDROCK_TEMPERATURE")
    )
    add_config_param(params, "top_p", config.get("top_p"), _env_float("AWS_BEDROCK_TOP_P"))
    config_stop = config.get("stop")
    effective_stop = config_stop if config_stop else (stop if stop else None)
    if effective_stop:
        add_config_param(params, "stop", effective_stop)
    add_config_param(
        params,
        "frequency_penalty",
        config.get("frequency_penalty"),
        _env_float("AWS_BEDROCK_FREQUENCY_PENALTY"),
    )
    add_config_param(
        params,
        "presence_penalty",
        config.get("presence_penalty"),
        _env_float("AWS_BEDROCK_PRESENCE_PENALTY"),
    )
    add_config_param(params, "reasoning_effort", config.get("reasoning_effort"))
    add_config_param(
        params, "tools", await maybe_load_tools_from_external_file(config.get("tools"), vars)
    )
    add_config_param(params, "tool_choice", config.get("tool_choice"))
    return params  # fmt: skip


_THINK_PREFIX = re.compile(r"^\s*<(think|reasoning)>[\s\S]*?</\1>", re.I)


def _openai_compat_output(config, response):
    if is_truthy(response.get("error")):
        raise ValueError(f"Bedrock API error: {js_string(response['error'])}")
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = (choice or {}).get("message") or {}
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        content = message.get("content")
        if isinstance(content, str) and config.get("showThinking") is False:
            content = re.sub(
                r"^\s*<(think|reasoning)>[\s\S]*?</\1>\s*", "", content, count=1, flags=re.I
            )
        calls = "\n".join(
            f"Called function {js_string((tc.get('function') or {}).get('name', _U))} with "
            f"arguments: {js_string((tc.get('function') or {}).get('arguments', _U))}"
            for tc in tool_calls
        )
        return f"{content}\n\n{calls}" if content else calls
    content = message.get("content")
    if not isinstance(content, str):
        return content
    if config.get("showThinking") is False:
        m = re.match(r"^\s*<(think|reasoning)>[\s\S]*?</\1>", content)
        if m:
            stripped = content[len(m.group(0)) :].strip()
            return content if stripped == "" else stripped
    return content


class _UndefinedStr:
    def __str__(self) -> str:
        return "undefined"


_U = _UndefinedStr()


# --- Llama -------------------------------------------------------------------------------------

LLAMA_VERSIONS = (2, 3, 3.1, 3.2, 3.3, 4)


def _dedent(result: str) -> str:
    """The `dedent` tag (v1.x) applied after interpolation."""
    lines = result.split("\n")
    mindent: int | None = None
    for line in lines:
        m = re.match(r"^(\s+)\S+", line)
        if m:
            mindent = len(m.group(1)) if mindent is None else min(mindent, len(m.group(1)))
    if mindent is not None:
        result = "\n".join(line[mindent:] if line[:1] in (" ", "\t") else line for line in lines)
    return result.strip().replace("\\n", "\n")


def _image_url_data(url: Any) -> str | None:
    if isinstance(url, str) and url.startswith("data:"):
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", url, re.S)
        if m:
            return m.group(1)
    return None


def _extract_image_data(block: dict[str, Any]) -> str | None:
    image_url = (
        (block.get("image_url") or {}).get("url")
        if isinstance(block.get("image_url"), dict)
        else None
    )
    if not image_url and block.get("type") == "image_url":
        image_url = block.get("url")
    data = _image_url_data(image_url)
    if data:
        return data
    image_data = block.get("image") if is_truthy(block.get("image")) else block
    source = (image_data or {}).get("source") or {}
    raw = source.get("bytes")
    if is_truthy(raw):
        if isinstance(raw, str):
            if raw.startswith("data:"):
                found = _image_url_data(raw)
                if found:
                    return found
            return raw
        if isinstance(raw, bytes | bytearray):
            import base64

            return base64.b64encode(bytes(raw)).decode()
    data: Any = source.get("data")
    if is_truthy(data):
        if data.startswith("data:"):
            found = _image_url_data(data)
            if found:
                return found
        return data
    return None  # fmt: skip


def _is_image_block(block: dict[str, Any]) -> bool:
    return (
        block.get("type") in ("image", "image_url")
        or is_truthy(block.get("image"))
        or is_truthy(block.get("image_url"))
    )


def extract_text_and_images(content: Any) -> dict[str, Any]:
    """Text with <|image|> tokens at image positions, plus base64 images (Llama 3.2 Vision)."""
    if isinstance(content, str):
        return {"text": content.strip(), "images": []}
    if isinstance(content, list):
        parts: list[str] = []
        images: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                block.get("type") == "text"
                and block.get("text")
                or block.get("text")
                and not block.get("type")
            ):
                parts.append(block["text"])
            elif _is_image_block(block):
                data = _extract_image_data(block)
                if data:
                    images.append(data)
                    parts.append("<|image|>")
        return {"text": "".join(parts).strip(), "images": images}
    return {"text": js_string(content).strip(), "images": []}


def extract_text_content(content: Any, model_name: str | None = None) -> str:
    """Text of a message; images are rejected because legacy InvokeModel Llama is text-only."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        has_non_text = False
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                block.get("type") == "text"
                and block.get("text")
                or block.get("text")
                and not block.get("type")
            ):
                parts.append(block["text"])
            elif _is_image_block(block):
                has_non_text = True
        if has_non_text:
            info = f" ({model_name})" if model_name else ""
            model = model_name or "<model-id>"
            raise ValueError(
                f"Multimodal content (images) detected but the legacy Bedrock Llama provider{info} "
                "does not support images. Please use the Converse API provider instead:\n\n"
                f"  Change: bedrock:{model}\n"
                f"  To:     bedrock:converse:{model}\n\n"
                "The Converse API supports multimodal content for vision-capable models like "
                "Llama 3.2 11B/90B."
            )
        return " ".join(parts).strip()
    return js_string(content).strip()


def format_prompt_llama2_chat(messages: list[dict[str, Any]], model_name: str | None = None) -> str:
    if not messages:
        return ""
    formatted = "<s>"
    system_included = False
    for i, message in enumerate(messages):
        text = extract_text_content(message.get("content"), model_name)
        role = message.get("role")
        if role == "system":
            if not system_included:
                formatted += f"[INST] <<SYS>>\n{text}\n<</SYS>>\n\n"
                system_included = True
        elif role == "user":
            if i == 0 and not system_included:
                formatted += f"[INST] {text} [/INST]"
            elif i == 0 and system_included:
                formatted += f"{text} [/INST]"
            elif i > 0 and messages[i - 1].get("role") == "assistant":
                formatted += f"<s>[INST] {text} [/INST]"
            else:
                formatted += f"{text} [/INST]"
        elif role == "assistant":
            formatted += f" {text} </s>"
        else:
            raise ValueError(f"Unexpected role: {js_string(role)}")
    return formatted


def _llama3_turn(role: Any, text: str) -> str:
    return _dedent(
        f"\n      <|start_header_id|>{js_string(role)}<|end_header_id|>\n\n      {text}<|eot_id|>"
    )


def format_prompt_llama3_instruct(
    messages: list[dict[str, Any]], model_name: str | None = None
) -> str:
    formatted = "<|begin_of_text|>"
    for message in messages:
        formatted += _llama3_turn(
            message.get("role"), extract_text_content(message.get("content"), model_name)
        )
    return formatted + "<|start_header_id|>assistant<|end_header_id|>"


def format_prompt_llama32_vision(messages: list[dict[str, Any]]) -> dict[str, Any]:
    formatted = "<|begin_of_text|>"
    all_images: list[str] = []
    for message in messages:
        extracted = extract_text_and_images(message.get("content"))
        all_images.extend(extracted["images"])
        formatted += _llama3_turn(message.get("role"), extracted["text"])
    formatted += "<|start_header_id|>assistant<|end_header_id|>"
    return {"prompt": formatted, "images": all_images}


def format_prompt_llama4(messages: list[dict[str, Any]], model_name: str | None = None) -> str:
    formatted = "<|begin_of_text|>"
    for message in messages:
        text = extract_text_content(message.get("content"), model_name)
        formatted += _dedent(
            f"<|header_start|>{js_string(message.get('role'))}<|header_end|>\n\n{text}<|eot|>"
        )
    return formatted + "<|header_start|>assistant<|header_end|>"


def _llama_token_usage(response: Any, _prompt: str = "") -> dict[str, Any]:
    usage = response.get("usage")
    if is_truthy(usage):
        return _usage(
            coerce_str_to_num(usage.get("prompt_tokens")),
            coerce_str_to_num(usage.get("completion_tokens")),
            coerce_str_to_num(usage.get("total_tokens")),
        )
    p, c = response.get("prompt_token_count"), response.get("generation_token_count")
    if p is not None and c is not None:
        pn, cn = coerce_str_to_num(p), coerce_str_to_num(c)
        return _usage(pn, cn, (pn or 0) + (cn or 0))
    return _missing_token_usage()


def get_llama_model_handler(version: float) -> BedrockModelHandler:
    if version not in LLAMA_VERSIONS:
        raise ValueError(f"Unsupported LLAMA version: {js_string(version)}")

    async def params(config, prompt, stop=None, model_name=None, vars=None):
        messages = _chat_messages(prompt)
        images: list[str] = []
        if version == 2:
            final_prompt = format_prompt_llama2_chat(messages, model_name)
        elif version in (3, 3.1, 3.3):
            final_prompt = format_prompt_llama3_instruct(messages, model_name)
        elif version == 3.2:
            # Only Llama 3.2 11B and 90B support vision; 1B and 3B are text-only.
            if model_name and re.search(r"11b|90b", model_name, re.I):
                result = format_prompt_llama32_vision(messages)
                final_prompt, images = result["prompt"], result["images"]
            else:
                final_prompt = format_prompt_llama3_instruct(messages, model_name)
        else:
            final_prompt = format_prompt_llama4(messages, model_name)
        out: dict[str, Any] = {"prompt": final_prompt}
        if images:
            out["images"] = images
        add_config_param(
            out, "temperature", config.get("temperature"), _env_float("AWS_BEDROCK_TEMPERATURE"), 0
        )
        add_config_param(out, "top_p", config.get("top_p"), _env_float("AWS_BEDROCK_TOP_P"), 1)
        add_config_param(
            out, "max_gen_len", config.get("max_gen_len"), _env_int("AWS_BEDROCK_MAX_GEN_LEN"), 1024
        )
        return out  # fmt: skip

    return BedrockModelHandler(
        name=f"LLAMA{js_string(version).replace('.', '_')}",
        params=params,
        output=lambda config, response: response.get("generation"),
        token_usage=_llama_token_usage,
    )


# --- Amazon Nova / Titan ---------------------------------------------------------------------


def _nova_messages(prompt: str) -> tuple[Any, Any]:
    try:
        parsed = json.loads(prompt)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        messages = [
            {
                "role": m.get("role"),
                "content": m["content"]
                if isinstance(m.get("content"), list)
                else [{"text": m.get("content")}],
            }
            for m in parsed
        ]
        messages = [m for m in messages if m["role"] != "system"]
        system_message = next((m for m in parsed if m.get("role") == "system"), None)
        system = [{"text": system_message.get("content")}] if system_message else None
        return messages, system
    result = nova_parse_messages(prompt)
    # Upstream wraps the already-wrapped system blocks again ([{text: [{text}]}]).
    return result["extractedMessages"], (
        [{"text": result["system"]}] if result.get("system") else None
    )


async def _nova_params(config, prompt, stop=None, model_name=None, vars=None):
    messages, system = _nova_messages(prompt)
    params: dict[str, Any] = {"messages": messages}
    if system:
        add_config_param(params, "system", system)
    interface = config.get("interfaceConfig")
    inference: dict[str, Any] = dict(interface) if is_truthy(interface) else {}
    add_config_param(
        inference,
        "max_new_tokens",
        (interface or {}).get("max_new_tokens"),
        _env_int("AWS_BEDROCK_MAX_TOKENS"),
    )
    add_config_param(
        inference,
        "temperature",
        (interface or {}).get("temperature"),
        _env_float("AWS_BEDROCK_TEMPERATURE"),
        0,
    )
    add_config_param(params, "inferenceConfig", inference)
    add_config_param(params, "toolConfig", config.get("toolConfig"))
    return params  # fmt: skip


async def _nova2_params(config, prompt, stop=None, model_name=None, vars=None):
    messages, system = _nova_messages(prompt)
    params: dict[str, Any] = {"messages": messages}
    if system:
        add_config_param(params, "system", system)
    reasoning = config.get("reasoningConfig") or {}
    reasoning_enabled = reasoning.get("type") == "enabled"
    high_effort = reasoning.get("maxReasoningEffort") == "high"
    interface = config.get("interfaceConfig") or {}
    inference = {k: v for k, v in interface.items() if k not in ("max_new_tokens", "temperature")}
    if not (reasoning_enabled and high_effort):
        add_config_param(
            inference,
            "max_new_tokens",
            interface.get("max_new_tokens"),
            _env_int("AWS_BEDROCK_MAX_TOKENS"),
        )
    if not reasoning_enabled:
        add_config_param(
            inference,
            "temperature",
            interface.get("temperature"),
            _env_float("AWS_BEDROCK_TEMPERATURE"),
            0,
        )
    add_config_param(params, "inferenceConfig", inference)
    add_config_param(params, "toolConfig", config.get("toolConfig"))
    if is_truthy(config.get("reasoningConfig")):
        add_config_param(params, "reasoningConfig", config["reasoningConfig"])
    return params  # fmt: skip


def _nova2_output(config, response):
    content = ((response.get("output") or {}).get("message") or {}).get("content")
    if not isinstance(content, list) or not content:
        if not isinstance(content, list):
            return nova_output_from_message(response)
    if any((block.get("toolUse") or {}).get("toolUseId") for block in content):
        parts = [
            None if is_truthy(b.get("text")) or "toolUse" not in b else json_stringify(b["toolUse"])
            for b in content
        ]
        return "\n\n".join(p for p in parts if p)
    show_thinking = config.get("showThinking") is not False
    out: list[str] = []
    for block in content:
        if is_truthy(block.get("reasoningContent")) and show_thinking:
            text = ((block["reasoningContent"].get("reasoningText")) or {}).get("text")
            if text:
                out.append(f"<thinking>\n{text}\n</thinking>")
        elif block.get("text"):
            out.append(block["text"])
    return "\n\n".join(out)


async def _titan_params(config, prompt, stop=None, model_name=None, vars=None):
    tgc = config.get("textGenerationConfig") or {}
    text_config: dict[str, Any] = {}
    add_config_param(
        text_config,
        "maxTokenCount",
        tgc.get("maxTokenCount"),
        _env_int("AWS_BEDROCK_MAX_TOKENS"),
        1024,
    )
    add_config_param(
        text_config, "temperature", tgc.get("temperature"), _env_float("AWS_BEDROCK_TEMPERATURE"), 0
    )
    add_config_param(text_config, "topP", tgc.get("topP"), _env_float("AWS_BEDROCK_TOP_P"), 1)
    add_config_param(text_config, "stopSequences", tgc.get("stopSequences"), None, stop)
    return {"inputText": prompt, "textGenerationConfig": text_config}  # fmt: skip


def _titan_output(config, response):
    results = response.get("results")
    first = results[0] if isinstance(results, list) and results else None
    return first.get("outputText") if isinstance(first, dict) else None


# --- Mistral text ------------------------------------------------------------------------------


async def _mistral_params(config, prompt, stop=None, model_name=None, vars=None):
    params: dict[str, Any] = {"prompt": prompt, "stop": stop}
    add_config_param(
        params, "max_tokens", config.get("max_tokens"), _env_int("MISTRAL_MAX_TOKENS"), 1024
    )
    add_config_param(
        params, "temperature", config.get("temperature"), _env_float("MISTRAL_TEMPERATURE"), 0
    )
    add_config_param(params, "top_p", config.get("top_p"), _env_float("MISTRAL_TOP_P"), 1)
    # Only send top_k when configured: Bedrock rejects top_k: 0.
    add_config_param(params, "top_k", config.get("top_k"), _env_float("MISTRAL_TOP_K"))
    return params  # fmt: skip


def _mistral_output(config, response):
    outputs = response.get("outputs")
    if not isinstance(outputs, list):
        return None
    return (outputs[0] or {}).get("text") if outputs else None


def _mistral_token_usage(response, _prompt=""):
    usage = response.get("usage")
    if is_truthy(usage):
        return _usage(
            coerce_str_to_num(usage.get("prompt_tokens")),
            coerce_str_to_num(usage.get("completion_tokens")),
            coerce_str_to_num(usage.get("total_tokens")),
        )
    if response.get("prompt_tokens") is not None and response.get("completion_tokens") is not None:
        p = coerce_str_to_num(response["prompt_tokens"])
        c = coerce_str_to_num(response["completion_tokens"])
        total = coerce_str_to_num(response.get("total_tokens"))
        if total is None:
            total = p + c if p is not None and c is not None else None
        return _usage(p, c, total)
    return _missing_token_usage()


# --- Claude Messages ---------------------------------------------------------------------------


def _claude_messages(prompt: str) -> tuple[Any, Any]:
    try:
        parsed = json.loads(prompt)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        system_messages = [m for m in parsed if m.get("role") == "system"]
        non_system = [m for m in parsed if m.get("role") != "system"]

        def content_of(msg: dict[str, Any]) -> Any:
            content = msg.get("content")
            return content if isinstance(content, list) else [{"type": "text", "text": content}]

        if len(system_messages) == 1 and not non_system:
            # Bedrock Claude rejects system-only prompts; send a lone system message as user.
            return [{"role": "user", "content": content_of(system_messages[0])}], None
        messages = [{"role": m.get("role"), "content": content_of(m)} for m in non_system]
        return messages, system_messages[0].get("content") if system_messages else None
    result = parse_messages(prompt)
    return result["extractedMessages"], result.get("system")


async def _claude_messages_params(config, prompt, stop=None, model_name=None, vars=None):
    from promptphoo.assertions.tools import maybe_load_tools_from_external_file

    messages, system = _claude_messages(prompt)
    params: dict[str, Any] = {"messages": messages}
    add_config_param(
        params, "anthropic_version", config.get("anthropic_version"), None, "bedrock-2023-05-31"
    )
    always_on = bool(model_name) and is_always_on_adaptive_thinking_claude_model(model_name)
    thinking_cfg = config.get("thinking")
    thinking_type = thinking_cfg.get("type") if isinstance(thinking_cfg, dict) else None
    thinks_by_default = always_on or (
        bool(model_name)
        and is_thinking_on_by_default_claude_model(model_name)
        and thinking_type not in ("disabled", "between_tools")
    )
    add_config_param(
        params,
        "max_tokens",
        config.get("max_tokens"),
        _env_int("AWS_BEDROCK_MAX_TOKENS"),
        2048 if thinks_by_default else 1024,
    )
    # InvokeModel exposes no effort field, so the effort-capped rules cannot apply here.
    thinking = (
        normalize_claude_thinking_config(model_name, thinking_cfg, None)
        if model_name
        else thinking_cfg
    )
    sampling, warnings = resolve_claude_sampling_params(
        {"temperature": config.get("temperature")},
        thinking_enabled=always_on or is_claude_thinking_enabled(thinking),
        sampling_params_deprecated=is_sampling_params_deprecated_claude_model(model_name)
        if model_name
        else False,
        default_temperature=0,
    )
    for warning in warnings:
        logger.warning(warning)
    add_config_param(params, "temperature", sampling.get("temperature"))
    add_config_param(
        params, "tools", await maybe_load_tools_from_external_file(config.get("tools"), vars)
    )
    tool_choice = config.get("tool_choice")
    tool_choice_type = tool_choice.get("type") if isinstance(tool_choice, dict) else None
    if (
        always_on
        or (bool(model_name) and is_forced_tool_choice_unsupported_claude_model(model_name))
    ) and tool_choice_type in ("any", "tool"):
        tool_choice = None
    add_config_param(params, "tool_choice", tool_choice)
    add_config_param(params, "thinking", thinking)
    if isinstance(params.get("max_tokens"), int | float) and not isinstance(
        params.get("max_tokens"), bool
    ):
        params["max_tokens"] = clamp_max_tokens_for_thinking_budget(params["max_tokens"], thinking)
    if is_truthy(system):
        add_config_param(params, "system", system)
    return params  # fmt: skip


def _claude_messages_output(config, response):
    show = config.get("showThinking")
    return output_from_message(response, True if show is None else show)


def _claude_messages_token_usage(response, _prompt=""):
    usage = response.get("usage") if isinstance(response, dict) else None
    if not is_truthy(usage):
        return _missing_token_usage()
    assert isinstance(usage, dict)
    normalized = {
        **usage,
        "input_tokens": usage.get("input_tokens")
        if usage.get("input_tokens") is not None
        else usage.get("prompt_tokens"),
        "output_tokens": usage.get("output_tokens")
        if usage.get("output_tokens") is not None
        else usage.get("completion_tokens"),
    }
    return {**get_token_usage({"usage": normalized}, False), "numRequests": 1}


def _unsupported_handler(name: str) -> BedrockModelHandler:
    async def params(config, prompt, stop=None, model_name=None, vars=None):
        raise ValueError(
            f"The Bedrock InvokeModel {name} handler is not supported by promptphoo "
            f"(model {model_name}). Use bedrock:converse:{model_name} instead."
        )

    return BedrockModelHandler(name=name, params=params, output=lambda c, r: None)


_MISTRAL_CHAT = BedrockModelHandler(
    "MISTRAL_CHAT", _mistral_chat_params, _mistral_chat_output, _mistral_chat_token_usage
)

BEDROCK_MODEL: dict[str, BedrockModelHandler] = {
    "AI21": BedrockModelHandler("AI21", _ai21_params, _ai21_output, _openai_compatible_token_usage),
    "AMAZON_NOVA": BedrockModelHandler(
        "AMAZON_NOVA",
        _nova_params,
        lambda c, r: nova_output_from_message(r),
        lambda r, p: _nova_token_usage(r),
    ),
    "AMAZON_NOVA_2": BedrockModelHandler(
        "AMAZON_NOVA_2", _nova2_params, _nova2_output, lambda r, p: _nova_token_usage(r)
    ),
    "CLAUDE_COMPLETION": _unsupported_handler("CLAUDE_COMPLETION"),
    "CLAUDE_MESSAGES": BedrockModelHandler(
        "CLAUDE_MESSAGES",
        _claude_messages_params,
        _claude_messages_output,
        _claude_messages_token_usage,
    ),
    "TITAN_TEXT": BedrockModelHandler(
        "TITAN_TEXT", _titan_params, _titan_output, _openai_compatible_token_usage
    ),
    "LLAMA2": get_llama_model_handler(2),
    "LLAMA3": get_llama_model_handler(3),
    "LLAMA3_1": get_llama_model_handler(3.1),
    "LLAMA3_2": get_llama_model_handler(3.2),
    "LLAMA3_3": get_llama_model_handler(3.3),
    "LLAMA4": get_llama_model_handler(4),
    "COHERE_COMMAND": _unsupported_handler("COHERE_COMMAND"),
    "COHERE_COMMAND_R": _unsupported_handler("COHERE_COMMAND_R"),
    "DEEPSEEK": _unsupported_handler("DEEPSEEK"),
    "MISTRAL": BedrockModelHandler(
        "MISTRAL", _mistral_params, _mistral_output, _mistral_token_usage
    ),
    "MISTRAL_CHAT": _MISTRAL_CHAT,
    "MISTRAL_LARGE_2407": _MISTRAL_CHAT,
    "DEEPSEEK_CHAT": BedrockModelHandler(
        "DEEPSEEK_CHAT",
        _deepseek_chat_params,
        _deepseek_chat_output,
        _openai_compatible_token_usage,
    ),
    "OPENAI": _unsupported_handler("OPENAI"),
    "QWEN": _unsupported_handler("QWEN"),
    "OPENAI_COMPAT": BedrockModelHandler(
        "OPENAI_COMPAT",
        _openai_compat_params,
        _openai_compat_output,
        _openai_compatible_token_usage,
    ),
}

_M = BEDROCK_MODEL
_CLAUDE = "CLAUDE_MESSAGES"
_NOVA = "AMAZON_NOVA"

AWS_BEDROCK_MODELS: dict[str, BedrockModelHandler] = {
    k: _M[v]
    for k, v in {
        "ai21.jamba-1-5-large-v1:0": "AI21",
        "ai21.jamba-1-5-mini-v1:0": "AI21",
        "amazon.nova-lite-v1:0": _NOVA,
        "amazon.nova-micro-v1:0": _NOVA,
        "amazon.nova-pro-v1:0": _NOVA,
        "amazon.nova-premier-v1:0": _NOVA,
        "amazon.nova-2-lite-v1:0": "AMAZON_NOVA_2",
        "anthropic.claude-3-5-sonnet-20240620-v1:0": _CLAUDE,
        "anthropic.claude-3-5-sonnet-20241022-v2:0": _CLAUDE,
        "anthropic.claude-3-7-sonnet-20250219-v1:0": _CLAUDE,
        "anthropic.claude-3-haiku-20240307-v1:0": _CLAUDE,
        "anthropic.claude-fable-5": _CLAUDE,
        "anthropic.claude-fable-5-1": _CLAUDE,
        "anthropic.claude-mythos-5-1": _CLAUDE,
        "anthropic.claude-opus-4-1-20250805-v1:0": _CLAUDE,
        "anthropic.claude-opus-4-6-v1": _CLAUDE,
        "anthropic.claude-opus-4-7": _CLAUDE,
        "anthropic.claude-opus-4-8": _CLAUDE,
        "anthropic.claude-opus-5": _CLAUDE,
        "anthropic.claude-opus-5-5": _CLAUDE,
        "anthropic.claude-opus-4-5-20251101-v1:0": _CLAUDE,
        "anthropic.claude-sonnet-5-5": _CLAUDE,
        "anthropic.claude-sonnet-5": _CLAUDE,
        "anthropic.claude-sonnet-4-6": _CLAUDE,
        "anthropic.claude-sonnet-4-5-20250929-v1:0": _CLAUDE,
        "anthropic.claude-haiku-4-5-20251001-v1:0": _CLAUDE,
        "anthropic.claude-sonnet-4-20250514-v1:0": _CLAUDE,
        "cohere.command-r-plus-v1:0": "COHERE_COMMAND_R",
        "cohere.command-r-v1:0": "COHERE_COMMAND_R",
        "deepseek.r1-v1:0": "DEEPSEEK",
        "deepseek.v3-v1:0": "DEEPSEEK_CHAT",
        "deepseek.v3.2": "DEEPSEEK_CHAT",
        "meta.llama3-1-405b-instruct-v1:0": "LLAMA3_1",
        "meta.llama3-1-70b-instruct-v1:0": "LLAMA3_1",
        "meta.llama3-1-8b-instruct-v1:0": "LLAMA3_1",
        "meta.llama3-2-3b-instruct-v1:0": "LLAMA3_2",
        "meta.llama3-3-70b-instruct-v1:0": "LLAMA3_3",
        "meta.llama3-70b-instruct-v1:0": "LLAMA3",
        "meta.llama3-8b-instruct-v1:0": "LLAMA3",
        "meta.llama4-scout-17b-instruct-v1:0": "LLAMA4",
        "meta.llama4-maverick-17b-instruct-v1:0": "LLAMA4",
        "mistral.mistral-7b-instruct-v0:2": "MISTRAL",
        "mistral.devstral-2-123b": "MISTRAL_CHAT",
        "mistral.magistral-small-2509": "MISTRAL_CHAT",
        "mistral.ministral-3-14b-instruct": "MISTRAL_CHAT",
        "mistral.ministral-3-8b-instruct": "MISTRAL_CHAT",
        "mistral.ministral-3-3b-instruct": "MISTRAL_CHAT",
        "mistral.mistral-large-2402-v1:0": "MISTRAL",
        "mistral.mistral-large-2407-v1:0": "MISTRAL_LARGE_2407",
        "mistral.mistral-large-3-675b-instruct": "MISTRAL_CHAT",
        "mistral.mistral-small-2402-v1:0": "MISTRAL",
        "mistral.mixtral-8x7b-instruct-v0:1": "MISTRAL",
        "mistral.pixtral-large-2502-v1:0": "MISTRAL_CHAT",
        "mistral.voxtral-mini-3b-2507": "MISTRAL_CHAT",
        "mistral.voxtral-small-24b-2507": "MISTRAL_CHAT",
        # APAC
        "apac.amazon.nova-lite-v1:0": _NOVA,
        "apac.amazon.nova-micro-v1:0": _NOVA,
        "apac.amazon.nova-pro-v1:0": _NOVA,
        "apac.amazon.nova-premier-v1:0": _NOVA,
        "apac.amazon.nova-2-lite-v1:0": "AMAZON_NOVA_2",
        "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": _CLAUDE,
        "apac.anthropic.claude-3-haiku-20240307-v1:0": _CLAUDE,
        "apac.anthropic.claude-opus-4-1-20250805-v1:0": _CLAUDE,
        "apac.anthropic.claude-opus-4-6-v1": _CLAUDE,
        "apac.anthropic.claude-opus-4-5-20251101-v1:0": _CLAUDE,
        "apac.anthropic.claude-sonnet-4-6": _CLAUDE,
        "apac.anthropic.claude-sonnet-4-5-20250929-v1:0": _CLAUDE,
        "apac.anthropic.claude-haiku-4-5-20251001-v1:0": _CLAUDE,
        "apac.anthropic.claude-sonnet-4-20250514-v1:0": _CLAUDE,
        "apac.meta.llama4-scout-17b-instruct-v1:0": "LLAMA4",
        "apac.meta.llama4-maverick-17b-instruct-v1:0": "LLAMA4",
        # AU
        "au.anthropic.claude-opus-4-7": _CLAUDE,
        "au.anthropic.claude-opus-4-8": _CLAUDE,
        "au.anthropic.claude-opus-5": _CLAUDE,
        "au.anthropic.claude-sonnet-5": _CLAUDE,
        # EU
        "eu.amazon.nova-lite-v1:0": _NOVA,
        "eu.amazon.nova-micro-v1:0": _NOVA,
        "eu.amazon.nova-pro-v1:0": _NOVA,
        "eu.amazon.nova-premier-v1:0": _NOVA,
        "eu.amazon.nova-2-lite-v1:0": "AMAZON_NOVA_2",
        "eu.anthropic.claude-3-5-sonnet-20240620-v1:0": _CLAUDE,
        "eu.anthropic.claude-3-7-sonnet-20250219-v1:0": _CLAUDE,
        "eu.anthropic.claude-3-haiku-20240307-v1:0": _CLAUDE,
        "eu.anthropic.claude-opus-4-1-20250805-v1:0": _CLAUDE,
        "eu.anthropic.claude-opus-4-6-v1": _CLAUDE,
        "eu.anthropic.claude-opus-4-7": _CLAUDE,
        "eu.anthropic.claude-opus-4-8": _CLAUDE,
        "eu.anthropic.claude-opus-5": _CLAUDE,
        "eu.anthropic.claude-opus-5-5": _CLAUDE,
        "eu.anthropic.claude-opus-4-5-20251101-v1:0": _CLAUDE,
        "eu.anthropic.claude-sonnet-5": _CLAUDE,
        "eu.anthropic.claude-sonnet-4-6": _CLAUDE,
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": _CLAUDE,
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0": _CLAUDE,
        "eu.anthropic.claude-sonnet-4-20250514-v1:0": _CLAUDE,
        "eu.meta.llama3-2-1b-instruct-v1:0": "LLAMA3_2",
        "eu.meta.llama3-2-3b-instruct-v1:0": "LLAMA3_2",
        "eu.meta.llama4-scout-17b-instruct-v1:0": "LLAMA4",
        "eu.meta.llama4-maverick-17b-instruct-v1:0": "LLAMA4",
        "eu.mistral.pixtral-large-2502-v1:0": "MISTRAL_CHAT",
        # GovCloud
        "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0": _CLAUDE,
        "us-gov.anthropic.claude-3-haiku-20240307-v1:0": _CLAUDE,
        # US
        "us.amazon.nova-lite-v1:0": _NOVA,
        "us.amazon.nova-micro-v1:0": _NOVA,
        "us.amazon.nova-pro-v1:0": _NOVA,
        "us.amazon.nova-premier-v1:0": _NOVA,
        "us.amazon.nova-2-lite-v1:0": "AMAZON_NOVA_2",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0": _CLAUDE,
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": _CLAUDE,
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0": _CLAUDE,
        "us.anthropic.claude-3-haiku-20240307-v1:0": _CLAUDE,
        "us.anthropic.claude-fable-5": _CLAUDE,
        "us.anthropic.claude-fable-5-1": _CLAUDE,
        "us.anthropic.claude-mythos-5-1": _CLAUDE,
        "us.anthropic.claude-opus-4-1-20250805-v1:0": _CLAUDE,
        "us.anthropic.claude-opus-4-6-v1": _CLAUDE,
        "us.anthropic.claude-opus-4-7": _CLAUDE,
        "us.anthropic.claude-opus-4-8": _CLAUDE,
        "us.anthropic.claude-opus-5": _CLAUDE,
        "us.anthropic.claude-opus-5-5": _CLAUDE,
        "us.anthropic.claude-opus-4-5-20251101-v1:0": _CLAUDE,
        "us.anthropic.claude-sonnet-5": _CLAUDE,
        "us.anthropic.claude-sonnet-4-6": _CLAUDE,
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": _CLAUDE,
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": _CLAUDE,
        "us.anthropic.claude-sonnet-4-20250514-v1:0": _CLAUDE,
        "us.deepseek.r1-v1:0": "DEEPSEEK",
        "us.meta.llama3-1-405b-instruct-v1:0": "LLAMA3_1",
        "us.meta.llama3-1-70b-instruct-v1:0": "LLAMA3_1",
        "us.meta.llama3-1-8b-instruct-v1:0": "LLAMA3_1",
        "us.meta.llama3-2-11b-instruct-v1:0": "LLAMA3_2",
        "us.meta.llama3-2-1b-instruct-v1:0": "LLAMA3_2",
        "us.meta.llama3-2-3b-instruct-v1:0": "LLAMA3_2",
        "us.meta.llama3-2-90b-instruct-v1:0": "LLAMA3_2",
        "us.meta.llama3-3-70b-instruct-v1:0": "LLAMA3_3",
        "us.meta.llama4-scout-17b-instruct-v1:0": "LLAMA4",
        "us.meta.llama4-maverick-17b-instruct-v1:0": "LLAMA4",
        "us.mistral.pixtral-large-2502-v1:0": "MISTRAL_CHAT",
        # OpenAI open-weight (gpt-oss) via InvokeModel
        "openai.gpt-oss-120b-1:0": "OPENAI",
        "openai.gpt-oss-20b-1:0": "OPENAI",
        "openai.gpt-oss-safeguard-120b": "OPENAI",
        "openai.gpt-oss-safeguard-20b": "OPENAI",
        # xAI Grok 4.6 profiles (served natively)
        "us.xai.grok-4.6": "OPENAI_COMPAT",
        "global.xai.grok-4.6": "OPENAI_COMPAT",
        # Qwen
        "qwen.qwen3-coder-next": "QWEN",
        "qwen.qwen3-coder-480b-a35b-v1:0": "QWEN",
        "qwen.qwen3-coder-30b-a3b-v1:0": "QWEN",
        "qwen.qwen3-next-80b-a3b": "QWEN",
        "qwen.qwen3-vl-235b-a22b": "QWEN",
        "qwen.qwen3-235b-a22b-2507-v1:0": "QWEN",
        "qwen.qwen3-32b-v1:0": "QWEN",
        # OpenAI chat-compatible families
        "zai.glm-5": "OPENAI_COMPAT",
        "zai.glm-4.7": "OPENAI_COMPAT",
        "zai.glm-4.7-flash": "OPENAI_COMPAT",
        "minimax.minimax-m2": "OPENAI_COMPAT",
        "minimax.minimax-m2.1": "OPENAI_COMPAT",
        "minimax.minimax-m2.5": "OPENAI_COMPAT",
        "moonshotai.kimi-k2.5": "OPENAI_COMPAT",
        "moonshot.kimi-k2-thinking": "OPENAI_COMPAT",
        "nvidia.nemotron-nano-9b-v2": "OPENAI_COMPAT",
        "nvidia.nemotron-nano-12b-v2": "OPENAI_COMPAT",
        "nvidia.nemotron-nano-3-30b": "OPENAI_COMPAT",
        "nvidia.nemotron-super-3-120b": "OPENAI_COMPAT",
        "us-gov.nvidia.nemotron-nano-9b-v2": "OPENAI_COMPAT",
        "us-gov.nvidia.nemotron-nano-12b-v2": "OPENAI_COMPAT",
        "us-gov.nvidia.nemotron-nano-3-30b": "OPENAI_COMPAT",
        "us-gov.nvidia.nemotron-super-3-120b": "OPENAI_COMPAT",
        "google.gemma-3-4b-it": "OPENAI_COMPAT",
        "google.gemma-3-12b-it": "OPENAI_COMPAT",
        "google.gemma-3-27b-it": "OPENAI_COMPAT",
        "us.writer.palmyra-x5-v1:0": "OPENAI_COMPAT",
        "us.writer.palmyra-x4-v1:0": "OPENAI_COMPAT",
        "writer.palmyra-vision-7b": "OPENAI_COMPAT",
        # Global / Japan cross-region profiles
        "global.amazon.nova-2-lite-v1:0": "AMAZON_NOVA_2",
        "global.anthropic.claude-opus-4-7": _CLAUDE,
        "jp.anthropic.claude-opus-4-7": _CLAUDE,
        "global.anthropic.claude-opus-4-8": _CLAUDE,
        "jp.anthropic.claude-opus-4-8": _CLAUDE,
        "global.anthropic.claude-opus-5": _CLAUDE,
        "global.anthropic.claude-opus-5-5": _CLAUDE,
        "jp.anthropic.claude-opus-5-5": _CLAUDE,
        "au.anthropic.claude-opus-5-5": _CLAUDE,
        "global.anthropic.claude-fable-5": _CLAUDE,
        "global.anthropic.claude-fable-5-1": _CLAUDE,
        "global.anthropic.claude-mythos-5-1": _CLAUDE,
        "global.anthropic.claude-sonnet-5": _CLAUDE,
        "global.anthropic.claude-sonnet-5-5": _CLAUDE,
    }.items()
}


def _canonical_messages_only_model(model_name: str) -> str | None:
    m = re.match(r"^(?:[^.]+\.)?(anthropic\.claude-mythos-(?:5|preview))$", model_name)
    if m:
        return m.group(1)
    if requires_bedrock_anthropic_messages_model(model_name):
        return model_name
    return None


# Model IDs AWS no longer serves (bare IDs; inference-profile prefixes are stripped).
RETIRED_BEDROCK_MODELS = {
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-instant-v1",
    "anthropic.claude-v1",
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "cohere.command-text-v14",
    "cohere.command-light-text-v14",
    "meta.llama2-13b-chat-v1",
    "meta.llama2-70b-chat-v1",
}


def assert_bedrock_model_is_available(model_name: str) -> None:
    """Reject withdrawn models before InvokeModel or Converse requests."""
    model_id = model_name
    if model_name.startswith("arn:"):
        m = re.search(r":(?:inference-profile|foundation-model)/([^/]+)$", model_name)
        model_id = m.group(1) if m else model_name
    if INFERENCE_PROFILE_PREFIX.sub("", model_id, count=1) in RETIRED_BEDROCK_MODELS:
        raise ValueError(f"Unknown Amazon Bedrock model: {model_name}")


_INFERENCE_TYPES = {
    "claude": "CLAUDE_MESSAGES",
    "nova": "AMAZON_NOVA",
    "llama": "LLAMA4",
    "llama2": "LLAMA2",
    "llama3": "LLAMA3",
    "llama3.1": "LLAMA3_1",
    "llama3_1": "LLAMA3_1",
    "llama3.2": "LLAMA3_2",
    "llama3_2": "LLAMA3_2",
    "llama3.3": "LLAMA3_3",
    "llama3_3": "LLAMA3_3",
    "llama4": "LLAMA4",
    "mistral": "MISTRAL",
    "cohere": "COHERE_COMMAND_R",
    "ai21": "AI21",
    "titan": "TITAN_TEXT",
    "deepseek": "DEEPSEEK",
    "openai": "OPENAI",
    "qwen": "QWEN",
    "nova2": "AMAZON_NOVA_2",
    "zai": "OPENAI_COMPAT",
    "minimax": "OPENAI_COMPAT",
    "moonshot": "OPENAI_COMPAT",
    "nvidia": "OPENAI_COMPAT",
    "writer": "OPENAI_COMPAT",
    "gemma": "OPENAI_COMPAT",
}


def get_handler_for_model(
    model_name: str, config: dict[str, Any] | None = None
) -> BedrockModelHandler:
    assert_bedrock_model_is_available(model_name)
    messages_only = _canonical_messages_only_model(model_name)
    if messages_only:
        raise ValueError(
            f'Amazon Bedrock model "{model_name}" uses Bedrock\'s Anthropic Messages API, not '
            f'InvokeModel. Load it as "bedrock:{messages_only}" instead.'
        )
    if "arn:" in model_name and "inference-profile" in model_name:
        inference_type = (config or {}).get("inferenceModelType")
        if not inference_type:
            raise ValueError(
                "Inference profile requires inferenceModelType to be specified in config. "
                "Options: claude, nova, nova2, llama (defaults to v4), llama2, llama3, llama3.1, "
                "llama3.2, llama3.3, llama4, mistral, cohere, ai21, titan, deepseek, openai, qwen, "
                "zai, minimax, moonshot, nvidia, writer, gemma"
            )
        handler = _INFERENCE_TYPES.get(inference_type)
        if handler is None:
            raise ValueError(f"Unknown inference model type: {js_string(inference_type)}")
        return BEDROCK_MODEL[handler]

    found = AWS_BEDROCK_MODELS.get(model_name)
    if found:
        return found
    for test, handler in (
        (lambda m: m.startswith("ai21."), "AI21"),
        (lambda m: "amazon.nova-2" in m, "AMAZON_NOVA_2"),
        (lambda m: "amazon.nova" in m, "AMAZON_NOVA"),
        (lambda m: "anthropic.claude" in m, "CLAUDE_MESSAGES"),
        (lambda m: m.startswith("meta.llama2"), "LLAMA2"),
        (lambda m: "meta.llama3-1" in m, "LLAMA3_1"),
        (lambda m: "meta.llama3-2" in m, "LLAMA3_2"),
        (lambda m: "meta.llama3-3" in m, "LLAMA3_3"),
        (lambda m: "meta.llama4" in m, "LLAMA4"),
        (lambda m: "meta.llama3" in m, "LLAMA3"),
        (lambda m: m.startswith("cohere.command-r"), "COHERE_COMMAND_R"),
        (lambda m: m.startswith("cohere.command"), "COHERE_COMMAND"),
        (lambda m: m.startswith("mistral."), "MISTRAL"),
        (lambda m: m.startswith("deepseek."), "DEEPSEEK"),
        (lambda m: m.startswith("qwen."), "QWEN"),
    ):
        if test(model_name):
            return BEDROCK_MODEL[handler]
    if "openai.gpt-oss" in model_name:
        if re.search(r"-\d+:\d+$", model_name):
            return BEDROCK_MODEL["OPENAI"]
        raise ValueError(
            f'Amazon Bedrock model "{model_name}" looks like the Bedrock Mantle id for an '
            "open-weight gpt-oss model. promptfoo's bedrock: provider calls the InvokeModel "
            "(bedrock-runtime) API, which uses the versioned runtime id — e.g. "
            '"bedrock:openai.gpt-oss-120b-1:0". See '
            "https://www.promptfoo.dev/docs/providers/aws-bedrock/#openai-models"
        )
    if "openai." in model_name:
        bare = re.sub(r"^[a-z]+\.(?=openai\.)", "", model_name, count=1)
        if re.match(r"^[a-z]+\.openai\.gpt-5\.6-(?:sol|terra|luna)$", model_name):
            raise ValueError(
                f'OpenAI model "{model_name}" is not served by Bedrock\'s InvokeModel API. For a '
                f'supported Runtime inference profile, use "bedrock:converse:{model_name}" with '
                f'AWS credentials, or "bedrock:{bare}" with AWS_BEARER_TOKEN_BEDROCK for Mantle '
                "Responses."
            )
        raise ValueError(
            f'OpenAI model "{model_name}" is not served by Bedrock\'s InvokeModel API. Frontier '
            "models (gpt-5.x) use the OpenAI-compatible Responses API — use "
            f'"bedrock:{bare}" and set AWS_BEARER_TOKEN_BEDROCK. See '
            "https://www.promptfoo.dev/docs/providers/aws-bedrock/#openai-models"
        )
    if "xai." in model_name or "grok" in model_name:
        raise ValueError(
            f'xAI model "{model_name}" is not served by Bedrock\'s InvokeModel API under that id. '
            "Grok 4.6 supports Runtime Converse through an inference profile — use "
            '"bedrock:converse:us.xai.grok-4.6" or "bedrock:converse:global.xai.grok-4.6" with '
            "ordinary AWS credentials. Other Grok models run on the OpenAI-compatible Responses "
            'API (mantle endpoint) — use the bare id such as "bedrock:xai.grok-4.3" and set '
            "AWS_BEARER_TOKEN_BEDROCK. See "
            "https://www.promptfoo.dev/docs/providers/aws-bedrock/#xai-grok-models"
        )
    raise ValueError(f"Unknown Amazon Bedrock model: {model_name}")


def _read_body(body: Any) -> str:
    raw = body.read() if hasattr(body, "read") else body
    return raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)


def _response_json_for_error(response: Any) -> str:
    if response is None:
        return "undefined"
    if isinstance(response, dict):
        return json_stringify({k: v for k, v in response.items() if k != "body"})
    return json_stringify(response)


class AwsBedrockCompletionProvider(AwsBedrockGenericProvider):
    AWS_BEDROCK_COMPLETION_MODELS = list(AWS_BEDROCK_MODELS)

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        env_stop = envars.get_env_string("AWS_BEDROCK_STOP")
        try:
            stop = json.loads(env_stop) if env_stop else []
        except ValueError as err:
            raise ValueError(
                f"BEDROCK_STOP is not a valid JSON string: SyntaxError: {err}"
            ) from err

        context = context or {}
        merged = {**self.config, **((context.get("prompt") or {}).get("config") or {})}
        model = get_handler_for_model(self.model_name, merged)
        params = await model.params(merged, prompt, stop, self.model_name, context.get("vars"))

        cache = cache_mod.get_cache()
        region = self.get_region()
        cache_key = (
            f"bedrock:{self.model_name}:{region}:"
            f"{create_bedrock_cache_key_hash(self.config, params, region)}"
        )
        if cache_mod.is_cache_enabled():
            cached = await cache.get(cache_key)
            if cached:
                return {
                    "output": model.output(merged, json.loads(cached)),
                    "tokenUsage": create_empty_token_usage(),
                    "cached": True,
                }

        request: dict[str, Any] = {"modelId": self.model_name}
        if self.config.get("guardrailIdentifier"):
            request["guardrailIdentifier"] = js_string(self.config["guardrailIdentifier"])
        if self.config.get("guardrailVersion"):
            request["guardrailVersion"] = js_string(self.config["guardrailVersion"])
        if self.config.get("trace"):
            request["trace"] = self.config["trace"]
        request.update(
            accept="application/json",
            contentType="application/json",
            body=json_stringify(params),
        )
        response: Any = None
        try:
            client = self.get_bedrock_instance()
            response = await asyncio.to_thread(client.invoke_model, **request)
            body_text = _read_body(response["body"])
        except Exception as err:
            return {"error": f"Bedrock API invoke model error: {aws_error_string(err)}"}

        if cache_mod.is_cache_enabled():
            try:
                await cache.set(cache_key, body_text)
            except Exception as err:
                logger.error("Failed to cache response: %s", err)
        try:
            output = json.loads(body_text)
            if model.token_usage:
                token_usage = model.token_usage(output, prompt)
            else:
                token_usage = _fallback_token_usage(output)
            if not token_usage.get("numRequests"):
                token_usage["numRequests"] = 1
            usage = (output.get("usage") if isinstance(output, dict) else None) or {}
            details = token_usage.get("completionDetails") or {}
            cache_read = details.get("cacheReadInputTokens")
            if cache_read is None:
                cache_read = coerce_str_to_num(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    if is_bedrock_grok46_profile(self.model_name)
                    else usage.get("cache_read_input_tokens")
                )
            if model is BEDROCK_MODEL["CLAUDE_MESSAGES"]:
                billable_prompt = coerce_str_to_num(
                    usage.get("input_tokens")
                    if usage.get("input_tokens") is not None
                    else usage.get("prompt_tokens")
                )
            elif (
                is_bedrock_grok46_profile(self.model_name) and token_usage.get("prompt") is not None
            ):
                billable_prompt = max(token_usage["prompt"] - (cache_read or 0), 0)
            else:
                billable_prompt = token_usage.get("prompt")
            cache_creation = details.get("cacheCreationInputTokens")
            if cache_creation is None:
                cache_creation = coerce_str_to_num(usage.get("cache_creation_input_tokens"))
            cost = calculate_bedrock_invoke_model_cost(
                self.model_name,
                billable_prompt,
                token_usage.get("completion"),
                cache_read,
                cache_creation,
                region,
                coerce_str_to_num(
                    (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens")
                ),
            )
            result: dict[str, Any] = {
                "output": model.output(merged, output),
                "tokenUsage": {k: v for k, v in token_usage.items() if v is not None},
            }
            if cost is not None:
                result["cost"] = cost
            guardrail_action = output.get("amazon-bedrock-guardrailAction")
            if is_truthy(guardrail_action):
                result["guardrails"] = {"flagged": guardrail_action == "INTERVENED"}
            return result
        except Exception as err:
            logger.error("Bedrock API response error: %s", err)
            return {
                "error": f"API response error: {type(err).__name__}: {err}: "
                f"{_response_json_for_error(response)}"
            }


def _fallback_token_usage(output: dict[str, Any]) -> dict[str, Any]:
    usage = output.get("usage") or {}

    def first(*values: Any) -> Any:
        return next((v for v in values if v is not None), None)

    p = coerce_str_to_num(
        first(
            usage.get("inputTokens"),
            usage.get("input_tokens"),
            usage.get("prompt_tokens"),
            output.get("prompt_tokens"),
            output.get("prompt_token_count"),
        )
    )
    c = coerce_str_to_num(
        first(
            usage.get("outputTokens"),
            usage.get("output_tokens"),
            usage.get("completion_tokens"),
            output.get("completion_tokens"),
            output.get("generation_token_count"),
        )
    )
    total = coerce_str_to_num(
        first(usage.get("totalTokens"), usage.get("total_tokens"), output.get("total_tokens"))
    )
    if total is None:
        total = p + c if p is not None and c is not None else None
    return _usage(p, c, total)


class AwsBedrockEmbeddingProvider(AwsBedrockGenericProvider):
    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("callApi is not implemented for embedding provider")

    async def call_embedding_api(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = (
            {"texts": [text], "input_type": self.config.get("input_type") or "search_document"}
            if "cohere.embed" in self.model_name
            else {"inputText": text}
        )
        try:
            client = self.get_bedrock_instance()
            response = await asyncio.to_thread(
                client.invoke_model,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json",
                body=json_stringify(params),
            )
            body_text = _read_body(response["body"])
        except Exception as err:
            return {"error": f"API call error: {aws_error_string(err)}"}
        try:
            data = json.loads(body_text)
            embeddings = data.get("embeddings")
            if isinstance(embeddings, dict) and embeddings.get("float") is not None:
                embeddings = embeddings["float"]
            embedding = data.get("embedding")
            if embedding is None and isinstance(embeddings, list) and len(embeddings) == 1:
                embedding = embeddings[0]
            if (
                not isinstance(embedding, list)
                or not embedding
                or not all(
                    isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
                    for v in embedding
                )
            ):
                raise ValueError("No valid embedding found in AWS Bedrock API response")
            return {"embedding": embedding}
        except Exception as err:
            return {"error": f"API response error: Error: {err}: {json_stringify(body_text)}"}
