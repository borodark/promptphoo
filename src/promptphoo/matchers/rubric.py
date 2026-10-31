"""Rubric prompt loading/rendering and the JSON grading call. Ports src/matchers/rubric.ts.

Includes the multimodal path: image outputs (and OpenAI-format audio) of the evaluated response
are attached to the grader prompt in the provider's content-part shape.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

from promptphoo import cli_state, envars
from promptphoo.assertions.refusal import js_trim
from promptphoo.files import is_javascript_file, maybe_load_from_external_file, parse_file_url
from promptphoo.jsonutil import extract_json_objects, js_json_parse
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify, number_to_string
from promptphoo.matchers.providers import call_provider_with_context, get_and_check_provider
from promptphoo.matchers.shared import grader_fail, normalize_matcher_token_usage
from promptphoo.sanitizer import safe_json_stringify
from promptphoo.templating import get_engine
from promptphoo.yamlload import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_GRADING_MAX_IMAGES = 4
DEFAULT_GRADING_IMAGE_MAX_BYTES = 20 * 1024 * 1024
GRADING_AUDIO_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_GRADING_IMAGE_MAX_TOTAL_BYTES = 20 * 1024 * 1024
DATA_URI_METADATA_MAX_CHARS = 256
DEFAULT_GRADING_IMAGE_MAX_RAW_CHARS = (
    math.ceil(DEFAULT_GRADING_IMAGE_MAX_BYTES / 3) * 4 + DATA_URI_METADATA_MAX_CHARS
)
DEFAULT_GRADING_IMAGE_MAX_TOTAL_RAW_CHARS = (
    math.ceil(DEFAULT_GRADING_IMAGE_MAX_TOTAL_BYTES / 3) * 4
    + DEFAULT_GRADING_MAX_IMAGES * DATA_URI_METADATA_MAX_CHARS
)
MULTIMODAL_GRADING_INSTRUCTION = (
    "The evaluated output includes the attached image(s). Treat the attached image(s) as primary "
    "evidence in <Output>. Inspect the visual content directly, and do not infer visual traits, "
    "demographics, safety issues, or rubric failures from the user prompt or from any "
    "base64/data URI text."
)
AUDIO_GRADING_INSTRUCTION = (
    "The evaluated output includes the attached audio. Listen to it as primary evidence for the "
    "rubric. Use the transcript only as supporting context; do not infer vocal delivery or sound "
    "quality from the transcript alone."
)
BLOB_HASH_REGEX = re.compile(r"^[a-f0-9]{64}\Z", re.IGNORECASE)
BLOB_URI_REGEX = re.compile(r"promptfoo://blob/([a-f0-9]{64})", re.IGNORECASE)
RESPONSES_PROVIDER_CLASS_NAMES = frozenset(
    {
        "AzureResponsesProvider",
        "BedrockOpenAiResponsesProvider",
        "GroqResponsesProvider",
        "OpenAiResponsesProvider",
        "OpenClawResponsesProvider",
        "XAIResponsesProvider",
    }
)


class LlmRubricProviderError(Exception):
    """The grading provider failed while `throwOnError` was requested."""


def _js_split2(text: str, sep: str) -> list[str]:
    """`text.split(sep, 2)`: JS drops everything after the second piece."""
    return text.split(sep)[:2]


def js_number(value: Any) -> float:
    """JS Number(value) for JSON-shaped values."""
    if value is None:
        return 0
    if value is UNDEFINED:
        return math.nan
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float):
        return value
    if isinstance(value, list):
        if not value:
            return 0
        return js_number(js_string(value[0])) if len(value) == 1 else math.nan
    if isinstance(value, dict):
        return math.nan
    text = js_trim(js_string(value))
    if text == "":
        return 0
    if re.fullmatch(r"0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+", text):
        return int(text, 0)
    if re.fullmatch(r"[+-]?Infinity", text):
        return -math.inf if text.startswith("-") else math.inf
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text):
        n = float(text)
        return int(n) if n.is_integer() and abs(n) < 2**53 and "." not in text else n
    return math.nan


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


# --- rubric prompt loading and rendering ------------------------------------------------------


async def load_rubric_prompt(rubric_prompt: Any, default_prompt: str) -> str:
    if not is_truthy(rubric_prompt):
        return default_prompt
    if isinstance(rubric_prompt, dict | list) and len(rubric_prompt) == 0:
        return default_prompt

    if isinstance(rubric_prompt, str) and rubric_prompt.startswith("file://"):
        base_path = cli_state.base_path() or ""
        rendered_file_path = get_engine().render_string(rubric_prompt, {})
        file_path, function_name = parse_file_url(rendered_file_path)
        resolved_path = os.path.abspath(os.path.join(base_path, file_path))
        if is_javascript_file(file_path):
            raise ValueError(
                f"JavaScript rubric prompt files are not supported by promptphoo: {rubric_prompt}. "
                "Use a .txt, .json or .yaml file instead."
            )
        try:
            with open(resolved_path, encoding="utf-8") as f:
                rubric_prompt = f.read()
        except FileNotFoundError:
            raise ValueError(f"File does not exist: {resolved_path}") from None
        del function_name
    else:
        rubric_prompt = maybe_load_from_external_file(rubric_prompt)

    if isinstance(rubric_prompt, dict | list):
        rubric_prompt = json_stringify(rubric_prompt)

    if not isinstance(rubric_prompt, str):
        raise ValueError("rubricPrompt must be a string")
    return rubric_prompt


def _process_context_for_templating(
    context: dict[str, Any], enable_object_access: bool
) -> dict[str, Any]:
    if enable_object_access:
        return context
    out: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, list):
            out[key] = [
                json_stringify(item) if isinstance(item, dict | list) else item for item in value
            ]
        elif isinstance(value, dict):
            out[key] = json_stringify(value)
        else:
            out[key] = value
    return out


def _render_json_strings(value: Any, engine: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return engine.render_string(value, context)
    if isinstance(value, list):
        return [_render_json_strings(v, engine, context) for v in value]
    if isinstance(value, dict):
        return {k: _render_json_strings(v, engine, context) for k, v in value.items()}
    return value


async def render_llm_rubric_prompt(rubric_prompt: str, context: dict[str, Any]) -> str:
    enable_object_access = envars.get_env_bool("PROMPTFOO_DISABLE_OBJECT_STRINGIFY", False)
    processed = _process_context_for_templating(context, enable_object_access)
    engine = get_engine(is_grader=True)
    try:
        parsed = js_json_parse(rubric_prompt)
        return json_stringify(_render_json_strings(parsed, engine, processed))
    except Exception as err:
        logger.debug("[Rubric] Rubric prompt is not valid JSON, using Nunjucks rendering: %s", err)
    return engine.render_string(rubric_prompt, processed)


# --- multimodal grading -----------------------------------------------------------------------


def _is_chat_message_array(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(m, dict) and isinstance(m.get("role"), str) for m in value
    )


def _is_valid_base64_payload(data: str) -> bool:
    if not data or len(data) % 4 == 1 or not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", data):
        return False
    idx = data.find("=")
    if idx == -1:
        return True
    padding = data[idx:]
    return len(padding) <= 2 and bool(re.fullmatch(r"=+", padding))


def _get_base64_decoded_bytes(base64_data: str) -> int:
    padding = 2 if base64_data.endswith("==") else 1 if base64_data.endswith("=") else 0
    return (len(base64_data) * 3) // 4 - padding


def _normalize_base64_image_data(data: str, mime_type: str | None = None) -> dict[str, Any]:
    trimmed = js_trim(data)
    if not trimmed:
        raise ValueError("Image output data must contain non-empty base64 image data.")
    normalized_mime = mime_type or "image/png"
    payload = trimmed
    if trimmed.startswith("data:"):
        parts = _js_split2(trimmed, ",")
        metadata = parts[0]
        raw_payload = parts[1] if len(parts) > 1 else ""
        if not raw_payload or ";base64" not in metadata.lower():
            raise ValueError(
                "Only base64-encoded data URI image outputs are supported for multimodal grading."
            )
        if not metadata.lower().startswith("data:image/"):
            raise ValueError("Only image data URI outputs are supported for multimodal grading.")
        normalized_mime = js_trim(metadata[len("data:") :].split(";")[0])
        payload = raw_payload
    base64_data = re.sub(r"\s", "", payload)
    if not _is_valid_base64_payload(base64_data):
        raise ValueError("Image output data must contain valid, non-empty base64 image data.")
    decoded = _get_base64_decoded_bytes(base64_data)
    if decoded <= 0:
        raise ValueError("Image output data must contain non-empty base64 image data.")
    return {
        "dataUri": f"data:{normalized_mime};base64,{base64_data}",
        "base64Data": base64_data,
        "mimeType": normalized_mime,
        "decodedBytes": decoded,
    }


def _has_blob_ref_image_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(BLOB_URI_REGEX.search(value))
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("hash"), str) and BLOB_HASH_REGEX.match(value["hash"]):
        return True
    if isinstance(value.get("uri"), str):
        return _has_blob_ref_image_value(value["uri"])
    return False


def _image_output_to_image_url(image: dict[str, Any]) -> dict[str, Any] | None:
    if image.get("blobRef") or _has_blob_ref_image_value(image.get("data")):
        raise ValueError(
            "Blob-backed image outputs are not supported for multimodal grading yet. Configure "
            "the image provider to return base64 or data URI image output."
        )
    if image.get("data"):
        data = js_trim(image["data"])
        if re.match(r"^https?://", data, re.IGNORECASE):
            raise ValueError(
                "Remote image URLs are not supported for multimodal grading. Provide local image "
                "output as a data URI or raw base64 string instead."
            )
        normalized = _normalize_base64_image_data(image["data"], image.get("mimeType"))
        return {
            "output": {"data": normalized["dataUri"], "mimeType": normalized["mimeType"]},
            "dataUri": normalized["dataUri"],
            "base64Data": normalized["base64Data"],
            "mimeType": normalized["mimeType"],
            "rawChars": len(image["data"]),
        }
    return None


def materialize_image_outputs_for_grading(images: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not images:
        return {"imageOutputs": [], "imageData": []}
    max_images = envars.get_env_int("PROMPTFOO_GRADING_MAX_IMAGES", DEFAULT_GRADING_MAX_IMAGES)
    if len(images) > max_images:
        raise ValueError(
            f"Too many images for multimodal grading: received {len(images)}, maximum is "
            f"{max_images}."
        )
    max_bytes = envars.get_env_int(
        "PROMPTFOO_GRADING_IMAGE_MAX_BYTES", DEFAULT_GRADING_IMAGE_MAX_BYTES
    )
    max_total_bytes = envars.get_env_int(
        "PROMPTFOO_GRADING_IMAGE_MAX_TOTAL_BYTES", DEFAULT_GRADING_IMAGE_MAX_TOTAL_BYTES
    )
    max_raw = envars.get_env_int(
        "PROMPTFOO_GRADING_IMAGE_MAX_RAW_CHARS", DEFAULT_GRADING_IMAGE_MAX_RAW_CHARS
    )
    max_total_raw = envars.get_env_int(
        "PROMPTFOO_GRADING_IMAGE_MAX_TOTAL_RAW_CHARS", DEFAULT_GRADING_IMAGE_MAX_TOTAL_RAW_CHARS
    )
    materialized = [m for m in (_image_output_to_image_url(i) for i in images) if m]
    total_bytes = 0
    total_raw = 0
    for image in materialized:
        decoded = _get_base64_decoded_bytes(image["base64Data"])
        if decoded > max_bytes:
            raise ValueError(
                f"Image output exceeds multimodal grading size limit: {decoded} bytes, maximum "
                f"is {max_bytes}."
            )
        raw_chars = max(image["rawChars"], len(image["dataUri"]))
        if raw_chars > max_raw:
            raise ValueError(
                f"Image output raw data exceeds multimodal grading size limit: {raw_chars} "
                f"characters, maximum is {max_raw}."
            )
        total_bytes += decoded
        if total_bytes > max_total_bytes:
            raise ValueError(
                f"Image outputs exceed multimodal grading total size limit: {total_bytes} bytes, "
                f"maximum is {max_total_bytes}."
            )
        total_raw += raw_chars
        if total_raw > max_total_raw:
            raise ValueError(
                f"Image outputs raw data exceeds multimodal grading total size limit: "
                f"{total_raw} characters, maximum is {max_total_raw}."
            )
    return {
        "imageOutputs": [m["output"] for m in materialized],
        "imageData": [
            {"dataUri": m["dataUri"], "base64Data": m["base64Data"], "mimeType": m["mimeType"]}
            for m in materialized
        ],
    }


def _stringify_content_part(part: Any) -> str:
    if part is UNDEFINED or callable(part):
        return js_string(part)
    return json_stringify(part)


def _build_text_part(text: str, fmt: str) -> dict[str, Any]:
    return {"type": "input_text" if fmt == "responses" else "text", "text": text}


def _build_image_parts(images: list[dict[str, Any]], fmt: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for image in images:
        if fmt == "responses":
            parts.append({"type": "input_image", "image_url": image["dataUri"]})
        elif fmt == "anthropic":
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image["mimeType"],
                        "data": image["base64Data"],
                    },
                }
            )
        elif fmt == "google":
            parts.append(
                {"inlineData": {"mimeType": image["mimeType"], "data": image["base64Data"]}}
            )
        else:
            parts.append({"type": "image_url", "image_url": {"url": image["dataUri"]}})
    return parts


def _base64_source(part: dict[str, Any]) -> dict[str, Any] | None:
    source = part.get("source")
    if (
        part.get("type") == "image"
        and isinstance(source, dict)
        and source.get("type") == "base64"
        and isinstance(source.get("media_type"), str)
        and isinstance(source.get("data"), str)
    ):
        return source
    return None


def _image_url_string(part: dict[str, Any]) -> str | None:
    image_url = part.get("image_url")
    if (
        part.get("type") == "image_url"
        and isinstance(image_url, dict)
        and isinstance(image_url.get("url"), str)
    ):
        return image_url["url"]
    return None


def _to_responses_content_part(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "input_text", "text": part}
    if not isinstance(part, dict):
        return {"type": "input_text", "text": _stringify_content_part(part)}
    ptype = part.get("type")
    if ptype in ("input_text", "text") and isinstance(part.get("text"), str):
        return {"type": "input_text", "text": part["text"]}
    if ptype == "input_image" and isinstance(part.get("image_url"), str):
        return {"type": "input_image", "image_url": part["image_url"]}
    url = _image_url_string(part)
    if url is not None:
        return {"type": "input_image", "image_url": url}
    source = _base64_source(part)
    if source is not None:
        return {
            "type": "input_image",
            "image_url": f"data:{source['media_type']};base64,{source['data']}",
        }
    return {"type": "input_text", "text": _stringify_content_part(part)}


def _data_uri_to_google_content_part(data_uri: str) -> dict[str, Any] | None:
    if not js_trim(data_uri).startswith("data:"):
        return None
    normalized = _normalize_base64_image_data(data_uri)
    return {"inlineData": {"mimeType": normalized["mimeType"], "data": normalized["base64Data"]}}


def _to_google_content_part(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": _stringify_content_part(part)}
    ptype = part.get("type")
    if ptype in ("text", "input_text") and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"]}
    url = _image_url_string(part)
    if url is not None:
        return _data_uri_to_google_content_part(url) or {
            "type": "text",
            "text": _stringify_content_part(part),
        }
    if ptype == "input_image" and isinstance(part.get("image_url"), str):
        return _data_uri_to_google_content_part(part["image_url"]) or {
            "type": "text",
            "text": _stringify_content_part(part),
        }
    source = _base64_source(part)
    if source is not None:
        return {"inlineData": {"mimeType": source["media_type"], "data": source["data"]}}
    inline = part.get("inlineData")
    if (
        isinstance(inline, dict)
        and isinstance(inline.get("mimeType"), str)
        and isinstance(inline.get("data"), str)
    ):
        return {"inlineData": {"mimeType": inline["mimeType"], "data": inline["data"]}}
    inline = part.get("inline_data")
    if (
        isinstance(inline, dict)
        and isinstance(inline.get("mime_type"), str)
        and isinstance(inline.get("data"), str)
    ):
        return {"inlineData": {"mimeType": inline["mime_type"], "data": inline["data"]}}
    return {"type": "text", "text": _stringify_content_part(part)}


def _append_media_to_content(
    content: Any, media_parts: list[dict[str, Any]], fmt: str
) -> list[Any]:
    if isinstance(content, list):
        if fmt == "responses":
            normalized = [_to_responses_content_part(c) for c in content]
        elif fmt == "google":
            normalized = [_to_google_content_part(c) for c in content]
        else:
            normalized = list(content)
        return [*normalized, *media_parts]
    if isinstance(content, str):
        return [_build_text_part(content, fmt), *media_parts]
    if content is None or content is UNDEFINED:
        return list(media_parts)
    return [_build_text_part(_stringify_content_part(content), fmt), *media_parts]


def _is_responses_compatible_provider(provider: Any, provider_id: str) -> bool:
    return bool(re.search(r"(^|:)responses(:|$)", provider_id, re.IGNORECASE)) or (
        type(provider).__name__ in RESPONSES_PROVIDER_CLASS_NAMES
    )


def _is_anthropic_compatible_provider_id(provider_id: str) -> bool:
    return bool(
        re.search(r"(^|:)anthropic(:|$)", provider_id, re.IGNORECASE)
        or re.match(
            r"^bedrock:(?:converse:|kb:)?(?:[^:]*\.)?anthropic\.claude", provider_id, re.IGNORECASE
        )
        or re.match(r"^vertex:claude", provider_id, re.IGNORECASE)
    )


def _is_google_compatible_provider_id(provider_id: str) -> bool:
    return bool(
        re.match(r"^google(:|$)", provider_id, re.IGNORECASE)
        or re.match(r"^vertex:(?:chat:)?gemini", provider_id, re.IGNORECASE)
    )


def _get_multimodal_prompt_format(provider: Any) -> str:
    try:
        provider_id = provider.id()
        if _is_responses_compatible_provider(provider, provider_id):
            return "responses"
        if _is_anthropic_compatible_provider_id(provider_id):
            return "anthropic"
        if _is_google_compatible_provider_id(provider_id):
            return "google"
    except Exception:
        pass
    return "openai"


def _append_media_to_chat_prompt(
    rendered_prompt: str, media_parts: list[dict[str, Any]], fmt: str
) -> str:
    parsed: Any = None
    if js_trim(rendered_prompt).startswith("- role:"):
        try:
            parsed = load_yaml(rendered_prompt)
        except Exception as err:
            raise ValueError(
                f"Chat Completion prompt is not a valid YAML string: {err}\n\n{rendered_prompt}"
            ) from err
    else:
        try:
            parsed = js_json_parse(rendered_prompt)
        except Exception:
            pass
    if _is_chat_message_array(parsed):
        messages = [dict(m) for m in parsed]
        roles = [m["role"] for m in messages]
        user_index = len(roles) - 1 - roles[::-1].index("user") if "user" in roles else -1
        if user_index >= 0:
            user_message = messages[user_index]
            messages[user_index] = {
                **user_message,
                "content": _append_media_to_content(
                    user_message.get("content", UNDEFINED), media_parts, fmt
                ),
            }
        else:
            messages.append({"role": "user", "content": media_parts})
        return json_stringify(messages)
    return json_stringify(
        [{"role": "user", "content": [_build_text_part(rendered_prompt, fmt), *media_parts]}]
    )


def _build_audio_grading_part(audio: dict[str, Any], provider: Any) -> dict[str, Any] | None:
    get_format = getattr(provider, "get_audio_input_format", None)
    if not callable(get_format) or get_format() != "openai":
        return None
    if audio.get("blobRef") or _has_blob_ref_image_value(audio.get("data")):
        raise ValueError(
            "Audio grading requires inline base64 audio; blob references are not supported."
        )
    if audio.get("format") not in ("wav", "mp3"):
        raise ValueError(
            "Audio grading requires WAV or MP3 output. Configure the target audio format."
        )
    data = re.sub(r"\s", "", audio.get("data") or "")
    if len(data) > math.ceil(GRADING_AUDIO_MAX_BYTES / 3) * 4:
        raise ValueError("Audio output exceeds the 20 MiB grading size limit.")
    if not _is_valid_base64_payload(data) or _get_base64_decoded_bytes(data) <= 0:
        raise ValueError("Audio grading requires non-empty, valid base64 audio data.")
    if _get_base64_decoded_bytes(data) > GRADING_AUDIO_MAX_BYTES:
        raise ValueError("Audio output exceeds the 20 MiB grading size limit.")
    return {"type": "input_audio", "input_audio": {"data": data, "format": audio["format"]}}


async def build_grading_provider_prompt(
    rendered_prompt: str,
    images: list[dict[str, Any]] | None = None,
    provider: Any = None,
    audio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_data = materialize_image_outputs_for_grading(images)["imageData"]
    audio_part = _build_audio_grading_part(audio, provider) if audio and provider else None
    fmt = "openai" if audio_part or not provider else _get_multimodal_prompt_format(provider)
    media_parts: list[dict[str, Any]] = (
        [
            _build_text_part(MULTIMODAL_GRADING_INSTRUCTION, fmt),
            *_build_image_parts(image_data, fmt),
        ]
        if image_data
        else []
    )
    if audio_part:
        media_parts.extend([_build_text_part(AUDIO_GRADING_INSTRUCTION, fmt), audio_part])
    return {
        "prompt": _append_media_to_chat_prompt(rendered_prompt, media_parts, fmt)
        if media_parts
        else rendered_prompt,
        "imageCount": len(image_data),
        "audioAttached": bool(audio_part),
    }


# --- JSON grading -----------------------------------------------------------------------------


def grader_failure_from_response(reason: str, response: dict[str, Any]) -> dict[str, Any]:
    failure = grader_fail(reason, response.get("tokenUsage"))
    if response.get("cached"):
        return {**failure, "metadata": {**failure["metadata"], "cachedResponse": True}}
    usage = failure["tokensUsed"]
    if (
        usage
        and not usage.get("numRequests")
        and not (usage.get("total") or 0)
        and not (usage.get("prompt") or 0)
        and not (usage.get("completion") or 0)
    ):
        return {**failure, "tokensUsed": {**usage, "numRequests": 1}}
    return failure


def _parse_json_grading_response(label: str, resp: dict[str, Any]) -> dict[str, Any]:
    output = resp.get("output")
    if isinstance(output, str):
        try:
            json_objects = extract_json_objects(output)
        except Exception as err:
            return {
                "failure": grader_failure_from_response(
                    f"{label} produced malformed response: {err}\n\n{output}", resp
                )
            }
        if not json_objects:
            return {
                "failure": grader_failure_from_response(
                    f"Could not extract JSON from {label} response", resp
                )
            }
    elif isinstance(output, dict):
        json_objects = [output]
    else:
        return {
            "failure": grader_failure_from_response(
                f"{label} produced malformed response - output must be string or object. "
                f"Output: {json_stringify(output)}",
                resp,
            )
        }
    parsed = json_objects[0]
    if not isinstance(parsed, dict):
        return {
            "failure": grader_failure_from_response(
                f"{label} produced malformed response. We were not able to parse the response "
                f"as JSON. Output: {json_stringify(output)}",
                resp,
            )
        }
    return {"parsed": parsed}


def _derive_verdict_from_grader(
    parsed: dict[str, Any], threshold: Any
) -> tuple[bool, float] | None:
    has_threshold = _is_finite_number(threshold)
    raw_pass = parsed.get("pass")
    raw_score = parsed.get("score", UNDEFINED)
    if raw_pass is None:
        if isinstance(raw_score, int | float) and not isinstance(raw_score, bool):
            score = raw_score
        elif isinstance(raw_score, str) and js_trim(raw_score) != "":
            score = js_number(raw_score)
        else:
            score = math.nan
        if not _is_finite_number(score):
            return None
        return (score >= threshold if has_threshold else score > 0), score

    passed = (
        raw_pass
        if isinstance(raw_pass, bool)
        else bool(re.fullmatch(r"true|yes|pass|y", js_string(raw_pass), re.IGNORECASE))
    )
    if isinstance(raw_score, int | float) and not isinstance(raw_score, bool):
        score = raw_score
    else:
        numeric = js_number(raw_score)
        score = numeric if _is_finite_number(numeric) else (1 if passed else 0)
    if has_threshold:
        passed = passed and score >= threshold
    return passed, score


async def run_json_grading_prompt(
    *,
    check_name: str,
    default_prompt: str,
    grading: dict[str, Any],
    label: str,
    vars: dict[str, Any],  # noqa: A002 - Node's parameter name
    assertion: dict[str, Any] | None = None,
    provider_call_context: dict[str, Any] | None = None,
    provider_prompt_config: dict[str, Any] | None = None,
    throw_on_error: bool = False,
    images: list[dict[str, Any]] | None = None,
    audio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from promptphoo.providers.defaults import get_default_providers

    rubric_prompt = await load_rubric_prompt(grading.get("rubricPrompt"), default_prompt)
    rendered_prompt = await render_llm_rubric_prompt(rubric_prompt, vars)

    defaults = await get_default_providers()
    default_provider = defaults.get("llmRubricProvider") or defaults.get("gradingJsonProvider")
    final_provider = await get_and_check_provider(
        "text", grading.get("provider"), default_provider, check_name
    )
    built = await build_grading_provider_prompt(rendered_prompt, images, final_provider, audio)
    resp = await call_provider_with_context(
        final_provider,
        built["prompt"],
        label,
        vars,
        provider_call_context,
        provider_prompt_config,
    )
    if resp.get("error") or not is_truthy(resp.get("output")):
        if throw_on_error:
            raise ValueError(resp.get("error") or "No output")
        return grader_failure_from_response(resp.get("error") or "No output", resp)

    parsed_result = _parse_json_grading_response(label, resp)
    parsed = parsed_result.get("parsed")
    if parsed is None:
        return parsed_result["failure"]

    threshold = (assertion or {}).get("threshold")
    if isinstance(threshold, str):
        threshold = js_number(threshold)

    verdict = _derive_verdict_from_grader(parsed, threshold)
    if verdict is None:
        return grader_failure_from_response(
            "Grader response contained neither a pass verdict nor a finite numeric score", resp
        )
    passed, score = verdict

    reason = parsed.get("reason")
    if not is_truthy(reason):
        reason = (
            "Grading passed"
            if passed
            else f"Score {number_to_string(score)} below threshold "
            f"{js_string(UNDEFINED if threshold is None else threshold)}"
        )

    response_metadata: dict[str, Any] = {}
    meta = resp.get("metadata")
    if isinstance(meta, dict):
        serialized = safe_json_stringify(meta)
        response_metadata = js_json_parse(serialized) if serialized else {}
    trusted_metadata = {k: v for k, v in response_metadata.items() if k != "cachedResponse"}

    token_usage = resp.get("tokenUsage") or {}
    parsed_tokens = parsed.get("tokensUsed")
    result: dict[str, Any] = {}
    if assertion is not None:
        result["assertion"] = assertion
    result.update(
        {
            "pass": passed,
            "score": score,
            "reason": reason,
            "tokensUsed": normalize_matcher_token_usage(
                {
                    **token_usage,
                    "completionDetails": token_usage.get("completionDetails")
                    or (
                        parsed_tokens.get("completionDetails")
                        if isinstance(parsed_tokens, dict)
                        else None
                    ),
                }
            ),
            "metadata": {
                **trusted_metadata,
                "renderedGradingPrompt": rendered_prompt,
                **(
                    {"renderedGradingPromptImages": built["imageCount"]}
                    if built["imageCount"] > 0
                    else {}
                ),
                **({"renderedGradingPromptAudio": True} if built["audioAttached"] else {}),
                **({"cachedResponse": True} if resp.get("cached") else {}),
            },  # fmt: skip
        }
    )
    return result
