"""TrueFoundry AI Gateway chat and embedding providers. Ports src/providers/truefoundry.ts and
the `truefoundry:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.compatible.common import (
    configured_provider_options,
    serialize_provider,
)
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider
from promptphoo.providers.openai.gpt6 import is_gpt6_model

TRUEFOUNDRY_GUARDRAIL_ERROR_TYPE = "guardrail_checks_failed"
DEFAULT_TRUEFOUNDRY_OPENAI_ACCOUNT_NAMES = frozenset(["openai-main"])
DOWNSTREAM_GUARDRAIL_ERROR_CODES = frozenset(["content_filter", "content_policy_violation"])
DOWNSTREAM_GUARDRAIL_MESSAGE_PATTERNS = [
    re.compile(r"\bresponse content blocked by label\b", re.I),
    re.compile(
        r"\b(?:prompt|input|response|output|completion) (?:was )?(?:blocked|filtered)\b", re.I
    ),
    re.compile(r"\bcontent management policy\b", re.I),
    re.compile(r"\bresponsible\s*ai\s*policy(?:\s*violation)?\b", re.I),
]  # fmt: skip


def _get_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def get_truefoundry_billing_model_name(
    model_name: str, openai_account_names: list[str] | None = None
) -> str:
    separator_index = model_name.find("/")
    if separator_index <= 0:
        return model_name
    account_name = model_name[:separator_index]
    is_openai_account = account_name in DEFAULT_TRUEFOUNDRY_OPENAI_ACCOUNT_NAMES or (
        isinstance(openai_account_names, list) and account_name in openai_account_names
    )
    return model_name[separator_index + 1 :] if is_openai_account else model_name


def _has_guardrail_check(value: Any) -> bool:
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(value)


def _has_filtered_content(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_filtered_content(v) for v in value)
    if not isinstance(value, dict):
        return False
    if value.get("filtered") is True:
        return True
    return any(_has_filtered_content(v) for v in value.values())


def _parse_http_error_body(response: ProviderResponse) -> dict[str, Any] | None:
    metadata = response.get("metadata")
    http = metadata.get("http") if isinstance(metadata, dict) else None
    status = http.get("status") if isinstance(http, dict) else None
    error = response.get("error")
    if status != 400 or not isinstance(error, str):
        return None
    body_start = error.find("\n")
    if body_start < 0:
        return None
    try:
        body = js_json_parse(error[body_start + 1 :])
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _get_inner_error(error: dict[str, Any]) -> dict[str, Any] | None:
    inner = error.get("innererror")
    if inner is None:
        inner = error.get("inner_error")
    return inner if isinstance(inner, dict) else None


def _is_guardrail_error(payload: dict[str, Any], error: dict[str, Any], message: str) -> bool:
    if _get_string(error.get("type")) == TRUEFOUNDRY_GUARDRAIL_ERROR_TYPE:
        return True
    inner = _get_inner_error(error)
    error_code = (_get_string(error.get("code")) or "").lower() or None
    inner_code = (_get_string((inner or {}).get("code")) or "").lower() or None
    if (
        error_code and error_code in DOWNSTREAM_GUARDRAIL_ERROR_CODES
    ) or inner_code == "responsibleaipolicyviolation":
        return True
    content_filter_results = [
        error.get("content_filter_result"),
        error.get("content_filter_results"),
        (inner or {}).get("content_filter_result"),
        (inner or {}).get("content_filter_results"),
        payload.get("content_filter_result"),
        payload.get("content_filter_results"),
    ]
    if any(_has_filtered_content(v) for v in content_filter_results):
        return True
    if error_code == "content_filter_error" or inner_code == "content_filter_error":
        return False
    return any(p.search(message) for p in DOWNSTREAM_GUARDRAIL_MESSAGE_PATTERNS)


def _get_guardrail_direction(payload: dict[str, Any], message: str) -> dict[str, bool]:
    checks = payload.get("guardrail_checks")
    if isinstance(checks, dict):
        flagged_input = _has_guardrail_check(checks.get("llm_input_guardrails"))
        flagged_output = _has_guardrail_check(checks.get("llm_output_guardrails"))
        if flagged_input or flagged_output:
            return {"flaggedInput": flagged_input, "flaggedOutput": flagged_output}
    if re.search(r"\b(prompt|input)\b", message, re.I):
        return {"flaggedInput": True, "flaggedOutput": False}
    if re.search(r"\b(response|output|completion)\b", message, re.I):
        return {"flaggedInput": False, "flaggedOutput": True}
    return {}


def normalize_guardrail_error_response(response: ProviderResponse) -> ProviderResponse:
    """Turn a blocked gateway or downstream-model 400 into a guardrail result; other 400s stay
    ordinary API errors."""
    payload = _parse_http_error_body(response)
    error = payload.get("error") if payload else None
    if not payload or not isinstance(error, dict):
        return response
    message = _get_string(error.get("message")) or "Content blocked by provider guardrail"
    if not _is_guardrail_error(payload, error, message):
        return response
    rest = {k: v for k, v in response.items() if k != "error"}
    return {
        **rest,
        "output": message,
        "isRefusal": True,
        "guardrails": {
            "flagged": True,
            **_get_guardrail_direction(payload, message),
            "reason": message,
        },
    }


def _get_truefoundry_provider_options(provider_options: dict[str, Any]) -> dict[str, Any]:
    config = provider_options.get("config") or {}
    return {
        **provider_options,
        "config": {
            **config,
            "apiKeyEnvar": "TRUEFOUNDRY_API_KEY",
            "apiBaseUrl": config.get("apiBaseUrl") or "https://llm-gateway.truefoundry.com",
        },
    }


def _tfy_headers(config: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    headers = dict(base or {})
    if config.get("metadata"):
        headers["X-TFY-METADATA"] = json_stringify(config["metadata"])
    if config.get("loggingConfig"):
        headers["X-TFY-LOGGING-CONFIG"] = json_stringify(config["loggingConfig"])
    return headers


class TrueFoundryProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, _get_truefoundry_provider_options(provider_options or {}))

    def get_billing_model_name(self, config: dict[str, Any]) -> str:
        return get_truefoundry_billing_model_name(
            super().get_billing_model_name(config), config.get("openaiAccountNames")
        )

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        """Detect OpenAI reasoning models behind TrueFoundry's `account/model` names."""
        actual = self.model_name.split("/")[-1] or self.model_name
        return actual.startswith(("o1", "o3", "o4", "gpt-5")) or is_gpt6_model(actual)

    async def get_openai_body(self, prompt, context=None, call_api_options=None):
        result = await super().get_openai_body(prompt, context, call_api_options)
        body, config = result["body"], result["config"]
        headers = _tfy_headers(self.config, config.get("headers"))
        tf_body = dict(body)
        # TrueFoundry takes metadata as the X-TFY-METADATA header, not a body field.
        if self.config.get("metadata") and tf_body.get("metadata"):
            del tf_body["metadata"]
        if self.config.get("mcp_servers"):
            tf_body["mcp_servers"] = self.config["mcp_servers"]
        if self.config.get("iteration_limit") is not None:
            tf_body["iteration_limit"] = self.config["iteration_limit"]
        return {"body": tf_body, "config": {**config, "headers": headers}}

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        response = await super().call_api(prompt, context, options)
        return normalize_guardrail_error_response(response)

    def default_id(self) -> str:
        return f"truefoundry:{self.model_name}"

    def __str__(self) -> str:
        return f"[TrueFoundry Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "truefoundry")


class TrueFoundryEmbeddingProvider(OpenAiEmbeddingProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, _get_truefoundry_provider_options(provider_options or {}))

    def get_billing_model_name(self, config: dict[str, Any]) -> str:
        return get_truefoundry_billing_model_name(
            super().get_billing_model_name(config), config.get("openaiAccountNames")
        )

    async def call_embedding_api(self, text, context=None, options=None):
        headers = _tfy_headers(self.config, self.config.get("headers"))
        # Keep generated headers local to this request.
        provider_for_request = TrueFoundryEmbeddingProvider(
            self.model_name, {"config": {**self.config, "headers": headers}, "env": self.env}
        )
        return await OpenAiEmbeddingProvider.call_embedding_api(
            provider_for_request, text, context, options
        )

    def default_id(self) -> str:
        return f"truefoundry:{self.model_name}"

    def __str__(self) -> str:
        return f"[TrueFoundry Embedding Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "truefoundry")


def create_truefoundry_provider(
    provider_path: str, options: dict[str, Any] | None = None
) -> TrueFoundryProvider | TrueFoundryEmbeddingProvider:
    options = options or {}
    model_name = ":".join(provider_path.split(":")[1:])
    outer = options.get("config") or {}
    inner = outer.get("config") or {}
    task = inner.get("task")
    # JS `undefined` is a missing key; an explicit null is rejected like any other value.
    if "task" in inner and task not in ("chat", "embedding"):
        raise ValueError('TrueFoundry config.task must be "chat" or "embedding"')
    # Account and deployment names are opaque, so explicit task selection must not rewrite them.
    is_embedding_model = task == "embedding" or (task is None and "embedding" in model_name.lower())
    provider_options = {**outer, "env": options.get("env")}
    if is_embedding_model:
        return TrueFoundryEmbeddingProvider(model_name, provider_options)
    return TrueFoundryProvider(model_name, provider_options)


def create_provider(
    provider_path: str, options: dict[str, Any]
) -> TrueFoundryProvider | TrueFoundryEmbeddingProvider:
    """The `truefoundry:` registry factory (configuredProviderFactory)."""
    return create_truefoundry_provider(provider_path, configured_provider_options(options))
