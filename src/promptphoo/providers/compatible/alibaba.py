"""Alibaba Cloud (DashScope) chat and embedding providers. Ports src/providers/alibaba.ts and the
`alibaba:` / `alicloud:` / `aliyun:` / `dashscope:` entry of src/providers/registry.ts.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider

logger = logging.getLogger(__name__)

# Insertion order matters: the unknown-model warning lists them as upstream does.
KNOWN_MODELS: tuple[str, ...] = tuple(
    """
    qwen3.8-max qwen3.8-max-0902 qwen3.8-max-2026-09-02 qwen3.8-flash qwen3.8-omni-flash
    qwen3.7-max qwen3.7-max-preview qwen3.7-max-2026-06-08 qwen3.7-max-2026-05-20
    qwen3.7-max-2026-05-17 qwen3.7-plus qwen3.7-plus-2026-05-26 qwen3.7-flash
    qwen3.7-flash-2026-07-15 qwen3.6-max-preview qwen3.6-plus qwen3.6-plus-2026-04-02
    qwen3.6-flash qwen3.6-flash-2026-04-16 qwen3.6-35b-a3b qwen3.5-plus qwen3.5-plus-2026-02-15
    qwen3.5-flash qwen3.5-flash-2026-02-23 qwen3.5-397b-a17b qwen3.5-122b-a10b qwen3.5-27b
    qwen3.5-35b-a3b qwen3-max qwen3-max-2026-01-23 qwen3-max-2025-09-23 qwen3-max-preview
    qwen-max qwen-max-latest qwen-max-2025-01-25 qwen-plus qwen-plus-latest
    qwen-plus-2025-12-01 qwen-plus-2025-09-11 qwen-plus-2025-07-28 qwen-plus-2025-07-14
    qwen-plus-2025-04-28 qwen-plus-2025-01-25 qwen-flash qwen-flash-2025-07-28 qwen-turbo
    qwen-turbo-latest qwen-turbo-2025-04-28 qwen-turbo-2024-11-01 qwq-plus qwen-long-latest
    qwen-long-2025-01-25 qwen3-omni-flash qwen3-omni-flash-2025-09-15 qwen3-omni-flash-realtime
    qwen3-omni-flash-realtime-2025-09-15 qvq-max qvq-max-latest qvq-max-2025-03-25
    qwen3-vl-plus qwen3-vl-plus-2025-09-23 qwen3-vl-flash qwen3-vl-flash-2025-10-15 qwen-vl-ocr
    qwen3-asr-flash qwen3-asr-flash-2025-09-08 qwen3-asr-flash-realtime
    qwen3-asr-flash-realtime-2025-10-27 qwen-math-plus qwen-math-plus-latest
    qwen-math-plus-2024-09-19 qwen-math-plus-2024-08-16 qwen-math-turbo qwen-math-turbo-latest
    qwen-math-turbo-2024-09-19 qwen3-coder-next qwen3-coder-plus qwen3-coder-plus-2025-09-23
    qwen3-coder-plus-2025-07-22 qwen3-coder-flash qwen3-coder-flash-2025-07-28 qwen-mt-plus
    qwen-mt-turbo qwen-doc-turbo qwen-deep-research qwen3-next-80b-a3b-thinking
    qwen3-next-80b-a3b-instruct qwen3-235b-a22b-thinking-2507 qwen3-235b-a22b-instruct-2507
    qwen3-30b-a3b-thinking-2507 qwen3-30b-a3b-instruct-2507 qwen3-235b-a22b qwen3-32b
    qwen3-30b-a3b qwen3-14b qwen3-8b qwen3-4b qwen3-1.7b qwen3-0.6b qwq-32b qwq-32b-preview
    qvq-72b-preview qwen2.5-omni-7b qwen3-omni-30b-a3b-captioner qwen3-vl-30b-a3b-thinking
    qwen3-vl-30b-a3b-instruct qwen3-vl-235b-a22b-thinking qwen3-vl-235b-a22b-instruct
    qwen3-vl-32b-thinking qwen3-vl-32b-instruct qwen3-vl-8b-thinking qwen3-vl-8b-instruct
    qwen2.5-math-72b-instruct qwen2.5-math-7b-instruct qwen2.5-math-1.5b-instruct
    qwen3-coder-480b-a35b-instruct qwen3-coder-30b-a3b-instruct deepseek-v4.1-flash
    deepseek-v4-pro-0813 deepseek-v4-pro deepseek-v4-flash deepseek-v3.2 deepseek-v3.2-exp
    deepseek-v3.1 deepseek-r1 deepseek-r1-0528 deepseek-v3 deepseek-r1-distill-qwen-1.5b
    deepseek-r1-distill-qwen-7b deepseek-r1-distill-qwen-14b deepseek-r1-distill-qwen-32b
    deepseek-r1-distill-llama-8b deepseek-r1-distill-llama-70b kimi-k3 kimi-k2.6
    moonshot-kimi-k2-instruct ZHIPU/GLM-5.3 glm-5.2 qwen-image-plus text-embedding-v3
    text-embedding-v4 qwen3.7-text-embedding
    """.split()  # noqa: SIM905 - a word list keeps the 150-model table readable
)
_KNOWN_MODEL_SET = frozenset(KNOWN_MODELS)

API_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def get_alibaba_provider_options(model_name: str, options: dict[str, Any]) -> dict[str, Any]:
    if not model_name:
        raise ValueError("Alibaba modelName is required")
    if model_name not in _KNOWN_MODEL_SET:
        logger.warning(
            "Unknown Alibaba Cloud model: %s. Known models: %s", model_name, ", ".join(KNOWN_MODELS)
        )
    config = options.get("config") or {}
    return {
        **options,
        "config": {
            **config,
            "apiBaseUrl": config["apiBaseUrl"]
            if config.get("apiBaseUrl") is not None
            else API_BASE_URL,
            "apiKeyEnvar": "DASHSCOPE_API_KEY",
        },
    }


class AlibabaChatCompletionProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, get_alibaba_provider_options(model_name, options or {}))

    def get_gen_ai_system(self) -> str:
        return "alibaba"


class AlibabaEmbeddingProvider(OpenAiEmbeddingProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, get_alibaba_provider_options(model_name, options or {}))


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `alibaba:` / `alicloud:` / `aliyun:` / `dashscope:` registry entry."""
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    model_name = ":".join(splits[2:])
    if model_type in ("embedding", "embeddings"):
        return AlibabaEmbeddingProvider(model_name or model_type, options)
    return AlibabaChatCompletionProvider(model_name or model_type, options)
