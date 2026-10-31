"""Azure OpenAI provider family. `create_provider` ports the `azure:`/`azureopenai:` factory in
src/providers/registry.ts.

Supported: chat (including "on your data" `data_sources`), completion, embedding(s) and
responses. Assistants, Foundry agents, image, video,
moderation (Content Safety) and realtime are not supported by promptphoo.
"""

from __future__ import annotations

from typing import Any

from promptphoo.providers.azure.chat import AzureChatCompletionProvider
from promptphoo.providers.azure.completion import AzureCompletionProvider
from promptphoo.providers.azure.embedding import AzureEmbeddingProvider
from promptphoo.providers.azure.generic import AzureGenericProvider
from promptphoo.providers.azure.responses import AzureResponsesProvider
from promptphoo.providers.shared import model_name_from_provider_path

__all__ = [
    "AzureChatCompletionProvider",
    "AzureCompletionProvider",
    "AzureEmbeddingProvider",
    "AzureGenericProvider",
    "AzureResponsesProvider",
    "create_provider",
]

_SUPPORTED_HINT = (
    "Use azure:chat:<deployment>, azure:completion:<deployment>, azure:embedding:<deployment> "
    "or azure:responses:<deployment>."
)


def _unsupported(provider_path: str, kind: str) -> ValueError:
    return ValueError(
        f"Azure {kind} providers are not supported by promptphoo ({provider_path}). "
        f"{_SUPPORTED_HINT}"
    )


def create_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    options = dict(options or {})
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    deployment_name = model_name_from_provider_path(provider_path, 2)
    is_azure_openai_prefix = provider_path.startswith("azureopenai:")

    if model_type == "decisions":
        raise ValueError(
            "The Decisions API is available through openai:decisions:<model name>, not the "
            "Azure provider."
        )

    def require_path_segment(type_: str, label: str, placeholder: str) -> None:
        if not deployment_name:
            raise ValueError(
                f"Azure {type_} provider requires {label}. Use azure:{type_}:<{placeholder}>."
            )

    if model_type == "moderation":
        if is_azure_openai_prefix:
            raise ValueError(
                "Azure OpenAI does not support moderation. Use azure:moderation instead, which "
                "routes to Azure Content Safety."
            )
        raise _unsupported(provider_path, "moderation (Content Safety)")
    if model_type == "chat":
        require_path_segment("chat", "a deployment name", "deployment")
        return AzureChatCompletionProvider(deployment_name, options)
    if model_type == "assistant":
        raise _unsupported(provider_path, "assistant")
    if model_type == "foundry-agent":
        raise _unsupported(provider_path, "Foundry agent")
    if model_type == "image":
        if is_azure_openai_prefix:
            raise ValueError(
                "azureopenai:image is not supported. MAI image models are Microsoft Foundry "
                "models — use azure:image:<deployment> instead."
            )
        raise _unsupported(provider_path, "image")
    if model_type in ("embedding", "embeddings"):
        return AzureEmbeddingProvider(deployment_name or "text-embedding-ada-002", options)
    if model_type == "completion":
        require_path_segment("completion", "a deployment name", "deployment")
        return AzureCompletionProvider(deployment_name, options)
    if model_type == "responses":
        return AzureResponsesProvider(deployment_name or "gpt-4.1-2025-04-14", options)
    if model_type == "live":
        raise ValueError(
            "GPT-Live is available through openai:live:<model name>, not the Azure provider."
        )
    if model_type == "realtime":
        raise _unsupported(provider_path, "realtime")
    if model_type == "video":
        raise _unsupported(provider_path, "video")
    raise ValueError(
        f"Unknown Azure model type: {model_type}. Use one of the following providers: "
        "azure:chat:<model name>, azure:assistant:<assistant id>, azure:completion:<model name>, "
        "azure:image:<deployment name>, azure:moderation:<model name>, "
        "azure:realtime:<deployment name>, azure:responses:<model name>, "
        "azure:video:<deployment name>"
    )
