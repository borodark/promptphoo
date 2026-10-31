"""Databricks Foundation Model APIs (Mosaic AI) through the serving-endpoints OpenAI route.
Ports src/providers/databricks.ts and the `databricks:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider


class DatabricksMosaicAiChatCompletionProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        config = provider_options.get("config") or {}
        resolved = resolve_provider_env(provider_options.get("env"), ["DATABRICKS_WORKSPACE_URL"])
        workspace_url = config.get("workspaceUrl") or (resolved["value"] if resolved else None)
        if not workspace_url:
            raise ValueError(
                "Databricks workspace URL is required. Set it in the config or "
                "DATABRICKS_WORKSPACE_URL environment variable."
            )
        clean_workspace_url = re.sub(r"/$", "", workspace_url)
        # The inherited chat transport appends /chat/completions to this base URL.
        api_base_url = f"{clean_workspace_url}/serving-endpoints"
        merged_config: dict[str, Any] = {
            **config,
            "apiKeyEnvar": config.get("apiKeyEnvar") or "DATABRICKS_TOKEN",
            "apiBaseUrl": api_base_url,
        }
        if config.get("usageContext"):
            merged_config["passthrough"] = {
                **(config.get("passthrough") or {}),
                "usage_context": config["usageContext"],
            }
        super().__init__(model_name, {**provider_options, "config": merged_config})
        self.config = merged_config


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `databricks:` registry factory."""
    model_name = ":".join(provider_path.split(":")[1:])
    return DatabricksMosaicAiChatCompletionProvider(
        model_name, {**options, "config": options.get("config") or {}}
    )
