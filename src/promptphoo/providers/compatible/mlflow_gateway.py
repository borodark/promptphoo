"""MLflow AI Gateway (MLflow >= 3.0) through its OpenAI-compatible endpoint. Ports
src/providers/mlflow-gateway.ts and the `mlflow-gateway:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo import envars
from promptphoo.providers.compatible.common import serialize_provider
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

MLFLOW_GATEWAY_URL_ENV_VAR = "MLFLOW_GATEWAY_URL"
MLFLOW_GATEWAY_API_KEY_ENV_VAR = "MLFLOW_GATEWAY_API_KEY"
MLFLOW_GATEWAY_API_PATH = "/gateway/mlflow/v1"


def normalize_gateway_url(gateway_url: str | None) -> str | None:
    trimmed = gateway_url.strip() if isinstance(gateway_url, str) else None
    return re.sub(r"/+$", "", trimmed) if trimmed else None


class MlflowGatewayChatCompletionProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        if not model_name or model_name.strip() == "":
            raise ValueError(
                "MLflow Gateway endpoint name is required. Use the format "
                '"mlflow-gateway:<endpoint-name>" where <endpoint-name> is the name of '
                "the gateway endpoint you created in the MLflow UI."
            )
        config = provider_options.get("config") or {}
        gateway_url = normalize_gateway_url(config.get("gatewayUrl"))
        if gateway_url is None:
            resolved = resolve_provider_env(
                provider_options.get("env"), [MLFLOW_GATEWAY_URL_ENV_VAR]
            )
            gateway_url = normalize_gateway_url(resolved["value"] if resolved else None)
        if not gateway_url:
            raise ValueError(
                "MLflow Gateway URL is required. Set it in the provider config or via the "
                f"{MLFLOW_GATEWAY_URL_ENV_VAR} environment variable (e.g., http://localhost:5000)."
            )
        merged_config: dict[str, Any] = {
            **config,
            "apiKeyEnvar": config.get("apiKeyEnvar") or MLFLOW_GATEWAY_API_KEY_ENV_VAR,
            "apiBaseUrl": f"{gateway_url}{MLFLOW_GATEWAY_API_PATH}",
            "apiKeyRequired": (
                config["apiKeyRequired"] if config.get("apiKeyRequired") is not None else False
            ),
        }
        super().__init__(
            model_name,
            {
                **provider_options,
                "id": provider_options.get("id") or f"mlflow-gateway:{model_name}",
                "config": merged_config,
            },
        )
        self.config = merged_config

    # Never fall back to OPENAI_API_KEY: the gateway is a different service.
    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        envar = self.config.get("apiKeyEnvar")
        if "apiKey" in self.config:
            return self.config["apiKey"]
        if not envar:
            return None
        value = envars.get_provider_env_string(self.env, envar)
        return value if value is not None else envars.get_env_string(envar)

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    # Never fall back to OPENAI_API_HOST / OPENAI_API_BASE_URL / OPENAI_BASE_URL.
    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        return self.config.get("apiBaseUrl") or self.get_api_url_default()

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        envar = self.config.get("apiKeyEnvar") or MLFLOW_GATEWAY_API_KEY_ENV_VAR
        return (
            f"MLflow Gateway Bearer token is not set. Set the {envar} "
            "environment variable or add `apiKey` to the provider config."
        )

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "mlflow-gateway")


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `mlflow-gateway:` registry factory."""
    model_name = ":".join(provider_path.split(":")[1:])
    return MlflowGatewayChatCompletionProvider(
        model_name, {**options, "config": options.get("config") or {}}
    )
