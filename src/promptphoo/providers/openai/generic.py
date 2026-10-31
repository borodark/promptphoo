"""Base class for OpenAI and OpenAI-compatible providers: API key, base URL, organization and
model-capability resolution. Ports src/providers/openai/index.ts.

OpenAI-compatible wrappers subclass a concrete provider and override `get_api_url_default()`,
`default_id()`, or set `config.apiKeyEnvar` / `config.apiBaseUrl`, as upstream subclasses do.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo import envars
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.credentials import resolve_provider_api_key, resolve_provider_env
from promptphoo.providers.openai.gpt6 import is_gpt6_model
from promptphoo.providers.openai.urls import url_hostname

OPENAI_ORIGINATOR_HEADER = "X-OpenAI-Originator"
OPENAI_ORGANIZATION_HEADER = "OpenAI-Organization"
DEFAULT_OPENAI_ORIGINATOR = "promptfoo"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1"

_FT_PREFIX = re.compile(r"(^|/)ft:")


def strip_fine_tune_prefix(model_name: str) -> str:
    return _FT_PREFIX.sub(r"\1", model_name)


def has_header_override(custom_headers: dict[str, str] | None, header_name: str) -> bool:
    """Whether `custom_headers` overrides `header_name`, case-insensitively."""
    target = header_name.lower()
    return any(key.lower() == target for key in (custom_headers or {}))


def resolve_openai_api_url(
    config: dict[str, Any] | None = None,
    env: dict[str, Any] | None = None,
    default_url: str = DEFAULT_OPENAI_API_URL,
) -> str:
    config = config or {}
    if config.get("apiHost"):
        return f"https://{config['apiHost']}/v1"
    if config.get("apiBaseUrl"):
        return config["apiBaseUrl"]
    endpoint = resolve_provider_env(
        env, ["OPENAI_API_HOST", "OPENAI_API_BASE_URL", "OPENAI_BASE_URL"]
    )
    if not endpoint:
        return default_url
    if endpoint["name"] == "OPENAI_API_HOST":
        return f"https://{endpoint['value']}/v1"
    return endpoint["value"]


class OpenAiGenericProvider(ApiProvider):
    """Shared OpenAI provider state. `config`/`env` are plain dicts, as in Node."""

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(options)
        self.model_name = model_name
        self.config = dict(options.get("config") or {})
        self.env = options.get("env")

    # Upstream: `id()` returns modelName for custom hosts, else `openai:<model>`.
    def default_id(self) -> str:
        if self.config.get("apiHost") or self.config.get("apiBaseUrl"):
            return self.model_name
        return f"openai:{self.model_name}"

    @property
    def modelName(self) -> str:  # noqa: N802 - upstream attribute name, for hooks/tests
        return self.model_name

    def get_gen_ai_system(self) -> str:
        """The backend family (`openai`, or the prefix of a subclass's default id)."""
        if type(self).default_id is OpenAiGenericProvider.default_id:
            return "openai"
        provider_id = self.default_id()
        return provider_id.split(":", 1)[0] if ":" in provider_id else "openai"

    def __str__(self) -> str:
        return f"[OpenAI Provider {self.model_name}]"

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        config = self.config if config is None else config
        return (
            config.get("organization")
            or (self.env or {}).get("OPENAI_ORGANIZATION")
            or envars.get_env_string("OPENAI_ORGANIZATION")
        )

    def get_openai_request_headers(
        self,
        custom_headers: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        *,
        use_config_headers: bool = True,
    ) -> dict[str, str]:
        """Default originator/organization headers; custom headers win case-insensitively.
        `custom_headers` defaults to `config.headers` (pass `use_config_headers=False` to send
        none)."""
        config = self.config if config is None else config
        if custom_headers is None and use_config_headers:
            custom_headers = self.config.get("headers")
        hostname = url_hostname(self.get_api_url(config))
        sends_to_openai_api = hostname is not None and hostname.lower() == "api.openai.com"
        has_originator = has_header_override(custom_headers, OPENAI_ORIGINATOR_HEADER)
        has_organization = has_header_override(custom_headers, OPENAI_ORGANIZATION_HEADER)
        organization = None if has_organization else self.get_organization(config)
        headers: dict[str, str] = {}
        if not has_originator and sends_to_openai_api:
            headers[OPENAI_ORIGINATOR_HEADER] = DEFAULT_OPENAI_ORIGINATOR
        if organization:
            headers[OPENAI_ORGANIZATION_HEADER] = organization
        headers.update(custom_headers or {})
        return headers

    def get_api_url_default(self) -> str:
        return DEFAULT_OPENAI_API_URL

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        config = self.config if config is None else config
        return resolve_openai_api_url(config, self.env, self.get_api_url_default())

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        config = self.config if config is None else config
        return resolve_provider_api_key(
            config,
            self.env,
            [] if config.get("useDefaultApiKey") is False else ["OPENAI_API_KEY"],
        )

    def requires_api_key(self) -> bool:
        required = self.config.get("apiKeyRequired")
        return True if required is None else required

    def get_request_authentication(self) -> Any:
        """Optional per-attempt auth headers hook (None for static-key providers)."""
        return None

    def get_capability_model_name(self) -> str:
        """Model id for capability and billing lookups; subclasses may strip a vendor prefix."""
        return self.model_name

    def normalize_capability_model_name(self, model_name: str) -> str:
        return model_name

    def is_gpt5_model(self, model_name: str | None = None) -> bool:
        model = strip_fine_tune_prefix(model_name or self.get_capability_model_name())
        return model.startswith("gpt-5") or "/gpt-5" in model

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        model = strip_fine_tune_prefix(model_name or self.get_capability_model_name())
        return (
            model.startswith(("o1", "o3", "o4"))
            or any(s in model for s in ("/o1", "/o3", "/o4"))
            or re.search(r"(^|/)gpt-daybreak-(?:blue|red)-latest$", model) is not None
            or self.is_gpt5_model(model)
            or is_gpt6_model(model)
        )

    def supports_temperature(self, model_name: str | None = None) -> bool:
        return not self.is_reasoning_model(model_name or self.get_capability_model_name())

    def get_billing_model_name(self, config: dict[str, Any]) -> str:
        return self.get_capability_model_name()

    def should_bust_cache(self, context: dict[str, Any] | None = None) -> bool | None:
        context = context or {}
        if context.get("bustCache") is not None:
            return context["bustCache"]
        return context.get("debug")

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        config = self.config if config is None else config
        envar = config.get("apiKeyEnvar") or "OPENAI_API_KEY"
        return (
            f"API key is not set. Set the {envar} environment variable or add `apiKey` to the "
            "provider config."
        )

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        raise NotImplementedError("Not implemented")
