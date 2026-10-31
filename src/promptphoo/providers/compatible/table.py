"""The data table of OpenAI-compatible families whose upstream factory only routes
`<prefix>:[chat|completion|embedding|embeddings:]<model>` to the OpenAI providers with a preset
base URL and API-key env var (see D31).

Ports src/providers/togetherai.ts, aimlapi.ts, cometapi.ts and nscale.ts. Each row records how
the upstream factory treats an explicit `config.apiBaseUrl` / `config.apiKeyEnvar` (kept or
overridden) and whether model parameters are copied into `passthrough`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

from promptphoo.providers.compatible.common import (
    PROVIDERS_BY_TYPE,
    not_supported,
    resolve_provider_creator_input,
    split_local_options,
)
from promptphoo.providers.credentials import resolve_provider_api_key
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

# The chat provider resolves these itself (loads file:// refs, renders vars, normalizes tools);
# `passthrough` is spread into the body last, so a raw copy would clobber the resolved value.
CHAT_RESOLVED_OPTION_NAMES = ("functions", "response_format", "tool_choice", "tools")


@dataclass(frozen=True)
class CompatibleFamily:
    prefix: str
    api_base_url: str
    # None leaves the OpenAI default (`OPENAI_API_KEY`) in place.
    api_key_envar: str | None
    # Whether `config.apiBaseUrl` / `config.apiKeyEnvar` win over the preset (`config.x || preset`).
    base_url_overridable: bool = True
    key_envar_overridable: bool = True
    # "none": config as given. "all": config plus model parameters copied into passthrough.
    # "local": only promptfoo's own options at top level, model parameters in passthrough.
    passthrough: Literal["none", "all", "local"] = "none"
    # Env vars read in order when the family overrides key resolution (nscale).
    api_key_envars: tuple[str, ...] = ()
    unsupported_types: dict[str, str] = field(default_factory=dict)

    def build_config(self, config: dict[str, Any], is_chat: bool) -> dict[str, Any]:
        base = config.get("apiBaseUrl") if self.base_url_overridable else None
        key = config.get("apiKeyEnvar") if self.key_envar_overridable else None
        params: dict[str, Any] = {}
        out = dict(config)
        if self.passthrough != "none":
            extra = CHAT_RESOLVED_OPTION_NAMES if is_chat and self.passthrough == "all" else ()
            local, params = split_local_options(config, extra)
            if self.passthrough == "local":
                out = dict(local)
        # Assignment order follows the upstream object literal; existing keys keep their place.
        out["apiBaseUrl"] = base or self.api_base_url
        if self.api_key_envar is not None:
            out["apiKeyEnvar"] = key or self.api_key_envar
        if self.passthrough != "none":
            out["passthrough"] = {**params, **(config.get("passthrough") or {})}
        return out


COMPATIBLE_FAMILIES: dict[str, CompatibleFamily] = {
    family.prefix: family
    for family in [
        # togetherai.ts: model parameters are passed through; connection options stay local.
        CompatibleFamily("togetherai", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
                         passthrough="all"),
        # aimlapi.ts
        CompatibleFamily("aimlapi", "https://api.aimlapi.com/v1", "AIML_API_KEY",
                         base_url_overridable=False, key_envar_overridable=False),
        # cometapi.ts
        CompatibleFamily("cometapi", "https://api.cometapi.com/v1", "COMETAPI_KEY",
                         base_url_overridable=False,
                         unsupported_types={"image": "The CometAPI image provider"}),
        # nscale.ts
        CompatibleFamily("nscale", "https://inference.api.nscale.com/v1", None,
                         passthrough="local",
                         api_key_envars=("NSCALE_SERVICE_TOKEN", "NSCALE_API_KEY"),
                         unsupported_types={"image": "The Nscale image provider"}),
    ]
}  # fmt: skip


@cache
def _with_api_key_envars(base: type, envars: tuple[str, ...]) -> type:
    """`withNscaleCredentials`: key lookup across several env vars, with a matching error."""

    def get_api_key(self, config=None):
        config = self.config if config is None else config
        return resolve_provider_api_key(config, self.env, list(envars))

    def get_missing_api_key_error_message(self, config=None):
        config = self.config if config is None else config
        envar = config.get("apiKeyEnvar") or envars[0]
        return (
            f"API key is not set. Set the {envar} environment variable or add `apiKey` to the "
            "provider config."
        )

    return type(
        base.__name__,
        (base,),
        {
            "get_api_key": get_api_key,
            "get_missing_api_key_error_message": get_missing_api_key_error_message,
            "__module__": __name__,
        },
    )


def create_family_provider(
    family: CompatibleFamily, provider_path: str, provider_options: dict[str, Any]
) -> Any:
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else None
    if model_type in family.unsupported_types:
        raise not_supported(provider_path, family.unsupported_types[model_type])
    # Without an explicit type the whole remainder is the model name and we default to chat.
    routed = PROVIDERS_BY_TYPE.get(model_type or "")
    cls = routed or OpenAiChatCompletionProvider
    model_name = ":".join(splits[2 if routed else 1 :])
    config = provider_options.get("config") or {}
    options = {
        **provider_options,
        "config": family.build_config(config, cls is OpenAiChatCompletionProvider),
    }
    if family.api_key_envars:
        cls = _with_api_key_envars(cls, family.api_key_envars)
    return cls(model_name, options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    # Every upstream factory here receives the loader's provider options unchanged.
    family = COMPATIBLE_FAMILIES[provider_path.split(":", 1)[0]]
    provider_options = resolve_provider_creator_input({"providerOptions": options})
    return create_family_provider(family, provider_path, provider_options)
