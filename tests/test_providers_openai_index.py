"""OpenAiGenericProvider and the `openai:*` dispatch. Ports deterministic cases from
test/providers/openai/index.test.ts and the OpenAI routes in test/providers/registry.test.ts
(routes promptphoo does not support yet must fail loudly instead)."""

from __future__ import annotations

import logging

import pytest
from openai_helpers import openai_env

from promptphoo.providers.openai import (
    OpenAiChatCompletionProvider,
    OpenAiCompletionProvider,
    OpenAiEmbeddingProvider,
    OpenAiModerationProvider,
    OpenAiResponsesProvider,
    create_provider,
)
from promptphoo.providers.openai.generic import (
    DEFAULT_OPENAI_ORIGINATOR,
    OPENAI_ORIGINATOR_HEADER,
    OpenAiGenericProvider,
)

pytestmark = pytest.mark.usefixtures(openai_env.__name__)


@pytest.fixture
def provider() -> OpenAiGenericProvider:
    return OpenAiGenericProvider(
        "test-model", {"config": {"apiKey": "test-key", "organization": "test-org"}}
    )


@pytest.fixture
def no_openai_key(openai_env: pytest.MonkeyPatch) -> None:
    openai_env.delenv("OPENAI_API_KEY")


# --- OpenAiGenericProvider ------------------------------------------------------------------


def test_default_api_url(provider):
    assert provider.get_api_url() == "https://api.openai.com/v1"


def test_custom_api_host():
    custom = OpenAiGenericProvider("test-model", {"config": {"apiHost": "custom.openai.com"}})
    assert custom.get_api_url() == "https://custom.openai.com/v1"


def test_custom_api_base_url():
    custom = OpenAiGenericProvider(
        "test-model", {"config": {"apiBaseUrl": "https://custom.api.com/openai"}}
    )
    assert custom.get_api_url() == "https://custom.api.com/openai"


def test_explicit_api_base_url_beats_env_api_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_HOST", "wrong.example")
    custom = OpenAiGenericProvider(
        "test-model", {"config": {"apiBaseUrl": "http://127.0.0.1:15500/proxy/openai/v1"}}
    )
    assert custom.get_api_url() == "http://127.0.0.1:15500/proxy/openai/v1"


def test_env_api_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_HOST", "env.example")
    assert OpenAiGenericProvider("test-model").get_api_url() == "https://env.example/v1"


def test_get_organization(provider):
    assert provider.get_organization() == "test-org"


def test_get_organization_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_ORGANIZATION", "env-org")
    assert OpenAiGenericProvider("test-model").get_organization() == "env-org"


def test_default_originator_header_and_explicit_override(provider):
    assert provider.get_openai_request_headers() == {
        OPENAI_ORIGINATOR_HEADER: DEFAULT_OPENAI_ORIGINATOR,
        "OpenAI-Organization": "test-org",
    }
    headers = provider.get_openai_request_headers({OPENAI_ORIGINATOR_HEADER: "custom-originator"})
    assert headers[OPENAI_ORIGINATOR_HEADER] == "custom-originator"


def test_originator_override_is_case_insensitive(provider):
    assert provider.get_openai_request_headers({"x-openai-originator": "custom-originator"}) == {
        "x-openai-originator": "custom-originator",
        "OpenAI-Organization": "test-org",
    }


def test_organization_override_is_case_insensitive(provider):
    assert provider.get_openai_request_headers({"openai-organization": "custom-org"}) == {
        "X-OpenAI-Originator": "promptfoo",
        "openai-organization": "custom-org",
    }


def test_compatible_endpoints_not_attributed_unless_configured():
    custom = OpenAiGenericProvider(
        "test-model", {"config": {"apiBaseUrl": "https://custom.api.com/openai"}}
    )
    assert OPENAI_ORIGINATOR_HEADER not in custom.get_openai_request_headers()
    headers = custom.get_openai_request_headers({OPENAI_ORIGINATOR_HEADER: "custom-originator"})
    assert headers[OPENAI_ORIGINATOR_HEADER] == "custom-originator"


def test_get_api_key(provider):
    assert provider.get_api_key() == "test-key"


def test_get_api_key_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert OpenAiGenericProvider("test-model").get_api_key() == "env-key"


def test_get_api_key_from_custom_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-key")
    custom = OpenAiGenericProvider("test-model", {"config": {"apiKeyEnvar": "CUSTOM_API_KEY"}})
    assert custom.get_api_key() == "custom-key"


def test_provider_scoped_custom_key_beats_suite_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "suite-key")
    scoped = OpenAiGenericProvider(
        "test-model",
        {
            "config": {"apiKeyEnvar": "DASHSCOPE_API_KEY"},
            "env": {"DASHSCOPE_API_KEY": "provider-key"},
        },
    )
    assert scoped.get_api_key() == "provider-key"


@pytest.mark.parametrize("use_default_api_key", [None, True, False])
def test_use_default_api_key_keeps_explicit_key_priority(
    monkeypatch: pytest.MonkeyPatch, use_default_api_key
):
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-key")
    monkeypatch.setenv("LOCAL_MODEL_KEY", "selected-key")
    monkeypatch.delenv("MISSING_LOCAL_MODEL_KEY", raising=False)
    flag = {} if use_default_api_key is None else {"useDefaultApiKey": use_default_api_key}
    config = {**flag, "apiKeyEnvar": "LOCAL_MODEL_KEY", "apiKey": "inline-key"}
    assert OpenAiGenericProvider("local", {"config": config}).get_api_key() == "inline-key"
    without_inline = {k: v for k, v in config.items() if k != "apiKey"}
    assert (
        OpenAiGenericProvider("local", {"config": without_inline}).get_api_key() == "selected-key"
    )
    missing = {**flag, "apiKeyEnvar": "MISSING_LOCAL_MODEL_KEY"}
    assert OpenAiGenericProvider("local", {"config": missing}).get_api_key() is None
    hosted = OpenAiGenericProvider(
        "local", {"config": flag, "env": {"OPENAI_API_KEY": "overridden-hosted-key"}}
    )
    assert hosted.get_api_key() == (
        None if use_default_api_key is False else "overridden-hosted-key"
    )


def test_id(provider):
    assert provider.id() == "openai:test-model"


def test_id_with_custom_api_host():
    custom = OpenAiGenericProvider("test-model", {"config": {"apiHost": "custom.openai.com"}})
    assert custom.id() == "test-model"


def test_explicit_id_wins():
    assert OpenAiGenericProvider("test-model", {"id": "my-id"}).id() == "my-id"


def test_string_representation(provider):
    assert str(provider) == "[OpenAI Provider test-model]"


def test_requires_api_key_by_default(provider):
    assert provider.requires_api_key() is True


def test_api_key_requirement_can_be_disabled():
    custom = OpenAiGenericProvider("test-model", {"config": {"apiKeyRequired": False}})
    assert custom.requires_api_key() is False


async def test_generic_call_api_not_implemented(provider):
    with pytest.raises(NotImplementedError, match="Not implemented"):
        await provider.call_api("test prompt")


def test_missing_key_message_names_custom_envar():
    custom = OpenAiGenericProvider("m", {"config": {"apiKeyEnvar": "MY_KEY"}})
    assert custom.get_missing_api_key_error_message() == (
        "API key is not set. Set the MY_KEY environment variable or add `apiKey` to the "
        "provider config."
    )


@pytest.mark.usefixtures("no_openai_key")
async def test_chat_without_key_and_api_key_required_false_does_not_raise_key_error():
    chat = OpenAiChatCompletionProvider(
        "gpt-4o-mini", {"config": {"apiKeyRequired": False, "apiBaseUrl": "http://localhost:1"}}
    )
    assert chat.get_api_key() is None
    assert chat.requires_api_key() is False


# --- create_provider dispatch ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "cls", "model"),
    [
        ("openai:chat", OpenAiChatCompletionProvider, "gpt-6-sol"),
        ("openai:chat:gpt-4o-mini", OpenAiChatCompletionProvider, "gpt-4o-mini"),
        ("openai:chat:gpt-3.5-turbo", OpenAiChatCompletionProvider, "gpt-3.5-turbo"),
        ("openai:completion", OpenAiCompletionProvider, "gpt-3.5-turbo-instruct"),
        ("openai:completion:text-davinci-003", OpenAiCompletionProvider, "text-davinci-003"),
        ("openai:embedding", OpenAiEmbeddingProvider, "text-embedding-3-large"),
        ("openai:embeddings:text-embedding-3-small", OpenAiEmbeddingProvider,
         "text-embedding-3-small"),
        ("openai:moderation", OpenAiModerationProvider, "omni-moderation-latest"),
        ("openai:moderation:omni-moderation-2024-09-26", OpenAiModerationProvider,
         "omni-moderation-2024-09-26"),
        ("openai:gpt-4o-mini", OpenAiChatCompletionProvider, "gpt-4o-mini"),
        ("openai:babbage-002", OpenAiCompletionProvider, "babbage-002"),
    ],
)  # fmt: skip
def test_create_provider_routes(path, cls, model):
    built = create_provider(path, {})
    assert type(built) is cls
    assert built.model_name == model


def test_fine_tuned_chat_model_id():
    built = create_provider("openai:chat:ft:gpt-4o-mini:company-name::ID:", {})
    assert isinstance(built, OpenAiChatCompletionProvider)
    assert built.id() == "openai:ft:gpt-4o-mini:company-name::ID:"


@pytest.mark.parametrize(
    "model",
    [
        "gpt-35-turbo",
        "gpt-35-turbo-0125",
        "gpt-4.1",
        "gpt-5",
        "gpt-5.5",
        "gpt-5.5-2026-04-23",
        "gpt-5.6custom",
        "GPT-5.6",
        "custom-gpt-6-astra",
        "gpt-oss-120b",
        "gpt-daybreak-blue-latest-custom",
        "custom-gpt-daybreak-red-latest",
        "openai/gpt-daybreak-blue-latest",
    ],
)
def test_bare_models_keep_chat_default(model):
    built = create_provider(f"openai:{model}", {})
    assert type(built) is OpenAiChatCompletionProvider
    assert built.model_name == model


@pytest.mark.parametrize(
    "model", ["gpt-5.6", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol", "gpt-6.1-sol", "gpt-7-mini"]
)
def test_explicit_chat_endpoint_honored_for_responses_default_models(model):
    built = create_provider(f"openai:chat:{model}", {})
    assert type(built) is OpenAiChatCompletionProvider
    assert built.model_name == model


def test_chat_endpoint_with_config_model():
    built = create_provider("openai:chat", {"config": {"model": "gpt-5.6"}})
    assert type(built) is OpenAiChatCompletionProvider
    assert built.model_name == "gpt-5.6"


def test_daybreak_shorthand_uses_chat_through_gateway():
    built = create_provider(
        "openai:gpt-daybreak-blue-latest",
        {"config": {"apiBaseUrl": "https://gateway.example/v1"}},
    )
    assert type(built) is OpenAiChatCompletionProvider


def test_options_are_preserved():
    options = {
        "id": "support-model",
        "label": "Support",
        "env": {"GATEWAY_API_KEY": "test-key"},
        "config": {"apiBaseUrl": "https://gateway.example/v1", "apiKeyEnvar": "GATEWAY_API_KEY"},
    }
    built = create_provider("openai:chat:gpt-4o", options)
    assert built.id() == "support-model"
    assert built.label == "Support"
    assert built.env == options["env"]
    assert built.config == options["config"]
    assert built.get_api_key() == "test-key"


@pytest.mark.parametrize(
    ("path", "model"),
    [
        ("openai:responses", "gpt-6-sol"),
        ("openai:responses:gpt-5.4", "gpt-5.4"),
        ("openai:responses:gpt-5-mini", "gpt-5-mini"),
        ("openai:gpt-5.6", "gpt-5.6"),
        ("openai:gpt-6-sol", "gpt-6-sol"),
        ("openai:gpt-6.1-sol-2026-09-29", "gpt-6.1-sol-2026-09-29"),
        ("openai:gpt-7-mini", "gpt-7-mini"),
        ("openai:gpt-5.5-pro", "gpt-5.5-pro"),
        ("openai:gpt-daybreak-blue-latest", "gpt-daybreak-blue-latest"),
    ],
)
def test_responses_routes(path, model):
    built = create_provider(path, {})
    assert type(built) is OpenAiResponsesProvider
    assert built.model_name == model
    assert built.id() == f"openai:{model}"


def test_responses_route_uses_config_model():
    built = create_provider("openai:responses", {"config": {"model": "gpt-5.4-mini"}})
    assert type(built) is OpenAiResponsesProvider
    assert built.model_name == "gpt-5.4-mini"


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("openai:assistant:asst_123", "Assistants"),
        ("openai:image:gpt-image-1", "image"),
        ("openai:video:sora-2", "video"),
        ("openai:tts:gpt-4o-mini-tts", "text-to-speech"),
        ("openai:transcription:gpt-4o-transcribe", "transcription"),
        ("openai:realtime:gpt-4o-mini-realtime-preview-2024-12-17", "Realtime"),
        ("openai:gpt-4o-mini-realtime-preview-2024-12-17", "Realtime"),
        ("openai:live:gpt-live-1", "GPT-Live"),
        ("openai:gpt-live-1", "GPT-Live"),
        ("openai:agents:my-agent", "Agents"),
        ("openai:agents-api", "Agents/Codex"),
        ("openai:codex-sdk", "Agents/Codex"),
        ("openai:codex-app-server:gpt-5.4", "Agents/Codex"),
    ],
)
def test_unsupported_routes_raise_clear_error(path, kind):
    with pytest.raises(ValueError) as excinfo:
        create_provider(path, {})
    assert str(excinfo.value).startswith(
        f"OpenAI {kind} providers are not supported by promptphoo ({path})."
    )


@pytest.mark.parametrize(
    "path",
    ["openai:chatkit", "openai:chatkit:", "openai:chatkit:wf_test", "openai:chatkit:wf_test:3"],
)
def test_chatkit_removed(path):
    with pytest.raises(ValueError, match="The openai:chatkit provider has been removed"):
        create_provider(path, {})


@pytest.mark.parametrize(
    "path",
    [
        "openai:gpt-live-transcribe",
        "openai:gpt-live-transcribe-2026-09-01",
        "openai:live:gpt-live-transcribe",
        "openai:transcription:gpt-live-transcribe",
        "openai:realtime:gpt-live-transcribe",
        "openai:chat:gpt-live-transcribe",
        "openai:responses:gpt-live-transcribe",
    ],
)
def test_live_transcription_rejected(path):
    with pytest.raises(ValueError, match="transcription session"):
        create_provider(path, {})


def test_live_transcription_rejected_despite_model_override():
    with pytest.raises(ValueError, match="transcription session"):
        create_provider("openai:gpt-live-transcribe", {"config": {"model": "gpt-live-1"}})


def test_live_transcription_rejected_in_passthrough_model():
    with pytest.raises(ValueError, match="transcription session"):
        create_provider(
            "openai:chat:gpt-4o", {"config": {"passthrough": {"model": "gpt-live-transcribe"}}}
        )


@pytest.mark.parametrize(
    "path",
    [
        "openai:chat:gpt-live-1",
        "openai:chat:gpt-live-1-2026-09-01",
        "openai:responses:gpt-live-1",
        "openai:completion:gpt-live-1",
        "openai:embedding:gpt-live-1",
        "openai:tts:gpt-live-1",
        "openai:image:gpt-live-1",
        "openai:realtime:gpt-live-1",
    ],
)
def test_incompatible_live_routes_rejected(path):
    with pytest.raises(ValueError, match="openai:live:"):
        create_provider(path, {})


def test_retired_model_rejected_on_first_party_api():
    with pytest.raises(ValueError) as excinfo:
        create_provider("openai:moderation:text-moderation-latest", {})
    assert str(excinfo.value) == (
        "OpenAI model text-moderation-latest has been retired and is no longer available from "
        "OpenAI's first-party API. Use a current model or configure a custom OpenAI-compatible "
        "apiBaseUrl."
    )


def test_retired_model_allowed_through_custom_gateway():
    built = create_provider(
        "openai:moderation:text-moderation-latest",
        {"config": {"apiBaseUrl": "https://gateway.example/v1"}},
    )
    assert type(built) is OpenAiModerationProvider


def test_codex_only_model_rejected_on_chat_route():
    with pytest.raises(ValueError, match="only available through openai:codex-sdk"):
        create_provider("openai:chat:gpt-5.3-codex-spark", {})


def test_unknown_model_type_falls_back_to_chat(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        built = create_provider("openai:my-custom-model", {})
    assert type(built) is OpenAiChatCompletionProvider
    assert built.model_name == "my-custom-model"
