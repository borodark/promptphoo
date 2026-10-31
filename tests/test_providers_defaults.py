"""Ports test/providers/defaults.test.ts and the selection cases of
test/providers/default-env-scope.test.ts.

Mistral, xAI, Voyage, Azure Content Safety and Codex are not ported, so their upstream cases
are replaced by tests that the branch is skipped (see D33).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from promptphoo import cli_state
from promptphoo.providers import defaults as d
from promptphoo.providers.anthropic.defaults import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicLlmRubricProvider,
    get_anthropic_providers,
)
from promptphoo.providers.anthropic.messages import AnthropicMessagesProvider
from promptphoo.providers.azure.chat import AzureChatCompletionProvider
from promptphoo.providers.azure.embedding import AzureEmbeddingProvider
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.google.ai_studio import AIStudioChatProvider, AIStudioEmbeddingProvider
from promptphoo.providers.google.vertex import VertexChatProvider, VertexEmbeddingProvider
from promptphoo.providers.openai import defaults as openai_defaults
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider
from promptphoo.providers.openai.responses import OpenAiResponsesProvider

_CREDENTIAL_PREFIXES = (
    "OPENAI_", "ANTHROPIC_", "MISTRAL_", "XAI_", "GEMINI_", "GOOGLE_", "PALM_", "AZURE_",
    "VOYAGE_", "GITHUB_", "VERTEX_", "GCLOUD_",
)  # fmt: skip


class MockProvider(ApiProvider):
    def __init__(self, pid: str) -> None:
        super().__init__()
        self._pid = pid

    def default_id(self) -> str:
        return self._pid


class AdcProbe:
    def __init__(self) -> None:
        self.result = False
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.result


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in list(os.environ):
        if key.startswith(_CREDENTIAL_PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setattr(d, "_default_completion_provider", None)
    monkeypatch.setattr(d, "_default_embedding_provider", None)


@pytest.fixture(autouse=True)
def adc(monkeypatch):
    probe = AdcProbe()
    monkeypatch.setattr(d, "has_google_default_credentials", probe)
    return probe


def _completion(providers: dict[str, Any]) -> list[ApiProvider]:
    return [providers[k] for k in ("gradingJsonProvider", "gradingProvider",
                                   "suggestionsProvider", "synthesizeProvider")]  # fmt: skip


# --- OpenAI fallback and overrides --------------------------------------------------------


async def test_openai_defaults_without_credentials():
    providers = await d.get_default_providers()
    assert providers["gradingProvider"] is openai_defaults.DefaultGradingProvider
    assert providers["gradingJsonProvider"] is openai_defaults.DefaultGradingJsonProvider
    assert providers["synthesizeProvider"] is openai_defaults.DefaultGradingJsonProvider
    assert providers["suggestionsProvider"] is openai_defaults.DefaultSuggestionsProvider
    assert providers["embeddingProvider"] is openai_defaults.DefaultEmbeddingProvider
    assert providers["moderationProvider"] is openai_defaults.DefaultModerationProvider
    assert providers["webSearchProvider"] is openai_defaults.DefaultWebSearchProvider
    assert "llmRubricProvider" not in providers
    assert providers["gradingProvider"].id() == "openai:gpt-6-sol"
    assert providers["gradingJsonProvider"].config["response_format"] == {"type": "json_object"}
    assert isinstance(providers["webSearchProvider"], OpenAiResponsesProvider)
    assert providers["webSearchProvider"].config["tools"] == [{"type": "web_search_preview"}]


@pytest.mark.parametrize("source", ["process", "scoped"])
async def test_ignores_github_models_token(monkeypatch, source):
    baseline = await d.get_default_providers()
    if source == "process":
        monkeypatch.setenv("GITHUB_TOKEN", "fixture-github-token")
    providers = await d.get_default_providers(
        {"GITHUB_TOKEN": "fixture-github-token"} if source == "scoped" else None
    )
    assert [p.id() for p in _completion(providers)] == [p.id() for p in _completion(baseline)]


async def test_completion_override():
    await d.set_default_completion_providers(MockProvider("test-completion-provider"))
    providers = await d.get_default_providers()
    for key in d.COMPLETION_PROVIDERS:
        assert providers[key].id() == "test-completion-provider"
    assert providers["embeddingProvider"].id() != "test-completion-provider"


async def test_embedding_override():
    await d.set_default_embedding_providers(MockProvider("test-embedding-provider"))
    providers = await d.get_default_providers()
    assert providers["embeddingProvider"].id() == "test-embedding-provider"
    assert all(p.id() != "test-embedding-provider" for p in _completion(providers))


async def test_both_overrides():
    await d.set_default_completion_providers(MockProvider("test-completion-provider"))
    await d.set_default_embedding_providers(MockProvider("test-embedding-provider"))
    providers = await d.get_default_providers()
    assert all(p.id() == "test-completion-provider" for p in _completion(providers))
    assert providers["embeddingProvider"].id() == "test-embedding-provider"


@pytest.mark.parametrize("scoped", [False, True])
async def test_azure_content_safety_is_not_ported_so_openai_moderation_stays(monkeypatch, scoped):
    env = {"AZURE_CONTENT_SAFETY_ENDPOINT": "https://test-endpoint.com", "AZURE_API_KEY": "k"}
    if not scoped:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    providers = await d.get_default_providers(env if scoped else None)
    assert providers["moderationProvider"].id() == openai_defaults.DefaultModerationProvider.id()


async def test_azure_content_safety_branch_uses_the_ported_class(monkeypatch):
    built: list[tuple] = []

    class FakeModeration(MockProvider):
        def __init__(self, model_name, options):
            super().__init__("azure:moderation:" + model_name)
            built.append((model_name, options))

    monkeypatch.setattr(d, "_get_azure_moderation_provider_class", lambda: FakeModeration)
    env = {"AZURE_CONTENT_SAFETY_ENDPOINT": "https://test-endpoint.com"}
    providers = await d.get_default_providers(env)
    assert isinstance(providers["moderationProvider"], FakeModeration)
    assert built == [("text-content-safety", {"env": env})]
    masked = await d.get_default_providers({"AZURE_CONTENT_SAFETY_ENDPOINT": ""})
    assert not isinstance(masked["moderationProvider"], FakeModeration)


@pytest.mark.parametrize(
    "key", ["ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "PALM_API_KEY"]
)
async def test_masked_key_is_ignored(monkeypatch, key):
    fallback = {"MISTRAL_API_KEY": "scoped-key"}
    expected = await d.get_default_providers(fallback)
    monkeypatch.setenv(key, "ambient-key")
    providers = await d.get_default_providers({**fallback, key: ""})
    assert providers["gradingProvider"].id() == expected["gradingProvider"].id()


async def test_masked_service_principal_does_not_select_azure(monkeypatch):
    for key, value in {"AZURE_CLIENT_ID": "c", "AZURE_CLIENT_SECRET": "s", "AZURE_TENANT_ID": "t",
                       "AZURE_DEPLOYMENT_NAME": "chat",
                       "AZURE_OPENAI_DEPLOYMENT_NAME": "chat"}.items():  # fmt: skip
        monkeypatch.setenv(key, value)
    assert isinstance(
        (await d.get_default_providers())["gradingProvider"], AzureChatCompletionProvider
    )
    providers = await d.get_default_providers(
        {"AZURE_CLIENT_SECRET": "", "ANTHROPIC_API_KEY": "scoped-key"}
    )
    assert providers["gradingProvider"].id() == f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"


@pytest.mark.parametrize("key", ["MISTRAL_API_KEY", "XAI_API_KEY"])
async def test_unported_mistral_and_xai_branches_fall_through_to_openai(monkeypatch, key):
    monkeypatch.setenv(key, "test-key")
    providers = await d.get_default_providers()
    assert providers["gradingProvider"] is openai_defaults.DefaultGradingProvider


async def test_mistral_branch_uses_a_ported_bundle(monkeypatch):
    bundle = {k: MockProvider(f"mistral:{k}") for k in (
        "embeddingProvider", "gradingJsonProvider", "gradingProvider", "suggestionsProvider",
        "synthesizeProvider")}  # fmt: skip
    monkeypatch.setattr(d, "_get_mistral_providers", lambda: lambda env: bundle)
    providers = await d.get_default_providers({"MISTRAL_API_KEY": "k", "XAI_API_KEY": "x"})
    for key, provider in bundle.items():
        assert providers[key] is provider
    assert providers["moderationProvider"].id() == openai_defaults.DefaultModerationProvider.id()
    assert "webSearchProvider" not in providers


async def test_xai_branch_uses_a_ported_bundle(monkeypatch):
    grader = MockProvider("xai:grok-4.3")
    bundle = {"gradingJsonProvider": grader, "gradingProvider": grader,
              "suggestionsProvider": grader, "synthesizeProvider": grader,
              "webSearchProvider": MockProvider("xai:responses:grok-4.3")}  # fmt: skip
    monkeypatch.setattr(d, "_get_xai_providers", lambda: lambda env: bundle)
    monkeypatch.setenv("XAI_API_KEY", "k")
    providers = await d.get_default_providers()
    assert providers["gradingProvider"] is grader
    assert providers["webSearchProvider"].id() == "xai:responses:grok-4.3"
    assert providers["embeddingProvider"] is openai_defaults.DefaultEmbeddingProvider


async def test_codex_credentials_never_select_codex(monkeypatch):
    monkeypatch.setattr(d, "has_codex_default_credentials", lambda env=None: True)
    providers = await d.get_default_providers()
    assert providers["gradingProvider"] is openai_defaults.DefaultGradingProvider


async def test_openai_key_wins_over_everything(monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    providers = await d.get_default_providers()
    assert providers["gradingProvider"] is openai_defaults.DefaultGradingProvider


# --- Google ADC probing ---------------------------------------------------------------------


async def test_probes_adc_once(adc):
    await d.get_default_providers()
    assert adc.calls == 1


async def test_no_adc_probe_with_azure_embedding_deployment(monkeypatch, adc):
    env = {
        "AZURE_OPENAI_API_KEY": "azure-key",
        "AZURE_DEPLOYMENT_NAME": "azure-chat",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "azure-chat",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "azure-vectors",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    await d.get_default_providers()
    assert adc.calls == 0


async def test_vertex_defaults_from_adc(adc):
    adc.result = True
    providers = await d.get_default_providers()
    assert isinstance(providers["gradingProvider"], VertexChatProvider)
    assert providers["gradingProvider"].config["region"] == "global"
    assert isinstance(providers["embeddingProvider"], VertexEmbeddingProvider)
    assert providers["moderationProvider"].id() == openai_defaults.DefaultModerationProvider.id()


# --- embeddings with Azure chat defaults ----------------------------------------------------

AZURE_ENV = {
    "AZURE_OPENAI_API_KEY": "fixture-azure-key",
    "AZURE_DEPLOYMENT_NAME": "tenant-chat",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "tenant-chat",
}


@pytest.mark.parametrize("source", ["process", "scoped"])
async def test_explicit_azure_embedding_deployment(monkeypatch, adc, source):
    env = {**AZURE_ENV, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "tenant-vectors-v1"}
    if source == "process":
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    providers = await d.get_default_providers(env if source == "scoped" else None)
    assert isinstance(providers["embeddingProvider"], AzureEmbeddingProvider)
    assert providers["embeddingProvider"].deployment_name == "tenant-vectors-v1"
    assert isinstance(providers["gradingProvider"], AzureChatCompletionProvider)
    assert providers["gradingProvider"].deployment_name == "tenant-chat"
    assert "webSearchProvider" not in providers
    assert adc.calls == 0


async def test_scoped_azure_embedding_deployment_wins(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "process-vectors")
    providers = await d.get_default_providers(
        {**AZURE_ENV, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "scoped-vectors"}
    )
    assert providers["embeddingProvider"].deployment_name == "scoped-vectors"


@pytest.mark.parametrize("key", ["GEMINI_API_KEY", "GOOGLE_API_KEY", "PALM_API_KEY"])
async def test_google_key_supplies_azure_embeddings(adc, key):
    env = {**AZURE_ENV, "GOOGLE_GENAI_USE_VERTEXAI": "true", key: "fixture-embedding-key"}
    providers = await d.get_default_providers(env)
    embedding = providers["embeddingProvider"]
    assert isinstance(embedding, AIStudioEmbeddingProvider)
    assert embedding.id() == "google:embedding:gemini-embedding-001"
    assert embedding.env is env
    assert embedding.config["vertexai"] is False
    assert embedding.get_api_key() == "fixture-embedding-key"
    assert providers["gradingProvider"].deployment_name == "tenant-chat"
    assert adc.calls == 0


@pytest.mark.parametrize("key", ["MISTRAL_API_KEY", "VOYAGE_API_KEY"])
async def test_unported_embedding_families_are_skipped(adc, key):
    providers = await d.get_default_providers({**AZURE_ENV, key: "fixture-embedding-key"})
    assert providers["embeddingProvider"].id() == openai_defaults.DefaultEmbeddingProvider.id()
    assert adc.calls == 1


async def test_ported_voyage_embedding_is_used(monkeypatch):
    built: list[tuple] = []

    class FakeVoyage(MockProvider):
        def __init__(self, model, config, env):
            super().__init__(f"voyage:{model}")
            built.append((model, config, env))

    monkeypatch.setattr(d, "_get_voyage_embedding_provider_class", lambda: FakeVoyage)
    env = {**AZURE_ENV, "GEMINI_API_KEY": "", "VOYAGE_API_KEY": "scoped-key"}
    providers = await d.get_default_providers(env)
    assert providers["embeddingProvider"].id() == "voyage:voyage-3.5"
    assert built == [("voyage-3.5", {}, env)]
    masked = await d.get_default_providers({**AZURE_ENV, "VOYAGE_API_KEY": ""})
    assert masked["embeddingProvider"].id() == openai_defaults.DefaultEmbeddingProvider.id()


async def test_masked_azure_embedding_deployment_is_skipped(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "ambient-vectors")
    providers = await d.get_default_providers(
        {**AZURE_ENV, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": ""}
    )
    assert not isinstance(providers["embeddingProvider"], AzureEmbeddingProvider)


async def test_vertex_embeddings_beside_azure_chat_with_adc(adc):
    adc.result = True
    providers = await d.get_default_providers(AZURE_ENV)
    assert isinstance(providers["embeddingProvider"], VertexEmbeddingProvider)
    assert providers["embeddingProvider"].id() == "vertex:gemini-embedding-001"
    assert isinstance(providers["gradingProvider"], AzureChatCompletionProvider)
    assert adc.calls == 1


async def test_openai_embedding_fallback_beside_azure_chat(adc):
    providers = await d.get_default_providers(AZURE_ENV)
    embedding = providers["embeddingProvider"]
    assert isinstance(embedding, OpenAiEmbeddingProvider)
    assert adc.calls == 1
    response = await embedding.call_embedding_api("hello")
    assert "API key" in response["error"]
    assert "embedding" not in response


async def test_embedding_override_skips_adc_probe(adc):
    override = MockProvider("custom:existing-vector-space")
    await d.set_default_embedding_providers(override)
    providers = await d.get_default_providers(AZURE_ENV)
    assert providers["embeddingProvider"] is override
    assert isinstance(providers["gradingProvider"], AzureChatCompletionProvider)
    assert adc.calls == 0
    providers = await d.get_default_providers(
        {**AZURE_ENV, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "tenant-vectors"}
    )
    assert providers["embeddingProvider"] is override


async def test_openai_key_disables_azure(monkeypatch):
    for key, value in {**AZURE_ENV, "OPENAI_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    providers = await d.get_default_providers()
    assert providers["gradingProvider"] is openai_defaults.DefaultGradingProvider


async def test_azure_needs_both_deployment_names():
    env = {k: v for k, v in AZURE_ENV.items() if k != "AZURE_DEPLOYMENT_NAME"}
    providers = await d.get_default_providers(env)
    assert not isinstance(providers["gradingProvider"], AzureChatCompletionProvider)


async def test_explicit_azure_deployments_win_over_the_suite_env():
    suite_env = {
        "AZURE_OPENAI_DEPLOYMENT_NAME": "suite-chat",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "suite-embedding",
    }
    with cli_state.with_env(suite_env):
        providers = await d.get_default_providers(
            {
                "AZURE_OPENAI_API_KEY": "fake-key",
                "AZURE_OPENAI_API_HOST": "example.test",
                "AZURE_DEPLOYMENT_NAME": "provider-chat",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "provider-chat",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "provider-embedding",
            }
        )
    assert "provider-chat" in providers["gradingProvider"].id()
    assert "provider-embedding" in providers["embeddingProvider"].id()


# --- Google AI Studio -----------------------------------------------------------------------


@pytest.mark.parametrize("key", ["GEMINI_API_KEY", "GOOGLE_API_KEY", "PALM_API_KEY"])
@pytest.mark.parametrize("scoped", [False, True])
async def test_google_ai_studio_defaults(monkeypatch, key, scoped):
    if not scoped:
        monkeypatch.setenv(key, "test-key")
    providers = await d.get_default_providers({key: "test-key"} if scoped else None)
    for role in ("gradingProvider", "gradingJsonProvider", "llmRubricProvider",
                 "suggestionsProvider", "synthesizeProvider"):  # fmt: skip
        assert isinstance(providers[role], AIStudioChatProvider)
    assert isinstance(providers["embeddingProvider"], VertexEmbeddingProvider)
    assert providers["gradingJsonProvider"].config["generationConfig"] == {
        "response_mime_type": "application/json"
    }


@pytest.mark.parametrize("other", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
async def test_google_ai_studio_loses_to_openai_and_anthropic(monkeypatch, other):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv(other, "test-key")
    providers = await d.get_default_providers()
    assert not any(isinstance(p, AIStudioChatProvider) for p in _completion(providers))


async def test_google_ai_studio_beats_mistral(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    providers = await d.get_default_providers()
    assert all(isinstance(p, AIStudioChatProvider) for p in _completion(providers))


# --- Anthropic ------------------------------------------------------------------------------


async def test_anthropic_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    providers = await d.get_default_providers()
    assert isinstance(providers["gradingProvider"], AnthropicMessagesProvider)
    assert providers["gradingProvider"] is providers["gradingJsonProvider"]
    assert isinstance(providers["llmRubricProvider"], AnthropicLlmRubricProvider)
    assert providers["llmRubricProvider"].config["tool_choice"] == {
        "type": "tool", "name": "grade_output"
    }  # fmt: skip
    assert providers["llmRubricProvider"].config["showThinking"] is False
    assert providers["webSearchProvider"].config["tools"][0]["type"] == "web_search_20260209"
    assert providers["embeddingProvider"] is openai_defaults.DefaultEmbeddingProvider
    assert providers["gradingProvider"].api_key == "test-key"


async def test_masked_openai_and_azure_ambient_keys_select_anthropic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    monkeypatch.setenv("AZURE_API_KEY", "shell-azure-key")
    providers = await d.get_default_providers(
        {
            "OPENAI_API_KEY": "",
            "AZURE_API_KEY": "",
            "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
            "AZURE_DEPLOYMENT_NAME": "azure-chat",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "azure-chat",
        }
    )
    assert isinstance(providers["gradingProvider"], AnthropicMessagesProvider)
    assert providers["gradingProvider"].api_key == "synthetic-anthropic-key"


async def test_overlapping_async_selections_are_isolated():
    async def select(name: str) -> dict[str, Any]:
        with cli_state.with_env({"ANTHROPIC_API_KEY": f"{name}-key"}):
            await asyncio.sleep(0)
            return await d.get_default_providers()

    first, second = await asyncio.gather(select("first"), select("second"))
    assert first["gradingProvider"].api_key == "first-key"
    assert second["gradingProvider"].api_key == "second-key"


def test_empty_anthropic_key_stays_masked(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-key")
    with cli_state.with_env({"ANTHROPIC_API_KEY": "suite-key"}):
        bundle = get_anthropic_providers({"ANTHROPIC_API_KEY": ""})
    for role in ("gradingProvider", "llmRubricProvider", "webSearchProvider"):
        assert not bundle[role].api_key


async def test_llm_rubric_provider_unwraps_the_tool_call(monkeypatch):
    responses: list[dict[str, Any]] = []

    async def fake_call_api(self, prompt, context=None, options=None):
        return responses.pop(0)

    monkeypatch.setattr(AnthropicMessagesProvider, "call_api", fake_call_api)
    provider = AnthropicLlmRubricProvider(
        DEFAULT_ANTHROPIC_MODEL, {"env": {"ANTHROPIC_API_KEY": "k"}}
    )
    tool_call = '{"type":"tool_use","id":"t","name":"grade_output","input":{"pass":true,"score":1,"reason":"ok"}}'  # noqa: E501
    responses[:] = [{"output": tool_call}, {"output": {"x": 1}}, {"output": "not json"}]
    assert await provider.call_api("p") == {"output": {"pass": True, "score": 1, "reason": "ok"}}
    assert (await provider.call_api("p"))["error"] == (
        'Anthropic LLM rubric grader - malformed non-string output\n\n{"x":1}'
    )
    assert (await provider.call_api("p"))["error"].startswith(
        "Anthropic LLM rubric grader - invalid JSON: "
    )


# --- explicit env bundles -------------------------------------------------------------------


async def test_explicit_openai_key_is_retained_by_the_bundle():
    bundles = await asyncio.gather(
        *(
            d.get_default_providers({"OPENAI_API_KEY": f"{name}-key"})
            for name in ("first", "second")
        )
    )
    with cli_state.with_env({"OPENAI_API_KEY": "later-suite-key"}):
        for i, bundle in enumerate(bundles):
            for role in ("gradingProvider", "gradingJsonProvider", "embeddingProvider",
                         "suggestionsProvider"):  # fmt: skip
                assert bundle[role].get_api_key() == ("first-key" if i == 0 else "second-key")


def test_openai_defaults_are_preserved_while_binding_explicit_bundles():
    defaults = openai_defaults.get_openai_providers()
    bound = openai_defaults.get_openai_providers({"OPENAI_API_KEY": "fixture-key"})
    assert defaults["gradingProvider"] is openai_defaults.DefaultGradingProvider
    assert openai_defaults.get_openai_providers() is not defaults
    for role, provider in bound.items():
        default_provider = defaults[role]
        assert provider is not default_provider
        assert provider.id() == default_provider.id()
        assert provider.config == default_provider.config
        assert provider.get_api_key() == "fixture-key"
    assert bound["gradingJsonProvider"].config["response_format"] == {"type": "json_object"}
