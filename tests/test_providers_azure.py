"""Azure provider base, authentication, cost, errors, warnings and factory. Ports the
deterministic cases of test/providers/azure/{generic,auth-lifecycle,util,errors,warnings,index}
.test.ts."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
import respx
from azure_helpers import azure_env, fake_az, ok

from promptphoo.providers.azure import (
    AzureChatCompletionProvider,
    AzureCompletionProvider,
    AzureEmbeddingProvider,
    AzureGenericProvider,
    create_provider,
    generic,
    identity,
)
from promptphoo.providers.azure.errors import (
    format_content_filter_response,
    is_content_filter_error,
    is_rate_limit_error,
    is_service_error,
)
from promptphoo.providers.azure.tables import AZURE_MODELS
from promptphoo.providers.azure.util import calculate_azure_cost, throw_configuration_error
from promptphoo.providers.azure.warnings import maybe_emit_azure_openai_warning
from promptphoo.providers.openai import OpenAiCompletionProvider

pytestmark = pytest.mark.usefixtures(azure_env.__name__)
_ = fake_az

HOUR_MS = 3_600_000


# --- getApiBaseUrl (baseUrlTests.ts) and endpoint/env resolution -------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"apiBaseUrl": "https://custom.azure.com"}, "https://custom.azure.com"),
        ({"apiBaseUrl": "https://custom.azure.com/"}, "https://custom.azure.com"),
        ({"apiHost": "api.azure.com"}, "https://api.azure.com"),
        ({"apiHost": "https://api.azure.com"}, "https://api.azure.com"),
        ({"apiHost": "api.azure.com/"}, "https://api.azure.com"),
        ({}, None),
    ],
)
def test_get_api_base_url(config, expected):
    provider = AzureGenericProvider("test-deployment", {"config": config})
    assert provider.get_api_base_url() == expected


@pytest.mark.parametrize(
    ("envar", "value", "expected"),
    [
        ("AZURE_API_HOST", "env.azure.com", "https://env.azure.com"),
        ("AZURE_OPENAI_API_HOST", "https://openai-env.azure.com/", "https://openai-env.azure.com"),
        ("AZURE_API_BASE_URL", "https://base.example.com/", "https://base.example.com"),
        ("AZURE_OPENAI_BASE_URL", "http://proxy.local:8080", "http://proxy.local:8080"),
    ],
)
def test_endpoint_from_env(azure_env, envar, value, expected):
    azure_env.setenv(envar, value)
    assert AzureGenericProvider("d").get_api_base_url() == expected


def test_base_url_envar_wins_over_host_envar(azure_env):
    azure_env.setenv("AZURE_API_HOST", "host.azure.com")
    azure_env.setenv("AZURE_API_BASE_URL", "https://base.azure.com")
    assert AzureGenericProvider("d").get_api_base_url() == "https://base.azure.com"


def test_config_endpoint_wins_over_env(azure_env):
    azure_env.setenv("AZURE_API_BASE_URL", "https://env.azure.com")
    provider = AzureGenericProvider("d", {"config": {"apiHost": "cfg.azure.com"}})
    assert provider.get_api_base_url() == "https://cfg.azure.com"


def test_provider_env_endpoint_wins_over_process_env(azure_env):
    azure_env.setenv("AZURE_API_HOST", "process.azure.com")
    provider = AzureGenericProvider("d", {"env": {"AZURE_API_HOST": "scoped.azure.com"}})
    assert provider.get_api_base_url() == "https://scoped.azure.com"


def test_id_and_str():
    provider = AzureChatCompletionProvider("my-deployment")
    assert provider.id() == "azure:my-deployment"
    assert str(provider) == "[Azure Provider my-deployment]"
    assert AzureChatCompletionProvider("x", {"id": "custom"}).id() == "custom"


# --- API key precedence ------------------------------------------------------------------------


def test_api_key_precedence(azure_env):
    azure_env.setenv("AZURE_API_KEY", "env-key")
    azure_env.setenv("AZURE_OPENAI_API_KEY", "openai-env-key")
    assert AzureGenericProvider("d", {"config": {"apiKey": "cfg"}}).get_api_key() == "cfg"
    assert AzureGenericProvider("d").get_api_key() == "env-key"
    azure_env.delenv("AZURE_API_KEY")
    assert AzureGenericProvider("d").get_api_key() == "openai-env-key"


def test_api_key_envar_selects_its_own_namespace(azure_env):
    azure_env.setenv("AZURE_API_KEY", "env-key")
    azure_env.setenv("MY_AZURE_KEY", "custom-key")
    provider = AzureGenericProvider("d", {"config": {"apiKeyEnvar": "MY_AZURE_KEY"}})
    assert provider.get_api_key() == "custom-key"
    missing = AzureGenericProvider("d", {"config": {"apiKeyEnvar": "UNSET_AZURE_KEY"}})
    assert missing.get_api_key() is None


def test_provider_env_api_key():
    provider = AzureGenericProvider("d", {"env": {"AZURE_OPENAI_API_KEY": "scoped"}})
    assert provider.get_api_key() == "scoped"


def test_get_api_key_or_throw():
    with pytest.raises(ValueError, match="Azure API key must be set.") as info:
        AzureGenericProvider("d").get_api_key_or_throw()
    assert "https://www.promptfoo.dev/docs/providers/azure/" in str(info.value)


async def test_api_key_auth_headers():
    provider = AzureGenericProvider("d", {"config": {"apiKey": "k"}})
    await provider.ensure_initialized()
    assert provider.auth_headers == {"api-key": "k"}


# --- Entra ID: credential selection (generic.test.ts) -------------------------------------------


class FakeCredential:
    def __init__(self, *tokens: Any) -> None:
        self.tokens = list(tokens)
        self.scopes: list[str] = []

    async def get_token(self, scope: str) -> Any:
        self.scopes.append(scope)
        token = self.tokens.pop(0) if self.tokens else _token("t")
        if isinstance(token, BaseException):
            raise token
        if isinstance(token, asyncio.Future):
            return await token
        return token


def _token(value: str, expires_in_ms: float = HOUR_MS) -> dict[str, Any]:
    return {"token": value, "expiresOnTimestamp": identity.now_ms() + expires_in_ms}


@pytest.fixture
def credentials(monkeypatch):
    """Replace both credential classes; records constructor calls."""
    made: dict[str, list[Any]] = {"secret": [], "cli": []}
    shared = FakeCredential()

    def secret(*args: Any) -> FakeCredential:
        made["secret"].append(args)
        return shared

    def cli(*args: Any) -> FakeCredential:
        made["cli"].append(args)
        return shared

    monkeypatch.setattr(identity, "ClientSecretCredential", secret)
    monkeypatch.setattr(identity, "AzureCliCredential", cli)
    made["credential"] = [shared]
    return made


CLIENT_CREDENTIALS = {
    "azureClientId": "private-client",
    "azureClientSecret": "private-secret",
    "azureTenantId": "private-tenant",
}


async def test_uses_service_principal_when_all_fields_set(credentials, caplog):
    caplog.set_level(logging.DEBUG, logger=generic.__name__)
    provider = AzureGenericProvider("d", {"config": CLIENT_CREDENTIALS})
    await provider.ensure_initialized()
    assert credentials["secret"] == [
        (
            "private-tenant",
            "private-client",
            "private-secret",
            {"authorityHost": "https://login.microsoftonline.com"},
        )
    ]
    assert credentials["cli"] == []
    assert provider.auth_headers == {"Authorization": "Bearer t"}
    assert "[Azure] Using service principal credentials" in caplog.messages
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "private-" not in caplog.text


async def test_service_principal_from_env(azure_env, credentials):
    azure_env.setenv("AZURE_CLIENT_ID", "c")
    azure_env.setenv("AZURE_CLIENT_SECRET", "s")
    azure_env.setenv("AZURE_TENANT_ID", "t")
    azure_env.setenv("AZURE_AUTHORITY_HOST", "https://login.microsoftonline.us")
    await AzureGenericProvider("d").ensure_initialized()
    assert credentials["secret"] == [
        ("t", "c", "s", {"authorityHost": "https://login.microsoftonline.us"})
    ]


async def test_shares_one_credential_per_provider(credentials):
    for deployment in ("first", "second"):
        provider = AzureGenericProvider(deployment, {"config": CLIENT_CREDENTIALS})
        results = await asyncio.gather(
            provider.ensure_initialized(),
            provider.get_azure_token_credential(),
            provider.get_azure_token_credential(),
        )
        assert results[1] is results[2]
    assert len(credentials["secret"]) == 2
    assert credentials["cli"] == []


async def test_warns_and_names_missing_fields_before_cli_fallback(credentials, caplog):
    provider = AzureGenericProvider(
        "d", {"config": {"azureClientId": "client", "azureTenantId": "tenant"}}
    )
    await provider.ensure_initialized()
    assert credentials["secret"] == []
    assert len(credentials["cli"]) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "azureClientSecret" in message
    assert "azureClientId" not in message
    assert "azureTenantId" not in message
    assert "Falling back to Azure CLI" in message


async def test_warns_once_per_instance(credentials, caplog):
    for count in (1, 2):
        provider = AzureGenericProvider("d", {"config": {"azureClientSecret": "private-secret"}})
        await asyncio.gather(
            provider.ensure_initialized(),
            provider.get_azure_token_credential(),
            provider.get_azure_token_credential(),
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == count
    assert warnings[-1].missing == [  # type: ignore[attr-defined]
        "azureClientId (AZURE_CLIENT_ID)",
        "azureTenantId (AZURE_TENANT_ID)",
    ]
    assert "private-secret" not in caplog.text


async def test_api_key_wins_over_partial_service_principal(credentials, caplog):
    provider = AzureGenericProvider(
        "d", {"config": {"apiKey": "configured-key", "azureClientId": "private-client"}}
    )
    await provider.ensure_initialized()
    assert provider.auth_headers == {"api-key": "configured-key"}
    assert credentials["cli"] == [] and credentials["secret"] == []
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


async def test_falls_back_to_cli_silently(credentials, caplog):
    await AzureGenericProvider("d").ensure_initialized()
    assert credentials["secret"] == []
    assert len(credentials["cli"]) == 1
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


async def test_retries_credential_construction_after_failure(monkeypatch):
    calls = []

    def cli() -> FakeCredential:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("temporary credential construction failure")
        return FakeCredential(_token("initial"))

    monkeypatch.setattr(identity, "AzureCliCredential", cli)
    provider = AzureGenericProvider("deployment")
    with pytest.raises(generic.AzureAuthenticationError, match="Azure Authentication failed"):
        await provider.ensure_initialized()
    assert provider.auth_headers is None
    await provider.ensure_initialized()
    assert provider.auth_headers == {"Authorization": "Bearer initial"}
    assert len(calls) == 2


def test_missing_auth_error_text():
    assert generic.AZURE_AUTHENTICATION_FAILED == (
        "Azure Authentication failed. \n"
        "Please choose one of the following options:\n"
        "  1. Set an API key via the AZURE_API_KEY environment variable.\n"
        "  2. Provide client credentials (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID).\n"
        "  3. Authenticate with Azure CLI using az login.\n"
        "    "
    )


async def test_no_credentials_raises_upstream_error(fake_az):
    with pytest.raises(generic.AzureAuthenticationError) as info:
        await AzureGenericProvider("d").ensure_initialized()
    assert str(info.value) == generic.AZURE_AUTHENTICATION_FAILED
    assert isinstance(info.value.__cause__, identity.CredentialUnavailableError)
    assert identity.AZURE_CLI_NOT_LOGGED_IN in str(info.value.__cause__)
    assert len(fake_az.calls) == 1


# --- Entra ID: token lifecycle (auth-lifecycle.test.ts) ----------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1_767_225_600_000.0  # 2026-01-01T00:00:00Z

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(identity, "now_ms", c)
    return c


@pytest.fixture
def cli_credential(monkeypatch, clock):
    credential = FakeCredential()
    made: list[Any] = []

    def cli() -> FakeCredential:
        made.append(1)
        return credential

    monkeypatch.setattr(identity, "AzureCliCredential", cli)
    monkeypatch.setattr(
        identity, "ClientSecretCredential", lambda *a: (made.append(a), credential)[1]
    )
    credential.made = made  # type: ignore[attr-defined]
    return credential


@pytest.mark.parametrize("auth", ["cli", "service principal"])
async def test_shares_initial_authentication_and_refresh(cli_credential, clock, auth):
    loop = asyncio.get_running_loop()
    initial = loop.create_future()
    cli_credential.tokens = [initial]
    config = {} if auth == "cli" else CLIENT_CREDENTIALS
    provider = AzureGenericProvider("deployment", {"config": config})
    first = [asyncio.ensure_future(provider.ensure_initialized()) for _ in range(20)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(cli_credential.scopes) == 1
    initial.set_result(_token("initial"))
    await asyncio.gather(*first)

    clock.now += 3_300_000
    refresh = loop.create_future()
    cli_credential.tokens = [refresh]
    nxt = [asyncio.ensure_future(provider.ensure_initialized()) for _ in range(20)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(cli_credential.scopes) == 2
    refresh.set_result(_token("refreshed"))
    await asyncio.gather(*nxt)

    assert provider.auth_headers == {"Authorization": "Bearer refreshed"}
    assert len(cli_credential.made) == 1
    await provider.ensure_initialized()
    assert len(cli_credential.scopes) == 2


async def test_shares_initial_failure_and_retries_same_credential(cli_credential):
    initial = asyncio.get_running_loop().create_future()
    cli_credential.tokens = [initial, _token("initial")]
    provider = AzureGenericProvider("deployment")
    requests = [asyncio.ensure_future(provider.ensure_initialized()) for _ in range(20)]
    await asyncio.sleep(0)
    initial.set_exception(RuntimeError("temporary authentication failure"))
    outcomes = await asyncio.gather(*requests, return_exceptions=True)
    assert all(isinstance(o, generic.AzureAuthenticationError) for o in outcomes)
    assert len(cli_credential.scopes) == 1
    assert provider.auth_headers is None

    await asyncio.gather(*(provider.ensure_initialized() for _ in range(20)))
    assert provider.auth_headers == {"Authorization": "Bearer initial"}
    assert len(cli_credential.scopes) == 2
    assert len(cli_credential.made) == 1


@pytest.mark.parametrize("elapsed", [3_300_000, 3_600_000])
async def test_failed_refresh_rejects_then_retries(cli_credential, clock, elapsed):
    provider = AzureGenericProvider("deployment")
    await provider.ensure_initialized()
    clock.now += elapsed
    cli_credential.tokens = [RuntimeError("temporary refresh failure"), _token("recovered")]
    outcomes = await asyncio.gather(
        *(provider.ensure_initialized() for _ in range(20)), return_exceptions=True
    )
    assert all(isinstance(o, generic.AzureAuthenticationError) for o in outcomes)
    assert len({id(o) for o in outcomes}) == 1
    assert len(cli_credential.scopes) == 2

    await asyncio.gather(*(provider.ensure_initialized() for _ in range(20)))
    assert provider.auth_headers == {"Authorization": "Bearer recovered"}
    assert len(cli_credential.scopes) == 3
    assert len(cli_credential.made) == 1


async def test_does_not_refetch_still_valid_token(cli_credential, clock):
    provider = AzureGenericProvider("deployment")
    await provider.ensure_initialized()
    clock.now += 3_299_999
    await asyncio.gather(*(provider.ensure_initialized() for _ in range(20)))
    assert len(cli_credential.scopes) == 1


async def test_keeps_tokens_with_unknown_expiry(cli_credential, clock):
    cli_credential.tokens = [{"token": "unknown-expiry"}]
    provider = AzureGenericProvider("deployment")
    await provider.ensure_initialized()
    clock.now += 86_400_000
    await provider.ensure_initialized()
    assert provider.auth_headers == {"Authorization": "Bearer unknown-expiry"}
    assert len(cli_credential.scopes) == 1


async def test_null_token_is_not_cached(cli_credential):
    cli_credential.tokens = [None]
    provider = AzureGenericProvider("deployment")
    with pytest.raises(generic.AzureAuthenticationError):
        await provider.ensure_initialized()
    await provider.ensure_initialized()
    assert provider.auth_headers == {"Authorization": "Bearer t"}
    assert len(cli_credential.scopes) == 2


async def test_api_key_never_refreshes(cli_credential, clock):
    provider = AzureGenericProvider(
        "deployment", {"config": {"apiKey": "test-key", "azureClientId": "test-client"}}
    )
    await asyncio.gather(*(provider.ensure_initialized() for _ in range(20)))
    clock.now += 86_400_000
    await provider.ensure_initialized()
    assert provider.auth_headers == {"api-key": "test-key"}
    assert cli_credential.made == [] and cli_credential.scopes == []


async def test_token_scope_precedence(cli_credential, azure_env):
    azure_env.setenv("AZURE_TOKEN_SCOPE", "https://process.example/.default")
    first = AzureGenericProvider(
        "first",
        {
            "config": {"azureTokenScope": "https://first.example/.default"},
            "env": {"AZURE_TOKEN_SCOPE": "https://ignored.example/.default"},
        },
    )
    await first.ensure_initialized()
    second = AzureGenericProvider(
        "second", {"env": {"AZURE_TOKEN_SCOPE": "https://second.example/.default"}}
    )
    await second.ensure_initialized()
    third = AzureGenericProvider("third")
    await third.ensure_initialized()
    assert cli_credential.scopes == [
        "https://first.example/.default",
        "https://second.example/.default",
        "https://process.example/.default",
    ]


async def test_default_token_scope(cli_credential):
    await AzureGenericProvider("d").ensure_initialized()
    assert cli_credential.scopes == ["https://cognitiveservices.azure.com/.default"]


# --- identity: client-secret over HTTP and the Azure CLI subprocess ----------------------------


TOKEN_URL = "https://login.microsoftonline.com/my-tenant/oauth2/v2.0/token"


@respx.mock
async def test_client_secret_credential_posts_client_credentials_grant(clock):
    route = respx.post(TOKEN_URL).mock(
        return_value=ok({"token_type": "Bearer", "expires_in": 3599, "access_token": "aad"})
    )
    credential = identity.ClientSecretCredential("my-tenant", "my-client", "my-secret")
    token = await credential.get_token("https://cognitiveservices.azure.com/.default")
    assert token == {"token": "aad", "expiresOnTimestamp": clock.now + 3_599_000}
    request = route.calls[0].request
    assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert parse_qs(request.content.decode()) == {
        "client_id": ["my-client"],
        "scope": ["https://cognitiveservices.azure.com/.default"],
        "grant_type": ["client_credentials"],
        "client_secret": ["my-secret"],
    }
    # Cached until it is within five minutes of expiry.
    clock.now += 3_599_000 - 5 * 60 * 1000 - 1
    assert (await credential.get_token("https://cognitiveservices.azure.com/.default")) is token
    assert route.call_count == 1
    clock.now += 1
    await credential.get_token("https://cognitiveservices.azure.com/.default")
    assert route.call_count == 2


@respx.mock
async def test_client_secret_credential_authority_host_and_error():
    route = respx.post("https://login.microsoftonline.us/t/oauth2/v2.0/token").mock(
        return_value=ok(
            {"error": "invalid_client", "error_description": "AADSTS7000215: bad secret"}, 401
        )
    )
    credential = identity.ClientSecretCredential(
        "t", "c", "s", {"authorityHost": "https://login.microsoftonline.us/"}
    )
    with pytest.raises(identity.AuthenticationError, match="AADSTS7000215: bad secret"):
        await credential.get_token("https://cognitiveservices.azure.com/.default")
    assert route.called


def test_client_secret_credential_rejects_bad_tenant():
    with pytest.raises(ValueError, match="Invalid tenant id"):
        identity.ClientSecretCredential("bad/tenant?x", "c", "s")


@respx.mock
async def test_entra_end_to_end_bearer_header(azure_env):
    azure_env.setenv("AZURE_CLIENT_ID", "c")
    azure_env.setenv("AZURE_CLIENT_SECRET", "s")
    azure_env.setenv("AZURE_TENANT_ID", "tenant")
    respx.post("https://login.microsoftonline.com/tenant/oauth2/v2.0/token").mock(
        return_value=ok({"expires_in": 3600, "access_token": "aad-token"})
    )
    chat = respx.post("https://test.azure.com/openai/deployments/dep/chat/completions").mock(
        return_value=ok({"choices": [{"message": {"content": "hi"}}], "usage": {}})
    )
    provider = AzureChatCompletionProvider(
        "dep", {"config": {"apiHost": "test.azure.com", "maxRetries": 0}}
    )
    result = await provider.call_api("hello")
    assert result["output"] == "hi"
    assert chat.calls[0].request.headers["Authorization"] == "Bearer aad-token"
    assert "api-key" not in chat.calls[0].request.headers


async def test_azure_cli_credential_runs_az(fake_az, clock):
    fake_az.reply("cli-token", expires_on=1_767_229_200)
    token = await identity.AzureCliCredential().get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    assert token == {"token": "cli-token", "expiresOnTimestamp": 1_767_229_200_000}
    assert fake_az.calls == [
        [
            "account",
            "get-access-token",
            "--output",
            "json",
            "--resource",
            "https://cognitiveservices.azure.com",
        ]
    ]


async def test_azure_cli_credential_parses_expires_on_string(fake_az):
    fake_az.replies.append(
        (0, '{"accessToken": "x", "expiresOn": "2026-01-01 01:00:00.000000"}', "")
    )
    token = await identity.AzureCliCredential().get_token("api://x/.default")
    assert token is not None and token["token"] == "x"
    assert isinstance(token["expiresOnTimestamp"], float)


async def test_azure_cli_not_installed(fake_az):
    fake_az.replies.append(FileNotFoundError("az"))
    with pytest.raises(identity.CredentialUnavailableError, match="Azure CLI could not be found"):
        await identity.AzureCliCredential().get_token("https://x/.default")
    fake_az.replies.append((127, "", "/bin/sh: 1: az: not found"))
    with pytest.raises(identity.CredentialUnavailableError, match="Azure CLI could not be found"):
        await identity.AzureCliCredential().get_token("https://x/.default")


async def test_azure_cli_not_logged_in(fake_az):
    with pytest.raises(identity.CredentialUnavailableError, match="az login"):
        await identity.AzureCliCredential().get_token("https://x/.default")


async def test_azure_cli_rejects_bad_scope(fake_az):
    with pytest.raises(ValueError, match="Invalid scope"):
        await identity.AzureCliCredential().get_token("https://x/.default; rm -rf /")
    assert fake_az.calls == []


async def test_entra_cli_end_to_end(fake_az):
    fake_az.reply("cli-token", expires_on=4_102_444_800)
    provider = AzureGenericProvider("d")
    await provider.ensure_initialized()
    assert provider.auth_headers == {"Authorization": "Bearer cli-token"}
    assert provider.auth_token_expires_on_timestamp == 4_102_444_800_000


# --- util.test.ts --------------------------------------------------------------------------------


def test_throw_configuration_error():
    with pytest.raises(ValueError) as info:
        throw_configuration_error("Test error message")
    assert str(info.value) == (
        "Test error message\n\nSee https://www.promptfoo.dev/docs/providers/azure/ to learn more "
        "about Azure configuration."
    )


def approx(value: float) -> Any:
    return pytest.approx(value, abs=1e-12)


def test_cost_long_context_pricing():
    assert calculate_azure_cost("gpt-5.4", {}, 272_000, 1_000) == approx(
        (272_000 * 2.5 + 1_000 * 15) / 1e6
    )
    assert calculate_azure_cost("gpt-5.4", {}, 272_001, 1_000) == approx(
        (272_001 * 5 + 1_000 * 22.5) / 1e6
    )
    assert calculate_azure_cost("gpt-5.4-pro", {}, 272_001, 1_000) == approx(
        (272_001 * 60 + 1_000 * 270) / 1e6
    )


@pytest.mark.parametrize(
    ("model_id", "inp", "out", "long_in", "long_out"),
    [
        ("gpt-5.6", 5, 30, 10, 45),
        ("gpt-5.6-sol", 5, 30, 10, 45),
        ("gpt-5.6-terra", 2.5, 15, 5, 22.5),
        ("gpt-5.6-luna", 1, 6, 2, 9),
        ("gpt-5.5-2026-04-24", 5, 30, 10, 45),
    ],
)
def test_cost_standard_and_long_context(model_id, inp, out, long_in, long_out):
    assert calculate_azure_cost(model_id, {}, 272_000, 1_000) == approx(
        (272_000 * inp + 1_000 * out) / 1e6
    )
    assert calculate_azure_cost(model_id, {}, 272_001, 1_000) == approx(
        (272_001 * long_in + 1_000 * long_out) / 1e6
    )


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.5-2026-04-24"])
def test_cost_short_context_priority_meters(model):
    config = {"passthrough": {"service_tier": "priority"}}
    assert calculate_azure_cost(model, config, 1_000, 0) == approx(0.0125)
    assert calculate_azure_cost(model, config, 1_000, 0, 1_000) == approx(0.00125)
    assert calculate_azure_cost(model, config, 0, 1_000) == approx(0.075)
    assert calculate_azure_cost(model, config, 272_001, 1_000, 1_000) == approx(
        (2 * (271_001 * 10 + 1_000 * 1 + 1_000 * 45)) / 1e6
    )


def test_cost_priority_only_from_passthrough():
    assert calculate_azure_cost(
        "gpt-5.4", {"service_tier": "priority"}, 272_001, 1_000, 1_000
    ) == approx((271_001 * 5 + 1_000 * 0.5 + 1_000 * 22.5) / 1e6)
    assert calculate_azure_cost(
        "gpt-5.1-codex-mini-2025-11-13",
        {"passthrough": {"service_tier": "priority"}},
        2_000,
        1_000,
        500,
    ) == approx((1.8 * (1_500 * 0.25 + 500 * 0.025 + 1_000 * 2)) / 1e6)


def test_cost_modalities():
    assert calculate_azure_cost(
        "gpt-realtime-mini-2025-10-06", {}, 1_000, 0, 500, 800, 0, 0, 500
    ) == approx((200 * 0.6 + 300 * 10 + 500 * 0.3) / 1e6)
    assert calculate_azure_cost("gpt-realtime", {}, 1_000, 0, 1_000, 1_000) == approx(
        (1_000 * 0.4) / 1e6
    )
    assert calculate_azure_cost(
        "gpt-realtime-mini-2025-10-06", {}, 1_000, 0, 500, 0, 0, 800
    ) == approx((200 * 0.06 + 500 * 0.8 + 300 * 0.08) / 1e6)


def test_cost_claude_and_mai():
    assert calculate_azure_cost("claude-fable-5", {}, 1000, 500) == pytest.approx(0.035, abs=1e-6)
    assert calculate_azure_cost("claude-fable-5-1", {}, 1000, 500, 500) == pytest.approx(
        0.030125, abs=1e-8
    )
    assert calculate_azure_cost("claude-sonnet-5", {}, 1000, 500, 500) == approx(
        (500 * 2 + 500 * 0.2 + 500 * 10) / 1e6
    )
    assert calculate_azure_cost("MAI-Image-2.5", {}, 0, 1000) == pytest.approx(0.033, abs=1e-6)


@pytest.mark.parametrize(
    "model_id", ["gpt-5.5-2026-04-23", "gpt-5.5-pro", "gpt-5.2-pro", "gpt-5-chat-latest"]
)
def test_cost_unpublished_ids(model_id):
    assert not any(m["id"] == model_id for m in AZURE_MODELS)
    assert calculate_azure_cost(model_id, {}, 100, 50) is None


def test_cost_edge_cases():
    assert calculate_azure_cost("unknown-model", {}, 100, 50) is None
    assert calculate_azure_cost("gpt-4", {}, None, None) is None
    assert calculate_azure_cost("gpt-4", {}, 0, 0) == 0
    assert isinstance(calculate_azure_cost("gpt-4", {}, 100, 50), float)
    assert calculate_azure_cost("gpt-4", {}, 100, None) is None
    assert calculate_azure_cost("gpt-4", {}, None, 50) is None


def test_every_model_has_positive_cost_and_unique_id():
    broken = [
        m["id"]
        for m in AZURE_MODELS
        if not (isinstance(c := calculate_azure_cost(m["id"], {}, 1000, 1000), float) and c > 0)
    ]
    assert broken == []
    ids = [m["id"] for m in AZURE_MODELS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("model_id", "expected"), [("gpt-5.4", 0.0175), ("gpt-4.1", 0.01), ("gpt-4o", 0.0125)]
)
def test_cost_spot_checks(model_id, expected):
    assert calculate_azure_cost(model_id, {}, 1000, 1000) == pytest.approx(expected, abs=1e-9)


# --- errors.test.ts ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "content_filter triggered",
        "Content filter blocked",
        "Response was filtered due to policy",
        "content filtering engaged",
        "inappropriate content detected",
        "violates safety guidelines",
        "Guardrail triggered",
    ],
)
def test_is_content_filter_error(message):
    assert is_content_filter_error(message)


def test_error_classifiers():
    assert not is_content_filter_error("network timeout")
    assert is_rate_limit_error("Rate limit reached")
    assert is_rate_limit_error("HTTP 429")
    assert not is_rate_limit_error("bad request")
    assert is_service_error("Service unavailable")
    assert is_service_error("Sorry, something went wrong")
    assert not is_service_error("bad request")


def test_format_content_filter_response():
    assert format_content_filter_response("The prompt was filtered")["guardrails"] == {
        "flagged": True,
        "flaggedInput": True,
        "flaggedOutput": False,
    }
    response = format_content_filter_response("The response was filtered")
    assert response["guardrails"]["flaggedOutput"] is True
    assert response["output"].startswith("The generated content was filtered")
    both = format_content_filter_response("input and output both filtered")
    assert both["guardrails"]["flaggedInput"] is True
    assert both["guardrails"]["flaggedOutput"] is False


# --- warnings.test.ts ----------------------------------------------------------------------------


def _suite(*providers: Any, default_test: Any = None) -> dict[str, Any]:
    return {"providers": list(providers), "defaultTest": default_test or {}, "prompts": []}


def test_warning_cases():
    azure = AzureCompletionProvider("foo", {"config": {"apiHost": "test.azure.com"}})
    graded = [{"assert": [{"type": "llm-rubric", "value": "foo bar"}]}]
    assert not maybe_emit_azure_openai_warning(_suite(OpenAiCompletionProvider("foo")), graded)
    assert not maybe_emit_azure_openai_warning(_suite(azure), [{"assert": [{"type": "equals"}]}])
    assert maybe_emit_azure_openai_warning(_suite(azure), graded)
    assert not maybe_emit_azure_openai_warning(
        _suite(azure, OpenAiCompletionProvider("foo")), graded
    )
    assert not maybe_emit_azure_openai_warning(
        _suite(azure, default_test={"options": {"provider": "azure:chat:x"}}), graded
    )
    assert not maybe_emit_azure_openai_warning(
        _suite(azure), [{"assert": [{"type": "llm-rubric", "provider": "azure:chat:x"}]}]
    )
    assert not maybe_emit_azure_openai_warning(
        _suite(azure),
        [{"assert": [{"type": "llm-rubric"}], "options": {"provider": "azure:chat:x"}}],
    )


# --- registry ------------------------------------------------------------------------------------


def test_factory_routes():
    assert isinstance(create_provider("azure:chat:my-dep", {}), AzureChatCompletionProvider)
    assert isinstance(create_provider("azureopenai:chat:my-dep", {}), AzureChatCompletionProvider)
    assert create_provider("azure:chat:my:dep", {}).deployment_name == "my:dep"
    assert isinstance(create_provider("azure:completion:d", {}), AzureCompletionProvider)
    embedding = create_provider("azure:embeddings", {})
    assert isinstance(embedding, AzureEmbeddingProvider)
    assert embedding.deployment_name == "text-embedding-ada-002"
    assert create_provider("azure:embedding:emb", {}).deployment_name == "emb"


def test_factory_via_registry():
    from promptphoo.providers.registry import find_factory

    factory = find_factory("azure:chat:x")
    assert factory is not None
    assert isinstance(factory("azure:chat:x", {}), AzureChatCompletionProvider)


@pytest.mark.parametrize(
    ("path", "error", "match"),
    [
        ("azure:chat", ValueError, r"Azure chat provider requires a deployment name\. "
         r"Use azure:chat:<deployment>\."),
        ("azure:completion", ValueError, "Azure completion provider requires"),
        ("azure:assistant:asst_1", ValueError, "not supported by promptphoo"),
        ("azure:foundry-agent:a", ValueError, "not supported by promptphoo"),
        ("azure:image:dalle", ValueError, "not supported by promptphoo"),
        ("azureopenai:image:x", ValueError, "azureopenai:image is not supported"),
        ("azure:video:sora", ValueError, "not supported by promptphoo"),
        ("azure:realtime:x", ValueError, "not supported by promptphoo"),
        ("azure:moderation", ValueError, "not supported by promptphoo"),
        ("azureopenai:moderation", ValueError, "Azure OpenAI does not support moderation"),
        ("azure:decisions:x", ValueError, "openai:decisions"),
        ("azure:live:x", ValueError, "openai:live"),
        ("azure:bogus:x", ValueError, "Unknown Azure model type: bogus"),
    ],
)  # fmt: skip
def test_factory_errors(path, error, match):
    with pytest.raises(error, match=match):
        create_provider(path, {})


def test_mcp_not_supported():
    with pytest.raises(ValueError, match="MCP"):
        AzureChatCompletionProvider("d", {"config": {"mcp": {"enabled": True}}})


@respx.mock
async def test_embedding_network_error_does_not_retry():
    route = respx.post(url__startswith="https://test.azure.com/").mock(
        side_effect=httpx.ConnectError("boom")
    )
    provider = AzureEmbeddingProvider(
        "emb", {"config": {"apiHost": "test.azure.com", "apiKey": "k", "maxRetries": 0}}
    )
    result = await provider.call_embedding_api("hello")
    assert result["error"].startswith("API call error:")
    assert route.call_count == 1
