"""Cloudflare AI, Cloudflare AI Gateway, Databricks, MLflow Gateway, Portkey and Snowflake
providers. Ports deterministic cases from test/providers/{cloudflare-ai,cloudflare-gateway,
databricks,mlflow-gateway,portkey,snowflake}.test.ts, with HTTP mocked by respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from openai_helpers import chat_response, ok, openai_env, sent_json

from promptphoo.providers.compatible import (
    cloudflare_ai,
    cloudflare_gateway,
    databricks,
    mlflow_gateway,
    portkey,
    snowflake,
)
from promptphoo.providers.compatible.cloudflare_ai import (
    CloudflareAiChatCompletionProvider,
    CloudflareAiCompletionProvider,
    CloudflareAiEmbeddingProvider,
    create_cloudflare_ai_provider,
)
from promptphoo.providers.compatible.cloudflare_gateway import (
    CloudflareGatewayAnthropicProvider,
    CloudflareGatewayOpenAiProvider,
    create_cloudflare_gateway_provider,
)
from promptphoo.providers.compatible.databricks import DatabricksMosaicAiChatCompletionProvider
from promptphoo.providers.compatible.mlflow_gateway import MlflowGatewayChatCompletionProvider
from promptphoo.providers.compatible.portkey import (
    PortkeyChatCompletionProvider,
    get_portkey_headers,
    to_kebab_case,
)
from promptphoo.providers.compatible.snowflake import SnowflakeCortexProvider

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

VENDOR_ENV = [
    "CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID", "CF_AIG_TOKEN",
    "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS",
    "GROQ_API_KEY", "DATABRICKS_WORKSPACE_URL", "DATABRICKS_TOKEN", "MLFLOW_GATEWAY_URL",
    "MLFLOW_GATEWAY_API_KEY", "PORTKEY_API_KEY", "PORTKEY_API_BASE_URL",
    "SNOWFLAKE_ACCOUNT_IDENTIFIER", "SNOWFLAKE_API_KEY",
]  # fmt: skip


@pytest.fixture(autouse=True)
def vendor_env(openai_env):
    for name in VENDOR_ENV:
        openai_env.delenv(name, raising=False)
    return openai_env


TOKEN_USAGE = {"total_tokens": 50, "prompt_tokens": 25, "completion_tokens": 25}
TOKEN_USAGE_RESULT = {"total": 50, "prompt": 25, "completion": 25, "numRequests": 1}
CF_MODEL = "@cf/meta/llama-2-7b-chat-fp16"
CF_MIN = {"accountId": "testAccountId", "apiKey": "testApiKey"}
CF_BASE = "https://api.cloudflare.com/client/v4/accounts/testAccountId/ai/v1"


def _chat(content: str = "Test text output") -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": TOKEN_USAGE}


# --- cloudflare-ai -------------------------------------------------------------------------------


@respx.mock
async def test_cloudflare_ai_chat():
    route = respx.post(f"{CF_BASE}/chat/completions").mock(return_value=ok(_chat()))
    provider = CloudflareAiChatCompletionProvider(CF_MODEL, {"config": CF_MIN})
    result = await provider.call_api("Test chat prompt")
    assert route.call_count == 1
    assert result["output"] == "Test text output"
    assert result["tokenUsage"] == TOKEN_USAGE_RESULT


@respx.mock
async def test_cloudflare_ai_passthrough_and_local_options():
    route = respx.post(url__regex=r".*/chat/completions").mock(return_value=ok(_chat()))
    provider = CloudflareAiChatCompletionProvider(
        CF_MODEL,
        {
            "config": {
                "accountId": "test-account",
                "apiKey": "test-key",
                "temperature": 0.8,
                "max_tokens": 1000,
                "seed": 12345,
                "custom_param": "custom_value",
                "basePath": "/Users/someone/secret-project",
                "headers": {"X-Tenant": "acme"},
                "maxRetries": 0,
            }
        },
    )
    await provider.call_api("Test passthrough")
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Tenant"] == "acme"
    body = sent_json(route)
    assert body["temperature"] == 0.8
    assert body["max_tokens"] == 1000
    assert body["seed"] == 12345
    assert body["custom_param"] == "custom_value"
    for key in ("accountId", "apiKey", "apiKeyEnvar", "apiBaseUrl", "basePath", "headers"):
        assert key not in body
    assert "maxRetries" not in body


@respx.mock
async def test_cloudflare_ai_custom_base_url():
    route = respx.post("https://custom.example.com/v1/chat/completions").mock(
        return_value=ok(_chat())
    )
    provider = CloudflareAiChatCompletionProvider(
        CF_MODEL, {"config": {**CF_MIN, "apiBaseUrl": "https://custom.example.com/v1"}}
    )
    await provider.call_api("hi")
    assert route.call_count == 1


def test_cloudflare_ai_identification_and_json():
    provider = CloudflareAiChatCompletionProvider(CF_MODEL, {"config": CF_MIN})
    assert provider.id() == f"cloudflare-ai:chat:{CF_MODEL}"
    assert str(provider) == f"[Cloudflare AI chat Provider {CF_MODEL}]"
    assert provider.get_api_key() == "testApiKey"
    as_json = provider.to_json()
    assert as_json["provider"] == "cloudflare-ai"
    assert as_json["model"] == CF_MODEL
    assert as_json["modelType"] == "chat"
    assert as_json["config"]["apiKeyEnvar"] == "CLOUDFLARE_API_KEY"
    assert "cloudflare.com" in as_json["config"]["apiBaseUrl"]
    assert "apiKey" not in as_json["config"]
    completion = CloudflareAiCompletionProvider(CF_MODEL, {"config": CF_MIN})
    assert completion.id() == f"cloudflare-ai:completion:{CF_MODEL}"
    assert completion.to_json()["modelType"] == "completion"
    embedding = CloudflareAiEmbeddingProvider("@cf/baai/bge", {"config": CF_MIN})
    assert str(embedding) == "[Cloudflare AI embedding Provider @cf/baai/bge]"


def test_cloudflare_ai_env_names(vendor_env):
    vendor_env.setenv("CUSTOM_CF_ACCOUNT", "custom-account-id")
    vendor_env.setenv("CUSTOM_CF_KEY", "custom-api-key")
    provider = CloudflareAiChatCompletionProvider(
        CF_MODEL,
        {"config": {"accountIdEnvar": "CUSTOM_CF_ACCOUNT", "apiKeyEnvar": "CUSTOM_CF_KEY"}},
    )
    assert provider.get_api_key() == "custom-api-key"
    assert provider.get_api_url() == (
        "https://api.cloudflare.com/client/v4/accounts/custom-account-id/ai/v1"
    )


def test_cloudflare_ai_env_fallback(vendor_env):
    vendor_env.setenv("CLOUDFLARE_ACCOUNT_ID", "env-account")
    vendor_env.setenv("CLOUDFLARE_API_KEY", "env-key")
    provider = create_cloudflare_ai_provider(f"cloudflare-ai:chat:{CF_MODEL}")
    assert provider.get_api_key() == "env-key"
    assert "env-account" in provider.get_api_url()
    explicit = CloudflareAiChatCompletionProvider(
        CF_MODEL, {"config": {"accountId": "cfg-account", "apiKey": "cfg-key"}}
    )
    assert explicit.get_api_key() == "cfg-key"
    assert "cfg-account" in explicit.get_api_url()


def test_cloudflare_ai_requires_token_and_account():
    with pytest.raises(ValueError, match="Cloudflare API token required"):
        CloudflareAiChatCompletionProvider(CF_MODEL, {"config": {"accountId": "test-account"}})
    with pytest.raises(ValueError, match="Cloudflare account ID required"):
        CloudflareAiChatCompletionProvider(CF_MODEL, {"config": {"apiKey": "k"}})


@respx.mock
async def test_cloudflare_ai_embeddings():
    route = respx.post(f"{CF_BASE}/embeddings").mock(
        return_value=ok({"data": [{"embedding": [0.1, 0.2]}], "usage": TOKEN_USAGE})
    )
    provider = CloudflareAiEmbeddingProvider("@cf/baai/bge-base-en-v1.5", {"config": CF_MIN})
    result = await provider.call_embedding_api("Embed me")
    assert route.call_count == 1
    assert result["embedding"] == [0.1, 0.2]


def test_cloudflare_ai_factory():
    with pytest.raises(ValueError, match="Model name is required"):
        create_cloudflare_ai_provider("cloudflare-ai:chat", {"config": CF_MIN})
    with pytest.raises(ValueError, match="Unknown Cloudflare AI model type: unknown"):
        create_cloudflare_ai_provider("cloudflare-ai:unknown:model", {"config": CF_MIN})
    opts = {"config": CF_MIN}
    assert isinstance(
        create_cloudflare_ai_provider(f"cloudflare-ai:chat:{CF_MODEL}", opts),
        CloudflareAiChatCompletionProvider,
    )
    assert isinstance(
        create_cloudflare_ai_provider("cloudflare-ai:completion:m", opts),
        CloudflareAiCompletionProvider,
    )
    assert isinstance(
        create_cloudflare_ai_provider("cloudflare-ai:embeddings:m", opts),
        CloudflareAiEmbeddingProvider,
    )
    provider = cloudflare_ai.create_provider("cloudflare-ai:chat:@cf/a:b:c", opts)
    assert provider.model_name == "@cf/a:b:c"


# --- cloudflare-gateway --------------------------------------------------------------------------

GW_MIN = {"accountId": "testAccountId", "gatewayId": "testGatewayId", "apiKey": "testApiKey"}
GW_BASE = "https://gateway.ai.cloudflare.com/v1/testAccountId/testGatewayId"


def test_gateway_factory_routes():
    opts = {"config": GW_MIN}
    provider = create_cloudflare_gateway_provider("cloudflare-gateway:openai:gpt-4o", opts)
    assert isinstance(provider, CloudflareGatewayOpenAiProvider)
    assert provider.id() == "cloudflare-gateway:openai:gpt-4o"
    anthropic = create_cloudflare_gateway_provider(
        "cloudflare-gateway:anthropic:claude-sonnet-4-20250514", opts
    )
    assert isinstance(anthropic, CloudflareGatewayAnthropicProvider)
    assert anthropic.id() == "cloudflare-gateway:anthropic:claude-sonnet-4-20250514"
    groq = create_cloudflare_gateway_provider(
        "cloudflare-gateway:groq:llama-3.3-70b-versatile", opts
    )
    assert groq.id() == "cloudflare-gateway:groq:llama-3.3-70b-versatile"
    colons = cloudflare_gateway.create_provider(
        "cloudflare-gateway:openai:gpt-4o:with:colons", opts
    )
    assert colons.id() == "cloudflare-gateway:openai:gpt-4o:with:colons"


def test_gateway_factory_errors():
    with pytest.raises(ValueError, match="Invalid cloudflare-gateway provider path"):
        create_cloudflare_gateway_provider("cloudflare-gateway:openai", {})
    with pytest.raises(ValueError, match="Unsupported Cloudflare AI Gateway provider"):
        create_cloudflare_gateway_provider("cloudflare-gateway:unsupported:m", {"config": GW_MIN})
    with pytest.raises(ValueError, match="Unsupported Cloudflare AI Gateway provider"):
        create_cloudflare_gateway_provider("cloudflare-gateway:bedrock:m", {"config": GW_MIN})
    with pytest.raises(ValueError, match="Cloudflare account ID required"):
        create_cloudflare_gateway_provider(
            "cloudflare-gateway:openai:gpt-4o", {"config": {"gatewayId": "g"}}
        )
    with pytest.raises(ValueError, match="Cloudflare AI Gateway ID required"):
        create_cloudflare_gateway_provider(
            "cloudflare-gateway:openai:gpt-4o", {"config": {"accountId": "a"}}
        )


@respx.mock
async def test_gateway_chat_url_and_headers():
    route = respx.post(f"{GW_BASE}/openai/chat/completions").mock(
        return_value=ok(_chat("Test response"))
    )
    provider = CloudflareGatewayOpenAiProvider(
        "openai",
        "gpt-4o",
        {
            "config": {
                **GW_MIN,
                "cfAigToken": "gateway-token",
                "headers": {"X-Custom-Header": "custom-value"},
                "temperature": 0.8,
                "max_tokens": 500,
            }
        },
    )
    result = await provider.call_api("Test prompt")
    assert result["output"] == "Test response"
    assert result["tokenUsage"] == TOKEN_USAGE_RESULT
    headers = route.calls[0].request.headers
    assert headers["cf-aig-authorization"] == "Bearer gateway-token"
    assert headers["X-Custom-Header"] == "custom-value"
    assert headers["Authorization"] == "Bearer testApiKey"
    body = sent_json(route)
    assert body["temperature"] == 0.8
    assert body["max_tokens"] == 500


@respx.mock
async def test_gateway_groq_url_and_errors():
    route = respx.post(f"{GW_BASE}/groq/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}})
    )
    provider = CloudflareGatewayOpenAiProvider("groq", "llama", {"config": GW_MIN})
    result = await provider.call_api("Test prompt")
    assert route.call_count == 1
    assert "API error: 401" in result["error"]


def test_gateway_identity_and_json():
    provider = CloudflareGatewayOpenAiProvider("openai", "gpt-4o", {"config": GW_MIN})
    assert str(provider) == "[Cloudflare AI Gateway openai Provider gpt-4o]"
    as_json = provider.to_json()
    assert as_json["provider"] == "cloudflare-gateway"
    assert as_json["underlyingProvider"] == "openai"
    assert as_json["model"] == "gpt-4o"
    assert "apiKey" not in as_json["config"]
    labelled = CloudflareGatewayOpenAiProvider(
        "groq", "m", {"id": "customer:custom-label", "config": GW_MIN}
    )
    assert labelled.id() == "customer:custom-label"
    assert labelled.get_gen_ai_system() == "groq"
    anthropic = CloudflareGatewayAnthropicProvider("claude-sonnet-4-20250514", {"config": GW_MIN})
    assert str(anthropic) == "[Cloudflare AI Gateway Anthropic Provider claude-sonnet-4-20250514]"
    assert anthropic.to_json()["underlyingProvider"] == "anthropic"
    assert "apiKey" not in anthropic.to_json()["config"]


def _anthropic_message(model: str = "claude-sonnet-4-20250514") -> dict:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Test"}],
        "model": model,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }


@respx.mock
async def test_gateway_anthropic_url_and_isolated_headers(vendor_env):
    vendor_env.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "Authorization: Bearer ambient-secret\nX-Proxy-Secret: ambient-proxy-secret",
    )
    route = respx.post(f"{GW_BASE}/anthropic/v1/messages").mock(
        return_value=ok(_anthropic_message())
    )
    provider = CloudflareGatewayAnthropicProvider(
        "claude-sonnet-4-20250514",
        {
            "config": GW_MIN,
            "env": {"ANTHROPIC_CUSTOM_HEADERS": "x-proxy-secret: scoped-proxy-secret"},
        },
    )
    result = await provider.call_api("Test prompt")
    assert result["output"] == "Test"
    headers = route.calls[0].request.headers
    assert "authorization" not in headers
    assert "x-proxy-secret" not in headers
    assert headers["x-api-key"] == "testApiKey"
    cached = await provider.call_api("Test prompt")
    assert cached["cached"] is True
    assert route.call_count == 1


@respx.mock
async def test_gateway_anthropic_future_claude_passthrough():
    route = respx.post(f"{GW_BASE}/anthropic/v1/messages").mock(
        return_value=ok(_anthropic_message("claude-haiku-5"))
    )
    provider = CloudflareGatewayAnthropicProvider(
        "claude-haiku-5",
        {"config": {**GW_MIN, "max_tokens": 10000, "temperature": 0.5, "top_p": 0.9}},
    )
    prompt = json.dumps(
        [
            {
                "role": "user",
                "content": "Test prompt",
                "thinking": {"type": "enabled", "budget_tokens": 5000},
            }
        ]
    )
    await provider.call_api(prompt)
    body = sent_json(route)
    assert body["model"] == "claude-haiku-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert "temperature" not in body
    assert "top_p" not in body


def test_gateway_env_ids(vendor_env):
    vendor_env.setenv("CLOUDFLARE_ACCOUNT_ID", "env-account")
    vendor_env.setenv("CLOUDFLARE_GATEWAY_ID", "env-gateway")
    vendor_env.setenv("OPENAI_API_KEY", "k")
    provider = CloudflareGatewayOpenAiProvider("openai", "gpt-4o", {})
    assert provider.get_api_url() == (
        "https://gateway.ai.cloudflare.com/v1/env-account/env-gateway/openai"
    )
    preferred = CloudflareGatewayOpenAiProvider(
        "openai", "gpt-4o", {"config": {"accountId": "cfg-a", "gatewayId": "cfg-g"}}
    )
    assert "cfg-a/cfg-g" in preferred.get_api_url()
    vendor_env.setenv("MY_ACCOUNT", "custom-a")
    vendor_env.setenv("MY_GATEWAY", "custom-g")
    custom = CloudflareGatewayOpenAiProvider(
        "openai",
        "gpt-4o",
        {"config": {"accountIdEnvar": "MY_ACCOUNT", "gatewayIdEnvar": "MY_GATEWAY"}},
    )
    assert "custom-a/custom-g" in custom.get_api_url()


def test_gateway_azure_requirements():
    with pytest.raises(ValueError, match="Azure OpenAI requires resourceName"):
        CloudflareGatewayOpenAiProvider("azure-openai", "gpt-4", {"config": GW_MIN})
    with pytest.raises(ValueError, match="Azure OpenAI requires deploymentName"):
        CloudflareGatewayOpenAiProvider(
            "azure-openai", "gpt-4", {"config": {**GW_MIN, "resourceName": "r"}}
        )
    no_key = {k: v for k, v in GW_MIN.items() if k != "apiKey"}
    with pytest.raises(ValueError, match="Azure OpenAI API key is required"):
        CloudflareGatewayOpenAiProvider(
            "azure-openai",
            "gpt-4",
            {"config": {**no_key, "resourceName": "r", "deploymentName": "d"}},
        )


@respx.mock
async def test_gateway_azure_url():
    route = respx.post(f"{GW_BASE}/azure-openai/my-resource/my-deployment/chat/completions").mock(
        return_value=ok(_chat("Test"))
    )
    provider = CloudflareGatewayOpenAiProvider(
        "azure-openai",
        "gpt-4",
        {
            "config": {
                **GW_MIN,
                "resourceName": "my-resource",
                "deploymentName": "my-deployment",
                "apiVersion": "2024-06-01",
            }
        },
    )
    await provider.call_api("Test prompt")
    request = route.calls[0].request
    assert request.url.params["api-version"] == "2024-06-01"
    assert request.headers["api-key"] == "testApiKey"


@respx.mock
async def test_gateway_workers_ai_url():
    model = "@cf/meta/llama-3.1-8b-instruct"
    route = respx.post(f"{GW_BASE}/workers-ai/{model}/chat/completions").mock(
        return_value=ok(_chat("Test"))
    )
    provider = create_cloudflare_gateway_provider(
        f"cloudflare-gateway:workers-ai:{model}", {"config": GW_MIN}
    )
    assert provider.id() == f"cloudflare-gateway:workers-ai:{model}"
    await provider.call_api("Test prompt")
    assert route.call_count == 1


# --- databricks ----------------------------------------------------------------------------------


def test_databricks_construction(vendor_env):
    provider = DatabricksMosaicAiChatCompletionProvider(
        "my-endpoint", {"config": {"workspaceUrl": "https://ws.cloud.databricks.com/"}}
    )
    assert provider.config["apiBaseUrl"] == "https://ws.cloud.databricks.com/serving-endpoints"
    assert provider.config["apiKeyEnvar"] == "DATABRICKS_TOKEN"
    assert provider.id() == "my-endpoint"
    vendor_env.setenv("DATABRICKS_WORKSPACE_URL", "https://env.cloud.databricks.com")
    env_provider = databricks.create_provider("databricks:m", {"config": {}})
    assert env_provider.get_api_url() == "https://env.cloud.databricks.com/serving-endpoints"
    custom = DatabricksMosaicAiChatCompletionProvider(
        "m", {"config": {"apiKeyEnvar": "CUSTOM_TOKEN"}}
    )
    assert custom.config["apiKeyEnvar"] == "CUSTOM_TOKEN"


def test_databricks_requires_workspace():
    with pytest.raises(ValueError, match="Databricks workspace URL is required"):
        DatabricksMosaicAiChatCompletionProvider("m", {"config": {}})


def test_databricks_usage_context():
    provider = DatabricksMosaicAiChatCompletionProvider(
        "m",
        {
            "config": {
                "workspaceUrl": "https://ws",
                "usageContext": {"project": "p"},
                "passthrough": {"extra": 1},
                "aiGatewayConfig": {"enableSafety": True},
            }
        },
    )
    assert provider.config["passthrough"] == {"extra": 1, "usage_context": {"project": "p"}}
    assert provider.config["aiGatewayConfig"] == {"enableSafety": True}


@respx.mock
async def test_databricks_call(vendor_env):
    vendor_env.setenv("DATABRICKS_TOKEN", "dapi-token")
    route = respx.post("https://ws/serving-endpoints/chat/completions").mock(
        return_value=ok(chat_response())
    )
    provider = DatabricksMosaicAiChatCompletionProvider(
        "databricks-meta-llama", {"config": {"workspaceUrl": "https://ws"}}
    )
    result = await provider.call_api("hi")
    assert result["output"] == "Test output"
    assert route.calls[0].request.headers["Authorization"] == "Bearer dapi-token"
    assert sent_json(route)["model"] == "databricks-meta-llama"


# --- mlflow-gateway ------------------------------------------------------------------------------


def _mlflow(model: str = "chat", **config) -> MlflowGatewayChatCompletionProvider:
    return MlflowGatewayChatCompletionProvider(
        model, {"config": {"gatewayUrl": "http://localhost:5000", **config}}
    )


def test_mlflow_construction(vendor_env):
    provider = _mlflow()
    assert provider.config["apiBaseUrl"] == "http://localhost:5000/gateway/mlflow/v1"
    assert provider.config["apiKeyEnvar"] == "MLFLOW_GATEWAY_API_KEY"
    assert provider.config["apiKeyRequired"] is False
    assert provider.id() == "mlflow-gateway:chat"
    assert _mlflow(gatewayUrl="http://gw:5000///").get_api_url() == (
        "http://gw:5000/gateway/mlflow/v1"
    )
    explicit = MlflowGatewayChatCompletionProvider(
        "chat", {"id": "custom-id", "config": {"gatewayUrl": "http://x"}}
    )
    assert explicit.id() == "custom-id"
    vendor_env.setenv("MLFLOW_GATEWAY_URL", "http://process:5000")
    assert MlflowGatewayChatCompletionProvider("e", {"config": {}}).get_api_url() == (
        "http://process:5000/gateway/mlflow/v1"
    )
    provider_env = MlflowGatewayChatCompletionProvider(
        "e", {"config": {}, "env": {"MLFLOW_GATEWAY_URL": "http://scoped:5000"}}
    )
    assert provider_env.get_api_url() == "http://scoped:5000/gateway/mlflow/v1"
    assert _mlflow(gatewayUrl="http://cfg").get_api_url() == "http://cfg/gateway/mlflow/v1"
    assert _mlflow(apiKeyEnvar="CUSTOM").config["apiKeyEnvar"] == "CUSTOM"
    assert _mlflow(apiKeyRequired=True).requires_api_key() is True


def test_mlflow_errors():
    with pytest.raises(ValueError, match="MLflow Gateway URL is required"):
        MlflowGatewayChatCompletionProvider("e", {"config": {}})
    for name in ("", "   "):
        with pytest.raises(ValueError, match="MLflow Gateway endpoint name is required"):
            _mlflow(name)


def test_mlflow_credentials(vendor_env):
    vendor_env.setenv("OPENAI_API_KEY", "sk-openai")
    vendor_env.setenv("OPENAI_ORGANIZATION", "org-x")
    vendor_env.setenv("OPENAI_API_HOST", "openai.example.com")
    vendor_env.setenv("OPENAI_BASE_URL", "https://openai.example.com/v1")
    provider = _mlflow()
    assert provider.get_api_key() is None
    assert provider.get_organization() is None
    assert provider.get_api_url() == "http://localhost:5000/gateway/mlflow/v1"
    assert provider.requires_api_key() is False
    vendor_env.setenv("MLFLOW_GATEWAY_API_KEY", "process-key")
    assert _mlflow().get_api_key() == "process-key"
    assert _mlflow(apiKey="cfg-key").get_api_key() == "cfg-key"
    scoped = MlflowGatewayChatCompletionProvider(
        "e",
        {"config": {"gatewayUrl": "http://x"}, "env": {"MLFLOW_GATEWAY_API_KEY": "scoped"}},
    )
    assert scoped.get_api_key() == "scoped"
    as_json = _mlflow(apiKey="secret").to_json()
    assert as_json["provider"] == "mlflow-gateway"
    assert "apiKey" not in as_json["config"]
    assert _mlflow(apiKeyRequired=True).get_missing_api_key_error_message() == (
        "MLflow Gateway Bearer token is not set. Set the MLFLOW_GATEWAY_API_KEY environment "
        "variable or add `apiKey` to the provider config."
    )


@respx.mock
async def test_mlflow_call(vendor_env):
    vendor_env.setenv("OPENAI_API_KEY", "sk-openai")
    route = respx.post("http://localhost:5000/gateway/mlflow/v1/chat/completions").mock(
        return_value=ok(chat_response("gateway"))
    )
    result = await mlflow_gateway.create_provider(
        "mlflow-gateway:chat", {"config": {"gatewayUrl": "http://localhost:5000"}}
    ).call_api("hi")
    assert result["output"] == "gateway"
    headers = route.calls[0].request.headers
    assert "Authorization" not in headers
    assert "X-OpenAI-Originator" not in headers
    secured = _mlflow(headers={"Authorization": "Bearer explicit"})
    await secured.call_api("hi")
    assert route.calls[1].request.headers["Authorization"] == "Bearer explicit"


@respx.mock
async def test_mlflow_http_error():
    respx.post("http://localhost:5000/gateway/mlflow/v1/chat/completions").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    result = await _mlflow().call_api("hi")
    assert result["error"].startswith("API error: 404")


async def test_mlflow_required_token_missing():
    with pytest.raises(ValueError, match="MLflow Gateway Bearer token is not set"):
        await _mlflow(apiKeyRequired=True).call_api("hi")


# --- portkey -------------------------------------------------------------------------------------


def test_portkey_kebab_case():
    assert to_kebab_case("apiKey") == "api-key"
    assert to_kebab_case("") == ""
    assert to_kebab_case("word") == "word"
    assert to_kebab_case("already-kebab") == "already-kebab"
    assert to_kebab_case("aB") == "a-b"


def test_portkey_headers():
    assert get_portkey_headers(
        {
            "portkeyApiKey": "test-api-key",
            "portkeyCustomHost": "custom.host.com",
            "portkeyMetadata": {"key1": "value1", "key2": "value2"},
        }
    ) == {
        "x-portkey-api-key": "test-api-key",
        "x-portkey-custom-host": "custom.host.com",
        "x-portkey-metadata": '{"key1":"value1","key2":"value2"}',
    }
    assert get_portkey_headers({"portkeyApiKey": "k", "portkeyMetadata": None}) == {
        "x-portkey-api-key": "k"
    }
    assert get_portkey_headers({}) == {}
    for foreign in (
        {"max_tokens": 512},
        {"basePath": "/home/user"},
        {"apiKey": "k"},
        {"instructions": "line one\nline two"},
    ):
        assert get_portkey_headers({"portkeyProvider": "@bedrock-eu", **foreign}) == {
            "x-portkey-provider": "@bedrock-eu"
        }
    assert get_portkey_headers({"portkeyApiKey": "k", "portkeyApiBaseUrl": "https://g/v1"}) == {
        "x-portkey-api-key": "k"
    }
    assert get_portkey_headers(
        {
            "portkeyTraceId": "generated",
            "headers": {"X-Portkey-Trace-Id": "explicit"},
        }
    ) == {"X-Portkey-Trace-Id": "explicit"}
    assert get_portkey_headers(
        {"headers": {"x-count": 5, "x-enabled": True, "x-obj": {"a": 1}, "x-skip": None}}
    ) == {"x-count": "5", "x-enabled": "true", "x-obj": '{"a":1}'}
    assert get_portkey_headers({"portkeyProvider": "@b", "headers": "not-an-object"}) == {
        "x-portkey-provider": "@b"
    }
    assert get_portkey_headers({"portkeyFeatureFlag": True, "portkeyTimeout": 1000}) == {
        "x-portkey-feature-flag": "true",
        "x-portkey-timeout": "1000",
    }


def test_portkey_credentials(vendor_env):
    provider = PortkeyChatCompletionProvider("gpt-4o", {"config": {"portkeyApiKey": "pk-cfg"}})
    assert provider.config["headers"]["x-portkey-api-key"] == "pk-cfg"
    assert provider.requires_api_key() is False
    vendor_env.setenv("PORTKEY_API_KEY", "pk-process")
    scoped = PortkeyChatCompletionProvider(
        "gpt-4o", {"config": {"portkeyProvider": "openai"}, "env": {"PORTKEY_API_KEY": "pk-o"}}
    )
    assert scoped.config["headers"]["x-portkey-api-key"] == "pk-o"
    env_provider = PortkeyChatCompletionProvider(
        "gpt-4o", {"config": {"portkeyProvider": "openai"}}
    )
    assert env_provider.config["headers"]["x-portkey-api-key"] == "pk-process"
    assert "portkeyApiKey" not in env_provider.config
    assert env_provider.get_api_key() == "test-api-key"  # OPENAI_API_KEY from the fixture
    vendor_env.delenv("OPENAI_API_KEY")
    assert env_provider.get_api_key() is None
    explicit_required = PortkeyChatCompletionProvider(
        "gpt-4o", {"config": {"portkeyApiKey": "pk", "apiKeyRequired": True}}
    )
    assert explicit_required.requires_api_key() is True
    header_key = PortkeyChatCompletionProvider(
        "@bedrock-eu/claude", {"config": {"headers": {"X-Portkey-Api-Key": "pk-header"}}}
    )
    vendor_env.delenv("PORTKEY_API_KEY")
    header_key = PortkeyChatCompletionProvider(
        "@bedrock-eu/claude", {"config": {"headers": {"X-Portkey-Api-Key": "pk-header"}}}
    )
    assert header_key.requires_api_key() is False
    headers = header_key.get_openai_request_headers()
    assert [k for k in headers if k.lower() == "x-portkey-api-key"] == ["X-Portkey-Api-Key"]


def test_portkey_managed_credentials(vendor_env):
    vendor_env.setenv("OPENAI_API_KEY", "sk-openai")
    explicit = PortkeyChatCompletionProvider(
        "gpt-4o", {"config": {"apiKey": "sk-explicit", "portkeyProvider": "openai"}}
    )
    assert explicit.get_api_key() == "sk-explicit"
    for model, config in (
        ("@bedrock-eu/claude", {"apiKey": "sk-inherited"}),
        ("claude", {"portkeyProvider": "@bedrock-eu"}),
        ("claude", {"portkeyVirtualKey": "bedrock-prod"}),
    ):
        provider = PortkeyChatCompletionProvider(
            model, {"config": {"portkeyApiKey": "pk", **config}}
        )
        assert provider.get_api_key() is None


@respx.mock
async def test_portkey_call_and_header_collisions():
    route = respx.post("https://api.portkey.ai/v1/chat/completions").mock(
        return_value=ok(chat_response())
    )
    provider = portkey.create_provider(
        "portkey:gpt-4o",
        {
            "config": {
                "portkeyApiKey": "pk-config-key",
                "portkeyProvider": "openai",
                "headers": {"authorization": "Bearer explicit"},
            }
        },
    )
    await provider.call_api("hi")
    request = route.calls[0].request
    assert request.headers.get_list("authorization") == ["Bearer explicit"]
    assert request.headers["x-portkey-api-key"] == "pk-config-key"
    assert request.headers["x-portkey-provider"] == "openai"
    # A prompt-level headers block keeps the Portkey credential.
    await provider.call_api(
        "hi", {"prompt": {"raw": "hi", "label": "p", "config": {"headers": {"x-other": "1"}}}}
    )
    request = route.calls[1].request
    assert request.headers["x-portkey-api-key"] == "pk-config-key"
    assert request.headers["x-other"] == "1"


def test_portkey_missing_key_message_and_colons():
    provider = portkey.create_provider("portkey:@bedrock-eu/anthropic.claude:v1", {"config": {}})
    assert provider.model_name == "@bedrock-eu/anthropic.claude:v1"
    assert provider.get_missing_api_key_error_message() == (
        "Portkey API key is not set. Set the PORTKEY_API_KEY environment variable or add "
        "`portkeyApiKey` to the provider config."
    )


# --- snowflake -----------------------------------------------------------------------------------

SF_URL = "https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/inference:complete"
SF_CONFIG = {"accountIdentifier": "myorg-myaccount", "apiKey": "test-token"}


def _snowflake(**config) -> SnowflakeCortexProvider:
    return SnowflakeCortexProvider("claude-sonnet-4-6", {"config": {**SF_CONFIG, **config}})


def test_snowflake_construction(vendor_env):
    provider = _snowflake()
    assert provider.config["apiBaseUrl"] == "https://myorg-myaccount.snowflakecomputing.com"
    assert provider.config["apiKeyEnvar"] == "SNOWFLAKE_API_KEY"
    assert provider.id() == "snowflake:claude-sonnet-4-6"
    assert str(provider) == "[Snowflake Cortex Provider claude-sonnet-4-6]"
    vendor_env.setenv("SNOWFLAKE_ACCOUNT_IDENTIFIER", "env-account")
    env_provider = snowflake.create_provider("snowflake:m", {"config": {}})
    assert env_provider.config["apiBaseUrl"] == "https://env-account.snowflakecomputing.com"
    custom = SnowflakeCortexProvider("m", {"config": {"apiBaseUrl": "https://custom.example"}})
    assert custom.config["apiBaseUrl"] == "https://custom.example"
    as_json = _snowflake().to_json()
    assert as_json["provider"] == "snowflake"
    assert as_json["model"] == "claude-sonnet-4-6"
    assert "apiKey" not in as_json["config"]


def test_snowflake_errors():
    with pytest.raises(ValueError, match="Snowflake provider requires an account identifier"):
        SnowflakeCortexProvider("m", {"config": {}})
    with pytest.raises(ValueError, match="Snowflake provider requires a model name"):
        snowflake.create_provider("snowflake:", {})


@respx.mock
async def test_snowflake_call():
    route = respx.post(SF_URL).mock(
        return_value=ok(
            {
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
    )
    result = await _snowflake().call_api("Test prompt")
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer test-token"
    assert sent_json(route)["model"] == "claude-sonnet-4-6"
    assert result["output"] == "Hello"
    assert result["finishReason"] == "stop"
    assert result["tokenUsage"]["total"] == 15


@respx.mock
async def test_snowflake_http_and_network_errors():
    respx.post(SF_URL).mock(return_value=httpx.Response(400, json={"message": "Bad"}))
    result = await _snowflake().call_api("p")
    assert result["error"].startswith("API error: 400")
    respx.post(SF_URL).mock(side_effect=httpx.ConnectError("Network error"))
    result = await _snowflake(maxRetries=0).call_api("p2")
    assert result["error"].startswith("API call error:")
    assert "Network error" in result["error"]


@respx.mock
async def test_snowflake_tool_calls():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    respx.post(SF_URL).mock(
        return_value=ok(
            {
                "choices": [
                    {
                        "message": {"content": "", "tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
    )
    result = await _snowflake().call_api("weather?")
    assert result["output"] == tool_calls


@respx.mock
async def test_snowflake_error_shapes():
    route = respx.post(SF_URL)
    route.mock(return_value=ok({"code": "390144", "message": "JWT token is invalid: secret"}))
    result = await _snowflake().call_api("a")
    assert result["error"] == "API error: Snowflake provider returned error code 390144"
    assert result["metadata"] == {"snowflakeErrorCode": "390144"}
    assert "secret" not in json.dumps(result)

    route.mock(
        return_value=ok(
            {
                "choices": [{"message": {"content": "partial"}, "finish_reason": "error"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
    )
    result = await _snowflake().call_api("b")
    assert result["error"] == "API error: Snowflake provider returned a generation error"
    assert result["finishReason"] == "error"

    for payload in ({"choices": []}, {"id": "x"}):
        route.mock(return_value=ok(payload))
        result = await _snowflake().call_api(json.dumps(payload))
        assert result["error"] == "Malformed response data: expected choices[0].message"
