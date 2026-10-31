"""Every registered OpenAI-compatible prefix resolves through `load_api_provider` to a provider
with upstream's base URL and API-key env var (see D31)."""

from __future__ import annotations

import pytest
from openai_helpers import openai_env

from promptphoo.providers import load_api_provider
from promptphoo.providers.openai import OpenAiGenericProvider

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

# (provider path, config, base URL, key env var or None when the key is not env-resolved)
CASES = [
    ("abliteration:m", {}, "https://api.abliteration.ai/v1", "ABLIT_KEY"),
    ("alibaba:qwen-max", {}, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
     "DASHSCOPE_API_KEY"),
    ("alicloud:qwen-max", {}, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
     "DASHSCOPE_API_KEY"),
    ("aliyun:embedding:text-embedding-v3", {},
     "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    ("dashscope:qwen-max", {}, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
     "DASHSCOPE_API_KEY"),
    ("aimlapi:chat:gpt-4o", {}, "https://api.aimlapi.com/v1", "AIML_API_KEY"),
    ("atlascloud:m", {}, "https://api.atlascloud.ai/v1", "ATLASCLOUD_API_KEY"),
    ("cerebras:gpt-oss-120b", {}, "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    ("cloudera:m", {"domain": "ml.example.com"},
     "https://ml.example.com/namespaces/serving-default/endpoints/m/v1", "CDP_TOKEN"),
    ("cloudflare-ai:chat:@cf/meta/llama", {"accountId": "acct"},
     "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1", "CLOUDFLARE_API_KEY"),
    ("cometapi:chat:gpt-4o", {}, "https://api.cometapi.com/v1", "COMETAPI_KEY"),
    ("databricks:m", {"workspaceUrl": "https://ws.example.com"},
     "https://ws.example.com/serving-endpoints", "DATABRICKS_TOKEN"),
    ("deepseek:deepseek-chat", {}, "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    ("docker:ai/smollm2", {}, "http://localhost:12434/engines/v1", None),
    ("envoy:m", {"apiBaseUrl": "https://gw.example/v1"}, "https://gw.example/v1", None),
    ("fireworks:accounts/fireworks/models/m", {}, "https://api.fireworks.ai/inference/v1",
     "FIREWORKS_API_KEY"),
    ("groq:llama-3.3-70b-versatile", {}, "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    ("helicone:openai/gpt-4o", {}, "http://localhost:8080/ai", "HELICONE_API_KEY"),
    ("hyperbolic:m", {}, "https://api.hyperbolic.xyz/v1", "HYPERBOLIC_API_KEY"),
    ("jfrog:m", {}, "https://models.qwak-prod.qwak.ai/v1/m", "QWAK_TOKEN"),
    ("qwak:m", {}, "https://models.qwak-prod.qwak.ai/v1/m", "QWAK_TOKEN"),
    ("litellm:gpt-4o", {}, "http://0.0.0.0:4000", "LITELLM_API_KEY"),
    ("llamaapi:Llama-4", {}, "https://api.llama.com/compat/v1", "LLAMA_API_KEY"),
    ("meta:chat:muse-spark", {}, "https://api.meta.ai/v1", "MODEL_API_KEY"),
    ("minimax:MiniMax-M2", {}, "https://api.minimax.io/v1", "MINIMAX_API_KEY"),
    ("mlflow-gateway:m", {"gatewayUrl": "http://mlflow:5000"},
     "http://mlflow:5000/gateway/mlflow/v1", "MLFLOW_GATEWAY_API_KEY"),
    ("moonshot:kimi-k2", {}, "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    ("novita:m", {}, "https://api.novita.ai/openai/v1", "NOVITA_API_KEY"),
    ("nscale:m", {}, "https://inference.api.nscale.com/v1", "NSCALE_SERVICE_TOKEN"),
    ("nvidia:m", {}, "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    ("openrouter:openai/gpt-4o", {}, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ("orcarouter:m", {}, "https://api.orcarouter.ai/v1", "ORCAROUTER_API_KEY"),
    ("perplexity:sonar", {}, "https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    ("portkey:gpt-4o", {}, "https://api.portkey.ai/v1", None),
    ("snowflake:mistral-large2", {"accountIdentifier": "acct"},
     "https://acct.snowflakecomputing.com", "SNOWFLAKE_API_KEY"),
    ("togetherai:meta-llama/Llama-3", {}, "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    ("truefoundry:openai-main/gpt-4o", {}, "https://llm-gateway.truefoundry.com",
     "TRUEFOUNDRY_API_KEY"),
    ("xai:grok-4", {}, "https://api.x.ai/v1", "XAI_API_KEY"),
]  # fmt: skip


@pytest.mark.parametrize(("path", "config", "base_url", "key_envar"), CASES)
async def test_prefix_resolves(monkeypatch, path, config, base_url, key_envar):
    if key_envar:
        monkeypatch.setenv(key_envar, f"key-for-{key_envar}")
    provider = await load_api_provider(path, options={"config": config})
    if path.startswith("litellm:"):
        # LiteLLM wraps the OpenAI provider to keep its own id (litellm.ts).
        assert provider.id() == path
        provider = provider.provider
    assert isinstance(provider, OpenAiGenericProvider)
    assert provider.get_api_url() == base_url
    if key_envar:
        assert provider.get_api_key() == f"key-for-{key_envar}"


@pytest.mark.parametrize(
    ("path", "provider_id"),
    [
        ("localai:chat:m", "localai:m"),
        ("llama:m", "llama:m"),
        ("cloudflare-gateway:openai:gpt-4o", None),
    ],
)
async def test_non_generic_prefixes_resolve(monkeypatch, path, provider_id):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw")
    provider = await load_api_provider(path)
    if provider_id:
        assert provider.id() == provider_id


@pytest.mark.parametrize(
    "path", ["xai:image:grok-2-image", "nscale:image:flux", "cometapi:image:dall-e-3",
             "hyperbolic:image:sdxl", "vercel:openai/gpt-4o", "mistral:mistral-large-latest",
             "ai21:jamba-large"],
)  # fmt: skip
async def test_unported_routes_fail_clearly(path):
    with pytest.raises(ValueError, match="not supported by promptphoo|not supported"):
        await load_api_provider(path)


async def test_retired_github_models_fails_with_upstream_message():
    with pytest.raises(ValueError, match="GitHub Models was retired on July 30, 2026"):
        await load_api_provider("github:openai/gpt-4.1")
