"""OpenAI-compatible families from the data table and the small subclasses. Ports deterministic
cases from test/providers/{deepseek,deepseek-request,cerebras,togetherai,togetherai-request,
nscale,nscaleRequest,aimlapi,cometapi,atlascloud,llamaApi,helicone,envoy}.test.ts."""

from __future__ import annotations

import math

import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo.providers import load_api_provider
from promptphoo.providers.compatible import simple, table
from promptphoo.providers.compatible.cerebras import (
    calculate_cerebras_cost,
    create_cerebras_provider,
)
from promptphoo.providers.compatible.deepseek import (
    DEEPSEEK_CHAT_MODELS,
    calculate_deepseek_cost,
    create_deepseek_provider,
)
from promptphoo.providers.openai import (
    OpenAiChatCompletionProvider,
    OpenAiCompletionProvider,
    OpenAiEmbeddingProvider,
)
from promptphoo.providers.registry import find_factory

VENDOR_ENV = [
    "DEEPSEEK_API_KEY", "CEREBRAS_API_KEY", "TOGETHER_API_KEY", "NSCALE_SERVICE_TOKEN",
    "NSCALE_API_KEY", "AIML_API_KEY", "COMETAPI_KEY", "ATLASCLOUD_API_KEY", "LLAMA_API_KEY",
    "HELICONE_API_KEY", "ENVOY_API_BASE_URL", "CDP_DOMAIN", "CDP_TOKEN", "QWAK_TOKEN",
]  # fmt: skip

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


pytestmark = pytest.mark.usefixtures(openai_env.__name__)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in VENDOR_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _chat(content="fixture answer", usage=None, **extra):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        **({"usage": usage} if usage is not None else {}),
        **extra,
    }


# --- DeepSeek -----------------------------------------------------------------------------------


def test_deepseek_usage_boundaries():
    assert calculate_deepseek_cost("deepseek-chat", {"inputCost": 0.01}, 10, 0) == pytest.approx(
        0.1
    )
    assert calculate_deepseek_cost("deepseek-chat", {"outputCost": 0.02}, 0, 10) == pytest.approx(
        0.2
    )
    assert calculate_deepseek_cost("deepseek-chat", {}, 0, 0) == 0


@pytest.mark.parametrize("count", [None, -1, math.nan, math.inf])
def test_deepseek_rejects_invalid_usage(count):
    assert calculate_deepseek_cost("deepseek-chat", {}, count, 1) is None
    assert calculate_deepseek_cost("deepseek-chat", {}, 1, count) is None


def test_deepseek_cost_rules():
    assert calculate_deepseek_cost("custom-model", {}, 100, 200) is None
    assert calculate_deepseek_cost("custom-model", {"inputCost": 1e-6}, 10, 1) is None
    assert calculate_deepseek_cost("custom-model", {"outputCost": 3e-6}, 1, 10) is None
    assert calculate_deepseek_cost("deepseek-flash", {}, 100, 100) == pytest.approx(0.00015)
    rates = {"inputCost": 0.01, "outputCost": 0.02}
    assert calculate_deepseek_cost("deepseek-flash", rates, 100, 100) == pytest.approx(3)
    rates["cacheReadCost"] = 0.001
    assert calculate_deepseek_cost("deepseek-flash", rates, 100, 100, 50) == pytest.approx(2.55)
    zero = {"cost": 0, "cacheReadCost": 0}
    assert calculate_deepseek_cost("deepseek-flash", zero, 100, 100, 50) == 0
    # Cached tokens are clamped to [0, promptTokens]; non-finite counts as no cache hits.
    full = calculate_deepseek_cost("deepseek-v4-pro", {}, 100, 0, 100)
    assert calculate_deepseek_cost("deepseek-v4-pro", {}, 100, 0, 500) == pytest.approx(full)
    none = calculate_deepseek_cost("deepseek-v4-pro", {}, 100, 0, 0)
    assert calculate_deepseek_cost("deepseek-v4-pro", {}, 100, 0, -5) == pytest.approx(none)
    assert calculate_deepseek_cost("deepseek-v4-pro", {}, 100, 0, math.nan) == pytest.approx(none)


def test_deepseek_model_table():
    pro = next(m for m in DEEPSEEK_CHAT_MODELS if m["id"] == "deepseek-v4-pro")
    assert pro["cost"] == {"input": 1.32e-6, "output": 3.96e-6, "cache_read": 0.044e-6}
    chat = next(m for m in DEEPSEEK_CHAT_MODELS if m["id"] == "deepseek-chat")
    assert chat["cost"]["cache_read"] == pytest.approx(0.0028e-6)


def test_deepseek_response_cost_hook():
    provider = create_deepseek_provider("deepseek:deepseek-v4-flash")
    data = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}
    assert provider.calculate_response_cost(data, {}, False) == pytest.approx(1.5)
    assert provider.calculate_response_cost(data, {}, True) == 0
    rates = {"inputCost": 1e-6, "outputCost": 3e-6, "cacheReadCost": 0.1e-6}
    pro = create_deepseek_provider("deepseek:deepseek-v4-pro")
    native = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000,
              "prompt_cache_hit_tokens": 400_000, "prompt_cache_miss_tokens": 600_000}  # fmt: skip
    assert pro.calculate_response_cost({"usage": native}, rates, False) == pytest.approx(2.14)
    gateway = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000,
               "prompt_tokens_details": {"cached_tokens": 400_000}}  # fmt: skip
    assert pro.calculate_response_cost({"usage": gateway}, rates, False) == pytest.approx(2.14)


@pytest.mark.parametrize("path", ["deepseek", "deepseek:"])
async def test_deepseek_bare_default(path):
    provider = create_deepseek_provider(path)
    assert provider.id() == "deepseek:deepseek-flash"
    body = (await provider.get_openai_body("Hello"))["body"]
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "disabled"}
    # DeepSeek chooses its own reasoning budget.
    assert "max_tokens" not in body


async def test_deepseek_thinking_defaults():
    named = create_deepseek_provider("deepseek:deepseek-flash")
    assert "thinking" not in (await named.get_openai_body("hello"))["body"]
    override = create_deepseek_provider(
        "deepseek:", {"config": {"config": {"passthrough": {"model": "deepseek-v4-pro"}}}}
    )
    body = (await override.get_openai_body("hello"))["body"]
    assert body["model"] == "deepseek-v4-pro"
    assert "thinking" not in body
    explicit = create_deepseek_provider(
        "deepseek:", {"config": {"config": {"passthrough": {"thinking": {"type": "enabled"}}}}}
    )
    assert (await explicit.get_openai_body("hello"))["body"]["thinking"] == {"type": "enabled"}
    replaced = create_deepseek_provider(
        "deepseek:", {"config": {"config": {"passthrough": {"trace_id": "provider-trace"}}}}
    )
    context = {"prompt": {"config": {"passthrough": {"trace_id": "prompt-trace"}}}}
    body = (await replaced.get_openai_body("hello", context))["body"]
    assert body["thinking"] == {"type": "disabled"}
    assert body["trace_id"] == "prompt-trace"


@pytest.mark.parametrize(
    ("path", "model", "expected"),
    [
        ("deepseek:", "deepseek-flash", 0.00004224),
        ("deepseek:deepseek-flash", "deepseek-flash", 0.00004224),
        ("deepseek:deepseek-v4-flash", "deepseek-v4-flash", 0.00004224),
        ("deepseek:deepseek-v4-flash-vision-exp", "deepseek-v4-flash-vision-exp", 0.00004224),
        ("deepseek:deepseek-v4-pro", "deepseek-v4-pro", 0.00016016),
        ("deepseek:custom-model", "custom-model", None),
        ("deepseek:deepseek-reasoner", "deepseek-reasoner", 0.000014112),
    ],
)
@respx.mock
async def test_deepseek_native_requests(path, model, expected):
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}
    route = respx.post(DEEPSEEK_URL).mock(return_value=ok(_chat(usage=usage)))
    provider = create_deepseek_provider(path, {"config": {"config": {"apiKey": "test-key"}}})
    result = await provider.call_api("Hello")
    body = sent_json(route)
    assert body["model"] == model
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert ("thinking" in body) == (path == "deepseek:")
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert "error" not in result
    assert result["output"] == "fixture answer"
    if expected is None:
        assert "cost" not in result
    else:
        assert result["cost"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("cache_usage", [{"prompt_cache_hit_tokens": 40},
                                         {"prompt_cache_miss_tokens": 60}])  # fmt: skip
@respx.mock
async def test_deepseek_prompt_level_rates(cache_usage):
    usage = {"prompt_tokens": 100, "completion_tokens": 20, **cache_usage}
    route = respx.post(DEEPSEEK_URL).mock(return_value=ok(_chat(usage=usage)))
    provider = create_deepseek_provider("deepseek:", {"config": {"config": {
        "apiKey": "k", "inputCost": 1e-6, "outputCost": 3e-6,
        "passthrough": {"thinking": {"type": "disabled"}},
    }}})  # fmt: skip
    prompt_config = {
        "inputCost": 2e-6, "outputCost": 4e-6, "cacheReadCost": 0.2e-6,
        "passthrough": {"model": "deepseek-v4-flash-vision-exp", "thinking": {"type": "enabled"},
                        "reasoning_effort": "high", "top_p": 0.97},
    }  # fmt: skip
    context = {"vars": {}, "prompt": {"raw": "Hello", "label": "Hello", "config": prompt_config}}
    result = await provider.call_api("Hello", context)
    body = sent_json(route)
    assert body["model"] == "deepseek-v4-flash-vision-exp"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["top_p"] == 0.97
    for key in ("inputCost", "outputCost", "cacheReadCost"):
        assert key not in body
    assert result["cost"] == pytest.approx(208e-6, abs=1e-12)


@respx.mock
async def test_deepseek_cached_response_is_free():
    usage = {"prompt_tokens": 100, "completion_tokens": 20}
    respx.post(DEEPSEEK_URL).mock(return_value=ok(_chat(usage=usage)))
    provider = create_deepseek_provider(
        "deepseek:deepseek-flash", {"config": {"config": {"apiKey": "k", "cost": 1e-6}}}
    )
    first = await provider.call_api("Hello")
    second = await provider.call_api("Hello")
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["cost"] == 0


def test_deepseek_provider_scoped_key_and_json():
    provider = create_deepseek_provider(
        "deepseek:",
        {"config": {"env": {"DEEPSEEK_API_KEY": "provider-key"}},
         "env": {"DEEPSEEK_API_KEY": "suite-key"}},
    )  # fmt: skip
    assert provider.get_api_key() == "provider-key"
    assert str(provider) == "[DeepSeek Provider deepseek-flash]"
    keyed = create_deepseek_provider("deepseek:x", {"config": {"config": {"apiKey": "secret"}}})
    serialized = keyed.to_json()
    assert serialized["provider"] == "deepseek"
    assert serialized["model"] == "x"
    assert "apiKey" not in serialized["config"]
    assert keyed.config["apiKey"] == "secret"


async def test_deepseek_missing_key_error():
    provider = create_deepseek_provider("deepseek:deepseek-flash")
    with pytest.raises(ValueError, match="Set the DEEPSEEK_API_KEY environment variable"):
        await provider.call_api("Hello")


# --- Cerebras -----------------------------------------------------------------------------------


def test_cerebras_create():
    provider = create_cerebras_provider("cerebras:llama3.1-8b")
    assert provider.id() == "llama3.1-8b"
    assert "OpenAI" in str(provider)
    assert provider.config["apiKeyEnvar"] == "CEREBRAS_API_KEY"
    assert provider.config["apiBaseUrl"] == "https://api.cerebras.ai/v1"
    assert provider.config["passthrough"] == {}
    custom = create_cerebras_provider(
        "cerebras:llama3.1-8b",
        {"config": {"config": {"basePath": "/custom/path", "temperature": 0.8}}},
    )
    assert custom.config["passthrough"] == {"temperature": 0.8}
    assert create_cerebras_provider("cerebras:model:with:colons").id() == "model:with:colons"


@respx.mock
async def test_cerebras_pricing_overrides_stay_local():
    url = "https://cerebras.example.invalid/v1/chat/completions"
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    route = respx.post(url).mock(return_value=ok(_chat("fixture response", usage=usage)))
    provider = create_cerebras_provider("cerebras:gpt-oss-120b", {"config": {"config": {
        "cost": 9e-6, "inputCost": 1e-6, "outputCost": 2e-6, "apiKey": "catalog-fixture-key",
        "apiBaseUrl": "https://cerebras.example.invalid/v1",
        "headers": {"x-catalog-fixture": "local-options"}, "temperature": 0.8,
        "passthrough": {"top_p": 0.6},
    }}})  # fmt: skip
    prepared = await provider.get_openai_body("test prompt")
    body, config = prepared["body"], prepared["config"]
    for key in ("cost", "inputCost", "outputCost"):
        assert key not in body
    assert body["temperature"] == 0.8
    assert config["passthrough"] == {"temperature": 0.8, "top_p": 0.6}
    assert provider.calculate_response_cost({"usage": usage}, config, False) == pytest.approx(3)
    result = await provider.call_api("test prompt")
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer catalog-fixture-key"
    assert request.headers["x-catalog-fixture"] == "local-options"
    sent = sent_json(route)
    assert sent["model"] == "gpt-oss-120b"
    assert sent["top_p"] == 0.6
    for key in ("apiKey", "apiBaseUrl", "headers", "cost", "inputCost", "outputCost"):
        assert key not in sent
    assert result["output"] == "fixture response"
    assert result["cost"] == pytest.approx(3)


async def test_cerebras_max_tokens():
    provider = create_cerebras_provider(
        "cerebras:llama3.1-8b", {"config": {"config": {"max_completion_tokens": 50}}}
    )
    body = (await provider.get_openai_body("p"))["body"]
    assert body["max_completion_tokens"] == 50
    assert "max_tokens" not in body
    plain = (await create_cerebras_provider("cerebras:llama3.1-8b").get_openai_body("p"))["body"]
    assert plain["max_tokens"] == 1024


@pytest.mark.parametrize(
    ("model", "cost"), [("gpt-oss-120b", 1.1), ("gemma-4-31b", 2.48), ("zai-glm-4.7", 5)]
)
def test_cerebras_published_cost(model, cost):
    assert calculate_cerebras_cost(model, {}, 1_000_000, 1_000_000) == pytest.approx(cost)


@pytest.mark.parametrize(
    ("config", "prompt", "completion", "cost"),
    [
        ({"inputCost": 1e-6, "outputCost": 2e-6}, 1e6, 1e6, 3),
        ({"cost": 2e-6}, 1e6, 1e6, 4),
        ({"cost": 2e-6, "inputCost": 0}, 1e6, 1e6, 2),
        ({"cost": 0}, 1e6, 1e6, 0),
        ({"inputCost": 1e-6}, 1e6, 0, 1),
        ({"outputCost": 2e-6}, 0, 1e6, 2),
        ({"cost": 1e-6}, 0, 0, 0),
    ],
)
def test_cerebras_uncatalogued_rates(config, prompt, completion, cost):
    assert calculate_cerebras_cost("dedicated-model", config, prompt, completion) == pytest.approx(
        cost
    )


def test_cerebras_cost_edge_cases():
    assert calculate_cerebras_cost("unknown-model", {}, 1e6, 1e6) is None
    assert calculate_cerebras_cost("gpt-oss-120b", {}, None, 1e6) is None
    for config in ({}, {"inputCost": 1e-6}, {"outputCost": 2e-6}):
        assert calculate_cerebras_cost("dedicated-model", config, 1e6, 1e6) is None
    for prompt, completion in [(None, 1e6), (math.nan, 1e6), (1e6, math.inf)]:
        for model in ("gpt-oss-120b", "dedicated-model"):
            assert calculate_cerebras_cost(model, {"cost": 1e-6}, prompt, completion) is None
    provider = create_cerebras_provider("cerebras:gpt-oss-120b")
    usage = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}
    assert provider.calculate_response_cost(usage, {}, False) == pytest.approx(1.1)
    prompt_level = {"inputCost": 3e-6, "outputCost": 4e-6,
                    "passthrough": {"inputCost": 1e-6, "outputCost": 2e-6}}  # fmt: skip
    assert provider.calculate_response_cost(usage, prompt_level, False) == pytest.approx(7)


async def test_cerebras_loads_from_registry():
    provider = await load_api_provider("cerebras:llama3.1-8b")
    assert provider.id() == "llama3.1-8b"
    assert provider.config["passthrough"] == {}
    # Families precede the JavaScript-file provider, as upstream's providerFamilies do.
    js_named = await load_api_provider("cerebras:model.js")
    assert js_named.id() == "model.js"


# --- Table families -----------------------------------------------------------------------------


def _make(path: str, config: dict | None = None, **options):
    return table.create_provider(path, {"config": config or {}, "env": None, **options})


@pytest.mark.parametrize("prefix", ["togetherai", "aimlapi", "cometapi", "nscale"])
@pytest.mark.parametrize(
    ("route", "cls"),
    [
        ("chat:", OpenAiChatCompletionProvider),
        ("completion:", OpenAiCompletionProvider),
        ("embedding:", OpenAiEmbeddingProvider),
        ("embeddings:", OpenAiEmbeddingProvider),
        ("", OpenAiChatCompletionProvider),
    ],
)
def test_table_routes(prefix, route, cls):
    provider = _make(f"{prefix}:{route}vendor/model:v1")
    assert isinstance(provider, cls)
    assert provider.model_name == "vendor/model:v1"
    assert provider.id() == "vendor/model:v1"
    assert provider.get_api_url() == table.COMPATIBLE_FAMILIES[prefix].api_base_url


def test_togetherai_passthrough():
    provider = _make(
        "togetherai:meta-llama/Llama-3",
        {"temperature": 0.7, "repetition_penalty": 1.1, "tools": [{"type": "function"}],
         "apiKey": "k", "headers": {"x": "y"}, "passthrough": {"top_k": 50}},
    )  # fmt: skip
    assert provider.config["apiBaseUrl"] == "https://api.together.xyz/v1"
    assert provider.config["apiKeyEnvar"] == "TOGETHER_API_KEY"
    # The chat provider resolves tools itself; connection options stay local.
    assert provider.config["passthrough"] == {
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "top_k": 50,
    }
    completion = _make("togetherai:completion:m", {"tools": [1], "apiKeyEnvar": "MY_KEY"})
    assert completion.config["passthrough"] == {"tools": [1]}
    assert completion.config["apiKeyEnvar"] == "MY_KEY"
    # A route segment naming an Object prototype member is part of the model name.
    assert _make("togetherai:constructor:x").model_name == "constructor:x"


@respx.mock
async def test_togetherai_request_body(_env):
    _env.setenv("TOGETHER_API_KEY", "together-key")
    route = respx.post("https://api.together.xyz/v1/chat/completions").mock(
        return_value=ok(_chat("hi"))
    )
    provider = await load_api_provider(
        "togetherai:meta-llama/Llama-3",
        options={"config": {"temperature": 0.3, "maxRetries": 0, "headers": {"X-T": "1"}}},
    )
    result = await provider.call_api("Hello")
    assert result["output"] == "hi"
    body = sent_json(route)
    assert body["temperature"] == 0.3
    for key in ("maxRetries", "headers", "basePath"):
        assert key not in body
    assert route.calls[0].request.headers["Authorization"] == "Bearer together-key"
    assert route.calls[0].request.headers["X-T"] == "1"


def test_aimlapi_and_cometapi_force_base_url():
    aiml = _make("aimlapi:chat:gpt-4o", {"apiBaseUrl": "https://other", "apiKeyEnvar": "X"})
    assert aiml.config["apiBaseUrl"] == "https://api.aimlapi.com/v1"
    assert aiml.config["apiKeyEnvar"] == "AIML_API_KEY"
    comet = _make("cometapi:chat:gpt-4o", {"apiBaseUrl": "https://other", "apiKeyEnvar": "X"})
    assert comet.config["apiBaseUrl"] == "https://api.cometapi.com/v1"
    assert comet.config["apiKeyEnvar"] == "X"
    with pytest.raises(ValueError, match="CometAPI image provider is not supported"):
        _make("cometapi:image:dall-e-3")


def test_nscale_config_and_credentials(_env):
    provider = _make(
        "nscale:chat:qwen/qwen-3",
        {"temperature": 0.5, "apiKey": "k", "basePath": "/x", "apiBaseUrl": "https://private/v1"},
    )
    assert provider.config == {
        "apiKey": "k",
        "basePath": "/x",
        "apiBaseUrl": "https://private/v1",
        "passthrough": {"temperature": 0.5},
    }
    _env.setenv("NSCALE_API_KEY", "legacy-key")
    assert _make("nscale:m").get_api_key() == "legacy-key"
    _env.setenv("NSCALE_SERVICE_TOKEN", "service-token")
    assert _make("nscale:m").get_api_key() == "service-token"
    assert _make("nscale:m", {"apiKey": "explicit"}).get_api_key() == "explicit"
    assert "NSCALE_SERVICE_TOKEN" in _make("nscale:m").get_missing_api_key_error_message()
    with pytest.raises(ValueError, match="Nscale image provider is not supported"):
        _make("nscale:image:flux")


@respx.mock
async def test_nscale_request_keeps_options_local(_env):
    _env.setenv("NSCALE_SERVICE_TOKEN", "service-token")
    route = respx.post("https://inference.api.nscale.com/v1/chat/completions").mock(
        return_value=ok(_chat("ok"))
    )
    provider = await load_api_provider(
        "nscale:qwen", options={"config": {"top_p": 0.4, "passthrough": {"seed": 1}}}
    )
    await provider.call_api("Hello")
    body = sent_json(route)
    assert body["top_p"] == 0.4
    assert body["seed"] == 1
    assert "basePath" not in body
    assert "passthrough" not in body
    assert route.calls[0].request.headers["Authorization"] == "Bearer service-token"


async def test_nscale_missing_key_raises():
    provider = _make("nscale:qwen")
    with pytest.raises(ValueError, match="Set the NSCALE_SERVICE_TOKEN environment variable"):
        await provider.call_api("Hello")


# --- Small subclasses ---------------------------------------------------------------------------


@respx.mock
async def test_atlascloud(_env):
    provider = simple.create_atlascloud_provider(
        "atlascloud:deepseek-ai/DeepSeek-V3", {"config": {"config": {"apiKey": "secret"}}}
    )
    assert provider.id() == "atlascloud:deepseek-ai/DeepSeek-V3"
    assert str(provider) == "[Atlas Cloud Provider deepseek-ai/DeepSeek-V3]"
    assert provider.to_json() == {
        "provider": "atlascloud",
        "model": "deepseek-ai/DeepSeek-V3",
        "config": {
            "apiBaseUrl": "https://api.atlascloud.ai/v1",
            "apiKeyEnvar": "ATLASCLOUD_API_KEY",
        },
    }
    custom = simple.create_atlascloud_provider(
        "atlascloud:m",
        {"config": {"config": {"apiBaseUrl": "https://custom/v1", "apiKeyEnvar": "MY"}}},
    )
    assert custom.config["apiBaseUrl"] == "https://custom/v1"
    assert custom.config["apiKeyEnvar"] == "MY"
    route = respx.post("https://api.atlascloud.ai/v1/chat/completions").mock(
        return_value=ok(_chat("atlas"))
    )
    assert (await provider.call_api("Hi"))["output"] == "atlas"
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret"
    env_provider = simple.create_atlascloud_provider(
        "atlascloud:m", {"config": {"config": {}}, "env": {"ATLASCLOUD_API_KEY": "env-key"}}
    )
    assert env_provider.get_api_key() == "env-key"


def test_llamaapi():
    provider = simple.create_llama_api_provider(
        "llamaapi:chat:Llama-4",
        {"config": {"id": "nested-id", "config": {
            "apiKey": "k", "apiBaseUrl": "https://ignored", "temperature": 0.5,
            "passthrough": {"top_p": 0.9},
        }}, "id": "factory-id"},
    )  # fmt: skip
    assert provider.model_name == "Llama-4"
    assert provider.id() == "factory-id"
    assert type(provider).__name__ == "LlamaApiProvider"
    assert str(provider) == "[Llama API Provider Llama-4]"
    assert provider.config["apiBaseUrl"] == "https://api.llama.com/compat/v1"
    assert provider.config["apiKeyEnvar"] == "LLAMA_API_KEY"
    assert provider.config["passthrough"] == {"top_p": 0.9}
    serialized = provider.to_json()
    assert serialized["provider"] == "llamaapi"
    assert "apiKey" not in serialized["config"]
    assert provider.config["apiKey"] == "k"
    bare = simple.create_llama_api_provider("llamaapi:Llama-3.3")
    assert bare.id() == "llamaapi:Llama-3.3"


def test_helicone():
    provider = simple.HeliconeGatewayProvider("openai/gpt-4o")
    assert provider.config["apiBaseUrl"] == "http://localhost:8080/ai"
    assert provider.id() == "helicone-gateway:openai/gpt-4o"
    assert str(provider) == "[Helicone AI Gateway http://localhost:8080/ai]"
    assert provider.get_api_key() == "placeholder-api-key"
    routed = simple.HeliconeGatewayProvider(
        "x", {"config": {"router": "prod", "baseUrl": "https://gw", "model": "anthropic/c"}}
    )
    assert routed.config["apiBaseUrl"] == "https://gw/router/prod"
    assert routed.id() == "helicone-gateway:prod:anthropic/c"
    scoped = simple.HeliconeGatewayProvider("m", {"env": {"HELICONE_API_KEY": "scoped"}})
    assert scoped.get_api_key() == "scoped"
    assert "apiKey" not in scoped.config
    with pytest.raises(ValueError, match="requires a model in format helicone"):
        simple.create_helicone("helicone:", {})


def test_cloudera_and_jfrog(_env):
    _env.setenv("CDP_DOMAIN", "ml.example.com")
    cloudera = simple.create_cloudera("cloudera:llama", {"config": {}})
    assert (
        cloudera.get_api_url()
        == "https://ml.example.com/namespaces/serving-default/endpoints/llama/v1"
    )
    assert cloudera.config["apiKeyEnvar"] == "CDP_TOKEN"
    jfrog = simple.create_jfrog("qwak:my-model", {"config": {"baseUrl": "https://jf.example/v1//"}})
    assert jfrog.get_api_url() == "https://jf.example/v1/my-model"
    assert jfrog.config["apiKeyEnvar"] == "QWAK_TOKEN"


@pytest.mark.parametrize(
    "url",
    [
        "https://gateway.example",
        "https://gateway.example/",
        "https://gateway.example//",
        "https://gateway.example/v1",
        "https://gateway.example/v1/",
        "https://gateway.example/v1//",
    ],
)
def test_envoy_environment_urls(_env, url):
    _env.setenv("ENVOY_API_BASE_URL", url)
    provider = simple.create_envoy_provider("envoy:route:stable")
    assert provider.get_api_url() == "https://gateway.example/v1"
    assert provider.model_name == "route:stable"


@respx.mock
async def test_envoy_requests(_env):
    _env.setenv("ENVOY_API_BASE_URL", "https://gateway.example?token=x#route")
    route = respx.post("https://gateway.example/v1/chat/completions?token=x").mock(
        return_value=ok(_chat("Hello"))
    )
    provider = simple.create_envoy_provider("envoy:route:stable")
    assert (await provider.call_api("Hello"))["output"] == "Hello"
    assert route.call_count == 1
    explicit = simple.create_envoy_provider(
        "envoy:route:stable",
        {"id": "custom-id", "config": {"config": {"apiBaseUrl": "https://configured.example"}}},
    )
    assert explicit.id() == "custom-id"
    assert explicit.get_api_url() == "https://configured.example"


async def test_envoy_errors(_env):
    with pytest.raises(ValueError, match="requires a model name"):
        simple.create_envoy_provider("envoy:")
    with pytest.raises(ValueError, match="requires a gateway URL"):
        simple.create_envoy_provider("envoy:m")
    _env.setenv("ENVOY_API_BASE_URL", "ftp://gateway")
    with pytest.raises(ValueError, match="requires a valid gateway URL"):
        simple.create_envoy_provider("envoy:m")
    _env.setenv("ENVOY_API_BASE_URL", "https://gateway.example")
    provider = await load_api_provider(
        "envoy:route:stable", options={"config": {"apiKeyEnvar": "MISSING_ENVOY_TEST_KEY"}}
    )
    with pytest.raises(ValueError, match="API key"):
        await provider.call_api("Hello")


@respx.mock
async def test_envoy_optional_auth_with_headers(_env):
    _env.setenv("ENVOY_API_BASE_URL", "https://gateway.example")
    route = respx.post("https://gateway.example/v1/chat/completions").mock(
        return_value=ok(_chat("Hello"))
    )
    provider = await load_api_provider(
        "envoy:route:stable",
        options={"config": {"apiKeyEnvar": "MISSING_ENVOY_TEST_KEY", "apiKeyRequired": False,
                            "headers": {"x-api-key": "test-header-key"}}},
    )  # fmt: skip
    assert (await provider.call_api("Hello"))["output"] == "Hello"
    headers = route.calls[0].request.headers
    assert headers["x-api-key"] == "test-header-key"
    assert "authorization" not in headers


# --- Skipped families ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("vercel:openai/gpt-4o", "Vercel AI SDK"),
        ("mistral:mistral-large-latest", "Mistral provider is not supported by promptphoo"),
        ("ai21:jamba-large", "AI21 provider is not supported by promptphoo"),
    ],
)
async def test_skipped_families_fail_clearly(path, message):
    assert find_factory(path) is not None
    with pytest.raises(ValueError, match=message):
        await load_api_provider(path)
