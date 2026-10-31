"""Groq, Fireworks, Hyperbolic, NVIDIA NIM, Moonshot and MiniMax providers. Ports the
deterministic cases from test/providers/groq/*.test.ts, fireworks.test.ts,
hyperbolic{,/chat}.test.ts, nvidia.test.ts, moonshot.test.ts, minimax.test.ts and
minimax-moonshot-request.test.ts, with HTTP mocked by respx."""

from __future__ import annotations

import httpx
import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo.jsvalues import json_stringify
from promptphoo.providers.compatible import fireworks, groq, hyperbolic, minimax, moonshot, nvidia
from promptphoo.providers.compatible.fireworks import (
    FireworksEmbeddingProvider,
    FireworksProvider,
    calculate_fireworks_cost,
    create_fireworks_provider,
    read_fireworks_cached_prompt_tokens,
)
from promptphoo.providers.compatible.groq import (
    GroqProvider,
    groq_supports_temperature,
    is_groq_reasoning_model,
)
from promptphoo.providers.compatible.hyperbolic import (
    HYPERBOLIC_CHAT_MODELS,
    HYPERBOLIC_REASONING_MODELS,
    HyperbolicProvider,
    calculate_hyperbolic_cost,
    create_hyperbolic_provider,
)
from promptphoo.providers.compatible.minimax import (
    MINIMAX_CHAT_MODELS,
    calculate_minimax_cost,
    create_minimax_provider,
)
from promptphoo.providers.compatible.moonshot import (
    calculate_moonshot_cost,
    create_moonshot_provider,
)
from promptphoo.providers.compatible.nvidia import (
    NvidiaProvider,
    calculate_nvidia_cost,
    create_nvidia_provider,
)
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

VENDOR_ENV = [
    "GROQ_API_KEY", "GROQ_PROXY_KEY", "GROQ_SCOPED_KEY", "GROQ_MISSING_KEY",
    "FIREWORKS_API_KEY", "FIREWORKS_API_BASE_URL", "CUSTOM_FIREWORKS_KEY",
    "HYPERBOLIC_API_KEY", "NVIDIA_API_KEY", "NVIDIA_API_BASE_URL", "CUSTOM_NVIDIA_KEY",
    "MOONSHOT_API_KEY", "CUSTOM_MOONSHOT_KEY", "MINIMAX_API_KEY", "CUSTOM_MINIMAX_KEY",
]  # fmt: skip

GROQ_API_BASE = "https://api.groq.com/openai/v1"
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
NVIDIA_NIM_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"


pytestmark = pytest.mark.usefixtures(openai_env.__name__)


@pytest.fixture(autouse=True)
def vendor_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in VENDOR_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _chat(content: str = "Hello", **usage) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _prompt_context(config: dict) -> dict:
    return {"vars": {}, "prompt": {"raw": "Hello", "label": "fixture", "config": config}}


async def _body(provider, prompt: str = "Hello", context=None) -> dict:
    return (await provider.get_openai_body(prompt, context))["body"]


def _stub_super_call(monkeypatch, response: dict) -> None:
    async def fake(self, prompt, context=None, options=None):
        return dict(response)

    monkeypatch.setattr(OpenAiChatCompletionProvider, "call_api", fake)


# --- Groq ----------------------------------------------------------------------------------


def test_groq_identification(vendor_env):
    vendor_env.setenv("GROQ_API_KEY", "test-key")
    provider = GroqProvider("mixtral-8x7b-32768", {})
    assert provider.model_name == "mixtral-8x7b-32768"
    assert provider.id() == "groq:mixtral-8x7b-32768"
    assert provider.get_gen_ai_system() == "groq"
    assert str(provider) == "[Groq Provider mixtral-8x7b-32768]"
    assert provider.config["apiKeyEnvar"] == "GROQ_API_KEY"
    assert provider.config["apiBaseUrl"] == GROQ_API_BASE
    custom = GroqProvider("mixtral-8x7b-32768", {"id": "customer:custom-label"})
    assert custom.id() == "customer:custom-label"
    assert custom.get_gen_ai_system() == "groq"


@pytest.mark.parametrize(
    ("model", "reasoning", "temperature"),
    [
        ("mixtral-8x7b-32768", False, True),
        ("deepseek-r1-distill-llama-70b", False, True),
        ("openai/gpt-oss-120b", True, True),
        ("qwen/qwen3.6-27b", True, True),
        ("o1-mini", True, False),
    ],
)
def test_groq_reasoning_and_temperature(model, reasoning, temperature):
    provider = GroqProvider(model, {})
    assert provider.is_reasoning_model() is reasoning
    assert provider.supports_temperature() is temperature


def test_groq_util_functions():
    assert not is_groq_reasoning_model("deepseek-r1-distill-llama-70b")
    assert not is_groq_reasoning_model("deepseek-r1")
    for model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "gpt-oss-safeguard"):
        assert is_groq_reasoning_model(model)
    for model in ("qwen/qwen3.6-27b", "qwen-2.5", "deepseek-r1-distill-qwen-32b"):
        assert is_groq_reasoning_model(model)
    for model in ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"):
        assert not is_groq_reasoning_model(model)
        assert not groq_supports_temperature(model)
    assert groq_supports_temperature("openai/gpt-oss-120b")


@pytest.mark.parametrize(
    ("model", "max_completion_tokens", "expected"),
    [
        ("openai/gpt-oss-120b", None, {"max_completion_tokens": 100}),
        ("qwen/qwen3.6-27b", None, {"max_completion_tokens": 100}),
        ("openai/gpt-oss-120b", 40, {"max_completion_tokens": 40}),
        ("openai/gpt-oss-120b", 0, {"max_completion_tokens": 0}),
        ("tenant/custom-served-model", None, {"max_tokens": 100}),
    ],
)
async def test_groq_preserves_token_limit(model, max_completion_tokens, expected):
    config = {"max_tokens": 100}
    if max_completion_tokens is not None:
        config["max_completion_tokens"] = max_completion_tokens
    body = await _body(GroqProvider(model, {"config": config}))
    assert body["model"] == model
    assert {k: body.get(k) for k in expected} == expected
    if "max_completion_tokens" in expected:
        assert "max_tokens" not in body


@pytest.mark.parametrize("model", ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"])
@pytest.mark.parametrize(
    ("env_cap", "omit_defaults"),
    [(None, False), ("321", False), ("0", False), ("321", True), (None, True)],
)
async def test_groq_reasoning_passthrough_omits_defaults(vendor_env, model, env_cap, omit_defaults):
    if env_cap is not None:
        vendor_env.setenv("OPENAI_MAX_TOKENS", env_cap)
    provider = GroqProvider(
        "llama-3.3-70b-versatile",
        {"config": {"omitDefaults": omit_defaults, "passthrough": {"model": model}}},
    )
    body = await _body(provider)
    assert body["model"] == model
    assert body.get("max_completion_tokens") is None
    assert "max_tokens" not in body


async def test_groq_explicit_passthrough_token_limit():
    provider = GroqProvider(
        "openai/gpt-oss-120b",
        {"config": {"max_tokens": 100, "passthrough": {"max_completion_tokens": 20}}},
    )
    assert (await _body(provider))["max_completion_tokens"] == 20


def test_groq_to_json():
    provider = GroqProvider(
        "mixtral-8x7b-32768", {"config": {"temperature": 0.7, "max_tokens": 100}}
    )
    assert provider.to_json() == {
        "provider": "groq",
        "model": "mixtral-8x7b-32768",
        "config": {
            "temperature": 0.7,
            "max_tokens": 100,
            "apiKeyEnvar": "GROQ_API_KEY",
            "apiBaseUrl": GROQ_API_BASE,
        },
    }
    secret = GroqProvider("m", {"config": {"apiKey": "secret-api-key", "temperature": 0.7}})
    assert "apiKey" not in secret.to_json()["config"]
    assert secret.config["apiKey"] == "secret-api-key"


@pytest.mark.parametrize(
    ("model", "effort", "reasoning"),
    [
        ("qwen/qwen3.6-27b", "none", True),
        ("openai/gpt-oss-120b", "high", True),
        ("llama-3.3-70b-versatile", "high", False),
    ],
)
@pytest.mark.parametrize(
    "selection", ["direct", "provider passthrough", "prompt passthrough", "unchanged passthrough"]
)
async def test_groq_model_request_contract(model, effort, reasoning, selection):
    if selection in ("direct", "unchanged passthrough"):
        configured = model
    else:
        configured = "qwen/qwen3.6-27b" if model == "openai/gpt-oss-120b" else "openai/gpt-oss-120b"
    config = {
        "reasoning_effort": effort,
        "max_completion_tokens": 4096,
        "max_tokens": 2048,
        "temperature": 0.6,
    }
    if selection != "direct":
        config["passthrough"] = {
            "model": configured if selection == "prompt passthrough" else model
        }
    context = (
        {
            "vars": {},
            "prompt": {
                "raw": "Test prompt",
                "label": "Test prompt",
                "config": {"passthrough": {"model": model}},
            },
        }
        if selection == "prompt passthrough"
        else None
    )
    provider = GroqProvider(configured, {"config": config})
    body = await _body(provider, "Test prompt", context)
    assert body["model"] == model
    assert body["messages"] == [{"role": "user", "content": "Test prompt"}]
    assert body["temperature"] == 0.6
    assert provider.model_name == configured
    assert provider.get_api_url() == GROQ_API_BASE
    if reasoning:
        assert body["reasoning_effort"] == effort
        assert body["max_completion_tokens"] == 4096
        assert "max_tokens" not in body
    else:
        assert body["max_tokens"] == 2048
        assert "max_completion_tokens" not in body
        assert "reasoning_effort" not in body


async def test_groq_service_tiers():
    for tier in ("auto", "on_demand", "flex", "performance"):
        provider = GroqProvider("openai/gpt-oss-120b", {"config": {"service_tier": tier}})
        assert (await _body(provider))["service_tier"] == tier
    provider = GroqProvider("openai/gpt-oss-120b", {"config": {"service_tier": "priority"}})
    with pytest.raises(ValueError, match='Invalid Groq Chat Completions service_tier "priority"'):
        await provider.get_openai_body("Test prompt")


async def test_groq_specific_parameters():
    compound = {
        "tools": {
            "enabled_tools": ["code_interpreter", "web_search"],
            "wolfram_settings": {"authorization": "test-key"},
        }
    }
    search = {
        "exclude_domains": ["example.com"],
        "include_domains": ["trusted.com"],
        "country": "US",
    }
    provider = GroqProvider(
        "openai/gpt-oss-120b",
        {
            "config": {
                "reasoning_format": "hidden",
                "include_reasoning": False,
                "compound_custom": compound,
                "search_settings": search,
            }
        },
    )
    body = await _body(provider, "Test prompt")
    assert body["reasoning_format"] == "hidden"
    assert body["include_reasoning"] is False
    assert body["compound_custom"] == compound
    assert body["search_settings"] == search
    plain = await _body(GroqProvider("mixtral-8x7b-32768", {}), "Test prompt")
    assert plain["model"] == "mixtral-8x7b-32768"
    assert plain["messages"] == [{"role": "user", "content": "Test prompt"}]


@pytest.mark.parametrize("model", [None, 123])
@respx.mock
async def test_groq_malformed_passthrough_model(model):
    error = {"error": {"message": "Fixture model must be a string"}}
    route = respx.post("https://groq.invalid/openai/v1/chat/completions").mock(
        return_value=ok(error, status=400)
    )
    provider = GroqProvider(
        "llama-3.3-70b-versatile",
        {
            "config": {
                "apiKey": "fixture-groq-key",
                "apiBaseUrl": "https://groq.invalid/openai/v1",
                "max_tokens": 100,
                "passthrough": {"model": model},
                "maxRetries": 0,
            }
        },
    )
    result = await provider.call_api("Hello")
    assert result["error"] == f"API error: 400 Bad Request\n{json_stringify(error)}"
    assert result["metadata"]["http"]["status"] == 400
    assert route.call_count == 1
    body = sent_json(route)
    assert body["model"] == model
    assert body["max_tokens"] == 100


@pytest.fixture
def groq_connection_env(request: pytest.FixtureRequest, vendor_env):
    request.getfixturevalue(openai_env.__name__)  # clear OPENAI_* first
    vendor_env.setenv("GROQ_API_KEY", "groq-process-key")
    vendor_env.setenv("GROQ_PROXY_KEY", "proxy-process-key")
    vendor_env.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    vendor_env.setenv("OPENAI_API_BASE_URL", "https://unrelated.invalid/v1")
    vendor_env.setenv("OPENAI_API_HOST", "unrelated.invalid")
    return vendor_env


@pytest.mark.usefixtures("groq_connection_env")
@pytest.mark.parametrize("api_key", [None, "explicit-key"])
@respx.mock
async def test_groq_custom_endpoint_and_key_variable(api_key):
    route = respx.post("http://127.0.0.1:9000/groq/chat/completions").mock(return_value=ok(_chat()))
    config = {"apiBaseUrl": "http://127.0.0.1:9000/groq", "apiKeyEnvar": "GROQ_PROXY_KEY"}
    if api_key:
        config["apiKey"] = api_key
    provider = GroqProvider("private/model:version", {"id": "custom-provider-id", "config": config})
    result = await provider.call_api("Hello")
    assert result["output"] == "Hello"
    assert result["tokenUsage"]["prompt"] == 5
    assert provider.id() == "custom-provider-id"
    request = route.calls[0].request
    assert request.headers["Authorization"] == f"Bearer {api_key or 'proxy-process-key'}"
    body = sent_json(route)
    assert body["model"] == "private/model:version"
    assert "apiBaseUrl" not in body and "apiKeyEnvar" not in body


@pytest.mark.usefixtures("groq_connection_env")
@respx.mock
async def test_groq_key_resolution_scopes():
    route = respx.post(f"{GROQ_API_BASE}/chat/completions").mock(return_value=ok(_chat()))
    await GroqProvider(
        "private/model",
        {"config": {"apiKeyEnvar": "GROQ_SCOPED_KEY"}, "env": {"GROQ_SCOPED_KEY": "scoped-key"}},
    ).call_api("Hello")
    assert route.calls[-1].request.headers["Authorization"] == "Bearer scoped-key"
    # Process env keeps precedence for the selected key variable.
    await GroqProvider(
        "private/model",
        {"config": {"apiKeyEnvar": "GROQ_PROXY_KEY"}, "env": {"GROQ_PROXY_KEY": "scoped-key"}},
    ).call_api("Hello")
    assert route.calls[-1].request.headers["Authorization"] == "Bearer proxy-process-key"
    await GroqProvider("private/model", {"config": {}}).call_api("Hello")
    assert route.calls[-1].request.headers["Authorization"] == "Bearer groq-process-key"


@pytest.mark.usefixtures("groq_connection_env")
@respx.mock
async def test_groq_explicit_key_and_api_host():
    custom = respx.post("http://127.0.0.1:9000/groq/chat/completions").mock(
        return_value=ok(_chat())
    )
    await GroqProvider(
        "private/model",
        {
            "config": {
                "apiBaseUrl": "http://127.0.0.1:9000/groq",
                "apiKeyEnvar": "GROQ_MISSING_KEY",
                "apiKey": "explicit-key",
            },
            "env": {"OPENAI_API_KEY": "unrelated-scoped-openai-key"},
        },
    ).call_api("Hello")
    assert custom.calls[0].request.headers["Authorization"] == "Bearer explicit-key"
    host = respx.post("https://proxy.invalid/v1/chat/completions").mock(return_value=ok(_chat()))
    await GroqProvider(
        "private/model",
        {"config": {"apiHost": "proxy.invalid", "apiBaseUrl": "https://secondary.invalid/v1"}},
    ).call_api("Hello")
    assert host.call_count == 1


@pytest.mark.parametrize("api_key_envar", [None, "GROQ_MISSING_KEY"])
@respx.mock
async def test_groq_missing_key_rejects(groq_connection_env, api_key_envar):
    if api_key_envar:
        groq_connection_env.setenv("GROQ_API_KEY", "unselected-groq-key")
    else:
        groq_connection_env.delenv("GROQ_API_KEY")
    config = {"apiBaseUrl": "http://127.0.0.1:9000/groq"}
    if api_key_envar:
        config["apiKeyEnvar"] = api_key_envar
    provider = GroqProvider(
        "private/model",
        {"config": config, "env": {"OPENAI_API_KEY": "unrelated-scoped-openai-key"}},
    )
    assert provider.get_api_key() is None
    with pytest.raises(ValueError, match=f"Set the {api_key_envar or 'GROQ_API_KEY'} environment"):
        await provider.call_api("Hello")


def test_groq_create_provider():
    assert groq.create_provider("groq:openai/gpt-oss-120b", {}).id() == "groq:openai/gpt-oss-120b"
    with pytest.raises(ValueError, match='Invalid groq provider path: "groq:"'):
        groq.create_provider("groq:", {})
    with pytest.raises(ValueError, match="Invalid groq:responses provider path"):
        groq.create_provider("groq:responses:", {})


# --- Fireworks -----------------------------------------------------------------------------


def test_fireworks_defaults_and_overrides(vendor_env):
    provider = FireworksProvider(FIREWORKS_MODEL, {})
    assert provider.model_name == FIREWORKS_MODEL
    assert provider.id() == f"fireworks:{FIREWORKS_MODEL}"
    assert str(provider) == f"[Fireworks AI Provider {FIREWORKS_MODEL}]"
    assert provider.config["apiBaseUrl"] == FIREWORKS_API_BASE
    assert provider.config["apiKeyEnvar"] == "FIREWORKS_API_KEY"
    empty = FireworksProvider(FIREWORKS_MODEL, {"config": {"apiBaseUrl": "", "apiKeyEnvar": ""}})
    assert empty.config["apiBaseUrl"] == FIREWORKS_API_BASE
    assert empty.config["apiKeyEnvar"] == "FIREWORKS_API_KEY"
    vendor_env.setenv("CUSTOM_FIREWORKS_KEY", "test-only-fake-key")
    custom = FireworksProvider(
        FIREWORKS_MODEL,
        {
            "config": {
                "apiBaseUrl": "https://proxy.example.com/fireworks/v1",
                "apiKeyEnvar": "CUSTOM_FIREWORKS_KEY",
            }
        },
    )
    assert custom.config["apiBaseUrl"] == "https://proxy.example.com/fireworks/v1"
    assert custom.get_api_key() == "test-only-fake-key"
    vendor_env.setenv("FIREWORKS_API_BASE_URL", "https://self-hosted.example.com/v1")
    assert (
        FireworksProvider(FIREWORKS_MODEL, {}).config["apiBaseUrl"]
        == "https://self-hosted.example.com/v1"
    )
    explicit = FireworksProvider(
        FIREWORKS_MODEL, {"config": {"apiBaseUrl": "https://explicit.example.com/v1"}}
    )
    assert explicit.config["apiBaseUrl"] == "https://explicit.example.com/v1"


def test_fireworks_never_uses_openai_settings(vendor_env):
    vendor_env.setenv("OPENAI_API_KEY", "sk-openai")
    vendor_env.setenv("OPENAI_ORGANIZATION", "org-openai")
    vendor_env.setenv("OPENAI_API_HOST", "openai-proxy.example.com")
    vendor_env.setenv("OPENAI_API_BASE_URL", "https://openai-proxy.example.com/v1")
    vendor_env.setenv("OPENAI_BASE_URL", "https://openai-proxy.example.com/v1")
    for cls in (FireworksProvider, FireworksEmbeddingProvider):
        provider = cls(FIREWORKS_MODEL, {"config": {"organization": "org-from-config"}})
        assert provider.get_api_key() is None
        assert provider.get_organization() is None
        assert provider.get_api_url() == FIREWORKS_API_BASE
    proxied = FireworksProvider(
        FIREWORKS_MODEL, {"config": {"apiHost": "fireworks-proxy.internal.example.com"}}
    )
    assert proxied.get_api_url() == "https://fireworks-proxy.internal.example.com/v1"
    vendor_env.setenv("FIREWORKS_API_KEY", "fw-embed-key")
    assert FireworksEmbeddingProvider("m", {}).get_api_key() == "fw-embed-key"


def test_fireworks_embedding_identity():
    model = "accounts/fireworks/models/qwen3-embedding-8b"
    provider = FireworksEmbeddingProvider(model, {})
    assert provider.id() == f"fireworks:embedding:{model}"
    assert str(provider) == f"[Fireworks AI Embedding Provider {model}]"


async def test_fireworks_cost_from_configured_rates(monkeypatch):
    _stub_super_call(monkeypatch, {"output": "ok", "tokenUsage": {"prompt": 10, "completion": 5}})
    provider = FireworksProvider(
        FIREWORKS_MODEL, {"config": {"inputCost": 0.001, "outputCost": 0.002}}
    )
    assert (await provider.call_api("hi"))["cost"] == 0.02
    _stub_super_call(
        monkeypatch,
        {"output": "ok", "tokenUsage": {"prompt": 10, "completion": 5}, "cost": 0.123},
    )
    assert (await FireworksProvider(FIREWORKS_MODEL, {}).call_api("hi"))["cost"] == 0.123


def test_calculate_fireworks_cost():
    assert calculate_fireworks_cost({"inputCost": 0.001}, 10, 5) is None
    assert calculate_fireworks_cost({"inputCost": 0.001, "outputCost": 0.002}, 10, 5) == 0.02
    assert calculate_fireworks_cost({"cost": 0.003}, 10, 5, True) == 0
    assert (
        calculate_fireworks_cost({"inputCost": 0.001, "outputCost": 0.002}, None, None, True) == 0
    )
    rates = {"inputCost": 0.001, "outputCost": 0.002}
    assert calculate_fireworks_cost(rates, 100, 10, False, 40) == pytest.approx(0.12)
    discounted = {**rates, "cacheReadInputCost": 0.0001}
    assert calculate_fireworks_cost(discounted, 100, 10, False, 40) == pytest.approx(
        0.001 * 60 + 0.0001 * 40 + 0.002 * 10
    )


def test_read_fireworks_cached_prompt_tokens():
    def resp(details=None, headers=None):
        out: dict = {"tokenUsage": {"completionDetails": details} if details else {}}
        if headers is not None:
            out["metadata"] = {"http": {"headers": headers}}
        return out

    assert read_fireworks_cached_prompt_tokens(resp({"cacheReadInputTokens": 30})) == 30
    assert (
        read_fireworks_cached_prompt_tokens(resp(None, {"fireworks-cached-prompt-tokens": "40"}))
        == 40
    )
    assert (
        read_fireworks_cached_prompt_tokens(resp(None, {"Fireworks-Cached-Prompt-Tokens": "12"}))
        == 12
    )
    both = resp({"cacheReadInputTokens": 30}, {"fireworks-cached-prompt-tokens": "50"})
    assert read_fireworks_cached_prompt_tokens(both) == 50
    assert read_fireworks_cached_prompt_tokens(resp()) == 0
    bad = resp(None, {"fireworks-cached-prompt-tokens": "not-a-number"})
    assert read_fireworks_cached_prompt_tokens(bad) == 0


def test_create_fireworks_provider():
    assert create_fireworks_provider(f"fireworks:{FIREWORKS_MODEL}", {}).id() == (
        f"fireworks:{FIREWORKS_MODEL}"
    )
    assert create_fireworks_provider("fireworks:vendor/model:tag-v1", {}).id() == (
        "fireworks:vendor/model:tag-v1"
    )
    with pytest.raises(ValueError, match="needs a model identifier"):
        create_fireworks_provider("fireworks:", {})
    with pytest.raises(ValueError, match="reserved for a future dedicated provider"):
        create_fireworks_provider("fireworks:image:some-model", {})
    for subtype in ("embedding", "embeddings"):
        provider = create_fireworks_provider(
            f"fireworks:{subtype}:accounts/fireworks/models/qwen3-embedding-8b", {}
        )
        assert isinstance(provider, FireworksEmbeddingProvider)
        assert provider.id() == "fireworks:embedding:accounts/fireworks/models/qwen3-embedding-8b"
    with pytest.raises(ValueError, match="embedding provider needs a model identifier"):
        create_fireworks_provider("fireworks:embedding:", {})
    provider = fireworks.create_provider(
        f"fireworks:{FIREWORKS_MODEL}",
        {"config": {"temperature": 0.5}, "env": {"FIREWORKS_API_KEY": "env-key"}},
    )
    assert isinstance(provider, FireworksProvider)
    assert provider.config["temperature"] == 0.5
    assert provider.get_api_key() == "env-key"


@respx.mock
async def test_fireworks_call_uses_cached_prompt_header():
    route = respx.post(f"{FIREWORKS_API_BASE}/chat/completions").mock(
        return_value=ok(
            _chat("hi", prompt_tokens=100, completion_tokens=10, total_tokens=110),
            headers={"fireworks-cached-prompt-tokens": "40"},
        )
    )
    provider = FireworksProvider(
        FIREWORKS_MODEL,
        {
            "config": {
                "apiKey": "fw-key",
                "inputCost": 0.001,
                "outputCost": 0.002,
                "cacheReadInputCost": 0.0001,
            }
        },
    )
    result = await provider.call_api("Hello")
    assert route.calls[0].request.headers["Authorization"] == "Bearer fw-key"
    assert "OpenAI-Organization" not in route.calls[0].request.headers
    assert result["cost"] == pytest.approx(0.001 * 60 + 0.0001 * 40 + 0.002 * 10)


# --- Hyperbolic ----------------------------------------------------------------------------


HYPERBOLIC_OPTIONS = {
    "config": {"config": {"apiKey": "test-key"}},
    "env": {"HYPERBOLIC_API_KEY": "test-key"},
}


def test_hyperbolic_identity_and_json():
    provider = HyperbolicProvider("deepseek-ai/DeepSeek-R1", HYPERBOLIC_OPTIONS)
    assert provider.id() == "hyperbolic:deepseek-ai/DeepSeek-R1"
    assert str(provider) == "[Hyperbolic Provider deepseek-ai/DeepSeek-R1]"
    assert provider.to_json() == {
        "provider": "hyperbolic",
        "model": "deepseek-ai/DeepSeek-R1",
        "config": {
            "apiKeyEnvar": "HYPERBOLIC_API_KEY",
            "apiBaseUrl": "https://api.hyperbolic.xyz/v1",
            "config": {"apiKey": "test-key"},
        },
    }
    assert provider.original_config == {"apiKey": "test-key"}
    assert HyperbolicProvider("deepseek-ai/DeepSeek-R1", {}).is_reasoning_model()
    assert HyperbolicProvider("qwen/QwQ-32B", {}).is_reasoning_model()
    assert not HyperbolicProvider("meta-llama/Llama-3.1-70B", {}).is_reasoning_model()


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (
            {"reasoning_tokens": 100, "accepted_prediction_tokens": 50,
             "rejected_prediction_tokens": 25},
            {"reasoning": 100, "acceptedPrediction": 50, "rejectedPrediction": 25},
        ),
        (
            {"reasoning_tokens": 0, "accepted_prediction_tokens": 0,
             "rejected_prediction_tokens": 0},
            {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
        ),
        (None, None),
    ],
)  # fmt: skip
@respx.mock
async def test_hyperbolic_call_normalizes_usage(details, expected):
    usage = {"total_tokens": 350, "prompt_tokens": 200, "completion_tokens": 150}
    if details:
        usage["completion_tokens_details"] = details
    route = respx.post("https://api.hyperbolic.xyz/v1/chat/completions").mock(
        return_value=ok({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                         "usage": usage})
    )  # fmt: skip
    result = await HyperbolicProvider("deepseek-ai/DeepSeek-R1", HYPERBOLIC_OPTIONS).call_api(
        "test prompt"
    )
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert result["output"] == "answer"
    assert result["tokenUsage"].get("completionDetails") == expected
    assert "raw" not in result
    assert result["cost"] == (0.5 / 1e6) * 200 + (2.18 / 1e6) * 150


async def test_hyperbolic_error_and_cost(monkeypatch):
    _stub_super_call(monkeypatch, {"error": "API error"})
    result = await HyperbolicProvider("deepseek-ai/DeepSeek-R1", {}).call_api("p")
    assert result == {"error": "API error"}
    _stub_super_call(
        monkeypatch, {"output": "x", "tokenUsage": {"prompt": 1000, "completion": 500}}
    )
    result = await HyperbolicProvider("deepseek-ai/DeepSeek-R1", {}).call_api("p")
    assert result["cost"] == (0.5 / 1e6) * 1000 + (2.18 / 1e6) * 500


def test_calculate_hyperbolic_cost():
    expected = (0.5 / 1e6) * 1000 + (2.18 / 1e6) * 500
    assert calculate_hyperbolic_cost("deepseek-ai/DeepSeek-R1", {}, 1000, 500) == expected
    assert calculate_hyperbolic_cost("DeepSeek-R1", {}, 1000, 500) == expected
    assert calculate_hyperbolic_cost("unknown-model", {}, 1000, 500) is None
    assert calculate_hyperbolic_cost("deepseek-ai/DeepSeek-R1", {}, None, 500) is None
    assert calculate_hyperbolic_cost("DeepSeek-R1", {"cost": 0.001}, 1000, 500) == (
        0.001 * 1000 + 0.001 * 500
    )
    assert calculate_hyperbolic_cost(
        "DeepSeek-R1", {"inputCost": 0.001, "outputCost": 0.003}, 1000, 500
    ) == pytest.approx(2.5)
    assert calculate_hyperbolic_cost(
        "DeepSeek-R1", {"cost": 0.5, "inputCost": 0.001, "outputCost": 0.003}, 1000, 500
    ) == pytest.approx(2.5)


def test_hyperbolic_tables_and_factory():
    for model in HYPERBOLIC_CHAT_MODELS:
        assert {"input", "output"} <= set(model["cost"])
        assert isinstance(model["aliases"], list)
    ids = {m["id"] for m in HYPERBOLIC_CHAT_MODELS}
    assert set(HYPERBOLIC_REASONING_MODELS) <= ids
    costs = [id(m["cost"]) for m in HYPERBOLIC_CHAT_MODELS]
    assert len(set(costs)) == len(costs)
    assert create_hyperbolic_provider("hyperbolic:deepseek-ai/DeepSeek-R1").id() == (
        "hyperbolic:deepseek-ai/DeepSeek-R1"
    )
    with pytest.raises(ValueError, match="Model name is required"):
        create_hyperbolic_provider("hyperbolic:")
    for route in ("image", "audio"):
        with pytest.raises(ValueError, match="not supported by promptphoo"):
            hyperbolic.create_provider(f"hyperbolic:{route}:model", {})


# --- NVIDIA NIM ----------------------------------------------------------------------------


def test_nvidia_identity_and_json():
    provider = NvidiaProvider(NVIDIA_MODEL, {})
    assert provider.model_name == NVIDIA_MODEL
    assert provider.id() == f"nvidia:{NVIDIA_MODEL}"
    assert str(provider) == f"[NVIDIA NIM Provider {NVIDIA_MODEL}]"
    secret = NvidiaProvider(NVIDIA_MODEL, {"config": {"apiKey": "nv-secret", "temperature": 0.2}})
    assert secret.to_json() == {
        "provider": "nvidia",
        "model": NVIDIA_MODEL,
        "config": {
            "temperature": 0.2,
            "apiBaseUrl": NVIDIA_NIM_API_BASE,
            "apiKeyEnvar": "NVIDIA_API_KEY",
        },
    }


def test_nvidia_defaults_and_overrides(vendor_env):
    provider = NvidiaProvider(NVIDIA_MODEL, {"config": {"apiBaseUrl": "", "apiKeyEnvar": ""}})
    assert provider.config["apiBaseUrl"] == NVIDIA_NIM_API_BASE
    assert provider.config["apiKeyEnvar"] == "NVIDIA_API_KEY"
    vendor_env.setenv("CUSTOM_NVIDIA_KEY", "test-only-fake-key")
    custom = NvidiaProvider(
        NVIDIA_MODEL,
        {
            "config": {
                "apiBaseUrl": "https://proxy.example.com/nvidia/v1",
                "apiKeyEnvar": "CUSTOM_NVIDIA_KEY",
            }
        },
    )
    assert custom.config["apiBaseUrl"] == "https://proxy.example.com/nvidia/v1"
    assert custom.get_api_key() == "test-only-fake-key"
    vendor_env.setenv("NVIDIA_API_BASE_URL", "https://self-hosted.example.com/v1")
    assert NvidiaProvider(NVIDIA_MODEL, {}).config["apiBaseUrl"] == (
        "https://self-hosted.example.com/v1"
    )
    explicit = NvidiaProvider(
        NVIDIA_MODEL, {"config": {"apiBaseUrl": "https://explicit.example.com/v1"}}
    )
    assert explicit.config["apiBaseUrl"] == "https://explicit.example.com/v1"


def test_nvidia_never_uses_openai_settings(vendor_env):
    vendor_env.setenv("OPENAI_API_KEY", "sk-openai")
    vendor_env.setenv("OPENAI_ORGANIZATION", "org-openai")
    vendor_env.setenv("OPENAI_API_HOST", "openai-proxy.example.com")
    vendor_env.setenv("OPENAI_BASE_URL", "https://openai-proxy.example.com/v1")
    provider = NvidiaProvider(NVIDIA_MODEL, {"config": {"organization": "org-from-config"}})
    assert provider.get_api_key() is None
    assert provider.get_organization() is None
    assert provider.get_api_url() == NVIDIA_NIM_API_BASE


async def test_nvidia_cost(monkeypatch):
    _stub_super_call(monkeypatch, {"output": "ok", "tokenUsage": {"prompt": 10, "completion": 5}})
    provider = NvidiaProvider(NVIDIA_MODEL, {"config": {"inputCost": 0.001, "outputCost": 0.002}})
    assert (await provider.call_api("hi"))["cost"] == 0.02
    assert calculate_nvidia_cost({"inputCost": 0.001}, 10, 5) is None
    assert calculate_nvidia_cost({"inputCost": 0.001, "outputCost": 0.002}, 10, 5) == 0.02
    assert calculate_nvidia_cost({"cost": 0.003}, 10, 5, True) == 0


def test_create_nvidia_provider():
    assert create_nvidia_provider(f"nvidia:{NVIDIA_MODEL}").id() == f"nvidia:{NVIDIA_MODEL}"
    assert create_nvidia_provider("nvidia:vendor/model:tag-v1").id() == (
        "nvidia:vendor/model:tag-v1"
    )
    with pytest.raises(ValueError, match='expected "nvidia:<model>"'):
        create_nvidia_provider("nvidia:")
    with pytest.raises(ValueError, match='Unsupported NVIDIA NIM provider subtype "embedding"'):
        create_nvidia_provider("nvidia:embedding:nvidia/nv-embed-v1")
    provider = nvidia.create_provider(
        f"nvidia:{NVIDIA_MODEL}",
        {"config": {"temperature": 0.5}, "env": {"NVIDIA_API_KEY": "env-key"}},
    )
    assert isinstance(provider, NvidiaProvider)
    assert provider.config["temperature"] == 0.5
    assert provider.get_api_key() == "env-key"


# --- Moonshot ------------------------------------------------------------------------------


def test_moonshot_routing():
    provider = create_moonshot_provider("moonshot:moonshot-v1-8k")
    assert provider.id() == "moonshot:moonshot-v1-8k"
    assert create_moonshot_provider("moonshot:chat:moonshot-v1-8k").id() == (
        "moonshot:moonshot-v1-8k"
    )
    assert create_moonshot_provider("moonshot:kimi-k2.6").model_name == "kimi-k2.6"
    for path in ("moonshot:", "moonshot:chat", "moonshot:chat:"):
        assert create_moonshot_provider(path).model_name == "kimi-k3"
    registered = moonshot.create_provider("moonshot:kimi-k2.6", {"id": "kimi", "config": {}})
    assert registered.id() == "kimi"


def test_moonshot_configuration():
    provider = create_moonshot_provider("moonshot:moonshot-v1-8k")
    assert provider.config["apiBaseUrl"] == "https://api.moonshot.ai/v1"
    assert provider.config["apiKeyEnvar"] == "MOONSHOT_API_KEY"
    assert provider.get_api_url() == "https://api.moonshot.ai/v1"
    cn = create_moonshot_provider(
        "moonshot:moonshot-v1-8k", {"config": {"apiBaseUrl": "https://api.moonshot.cn/v1"}}
    )
    assert cn.get_api_url() == "https://api.moonshot.cn/v1"
    assert str(provider) == "[Moonshot Provider moonshot-v1-8k]"
    secret = create_moonshot_provider(
        "moonshot:moonshot-v1-8k", {"config": {"apiKey": "sk-secret", "temperature": 0.2}}
    )
    json_out = secret.to_json()
    assert json_out["provider"] == "moonshot" and json_out["model"] == "moonshot-v1-8k"
    assert "apiKey" not in json_out["config"] and json_out["config"]["temperature"] == 0.2


def test_moonshot_key_resolution(vendor_env):
    assert (
        create_moonshot_provider(
            "moonshot:m", {"config": {"apiKey": "sk-from-config"}}
        ).get_api_key()
        == "sk-from-config"
    )
    vendor_env.setenv("OPENAI_ORGANIZATION", "org-openai-secret")
    provider = create_moonshot_provider("moonshot:m")
    assert provider.get_api_key() is None
    assert provider.get_organization() is None
    assert provider.get_missing_api_key_error_message() == (
        "Moonshot API key is not set. Set the MOONSHOT_API_KEY environment variable or add "
        "`apiKey` to the provider config."
    )
    vendor_env.setenv("MOONSHOT_API_KEY", "sk-from-env")
    assert create_moonshot_provider("moonshot:m").get_api_key() == "sk-from-env"
    vendor_env.setenv("CUSTOM_MOONSHOT_KEY", "sk-custom")
    custom = create_moonshot_provider(
        "moonshot:m", {"config": {"apiKeyEnvar": "CUSTOM_MOONSHOT_KEY"}}
    )
    assert custom.get_api_key() == "sk-custom"


async def test_moonshot_kimi_sampling(vendor_env):
    body = await _body(create_moonshot_provider("moonshot:kimi-k2.6"))
    for key in ("temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"):
        assert key not in body
    body = await _body(
        create_moonshot_provider(
            "moonshot:kimi-k2.6", {"config": {"temperature": 1, "max_tokens": 2048}}
        )
    )
    assert body["temperature"] == 1 and body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body
    body = await _body(
        create_moonshot_provider("moonshot:kimi-k2.6", {"config": {"max_completion_tokens": 4096}})
    )
    assert body["max_completion_tokens"] == 4096 and "max_tokens" not in body
    vendor_env.setenv("OPENAI_TOP_P", "0.5")
    vendor_env.setenv("OPENAI_PRESENCE_PENALTY", "0.7")
    vendor_env.setenv("OPENAI_FREQUENCY_PENALTY", "0.9")
    body = await _body(create_moonshot_provider("moonshot:kimi-k2.6"))
    assert not {"top_p", "presence_penalty", "frequency_penalty"} & set(body)


async def test_moonshot_reasoning_effort():
    max_effort = {"config": {"reasoning_effort": "max"}}
    body = await _body(create_moonshot_provider("moonshot:kimi-k3", max_effort))
    assert body["reasoning_effort"] == "max"
    assert "temperature" not in body and "max_tokens" not in body
    assert "reasoning_effort" not in await _body(create_moonshot_provider("moonshot:kimi-k3"))
    templated = create_moonshot_provider(
        "moonshot:kimi-k3", {"config": {"reasoning_effort": "{{ effort }}"}}
    )
    body = await _body(templated, "Hello", {"prompt": {"raw": "Hello", "label": "t"},
                                            "vars": {"effort": "max"}})  # fmt: skip
    assert body["reasoning_effort"] == "max"
    with pytest.raises(ValueError, match=r"kimi-k2\.6 does not support reasoning_effort"):
        await create_moonshot_provider("moonshot:kimi-k2.6", max_effort).get_openai_body("Hello")
    with pytest.raises(ValueError, match="does not support reasoning_effort"):
        await create_moonshot_provider("moonshot:moonshot-v1-8k", max_effort).get_openai_body("H")
    forced = create_moonshot_provider(
        "moonshot:kimi-k2.6", {"config": {"passthrough": {"reasoning_effort": "max"}}}
    )
    assert (await _body(forced))["reasoning_effort"] == "max"
    via_passthrough_model = create_moonshot_provider(
        "moonshot:kimi-k2.6",
        {"config": {"reasoning_effort": "low", "passthrough": {"model": "kimi-k3-private"}}},
    )
    body = await _body(via_passthrough_model)
    assert body["model"] == "kimi-k3-private" and body["reasoning_effort"] == "low"
    with pytest.raises(ValueError, match="kimi-k2.6 does not support reasoning_effort"):
        await create_moonshot_provider(
            "moonshot:kimi-k3",
            {"config": {"reasoning_effort": "max", "passthrough": {"model": "kimi-k2.6"}}},
        ).get_openai_body("Hello")


async def test_moonshot_prompt_level_limits_and_v1_defaults():
    provider = create_moonshot_provider(
        "moonshot:kimi-k3", {"config": {"max_completion_tokens": 4096}}
    )
    body = await _body(provider, "Hello", _prompt_context({"max_tokens": 1000}))
    assert body["max_completion_tokens"] == 1000 and "max_tokens" not in body
    provider = create_moonshot_provider("moonshot:kimi-k3", {"config": {"max_tokens": 2048}})
    body = await _body(provider, "Hello", _prompt_context({"max_completion_tokens": 900}))
    assert body["max_completion_tokens"] == 900
    body = await _body(create_moonshot_provider("moonshot:moonshot-v1-8k"))
    assert body["temperature"] == 0 and body["max_tokens"] == 1024


def test_calculate_moonshot_cost():
    assert calculate_moonshot_cost({}, 1000, 500) is None
    assert calculate_moonshot_cost({"cost": 0.000002}, None, 500) is None
    assert calculate_moonshot_cost({"cost": 0.000002}, 1000, None) is None
    assert calculate_moonshot_cost({"cost": 0.000002}, 1000, 500) == pytest.approx(0.003)
    assert calculate_moonshot_cost(
        {"cost": 0.000002, "inputCost": 0.000001, "outputCost": 0.000004}, 1000, 500
    ) == pytest.approx(1000 * 0.000001 + 500 * 0.000004)
    rates = {"inputCost": 0.95 / 1e6, "cacheReadCost": 0.16 / 1e6, "outputCost": 4 / 1e6}
    assert calculate_moonshot_cost(rates, 1_000, 500, 400) == 0.002634


MOONSHOT_URL = "https://api.moonshot.ai/v1/chat/completions"
MOONSHOT_USAGE = {
    "total_tokens": 15,
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "prompt_tokens_details": {"cached_tokens": 4},
}
PRICING = {"apiKey": "k", "inputCost": 0.000002, "outputCost": 0.000004}


@respx.mock
async def test_moonshot_call_cost():
    respx.post(MOONSHOT_URL).mock(return_value=ok(_chat("hi", **MOONSHOT_USAGE)))
    provider = create_moonshot_provider("moonshot:moonshot-v1-8k", {"config": PRICING})
    result = await provider.call_api("Say hi")
    assert result["cost"] == calculate_moonshot_cost(PRICING, 10, 5, 4)
    result = await provider.call_api("Say hi twice", _prompt_context({"inputCost": 0.00002}))
    assert result["cost"] == calculate_moonshot_cost(
        {"inputCost": 0.00002, "outputCost": 0.000004}, 10, 5, 4
    )
    unpriced = create_moonshot_provider("moonshot:moonshot-v1-8k", {"config": {"apiKey": "k"}})
    assert "cost" not in await unpriced.call_api("Say hi again")
    # A promptfoo cache hit leaves the cost undefined.
    cached = await provider.call_api("Say hi")
    assert cached["cached"] is True
    assert "cost" not in cached


@respx.mock
async def test_moonshot_top_level_cached_tokens():
    usage = {"total_tokens": 15, "prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 4}
    respx.post(MOONSHOT_URL).mock(return_value=ok(_chat("hi", **usage)))
    provider = create_moonshot_provider(
        "moonshot:moonshot-v1-8k",
        {"config": {"apiKey": "k", "inputCost": 0.000002, "cacheReadCost": 0.0000002,
                    "outputCost": 0.000004}},
    )  # fmt: skip
    result = await provider.call_api("Say hi")
    assert result["cost"] == pytest.approx(6 * 0.000002 + 4 * 0.0000002 + 5 * 0.000004)
    assert result["tokenUsage"]["completionDetails"]["cacheReadInputTokens"] == 4


# --- MiniMax -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "prompt", "completion", "cached", "expected"),
    [
        ("MiniMax-M3", 500000, 1000000, None, 1.35),
        ("MiniMax-M3", 500000, 1000000, 250000, 1.29),
        ("MiniMax-M3", 1000000, 1000000, 500000, 2.76),
        ("MiniMax-M3", 512_000, 1_000_000, None, (0.3 * 512_000 + 1.2 * 1_000_000) / 1e6),
        ("MiniMax-M3", 512_001, 1_000_000, None, (0.6 * 512_001 + 2.4 * 1_000_000) / 1e6),
        ("MiniMax-M2.7", 1000000, 1000000, None, 1.5),
        ("MiniMax-M2.7", 1000000, 1000000, 500000, 1.38),
        ("MiniMax-M2.7-highspeed", 1000000, 1000000, None, 3.0),
        ("MiniMax-M2.7-highspeed", 1000000, 1000000, 500000, 2.73),
        ("MiniMax-M2.7", 1000000, 0, None, 0.3),
        ("MiniMax-M2.7", 1000000, 0, 2000000, 0.06),
        ("MiniMax-M2.7", 1000000, 1000000, 1000000, 1.26),
    ],
)
def test_calculate_minimax_cost(model, prompt, completion, cached, expected):
    assert calculate_minimax_cost(model, {}, prompt, completion, cached) == pytest.approx(expected)


def test_calculate_minimax_cost_edges():
    assert calculate_minimax_cost("MiniMax-M2.7", {}, None, 1000000) is None
    assert calculate_minimax_cost("MiniMax-M2.7", {}, 1000000, None) is None
    assert calculate_minimax_cost("unknown-model", {}, 1000000, 1000000) is None
    assert calculate_minimax_cost(
        "MiniMax-M2.7", {"cost": 1.0 / 1e6}, 1000000, 1000000
    ) == pytest.approx(2.0)
    assert calculate_minimax_cost(
        "MiniMax-M2.7", {"inputCost": 1.0 / 1e6, "outputCost": 2.0 / 1e6}, 1000000, 1000000
    ) == pytest.approx(3.0)
    priority = {"service_tier": "priority", "inputCost": 0.01, "outputCost": 0.02,
                "cacheReadCost": 0.001}  # fmt: skip
    assert calculate_minimax_cost("MiniMax-M3", priority, 100, 50, 40) == pytest.approx(1.64)
    assert calculate_minimax_cost(
        "MiniMax-M3", {"service_tier": "priority"}, 512001, 0
    ) == pytest.approx((512001 * 0.6 * 1.5) / 1e6)
    assert calculate_minimax_cost("private-deployment", {"inputCost": 0.01}, 100, 50) is None
    assert [m["id"] for m in MINIMAX_CHAT_MODELS] == [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    ]
    assert MINIMAX_CHAT_MODELS[0]["cost"]["longContext"] == {
        "threshold": 512_000,
        "input": 0.6 / 1e6,
        "output": 2.4 / 1e6,
        "cache_read": 0.12 / 1e6,
    }


def test_minimax_factory_and_identity(vendor_env):
    assert create_minimax_provider("minimax").id() == "minimax:MiniMax-M3"
    assert create_minimax_provider("minimax:MiniMax-M2.7-highspeed").id() == (
        "minimax:MiniMax-M2.7-highspeed"
    )
    provider = create_minimax_provider("minimax:MiniMax-M2.7")
    assert str(provider) == "[MiniMax Provider MiniMax-M2.7]"
    json_out = provider.to_json()
    assert json_out["provider"] == "minimax" and json_out["model"] == "MiniMax-M2.7"
    assert json_out["config"]["apiBaseUrl"] == "https://api.minimax.io/v1"
    assert json_out["config"]["apiKeyEnvar"] == "MINIMAX_API_KEY"
    secret = create_minimax_provider(
        "minimax:MiniMax-M2.7", {"config": {"config": {"apiKey": "secret"}}}
    )
    assert "apiKey" not in secret.to_json()["config"]
    vendor_env.setenv("CUSTOM_MINIMAX_KEY", "custom-minimax-key")
    custom = create_minimax_provider(
        "minimax:MiniMax-M2.7",
        {"config": {"config": {"apiBaseUrl": "https://proxy.example.com/minimax/v1",
                               "apiKeyEnvar": "CUSTOM_MINIMAX_KEY"}}},
    )  # fmt: skip
    assert custom.get_api_url() == "https://proxy.example.com/minimax/v1"
    assert custom.get_api_key() == "custom-minimax-key"
    vendor_env.setenv("OPENAI_ORGANIZATION", "org-openai")
    vendor_env.setenv("OPENAI_BASE_URL", "https://openai-proxy.example.com/v1")
    plain = create_minimax_provider("minimax:MiniMax-M2.7")
    assert plain.get_api_key() is None
    assert plain.get_organization() is None
    assert plain.get_api_url() == "https://api.minimax.io/v1"
    # The registry factory wraps the loader options as {config: providerOptions, env}.
    registered = minimax.create_provider(
        "minimax:MiniMax-M2.7", {"id": "my-minimax", "config": {"temperature": 0.3}, "env": None}
    )
    assert registered.id() == "my-minimax"
    assert registered.config["temperature"] == 0.3


async def test_minimax_request_body(vendor_env):
    with pytest.raises(ValueError, match="MiniMax does not support function_call"):
        await create_minimax_provider(
            "minimax:MiniMax-M2.7", {"config": {"config": {"function_call": "auto"}}}
        ).get_openai_body("Call a tool")
    body = await _body(
        create_minimax_provider("minimax:MiniMax-M2.7", {"config": {"config": {"max_tokens": 256}}})
    )
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body and "temperature" not in body
    body = await _body(
        create_minimax_provider(
            "minimax:MiniMax-M2.7",
            {"config": {"config": {"max_completion_tokens": 512, "temperature": 0.7}}},
        )
    )
    assert body["max_completion_tokens"] == 512 and body["temperature"] == 0.7
    vendor_env.setenv("OPENAI_TOP_P", "0.5")
    vendor_env.setenv("OPENAI_PRESENCE_PENALTY", "0.7")
    vendor_env.setenv("OPENAI_FREQUENCY_PENALTY", "0.9")
    body = await _body(create_minimax_provider("minimax:MiniMax-M2.7"))
    assert not {"top_p", "presence_penalty", "frequency_penalty"} & set(body)
    explicit = {"top_p": 0.95, "presence_penalty": 0.1, "frequency_penalty": 0.2}
    body = await _body(
        create_minimax_provider("minimax:MiniMax-M2.7", {"config": {"config": explicit}})
    )
    assert {k: body[k] for k in explicit} == explicit


MINIMAX_PROXY = "https://proxy.example.com/minimax/v1/chat/completions"
MINIMAX_URL = "https://api.minimax.io/v1/chat/completions"


@respx.mock
async def test_minimax_call_endpoint_and_cache_reads():
    route = respx.post(MINIMAX_PROXY).mock(
        return_value=ok(_chat("MiniMax output", **MOONSHOT_USAGE))
    )
    provider = create_minimax_provider(
        "minimax:MiniMax-M2.7",
        {"config": {"config": {"apiKey": "minimax-key",
                               "apiBaseUrl": "https://proxy.example.com/minimax/v1",
                               "max_tokens": 256, "temperature": 0.7}}},
    )  # fmt: skip
    result = await provider.call_api("Test prompt")
    assert route.calls[0].request.headers["Authorization"] == "Bearer minimax-key"
    body = sent_json(route)
    assert body["max_completion_tokens"] == 256 and body["temperature"] == 0.7
    assert "max_tokens" not in body
    assert result["output"] == "MiniMax output"
    assert result["cost"] == calculate_minimax_cost("MiniMax-M2.7", {}, 10, 5, 4)


@respx.mock
async def test_minimax_errors():
    respx.post(MINIMAX_URL).mock(return_value=ok({"error": {"message": "invalid model"}}, 400))
    provider = create_minimax_provider(
        "minimax:MiniMax-M2.7", {"config": {"config": {"apiKey": "minimax-key", "maxRetries": 0}}}
    )
    result = await provider.call_api("Test prompt")
    assert "API error: 400 Bad Request" in result["error"]
    assert "invalid model" in result["error"]


@pytest.mark.parametrize("cached_tokens", [0, 25, 100])
@respx.mock
async def test_minimax_native_cached_usage(cached_tokens):
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
             "prompt_tokens_details": {"cached_tokens": cached_tokens}}  # fmt: skip
    respx.post(MINIMAX_URL).mock(return_value=ok(_chat("MiniMax response", **usage)))
    provider = create_minimax_provider(
        "minimax:MiniMax-M2.7", {"config": {"config": {"apiKey": "fixture-key"}}}
    )
    result = await provider.call_api("Test prompt")
    assert result["cost"] == calculate_minimax_cost("MiniMax-M2.7", {}, 100, 50, cached_tokens)
    assert result["tokenUsage"]["prompt"] == 100


def _minimax(config: dict):
    return create_minimax_provider("minimax:MiniMax-M3", {"config": {"config": config}})


def _minimax_reply(service_tier=None, cache_usage=None) -> httpx.Response:
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
             **(cache_usage if cache_usage is not None
                else {"prompt_tokens_details": {"cached_tokens": 40}})}  # fmt: skip
    data = _chat("Hello", **usage)
    if service_tier:
        data["service_tier"] = service_tier
    return ok(data)


@respx.mock
async def test_minimax_bills_prompt_override_model():
    respx.post(MINIMAX_URL).mock(return_value=_minimax_reply())
    result = await _minimax({"apiKey": "fixture", "inputCost": 0.01}).call_api(
        "Hello",
        _prompt_context({"inputCost": 0.02, "passthrough": {"model": "MiniMax-M2.7-highspeed"}}),
    )
    assert result["cost"] == pytest.approx(60 * 0.02 + (40 * 0.06) / 1e6 + (50 * 2.4) / 1e6)


@pytest.mark.parametrize("config", [{}, {"passthrough": {"service_tier": "priority"}}])
@respx.mock
async def test_minimax_priority_billing(config):
    respx.post(MINIMAX_URL).mock(return_value=_minimax_reply(None if config else "priority"))
    result = await _minimax({"apiKey": "fixture", **config}).call_api("Hello")
    assert result["cost"] == pytest.approx(((60 * 0.3 + 40 * 0.06 + 50 * 1.2) * 1.5) / 1e6)


@pytest.mark.parametrize(
    "cache_usage",
    [
        {"prompt_tokens_details": {"cached_tokens": 40}},
        {"input_tokens_details": {"cached_tokens": 40}},
        {"cached_tokens": 40},
    ],
)
@respx.mock
async def test_minimax_normalized_cached_billing(cache_usage):
    respx.post(MINIMAX_URL).mock(return_value=_minimax_reply(None, cache_usage))
    result = await _minimax({"apiKey": "fixture"}).call_api("Hello")
    assert result["tokenUsage"]["completionDetails"]["cacheReadInputTokens"] == 40
    assert result["cost"] == pytest.approx(0.0000804, abs=1e-14)


@respx.mock
async def test_minimax_native_zero_wins():
    respx.post(MINIMAX_URL).mock(
        return_value=_minimax_reply(
            None, {"prompt_tokens_details": {"cached_tokens": 0}, "cached_tokens": 40}
        )
    )
    result = await _minimax({"apiKey": "fixture"}).call_api("Hello")
    assert result["cost"] == pytest.approx(0.00009, abs=1e-14)


@pytest.mark.parametrize(
    ("config", "cached_tokens", "expected"),
    [
        ({"inputCost": 0.01, "outputCost": 0.02, "cacheReadCost": 0.001}, 40, 1.64),
        ({"inputCost": 0.01, "outputCost": 0.02}, 0, 2),
        ({"inputCost": 0.01, "outputCost": 0.02}, 40, 2),
        ({"cost": 0.0001, "cacheReadCost": 0}, 40, 0.011),
        ({"inputCost": 0, "outputCost": 0, "cacheReadCost": 0}, 40, 0),
    ],
)
@respx.mock
async def test_minimax_custom_pricing_for_unlisted_alias(config, cached_tokens, expected):
    respx.post(MINIMAX_URL).mock(
        return_value=_minimax_reply(
            "priority", {"input_tokens_details": {"cached_tokens": cached_tokens}}
        )
    )
    result = await _minimax({"apiKey": "fixture"}).call_api(
        "Hello",
        _prompt_context({**config, "passthrough": {"model": "private-minimax-deployment"}}),
    )
    assert result["cost"] == pytest.approx(expected, abs=1e-14)


@respx.mock
async def test_minimax_cache_hit_costs_zero():
    respx.post(MINIMAX_URL).mock(return_value=_minimax_reply())
    provider = _minimax({"apiKey": "fixture", "service_tier": "priority"})
    await provider.call_api("Hello")
    result = await provider.call_api("Hello")
    assert result["cached"] is True
    assert result["cost"] == 0


# --- MiniMax and Moonshot shared request rules ---------------------------------------------


@pytest.fixture
def sampling_env(request: pytest.FixtureRequest, vendor_env):
    request.getfixturevalue(openai_env.__name__)  # clear OPENAI_* first
    for name, value in {"OPENAI_TEMPERATURE": "0", "OPENAI_TOP_P": "0.1",
                        "OPENAI_MAX_TOKENS": "12", "OPENAI_PRESENCE_PENALTY": "0.2",
                        "OPENAI_FREQUENCY_PENALTY": "0.3"}.items():  # fmt: skip
        vendor_env.setenv(name, value)
    return vendor_env


def _make(name: str, config: dict):
    if name == "minimax":
        return create_minimax_provider("minimax:MiniMax-M3", {"config": {"config": config}})
    return create_moonshot_provider("moonshot:kimi-k3", {"config": config})


@pytest.mark.usefixtures("sampling_env")
@pytest.mark.parametrize("name", ["minimax", "moonshot"])
async def test_explicit_passthrough_sampling(name):
    body = await _body(
        _make(name, {"passthrough": {"temperature": 0, "top_p": 0.7, "frequency_penalty": 0}})
    )
    assert body["temperature"] == 0 and body["top_p"] == 0.7 and body["frequency_penalty"] == 0
    assert not {"presence_penalty", "max_tokens", "max_completion_tokens"} & set(body)
    for field in ("max_tokens", "max_completion_tokens"):
        body = await _body(_make(name, {"passthrough": {field: 17}}))
        assert body["max_completion_tokens"] == 17 and "max_tokens" not in body
    body = await _body(
        _make(name, {"max_completion_tokens": 100, "passthrough": {"max_tokens": 9}})
    )
    assert body["max_completion_tokens"] == 9


@pytest.mark.usefixtures("sampling_env")
@pytest.mark.parametrize("name", ["minimax", "moonshot"])
@pytest.mark.parametrize(
    ("config", "prompt_config"),
    [
        ({"max_completion_tokens": 100}, {"max_tokens": 7}),
        ({"max_tokens": 100}, {"max_completion_tokens": 7}),
        ({"passthrough": {"max_completion_tokens": 100}}, {"max_tokens": 7}),
        ({"max_completion_tokens": 100}, {"passthrough": {"max_tokens": 7}}),
    ],
)
async def test_prompt_token_limits_win(name, config, prompt_config):
    body = await _body(_make(name, config), "Hello", _prompt_context(prompt_config))
    assert body["max_completion_tokens"] == 7 and "max_tokens" not in body


@pytest.mark.usefixtures("sampling_env")
@pytest.mark.parametrize(
    ("config", "prompt_config", "expected"),
    [
        ({"max_completion_tokens": 2000}, {}, 2000),
        ({"max_tokens": 2000}, {"max_completion_tokens": 7}, 7),
        ({"max_completion_tokens": 2000},
         {"passthrough": {"model": "moonshot-v1-8k", "max_tokens": 9}}, 9),
    ],
)  # fmt: skip
async def test_moonshot_legacy_model_override_limits(config, prompt_config, expected):
    provider = _make("moonshot", {**config, "passthrough": {"model": "moonshot-v1-8k"}})
    body = await _body(provider, "Hello", _prompt_context(prompt_config))
    assert body["model"] == "moonshot-v1-8k" and body["temperature"] == 0
    assert body["max_completion_tokens"] == expected and "max_tokens" not in body


@pytest.mark.usefixtures("sampling_env")
async def test_moonshot_legacy_override_keeps_sampling():
    body = await _body(_make("moonshot", {"passthrough": {"model": "moonshot-v1-private"}}))
    assert body["model"] == "moonshot-v1-private"
    assert body["temperature"] == 0 and body["max_tokens"] == 12
