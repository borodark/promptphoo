"""xAI chat and Perplexity providers. Ports deterministic cases from
test/providers/xai/chat.test.ts and test/providers/perplexity.test.ts, with HTTP mocked by
respx."""

from __future__ import annotations

import httpx
import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo import cache
from promptphoo.providers.compatible import perplexity as pplx
from promptphoo.providers.compatible import xai
from promptphoo.providers.compatible.perplexity import (
    PerplexityProvider,
    calculate_perplexity_cost,
    create_perplexity_provider,
)
from promptphoo.providers.compatible.xai import (
    GROK_3_MINI_MODELS,
    GROK_4_MODELS,
    GROK_REASONING_EFFORT_MODELS,
    GROK_REASONING_MODELS,
    XAI_CHAT_MODELS,
    calculate_xai_cost,
    create_xai_provider,
    get_xai_cost_in_usd,
)

XAI_URL = "https://api.x.ai/v1/chat/completions"
PPLX_URL = "https://api.perplexity.ai/chat/completions"
VENDOR_ENV = [
    "XAI_API_KEY",
    "PERPLEXITY_API_KEY",
    "OPENAI_MAX_TOKENS",
    "OPENAI_MAX_COMPLETION_TOKENS",
    "PROMPTFOO_DISABLE_TEMPLATING",
]

pytestmark = pytest.mark.usefixtures(openai_env.__name__)


@pytest.fixture(autouse=True)
def _vendor_env(openai_env):
    for name in VENDOR_ENV:
        openai_env.delenv(name, raising=False)
    openai_env.setattr(cache, "_enabled", False)
    return openai_env


def _mock_response(content="Mock response", usage=None, **extra):
    usage = (
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        if usage is None
        else usage
    )
    return {"choices": [{"message": {"content": content}}], "usage": usage, **extra}


def _xai(model: str, outer: dict | None = None, **nested):
    options: dict = dict(outer or {})
    if nested:
        options["config"] = nested
    return create_xai_provider(f"xai:{model}", {"config": options} if options else {})


async def _xbody(provider, **context):
    return (await provider.get_openai_body("hello", context or None))["body"]


# ---------------------------------------------------------------- xAI: creation


@respx.mock
async def test_grok47_uses_configured_endpoint_before_region_default():
    route = respx.post("http://localhost:8080/v1/chat/completions").mock(
        return_value=ok(_mock_response())
    )
    provider = _xai(
        "grok-4.7", apiKey="test-key", apiBaseUrl="http://localhost:8080/v1", region="us"
    )
    result = await provider.call_api("Hello")
    assert "error" not in result
    assert "grok-4.7" in route.calls[0].request.content.decode()


def test_requires_model_name():
    with pytest.raises(ValueError, match="Model name is required"):
        create_xai_provider("xai:")


def test_creates_provider_with_model_and_defaults():
    provider = create_xai_provider("xai:grok-2")
    assert provider.id() == "xai:grok-2"
    assert provider.model_name == "grok-2"
    assert provider.config["apiBaseUrl"] == "https://api.x.ai/v1"
    assert provider.config["apiKeyEnvar"] == "XAI_API_KEY"
    assert str(provider) == "[xAI Provider grok-2]"


def test_region_base_url():
    provider = _xai("grok-2", region="eu-west-1")
    assert provider.config["apiBaseUrl"] == "https://eu-west-1.api.x.ai/v1"


def test_merges_outer_options():
    provider = create_xai_provider(
        "xai:grok-2", {"config": {"temperature": 0.7, "max_tokens": 100}, "id": "custom-id"}
    )
    assert provider.config["apiBaseUrl"] == "https://api.x.ai/v1"
    assert provider.config["temperature"] == 0.7
    assert provider.config["max_tokens"] == 100


def test_multiple_options_for_mini_models():
    provider = create_xai_provider(
        "xai:grok-3-mini-beta",
        {
            "config": {
                "config": {"reasoning_effort": "high", "region": "us-east-1"},
                "temperature": 0.5,
                "max_tokens": 1000,
            }
        },
    )
    assert provider.config["apiBaseUrl"] == "https://us-east-1.api.x.ai/v1"
    assert provider.config["temperature"] == 0.5
    assert provider.config["max_tokens"] == 1000


def test_stores_original_config():
    nested = {"search_parameters": {"mode": "test"}, "region": "eu-west-1"}
    provider = create_xai_provider("xai:grok-2", {"config": {"config": nested}})
    assert provider.original_config == nested


def test_to_json_masks_api_key():
    provider = create_xai_provider(
        "xai:grok-3-beta", {"config": {"temperature": 0.7, "apiKey": "secret"}}
    )
    json = provider.to_json()
    assert json["provider"] == "xai"
    assert json["model"] == "grok-3-beta"
    assert "apiKey" not in json["config"]


def test_models_with_cost_are_independent():
    grok46 = next(m for m in XAI_CHAT_MODELS if m["id"] == "grok-4.6")
    grok47 = next(m for m in XAI_CHAT_MODELS if m["id"] == "grok-4.7")
    assert grok46["cost"] is not grok47["cost"]
    assert grok46["cost"]["longContext"] is not grok47["cost"]["longContext"]
    assert "aliases" not in grok47
    assert not any(m["id"] == "grok-4.7-fast" for m in XAI_CHAT_MODELS)


def test_model_constants():
    assert "grok-3-mini-beta" in GROK_3_MINI_MODELS
    assert "grok-2-1212" not in GROK_3_MINI_MODELS
    for model in ("grok-build-0.1", "grok-code-fast-1", "grok-4-0709", "grok-4.3"):
        assert model in GROK_REASONING_MODELS
    assert "grok-4-1-fast-non-reasoning" not in GROK_REASONING_MODELS
    assert "grok-4-1-fast-non-reasoning" in GROK_4_MODELS
    assert "grok-4-0709" not in GROK_REASONING_EFFORT_MODELS
    assert "grok-4.5" in GROK_REASONING_EFFORT_MODELS


def test_create_provider_routes():
    with pytest.raises(ValueError, match="does not currently expose a public embeddings API"):
        xai.create_provider("xai:embedding:foo", {})
    for route in ("image", "video", "responses", "voice"):
        with pytest.raises(ValueError, match="not supported by promptphoo"):
            xai.create_provider(f"xai:{route}:grok-4", {})
    provider = xai.create_provider("xai:grok-4", {"config": {"temperature": 0.2}, "env": None})
    assert provider.id() == "xai:grok-4"
    assert provider.config["temperature"] == 0.2


# ---------------------------------------------------------------- xAI: request body


async def test_service_tier_validation():
    provider = create_xai_provider("xai:grok-4.5", {"config": {"service_tier": "priority"}})
    assert (await _xbody(provider))["service_tier"] == "priority"
    invalid = create_xai_provider("xai:grok-4.5", {"config": {"service_tier": "flex"}})
    with pytest.raises(ValueError, match="Invalid xAI service_tier."):
        await _xbody(invalid)


async def test_search_parameters_rendered():
    provider = _xai("grok-3-beta", search_parameters={"mode": "{{mode}}", "filter": "{{filter}}"})
    body = await _xbody(provider, vars={"mode": "advanced", "filter": "latest"})
    assert body["search_parameters"] == {"mode": "advanced", "filter": "latest"}
    assert "search_parameters" not in await _xbody(create_xai_provider("xai:grok-3-beta"))


async def test_temperature_zero_sent():
    provider = create_xai_provider("xai:grok-3-beta", {"config": {"temperature": 0}})
    assert (await _xbody(provider))["temperature"] == 0


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
@respx.mock
async def test_grok47_reasoning_effort_sent(effort):
    route = respx.post(XAI_URL).mock(return_value=ok(_mock_response()))
    provider = _xai("grok-4.7", apiKey="test-key", reasoning_effort=effort)
    response = await provider.call_api("hello")
    assert sent_json(route)["model"] == "grok-4.7"
    assert sent_json(route)["reasoning_effort"] == effort
    assert response["output"] == "Mock response"


async def test_grok47_default_effort_and_sampling(monkeypatch):
    provider = _xai("grok-4.7", max_completion_tokens=4096)
    body = await _xbody(provider)
    assert "reasoning_effort" not in body
    assert body["max_completion_tokens"] == 4096
    body = await _xbody(
        provider,
        prompt={
            "config": {
                "max_completion_tokens": 32,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.2,
                "stop": ["end"],
                "temperature": 0.8,
            }
        },
    )
    assert body["model"] == "grok-4.7"
    assert body["max_completion_tokens"] == 32
    assert body["temperature"] == 0.8
    for key in ("max_tokens", "presence_penalty", "frequency_penalty", "stop"):
        assert key not in body
    for config in (
        {"max_tokens": 64},
        {"max_completion_tokens": 64, "passthrough": {"model": "grok-4.7"}},
        {"passthrough": {"model": "grok-4.7", "max_tokens": 64}},
    ):
        overridden = await _xbody(provider, prompt={"config": config})
        assert overridden["max_completion_tokens"] == 64
        assert "max_tokens" not in overridden
    with pytest.raises(xai.XAIRequestConfigError, match="conflicting max_tokens"):
        await _xbody(provider, prompt={"config": {"max_completion_tokens": 64, "max_tokens": 32}})
    assert (await _xbody(provider, test={"options": {"max_tokens": 64}}))[
        "max_completion_tokens"
    ] == 64
    with pytest.raises(xai.XAIRequestConfigError, match="conflicting max_tokens"):
        await _xbody(provider, test={"options": {"max_completion_tokens": 2048, "max_tokens": 64}})
    nullish = _xai("grok-4.7", max_tokens=128, max_completion_tokens=None)
    assert (await _xbody(nullish))["max_completion_tokens"] == 128
    assert (await _xbody(nullish, test={"options": {"max_completion_tokens": None}}))[
        "max_completion_tokens"
    ] == 128
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "37")
    assert (await _xbody(create_xai_provider("xai:grok-4.7")))["max_completion_tokens"] == 37


async def test_grok47_passthrough_replaced_by_prompt_or_test():
    provider = _xai(
        "grok-4.7",
        passthrough={"model": "grok-4.7", "reasoning_effort": "high", "max_completion_tokens": 256},
    )
    partial = {"passthrough": {"model": "grok-4.7"}}
    for context in ({"prompt": {"config": partial}}, {"test": {"options": partial}}):
        body = await _xbody(provider, **context)
        assert "reasoning_effort" not in body
        assert "max_completion_tokens" not in body


async def test_grok47_null_passthrough_output_cap():
    body = await _xbody(_xai("grok-4.7", passthrough={"max_completion_tokens": None}))
    assert "max_completion_tokens" not in body
    assert "max_tokens" not in body


@pytest.mark.parametrize(("model", "effort"), [("grok-4.6", "xhigh"), ("grok-4.3", "none")])
async def test_passthrough_model_keeps_reasoning_options(model, effort):
    provider = _xai(
        "grok-4.7", reasoning_effort=effort, max_completion_tokens=256, passthrough={"model": model}
    )
    body = await _xbody(provider)
    assert body["model"] == model
    assert body["reasoning_effort"] == effort
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body


async def test_prompt_and_test_options_override_passthrough_defaults():
    provider = _xai(
        "grok-4.7",
        reasoning_effort="low",
        max_completion_tokens=256,
        passthrough={"model": "grok-4.6"},
    )
    body = await _xbody(
        provider, prompt={"config": {"reasoning_effort": "xhigh", "max_completion_tokens": 128}}
    )
    assert (body["model"], body["reasoning_effort"], body["max_completion_tokens"]) == (
        "grok-4.6",
        "xhigh",
        128,
    )
    provider = _xai(
        "grok-4.7",
        passthrough={"model": "grok-4.6", "reasoning_effort": "high", "max_completion_tokens": 256},
    )
    body = await _xbody(
        provider, test={"options": {"reasoning_effort": "low", "max_completion_tokens": 128}}
    )
    assert (body["reasoning_effort"], body["max_completion_tokens"]) == ("low", 128)
    assert "max_tokens" not in body


@respx.mock
async def test_grok47_effort_variables_validated_before_request():
    route = respx.post(XAI_URL).mock(return_value=ok(_mock_response()))
    provider = _xai("grok-4.7", apiKey="test-key")
    options = {"reasoning_effort": "{{ effort }}", "passthrough": {"model": "grok-4.7"}}
    accepted = await provider.call_api(
        "hello",
        {
            "prompt": {"raw": "hello", "label": "hello", "config": options},
            "vars": {"effort": "xhigh"},
            "test": {"options": options},
        },
    )
    assert accepted["output"] == "Mock response"
    assert sent_json(route)["reasoning_effort"] == "xhigh"
    for effort in ("none", "{{ candidate | upper }}", "{{ missing }}", "private-marker"):
        bad = _xai("grok-4.7", apiKey="test-key", reasoning_effort=effort)
        result = await bad.call_api(
            "hello", {"prompt": {"raw": "hello", "label": "hello"}, "vars": {"candidate": "x"}}
        )
        assert "reasoning" in result["error"]
        assert "private-marker" not in result["error"]
        assert "API key" not in result["error"]
    assert route.call_count == 1


@respx.mock
async def test_test_effort_priority_and_disabled_templating(monkeypatch):
    route = respx.post(XAI_URL).mock(return_value=ok(_mock_response()))
    provider = _xai("grok-4.7", apiKey="test-key", passthrough={"reasoning_effort": "high"})
    context = {
        "prompt": {"raw": "hello", "label": "hello"},
        "vars": {"effort": "xhigh"},
        "test": {"options": {"reasoning_effort": "low"}},
    }
    await provider.call_api("hello", context)
    assert sent_json(route)["reasoning_effort"] == "low"
    conflicting = await provider.call_api(
        "hello",
        {
            **context,
            "test": {
                "options": {"reasoning_effort": "low", "passthrough": {"reasoning_effort": "xhigh"}}
            },
        },
    )
    assert "both test options and test passthrough" in conflicting["error"]
    monkeypatch.setenv("PROMPTFOO_DISABLE_TEMPLATING", "true")
    result = await provider.call_api(
        "hello", {**context, "test": {"options": {"reasoning_effort": "{{ effort }}"}}}
    )
    assert "reasoning effort" in result["error"]
    assert route.call_count == 1


async def test_effort_validation():
    provider = create_xai_provider("xai:grok-4.6")
    assert (await _xbody(provider, prompt={"config": {"reasoning_effort": "xhigh"}}))[
        "reasoning_effort"
    ] == "xhigh"
    for effort in ("none", "minimal"):
        with pytest.raises(xai.XAIRequestConfigError, match="grok-4.6 does not support"):
            await _xbody(provider, prompt={"config": {"reasoning_effort": effort}})
    for model in ("grok-4.5", "grok-4.5-latest", "grok-build-latest"):
        provider = create_xai_provider(f"xai:{model}")
        assert provider.is_reasoning_model()
        assert provider.supports_reasoning_effort()
        for effort in ("none", "xhigh", "minimal"):
            with pytest.raises(xai.XAIRequestConfigError, match=f"{model} does not support"):
                await _xbody(provider, prompt={"config": {"reasoning_effort": effort}})
    for model in ("grok-4.3", "grok-4.3-latest", "grok-latest"):
        body = await _xbody(
            create_xai_provider(f"xai:{model}"), prompt={"config": {"reasoning_effort": "none"}}
        )
        assert body["reasoning_effort"] == "none"


@respx.mock
async def test_passthrough_grok43_effort_validated():
    route = respx.post(XAI_URL).mock(return_value=ok(_mock_response()))
    provider = _xai(
        "grok-4.7",
        apiKey="test-key",
        passthrough={"model": "grok-4.3", "reasoning_effort": "xhigh"},
    )
    result = await provider.call_api("hello")
    assert "grok-4.3 does not support reasoning_effort" in result["error"]
    assert route.call_count == 0


@pytest.mark.parametrize("model", ["grok-4.6", "grok-4.5", "grok-4.3"])
async def test_reasoning_models_strip_unsupported_sampling(model):
    body = await _xbody(
        create_xai_provider(f"xai:{model}"),
        prompt={
            "config": {
                "reasoning_effort": "high",
                "presence_penalty": 0.5,
                "frequency_penalty": 0.7,
                "stop": ["\\n"],
                "temperature": 0.8,
            }
        },
    )
    assert body["reasoning_effort"] == "high"
    for key in ("presence_penalty", "frequency_penalty", "stop"):
        assert key not in body
    assert body["temperature"] == 0.8


def test_grok4_reasoning_flags():
    provider = create_xai_provider("xai:grok-4-0709")
    assert provider.is_reasoning_model()
    assert not provider.supports_reasoning_effort()
    assert create_xai_provider("xai:grok-4").is_reasoning_model()


@pytest.mark.parametrize("model", ["grok-4", "grok-4-1-fast-reasoning", "grok-4-fast-reasoning"])
async def test_grok4_family_strips_reasoning_effort_and_sampling(model):
    body = await _xbody(
        create_xai_provider(f"xai:{model}"),
        prompt={
            "config": {
                "presence_penalty": 0.5,
                "frequency_penalty": 0.7,
                "stop": ["\\n"],
                "reasoning_effort": "high",
                "temperature": 0.7,
                "max_completion_tokens": 2048,
            }
        },
    )
    for key in ("presence_penalty", "frequency_penalty", "stop", "reasoning_effort"):
        assert key not in body
    assert body["temperature"] == 0.7
    assert body["max_completion_tokens"] == 2048


async def test_non_reasoning_variant_strips_sampling():
    body = await _xbody(
        create_xai_provider("xai:grok-4-1-fast-non-reasoning"),
        prompt={
            "config": {
                "presence_penalty": 0.5,
                "frequency_penalty": 0.7,
                "stop": ["\\n"],
                "temperature": 0.5,
            }
        },
    )
    for key in ("presence_penalty", "frequency_penalty", "stop"):
        assert key not in body
    assert body["temperature"] == 0.5


@pytest.mark.parametrize(
    "model", ["grok-build-0.1", "grok-code-fast-1", "grok-code-fast", "grok-code-fast-1-0825"]
)
async def test_grok_build_uses_max_tokens(model, monkeypatch):
    for config in (
        {"max_tokens": 321},
        {"passthrough": {"max_tokens": 321}},
        {"max_completion_tokens": 321},
        {"passthrough": {"max_completion_tokens": 321}},
    ):
        body = await _xbody(create_xai_provider(f"xai:{model}"), prompt={"config": config})
        assert body["max_tokens"] == 321
        assert "max_completion_tokens" not in body
    body = await _xbody(
        create_xai_provider(f"xai:{model}"),
        prompt={
            "config": {
                "max_tokens": 111,
                "max_completion_tokens": 222,
                "passthrough": {"max_completion_tokens": 333},
            }
        },
    )
    assert body["max_tokens"] == 333
    body = await _xbody(
        _xai(model, max_tokens=111), prompt={"config": {"max_completion_tokens": 222}}
    )
    assert body["max_tokens"] == 222
    assert "max_completion_tokens" not in body
    monkeypatch.setenv("OPENAI_MAX_COMPLETION_TOKENS", "654")
    body = await _xbody(create_xai_provider(f"xai:{model}"))
    assert body["max_tokens"] == 654
    assert "max_completion_tokens" not in body


async def test_effective_passthrough_model_token_limits():
    provider = _xai("grok-4.3", omitDefaults=True, passthrough={"max_tokens": 111})
    body = await _xbody(provider, prompt={"config": {"passthrough": {"model": "grok-build-0.1"}}})
    assert body["model"] == "grok-build-0.1"
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body

    body = await _xbody(
        create_xai_provider("xai:grok-4.3"),
        prompt={
            "config": {"passthrough": {"model": "grok-build-0.1", "max_completion_tokens": 333}}
        },
    )
    assert body["max_tokens"] == 333
    assert "max_completion_tokens" not in body
    body = await _xbody(
        create_xai_provider("xai:grok-build-0.1"),
        prompt={"config": {"passthrough": {"model": "grok-4.3", "max_completion_tokens": 444}}},
    )
    assert body["max_completion_tokens"] == 444
    assert "max_tokens" not in body
    body = await _xbody(
        create_xai_provider("xai:grok-2"),
        prompt={"config": {"max_completion_tokens": 777, "passthrough": {"model": "grok-4.5"}}},
    )
    assert body["max_completion_tokens"] == 777
    assert "max_tokens" not in body
    body = await _xbody(
        _xai("grok-build-0.1", max_completion_tokens=111),
        prompt={"config": {"max_tokens": 222, "passthrough": {"model": "grok-4.3"}}},
    )
    assert body["max_completion_tokens"] == 222
    assert "max_tokens" not in body


async def test_effective_passthrough_model_sampling_filters():
    sampling = {
        "reasoning_effort": "medium",
        "presence_penalty": 0.5,
        "frequency_penalty": 0.7,
        "stop": ["\\n"],
    }
    body = await _xbody(
        create_xai_provider("xai:grok-2"),
        prompt={"config": {"passthrough": {"model": "grok-4.5", **sampling}}},
    )
    assert body["model"] == "grok-4.5"
    assert body["reasoning_effort"] == "medium"
    assert "presence_penalty" not in body
    body = await _xbody(
        create_xai_provider("xai:grok-4.5"),
        prompt={"config": {"passthrough": {"model": "grok-2", **sampling}}},
    )
    assert body["model"] == "grok-2"
    assert "reasoning_effort" not in body
    assert body["presence_penalty"] == 0.5
    assert body["stop"] == ["\\n"]
    body = await _xbody(
        create_xai_provider("xai:grok-2"),
        prompt={"config": {"reasoning_effort": "high", "passthrough": {"model": "grok-4.5"}}},
    )
    assert body["reasoning_effort"] == "high"


@pytest.mark.parametrize(("configured", "effort"), [("grok-2", "none"), ("grok-4.6", "xhigh")])
async def test_validates_against_effective_grok45(configured, effort):
    with pytest.raises(xai.XAIRequestConfigError, match="grok-4.5 does not support"):
        await _xbody(
            create_xai_provider(f"xai:{configured}"),
            prompt={"config": {"passthrough": {"model": "grok-4.5", "reasoning_effort": effort}}},
        )


# ---------------------------------------------------------------- xAI: cost


def test_cost_ticks():
    assert get_xai_cost_in_usd({"cost_in_usd_ticks": 37_756_000}) == 0.0037756
    assert get_xai_cost_in_usd({"cost_in_usd_ticks": -1}) is None
    assert get_xai_cost_in_usd({"cost_in_usd_ticks": float("nan")}) is None
    assert get_xai_cost_in_usd(None) is None


def test_cost_basics():
    assert calculate_xai_cost("grok-2-1212", {}, 600, 400) is not None
    assert calculate_xai_cost("grok-4", {}, 600, 400) is not None
    assert calculate_xai_cost("grok-3-mini", {}, 600, 400) is not None
    assert calculate_xai_cost("invalid-model", {}, 600, 400) is None
    assert calculate_xai_cost("grok-2-1212", {}) is None
    assert calculate_xai_cost("grok-2-1212", {}, 0, 0) is None
    assert calculate_xai_cost("grok-2-1212", {}, 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert (
        calculate_xai_cost("grok-2-1212", {"inputCost": 0.001, "outputCost": 0.003}, 1000, 500)
        == 2.5
    )
    assert (
        calculate_xai_cost(
            "grok-2-1212", {"cost": 0.02, "inputCost": 0.001, "outputCost": 0.003}, 1000, 500
        )
        == 2.5
    )


def test_cache_read_rates():
    rates = {"inputCost": 3e-6, "outputCost": 15e-6, "cacheReadCost": 0.75e-6}
    assert calculate_xai_cost("grok-4-0709", rates, 1000, 500, 0, 800) == pytest.approx(0.0087)
    assert calculate_xai_cost("grok-4-0709", rates, 1000, 500) == pytest.approx(0.0105)
    assert calculate_xai_cost(
        "grok-4-0709", {**rates, "cacheReadCost": 0}, 1000, 500, 0, 800
    ) == pytest.approx(0.0081)
    for cached, expected in ((1000, 0.00825), (2000, 0.00825), (-1, 0.0105)):
        assert calculate_xai_cost("grok-4-0709", rates, 1000, 500, 0, cached) == pytest.approx(
            expected
        )
    for cached in (float("nan"), float("inf")):
        assert calculate_xai_cost("grok-4-0709", rates, 1000, 500, 0, cached) == pytest.approx(
            0.0105
        )
    with_cache = calculate_xai_cost("grok-2-1212", {}, 1000, 500, 0, 800)
    assert with_cache == pytest.approx(0.007)
    assert with_cache == calculate_xai_cost("grok-2-1212", {}, 1000, 500)
    assert calculate_xai_cost("grok-4-0709", {"cost": 0.001}, 1000, 500, 0, 800) == 1.5


def test_redirected_and_build_pricing():
    for model in ("grok-4-1-fast-reasoning", "grok-4", "grok-3", "grok-3-fast-beta"):
        assert calculate_xai_cost(model, {}, 1_000_000, 1_000_000) == pytest.approx(7.5)
    for model in ("grok-build-0.1", "grok-code-fast-1", "grok-code-fast"):
        assert calculate_xai_cost(model, {}, 100_000, 1_000_000) == pytest.approx(2.1)
        assert calculate_xai_cost(model, {}, 199_999, 1_000) == pytest.approx(
            (199_999 * 1 + 1_000 * 2) / 1e6
        )
        assert calculate_xai_cost(model, {}, 200_000, 1_000) == pytest.approx(
            (200_000 * 2 + 1_000 * 4) / 1e6
        )
    assert calculate_xai_cost("grok-build-0.1", {}, 200_001, 1_000, 0, 100_000) == pytest.approx(
        (100_001 * 2 + 100_000 * 0.4 + 1_000 * 4) / 1e6
    )


def test_grok45_46_47_pricing():
    for model in ("grok-4.5", "grok-4.5-latest", "grok-build-latest"):
        assert calculate_xai_cost(model, {}, 100_000, 1_000_000) == pytest.approx(6.2)
    assert calculate_xai_cost("grok-4.5", {}, 200_000, 1_000) == pytest.approx(
        (200_000 * 4 + 1_000 * 12) / 1e6
    )
    assert calculate_xai_cost("grok-4.5", {}, 1_000, 500, 0, 800) == pytest.approx(
        (200 * 2 + 800 * 0.3 + 500 * 6) / 1e6
    )
    assert calculate_xai_cost(
        "grok-4.6", {}, 1_000, 500, 0, 800, {"apiUrl": "https://us.api.x.ai/v1"}
    ) == pytest.approx(((200 * 2 + 800 * 0.5 + 500 * 6) / 1e6) * 1.1)
    assert calculate_xai_cost("grok-4.7", {}, 199_999, 1_000, 0, 100_000) == pytest.approx(0.255998)
    assert calculate_xai_cost("grok-4.7", {}, 200_000, 1_000, 0, 100_000) == pytest.approx(0.512)
    options = {"apiUrl": "https://us.api.x.ai/v1", "reasoningBilledSeparately": True}
    assert calculate_xai_cost("grok-4.7", {}, 1_000, 500, 20, 800, options) == pytest.approx(
        0.004312
    )
    assert calculate_xai_cost("grok-4.7", {}, 200_000, 1_000, 0, 100_000, options) == pytest.approx(
        0.5632
    )
    for api_url in (
        "https://api.x.ai/v1",
        "https://eu-west-1.api.x.ai/v1",
        "https://example.com/v1",
        "https://us.api.x.ai.example.com/v1",
    ):
        assert calculate_xai_cost(
            "grok-4.7", {}, 1_000, 500, 20, 800, {**options, "apiUrl": api_url}
        ) == pytest.approx(0.00392)
    assert calculate_xai_cost("grok-test-model", {"cost": 0.001}, 10, 20) == pytest.approx(0.03)
    assert calculate_xai_cost("grok-test-model", {"inputCost": 0.001}, 10, 20) is None


def test_reasoning_token_billing():
    sep = {"reasoningBilledSeparately": True}
    assert calculate_xai_cost("grok-4.5", {}, 219, 1, 50, 128, sep) == pytest.approx(0.0005264)
    assert calculate_xai_cost("grok-4.5", {}, 219, 51, 50, 128) == pytest.approx(0.0005264)
    assert calculate_xai_cost("grok-4.5", {}, 219, 0, 50, 128, sep) == pytest.approx(
        (91 * 2 + 128 * 0.3 + 50 * 6) / 1e6
    )
    assert calculate_xai_cost("grok-4.3", {}, 10, 0, 0, 8) == pytest.approx(0.0000041)
    assert calculate_xai_cost("grok-3-mini-beta", {}, 500, 500, 200) == pytest.approx(0.0004)
    assert calculate_xai_cost("grok-3-mini-beta", {}, 500, 0, 200) == pytest.approx(0.00025)
    assert calculate_xai_cost("grok-3-mini-beta", {}, 0, 200, 50) == pytest.approx(0.0001)


@pytest.mark.parametrize(
    ("config", "prompt", "output", "cached", "cost"),
    [
        ({"inputCost": 0.002}, 10, 0, 0, 0.02),
        ({"inputCost": 0.002, "cacheReadCost": 0.0005}, 10, 0, 8, 0.008),
        ({"cacheReadCost": 0.0005}, 10, 0, 10, 0.005),
        ({"outputCost": 0.003}, 0, 5, 0, 0.015),
        ({"inputCost": 0}, 10, 0, 0, 0),
    ],
)
def test_custom_rates_for_used_token_classes(config, prompt, output, cached, cost):
    assert calculate_xai_cost("custom-model", config, prompt, output, 0, cached) == pytest.approx(
        cost
    )


@pytest.mark.parametrize(
    ("config", "prompt", "output", "cached"),
    [
        ({"inputCost": 0.002}, 10, 1, 0),
        ({"cacheReadCost": 0.0005}, 10, 0, 9),
        ({"outputCost": 0.003}, 1, 5, 0),
    ],
)
def test_custom_rates_missing(config, prompt, output, cached):
    assert calculate_xai_cost("custom-model", config, prompt, output, 0, cached) is None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, (200 * 4 + 800 * 0.6 + 500 * 12) / 1e6),
        ({"cost": 0.001}, 1.5),
        ({"inputCost": 0.001}, 1 + (500 * 12) / 1e6),
        ({"outputCost": 0.001}, (200 * 4 + 800 * 0.6) / 1e6 + 0.5),
        ({"cacheReadCost": 0.001}, (200 * 4 + 500 * 12) / 1e6 + 0.8),
        ({"cost": 0.002, "inputCost": 0.001, "outputCost": 0.003, "cacheReadCost": 0.0005}, 2.1),
        ({"cost": 0}, 0),
        ({"cacheReadCost": 0}, (200 * 4 + 500 * 12) / 1e6),
    ],
)
def test_priority_pricing(config, expected):
    assert calculate_xai_cost(
        "grok-4.5", config, 1_000, 500, 0, 800, {"serviceTier": "priority"}
    ) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "model", ["grok-4.3", "grok-4.20-0309-non-reasoning", "grok-4.20-multi-agent-0309", "grok-4.20"]
)
def test_long_context_pricing_grok43_420(model):
    assert calculate_xai_cost(model, {}, 199_999, 1_000, 0, 100_000) == pytest.approx(
        (99_999 * 1.25 + 100_000 * 0.2 + 1_000 * 2.5) / 1e6
    )
    assert calculate_xai_cost(model, {}, 200_000, 1_000, 0, 100_000) == pytest.approx(
        (100_000 * 2.5 + 100_000 * 0.4 + 1_000 * 5) / 1e6
    )


# ---------------------------------------------------------------- xAI: call_api


@respx.mock
async def test_network_error_passes_through_parent():
    respx.post(XAI_URL).mock(side_effect=httpx.ConnectError("Network timeout"))
    provider = create_xai_provider(
        "xai:grok-4", {"config": {"apiKey": "test-key", "maxRetries": 0}}
    )
    result = await provider.call_api("test prompt")
    assert "API call error:" in result["error"]


@respx.mock
async def test_502_error_enhanced():
    respx.post(XAI_URL).mock(return_value=ok({"error": {"message": "Bad Gateway"}}, status=502))
    provider = create_xai_provider(
        "xai:grok-4", {"config": {"apiKey": "test-key", "maxRetries": 0}}
    )
    result = await provider.call_api("test prompt")
    assert "x.ai API error:" in result["error"]
    assert "XAI_API_KEY" in result["error"]


@respx.mock
async def test_401_error():
    respx.post(XAI_URL).mock(return_value=ok({"error": {"message": "Invalid API key"}}, status=401))
    provider = create_xai_provider("xai:grok-4", {"config": {"apiKey": "test-key"}})
    result = await provider.call_api("test prompt")
    assert "API error: 401" in result["error"]


async def test_missing_api_key_is_reported():
    result = await create_xai_provider("xai:grok-4").call_api("test prompt")
    assert result["error"].startswith("x.ai API error: API key is not set. Set the XAI_API_KEY")


@respx.mock
async def test_success_usage_and_cost():
    respx.post(XAI_URL).mock(
        return_value=ok(
            _mock_response(
                "Response with tokens",
                {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            )
        )
    )
    provider = create_xai_provider("xai:grok-4", {"config": {"apiKey": "test-key"}})
    result = await provider.call_api("test prompt")
    assert result["output"] == "Response with tokens"
    assert "error" not in result
    assert result["tokenUsage"]["prompt"] == 1000
    assert result["tokenUsage"]["completion"] == 500
    assert result["tokenUsage"]["total"] == 1500
    # grok-4 redirects to grok-4.3 pricing.
    assert result["cost"] == pytest.approx(1000 * 1.25e-6 + 500 * 2.5e-6)


@respx.mock
async def test_effective_passthrough_model_fallback_pricing():
    respx.post(XAI_URL).mock(
        return_value=ok(
            _mock_response(
                "Build response",
                {"prompt_tokens": 100_000, "completion_tokens": 100_000, "total_tokens": 200_000},
            )
        )
    )
    provider = create_xai_provider("xai:grok-4.3", {"config": {"apiKey": "test-key"}})
    result = await provider.call_api(
        "test prompt",
        {
            "prompt": {
                "raw": "test prompt",
                "label": "test prompt",
                "config": {"passthrough": {"model": "grok-build-0.1"}},
            },
            "vars": {},
        },
    )
    assert result["cost"] == pytest.approx(0.3)


@respx.mock
async def test_cache_read_and_reasoning_from_usage():
    respx.post(XAI_URL).mock(
        return_value=ok(
            _mock_response(
                "Cached response",
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            )
        )
    )
    provider = _xai(
        "grok-4-0709", apiKey="test-key", inputCost=3e-6, outputCost=15e-6, cacheReadCost=0.75e-6
    )
    result = await provider.call_api("test prompt")
    assert result["tokenUsage"]["completionDetails"]["cacheReadInputTokens"] == 800
    assert result["cost"] == pytest.approx(0.0087)


@respx.mock
async def test_missing_usage_leaves_cost_undefined():
    respx.post(XAI_URL).mock(
        return_value=ok({"choices": [{"message": {"content": "Response without usage"}}]})
    )
    provider = create_xai_provider("xai:grok-4-0709", {"config": {"apiKey": "test-key"}})
    result = await provider.call_api("test prompt")
    assert result["output"] == "Response without usage"
    assert "cost" not in result


@pytest.mark.parametrize(
    ("model", "usage", "cost"),
    [
        (
            "grok-3-mini-beta",
            {
                "prompt_tokens": 500,
                "completion_tokens": 500,
                "total_tokens": 1200,
                "completion_tokens_details": {"reasoning_tokens": 200},
            },
            0.0005,
        ),
        (
            "grok-4.5",
            {
                "prompt_tokens": 4511,
                "completion_tokens": 1,
                "total_tokens": 4570,
                "prompt_tokens_details": {"cached_tokens": 384},
                "completion_tokens_details": {"reasoning_tokens": 58},
            },
            0.0087232,
        ),
        (
            "grok-4.6",
            {
                "prompt_tokens": 4653,
                "completion_tokens": 1,
                "total_tokens": 4954,
                "prompt_tokens_details": {"cached_tokens": 4608},
                "completion_tokens_details": {"reasoning_tokens": 300},
            },
            0.0042,
        ),
        (
            "grok-4.5",
            {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "cost_in_usd_ticks": 123_456_789,
            },
            0.0123456789,
        ),
    ],
)
@respx.mock
async def test_live_verified_costs(model, usage, cost):
    respx.post(XAI_URL).mock(return_value=ok(_mock_response("4", usage)))
    provider = create_xai_provider(f"xai:{model}", {"config": {"apiKey": "test-key"}})
    result = await provider.call_api("test prompt")
    assert result["cost"] == pytest.approx(cost, abs=1e-12)


@respx.mock
async def test_billed_ticks_on_refusals():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "cost_in_usd_ticks": 123_456_789,
    }
    respx.post(XAI_URL).mock(
        side_effect=[
            ok(
                {
                    "choices": [
                        {
                            "message": {"content": None, "refusal": "I cannot help with that."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            ),
            ok(
                {
                    "error": {
                        "code": "invalid_prompt",
                        "message": "The prompt was rejected by the safety system.",
                    },
                    "usage": usage,
                },
                status=400,
            ),
        ]
    )
    provider = create_xai_provider("xai:grok-4.5", {"config": {"apiKey": "test-key"}})
    refusal = await provider.call_api("test prompt")
    assert refusal["isRefusal"] is True
    assert refusal["cost"] == 0.0123456789
    blocked = await provider.call_api("blocked prompt")
    assert "error" not in blocked
    assert blocked["isRefusal"] is True
    assert blocked["cached"] is False
    assert blocked["tokenUsage"]["prompt"] == 100
    assert blocked["cost"] == 0.0123456789


@pytest.mark.parametrize("service_tier", ["default", "priority"])
@respx.mock
async def test_custom_cost_overrides_beat_ticks(service_tier):
    respx.post(XAI_URL).mock(
        return_value=ok(
            _mock_response(
                "custom-priced response",
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 8},
                    "cost_in_usd_ticks": 123_456_789,
                },
                service_tier=service_tier,
            )
        )
    )
    provider = _xai("grok-4.5", apiKey="test-key", cost=0.001, service_tier="priority")
    result = await provider.call_api("test prompt")
    assert result["cost"] == pytest.approx(0.015)


@respx.mock
async def test_priority_premium_on_fallback_pricing():
    respx.post(XAI_URL).mock(
        return_value=ok(
            _mock_response(
                "priority response",
                {"prompt_tokens": 100_000, "completion_tokens": 100_000, "total_tokens": 200_000},
                service_tier="priority",
            )
        )
    )
    provider = create_xai_provider(
        "xai:grok-4.5", {"config": {"apiKey": "test-key", "service_tier": "priority"}}
    )
    result = await provider.call_api("test prompt")
    assert result["cost"] == pytest.approx(1.6)


@respx.mock
async def test_grok47_us_region_pricing_and_billed_cost():
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 500,
        "total_tokens": 1_520,
        "prompt_tokens_details": {"cached_tokens": 800},
        "completion_tokens_details": {"reasoning_tokens": 20},
    }
    route = respx.post("https://us.api.x.ai/v1/chat/completions").mock(
        side_effect=[
            ok(_mock_response("result", usage)),
            ok(_mock_response("result", {**usage, "cost_in_usd_ticks": 123_000})),
        ]
    )
    provider = _xai("grok-4.7", apiKey="test-key", region="us")
    assert (await provider.call_api("estimate"))["cost"] == pytest.approx(0.004312)
    assert (await provider.call_api("billed"))["cost"] == pytest.approx(0.0000123)
    assert route.call_count == 2


# ---------------------------------------------------------------- Perplexity


def test_perplexity_creation():
    provider = create_perplexity_provider("perplexity:sonar")
    assert isinstance(provider, PerplexityProvider)
    assert create_perplexity_provider("perplexity:").model_name == "sonar"
    for model in ("sonar-pro", "sonar-reasoning", "sonar-deep-research", "r1-1776"):
        assert create_perplexity_provider(f"perplexity:{model}").model_name == model
    config = {"temperature": 0.7, "max_tokens": 1000, "search_recency_filter": "week"}
    provider = create_perplexity_provider("perplexity:sonar", {"config": {"config": config}})
    assert config.items() <= provider.config.items()


def test_perplexity_defaults_id_and_json():
    provider = PerplexityProvider("sonar")
    assert provider.config["apiBaseUrl"] == "https://api.perplexity.ai"
    assert provider.config["apiKeyEnvar"] == "PERPLEXITY_API_KEY"
    assert str(provider) == "[Perplexity Provider sonar]"
    assert PerplexityProvider("sonar-pro").id() == "sonar-pro"
    provider = PerplexityProvider(
        "sonar-pro", {"config": {"temperature": 0.7, "max_tokens": 1000, "apiKey": "k"}}
    )
    json = provider.to_json()
    assert json["provider"] == "perplexity"
    assert json["model"] == "sonar-pro"
    assert json["config"]["temperature"] == 0.7
    assert "apiKey" not in json["config"]


def test_perplexity_create_provider_from_registry_options():
    provider = pplx.create_provider(
        "perplexity:sonar-pro", {"config": {"temperature": 0.1}, "env": None}
    )
    assert provider.model_name == "sonar-pro"
    assert provider.config["temperature"] == 0.1
    assert provider.id() == "sonar-pro"


async def _pbody(provider, prompt_config=None):
    context = {"prompt": {"raw": "Test prompt", "label": "Test prompt", "config": prompt_config}}
    context["vars"] = {}
    return (await provider.get_openai_body("Test prompt", context if prompt_config else None))[
        "body"
    ]


async def test_perplexity_forwards_search_options():
    provider = PerplexityProvider(
        "sonar-pro",
        {
            "config": {
                "search_domain_filter": ["example.com"],
                "search_recency_filter": "week",
                "return_related_questions": True,
                "return_images": True,
                "search_after_date_filter": "01/01/2026",
                "search_before_date_filter": "02/01/2026",
                "web_search_options": {"search_context_size": "high"},
            }
        },
    )
    body = await _pbody(
        provider,
        {
            "search_domain_filter": ["prompt.example"],
            "web_search_options": {"search_context_size": "low"},
        },
    )
    assert body["search_domain_filter"] == ["prompt.example"]
    assert body["search_recency_filter"] == "week"
    assert body["return_related_questions"] is True
    assert body["return_images"] is True
    assert body["search_after_date_filter"] == "01/01/2026"
    assert body["web_search_options"] == {"search_context_size": "low"}
    body = await _pbody(
        provider,
        {
            "search_domain_filter": ["prompt.example"],
            "passthrough": {
                "search_domain_filter": ["passthrough.example"],
                "web_search_options": {"search_context_size": "medium"},
            },
        },
    )
    assert body["search_domain_filter"] == ["passthrough.example"]
    assert body["web_search_options"] == {"search_context_size": "medium"}


async def test_perplexity_prompt_options_beat_provider_passthrough():
    provider = PerplexityProvider(
        "sonar-pro",
        {
            "config": {
                "passthrough": {
                    "search_domain_filter": ["provider.example"],
                    "return_images": True,
                    "web_search_options": {"search_context_size": "high"},
                }
            }
        },
    )
    body = await _pbody(
        provider,
        {
            "search_domain_filter": ["prompt.example"],
            "return_images": False,
            "web_search_options": {"search_context_size": "low"},
        },
    )
    assert body["search_domain_filter"] == ["prompt.example"]
    assert body["return_images"] is False
    assert body["web_search_options"] == {"search_context_size": "low"}


async def test_perplexity_prompt_passthrough_replaces_provider_passthrough():
    provider = PerplexityProvider(
        "sonar-pro",
        {
            "config": {
                "passthrough": {
                    "search_domain_filter": ["private.example"],
                    "web_search_options": {"search_context_size": "high"},
                }
            }
        },
    )
    body = await _pbody(provider, {"passthrough": {"model": "sonar-pro"}})
    assert body["model"] == "sonar-pro"
    assert "search_domain_filter" not in body
    assert "web_search_options" not in body


async def test_perplexity_null_search_options():
    search = {
        "search_domain_filter": ["provider.example"],
        "web_search_options": {"search_context_size": "high"},
    }
    cleared = {"search_domain_filter": None, "web_search_options": None}
    body = await _pbody(
        PerplexityProvider("sonar-pro", {"config": search}), {"passthrough": cleared}
    )
    assert body["search_domain_filter"] is None
    assert body["web_search_options"] is None
    provider = PerplexityProvider(
        "sonar-pro",
        {
            "config": {
                **search,
                "search_recency_filter": "week",
                "passthrough": {**cleared, "search_recency_filter": None},
            }
        },
    )
    body = await _pbody(provider)
    assert body["search_domain_filter"] is None
    assert body["search_recency_filter"] is None
    assert body["web_search_options"] is None
    body = await _pbody(
        PerplexityProvider("sonar-pro", {"config": {"passthrough": search}}), cleared
    )
    assert body["search_domain_filter"] is None
    assert body["web_search_options"] is None


def _pplx_response(usage=None, **extra):
    data = {
        "choices": [{"message": {"content": "Test output"}, "finish_reason": "stop"}],
        **extra,
    }
    if usage is not None:
        data["usage"] = usage
    return data


def _pplx(model="sonar-pro"):
    return PerplexityProvider(model, {"config": {"apiKey": "test-key"}})


@respx.mock
async def test_perplexity_reported_total_cost():
    route = respx.post(PPLX_URL).mock(
        return_value=ok(
            _pplx_response(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                    "cost": {"request_cost": 0.01, "total_cost": 0.01018},
                }
            )
        )
    )
    result = await _pplx().call_api("Test prompt")
    assert result["output"] == "Test output"
    assert result["cost"] == 0.01018
    assert result["tokenUsage"]["prompt"] == 10
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize(
    "model", ["sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-deep-research", "gpt-4o"]
)
@respx.mock
async def test_perplexity_cost_unknown_without_total(model):
    respx.post(PPLX_URL).mock(
        return_value=ok(
            _pplx_response({"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200})
        )
    )
    result = await _pplx(model).call_api("Test prompt")
    assert result["output"] == "Test output"
    assert "cost" not in result


@pytest.mark.parametrize("cost", [0, 0.123])
@respx.mock
async def test_perplexity_total_without_token_counts(cost):
    respx.post(PPLX_URL).mock(return_value=ok(_pplx_response({"cost": {"total_cost": cost}})))
    assert (await _pplx("custom-model").call_api("Test prompt"))["cost"] == cost


@pytest.mark.parametrize("total", [-1, "0.01", None, {}, True])
@respx.mock
async def test_perplexity_invalid_total(total):
    respx.post(PPLX_URL).mock(
        return_value=ok(
            _pplx_response(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                    "cost": {"total_cost": total},
                }
            )
        )
    )
    result = await _pplx().call_api("Test prompt")
    assert result["output"] == "Test output"
    assert "cost" not in result


@pytest.mark.parametrize("usage", [None, {}, {"total_tokens": 0}, {"cost": None}])
@respx.mock
async def test_perplexity_missing_billing_data(usage):
    respx.post(PPLX_URL).mock(return_value=ok(_pplx_response(usage)))
    result = await _pplx().call_api("Test prompt")
    assert result["output"] == "Test output"
    assert "cost" not in result


@pytest.mark.parametrize("status", [400, 429, 500])
@respx.mock
async def test_perplexity_http_errors_have_no_cost(status):
    respx.post(PPLX_URL).mock(
        return_value=ok(
            {"error": {"message": "API error"}, "usage": {"cost": {"total_cost": 0.01}}},
            status=status,
        )
    )
    provider = PerplexityProvider("sonar-pro", {"config": {"apiKey": "k", "maxRetries": 0}})
    result = await provider.call_api("Test prompt")
    # A 429 surfaces through the fetch layer's rate-limit error once retries are exhausted.
    expected = "HTTP 429" if status == 429 else f"API error: {status}"
    assert expected in result["error"]
    assert "cost" not in result


@respx.mock
async def test_perplexity_cache_hit_keeps_reported_total(openai_env):
    openai_env.setattr(cache, "_enabled", True)
    route = respx.post(PPLX_URL).mock(
        return_value=ok(
            _pplx_response(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                    "cost": {"total_cost": 0.01418},
                },
                citations=["https://example.com/cached-source"],
            )
        )
    )
    provider = _pplx()
    first = await provider.call_api("Test prompt")
    second = await provider.call_api("Test prompt")
    expected_metadata = {
        "citations": [
            {
                "url": "https://example.com/cached-source",
                "content": "https://example.com/cached-source",
            }
        ],
        "perplexity": {"citations": ["https://example.com/cached-source"]},
    }
    assert first["cached"] is False
    assert first["cost"] == 0.01418
    assert second["cached"] is True
    assert second["cost"] == 0.01418
    assert second["tokenUsage"]["cached"] == 20
    for response in (first, second):
        for key, value in expected_metadata.items():
            assert response["metadata"][key] == value
    assert route.call_count == 1


@respx.mock
async def test_perplexity_search_artifacts_in_metadata():
    images = [{"image_url": "https://example.com/image.jpg", "width": 640}]
    citations = ["https://example.com/article", "not-a-url"]
    search_results = [{"title": "Example article", "url": "https://example.com/article"}]
    respx.post(PPLX_URL).mock(
        return_value=ok(
            _pplx_response(
                citations=citations,
                search_results=search_results,
                images=images,
                related_questions=["What happened next?"],
            )
        )
    )
    provider = PerplexityProvider(
        "sonar-pro",
        {"config": {"apiKey": "test-key", "return_images": True, "return_related_questions": True}},
    )
    result = await provider.call_api("Test prompt")
    assert result["output"] == "Test output"
    assert result["metadata"]["citations"] == [
        {"url": citations[0], "content": citations[0]},
        "not-a-url",
    ]
    assert result["metadata"]["perplexity"] == {
        "citations": citations,
        "search_results": search_results,
        "images": images,
        "related_questions": ["What happened next?"],
    }


def test_perplexity_calculate_response_cost_prefers_total():
    provider = PerplexityProvider("sonar-pro")
    cost = provider.calculate_response_cost(
        {"usage": {"prompt_tokens": 10, "cost": {"total_cost": 0.01418}}}, {}, False
    )
    assert cost == 0.01418
    for total in (float("nan"), float("inf"), float("-inf")):
        data = {"usage": {"cost": {"total_cost": total}}}
        assert provider.calculate_response_cost(data, {}, False) is None


def test_calculate_perplexity_cost():
    assert calculate_perplexity_cost("sonar") == 0
    assert calculate_perplexity_cost("sonar", 0, 0) == 0
    assert calculate_perplexity_cost("sonar", 1_000_000, 1_000_000) == 2
    assert calculate_perplexity_cost("sonar-pro", 1_000_000, 1_000_000) == 18
    assert calculate_perplexity_cost("sonar-pro", 0, 2_000_000) == 30
    assert calculate_perplexity_cost("sonar-reasoning", 1_000_000, 1_000_000) == 6
    assert calculate_perplexity_cost("sonar-reasoning-pro", 1_000_000, 1_000_000) == 10
    assert calculate_perplexity_cost("sonar-deep-research", 500_000, 500_000) == 5
    assert calculate_perplexity_cost("r1-1776", 0, 2_000_000) == 16
    for model in ("unknown-model", "sonar-pro-future", "my-sonar"):
        assert calculate_perplexity_cost(model, 1_000_000, 1_000_000) is None
    assert calculate_perplexity_cost("SONAR-PRO", 1_000_000, 1_000_000) == 18
    assert calculate_perplexity_cost("sonar-pro", 1_000_000, 1_000_000, "high") == 18
