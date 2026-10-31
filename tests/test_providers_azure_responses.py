"""Azure Responses provider. Ports the deterministic cases of
test/providers/azure/responses.test.ts and responses-nested-schema.integration.test.ts, against
respx-mocked endpoints."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from azure_helpers import azure_env, fake_az, ok, sent_json

from promptphoo.providers.azure import AzureResponsesProvider, create_provider
from promptphoo.providers.azure import responses as azure_responses

pytestmark = pytest.mark.usefixtures(azure_env.__name__)

HOST = "test.openai.azure.com"
URL = f"https://{HOST}/openai/v1/responses"


@pytest.fixture(autouse=True)
def _env(azure_env):
    azure_env.setenv("AZURE_API_KEY", "test-key")
    azure_env.setenv("AZURE_API_HOST", HOST)


def message(text: str) -> dict[str, Any]:
    content = [{"type": "output_text", "text": text}]
    return {"type": "message", "role": "assistant", "content": content}


def cached_token_response() -> dict[str, Any]:
    return {
        "output": [message("4")],
        "usage": {
            "input_tokens": 2_000,
            "input_tokens_details": {"cached_tokens": 500},
            "output_tokens": 1_000,
        },
    }


BASE_COST = (1_500 * 5 + 500 * 0.5 + 1_000 * 30) / 1e6


def provider(deployment: str = "gpt-4.1-test", **config: Any) -> AzureResponsesProvider:
    return AzureResponsesProvider(deployment, {"config": config} if config else {})


# --- constructor, factory, id ------------------------------------------------------------------


def test_constructor_and_shape():
    p = provider()
    assert isinstance(p, AzureResponsesProvider)
    assert p.deploymentName == "gpt-4.1-test"
    assert p.id() == "azure:gpt-4.1-test"
    assert p.to_json() == {"id": "azure:gpt-4.1-test"}
    assert str(p) == "[Azure Provider gpt-4.1-test]"
    labelled = AzureResponsesProvider("d", {"id": "custom", "label": "L"})
    assert labelled.id() == "custom"
    assert labelled.to_json()["label"] == "L"


def test_factory_routes_responses():
    p = create_provider("azure:responses:my-dep", {})
    assert isinstance(p, AzureResponsesProvider)
    assert p.deployment_name == "my-dep"
    assert create_provider("azureopenai:responses:x:y", {}).deployment_name == "x:y"
    assert create_provider("azure:responses", {}).deployment_name == "gpt-4.1-2025-04-14"


# --- isReasoningModel / supportsTemperature ----------------------------------------------------


@pytest.mark.parametrize(
    ("deployment", "config", "expected"),
    [
        ("o1-preview", {}, True),
        ("o3-mini", {}, True),
        ("gpt-5", {}, True),
        ("gpt-4.1", {}, False),
        ("my-custom-deployment", {"isReasoningModel": True}, True),
        ("my-custom-deployment", {"o1": True}, True),
        ("my-custom-deployment", {}, False),
    ],
)
def test_is_reasoning_model(deployment, config, expected):
    assert provider(deployment, **config).is_reasoning_model() is expected


def test_supports_temperature():
    assert provider("gpt-4.1").supports_temperature() is True
    assert provider("o1-preview").supports_temperature() is False


# --- getAzureResponsesBody ---------------------------------------------------------------------


async def test_basic_body():
    body = await provider().get_azure_responses_body("Hello world")
    assert body == {
        "model": "gpt-4.1-test",
        "input": "Hello world",
        "max_output_tokens": 1024,
        "temperature": 0,
        "text": {"format": {"type": "text"}},
    }


async def test_json_array_prompt_is_input_list():
    body = await provider().get_azure_responses_body('[{"role": "user", "content": "hi"}]')
    assert body["input"] == [{"role": "user", "content": "hi"}]


async def test_external_response_format_loaded_once(monkeypatch):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "schema": {"type": "object", "properties": {"test": {"type": "string"}}},
        },
    }
    calls: list[tuple[Any, Any]] = []

    def fake_load(value, vars=None):
        calls.append((value, vars))
        return schema

    monkeypatch.setattr(azure_responses, "maybe_load_response_format_from_external_file", fake_load)
    body = await provider(response_format="file://test-schema.json").get_azure_responses_body("x")
    assert calls == [("file://test-schema.json", None)]
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "test_schema",
        "schema": schema["json_schema"]["schema"],
        "strict": True,
    }


async def test_inline_response_format():
    fmt = {
        "type": "json_schema",
        "json_schema": {
            "name": "inline_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }
    body = await provider(response_format=fmt).get_azure_responses_body("x")
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "inline_schema",
        "schema": fmt["json_schema"]["schema"],
        "strict": True,
    }


async def test_json_object_response_format():
    body = await provider(response_format={"type": "json_object"}).get_azure_responses_body("x")
    assert body["text"] == {"format": {"type": "json_object"}}


async def test_temperature_by_model():
    body = await provider("o1-preview", temperature=0.7).get_azure_responses_body("x")
    assert "temperature" not in body
    body = await provider(temperature=0.7).get_azure_responses_body("x")
    assert body["temperature"] == 0.7
    body = await provider(temperature=0).get_azure_responses_body("x")
    assert body["temperature"] == 0


@pytest.mark.parametrize("deployment", ["gpt-chat-latest", "gpt-chat-latest-2026-06-24"])
async def test_fixed_reasoning_alias(deployment):
    p = provider(deployment, max_output_tokens=2_000, reasoning_effort="high", temperature=0.7)
    body = await p.get_azure_responses_body("Hello world")
    assert body["max_output_tokens"] == 2_000
    assert "effort" not in (body.get("reasoning") or {})
    assert "temperature" not in body


async def test_fixed_reasoning_keeps_summary_from_prompt_passthrough():
    direct = await provider("gpt-chat-latest").get_azure_responses_body("hello")
    assert "effort" not in (direct.get("reasoning") or {})

    p = provider("opaque-deployment", modelName="gpt-4.1")
    passthrough = {"reasoning": {"effort": "high", "summary": "auto"}}
    body = await p.get_azure_responses_body(
        "hello",
        {
            "vars": {},
            "prompt": {
                "raw": "hello",
                "label": "override",
                "config": {"modelName": "gpt-chat-latest", "passthrough": passthrough},
            },
        },
    )
    assert body["model"] == "opaque-deployment"
    assert body["reasoning"] == {"summary": "auto"}
    assert "temperature" not in body
    assert passthrough["reasoning"] == {"effort": "high", "summary": "auto"}


async def test_reasoning_effort_rendered():
    p = provider("o3-mini", reasoning_effort="{{ effort }}")
    body = await p.get_azure_responses_body("x", {"vars": {"effort": "low"}})
    assert body["reasoning"] == {"effort": "low"}


async def test_omit_defaults(azure_env):
    body = await provider(omitDefaults=True).get_azure_responses_body("x")
    assert "temperature" not in body and "max_output_tokens" not in body

    azure_env.setenv("OPENAI_TEMPERATURE", "0.5")
    azure_env.setenv("OPENAI_MAX_TOKENS", "2048")
    body = await provider(omitDefaults=True).get_azure_responses_body("x")
    assert body["temperature"] == 0.5
    assert body["max_output_tokens"] == 2048


async def test_reasoning_max_output_tokens_env(azure_env):
    body = await provider("o1-preview").get_azure_responses_body("x")
    assert "max_output_tokens" not in body
    body = await provider("o1-preview", omitDefaults=True).get_azure_responses_body("x")
    assert "max_output_tokens" not in body

    azure_env.setenv("OPENAI_MAX_TOKENS", "2048")
    assert (await provider("o1-preview").get_azure_responses_body("x"))["max_output_tokens"] == 2048
    body = await provider("o1-preview", omitDefaults=True).get_azure_responses_body("x")
    assert body["max_output_tokens"] == 2048

    azure_env.setenv("OPENAI_MAX_COMPLETION_TOKENS", "4096")
    body = await provider("o1-preview", omitDefaults=True).get_azure_responses_body("x")
    assert body["max_output_tokens"] == 4096


async def test_max_output_tokens_zero_is_sent():
    body = await provider(max_output_tokens=0).get_azure_responses_body("x")
    assert body["max_output_tokens"] == 0


async def test_verbosity():
    body = await provider("gpt-5", verbosity="high").get_azure_responses_body("x")
    assert body["text"] == {"format": {"type": "text"}, "verbosity": "high"}
    p = provider("my-custom-deployment", isReasoningModel=True, verbosity="medium")
    assert (await p.get_azure_responses_body("x"))["text"]["verbosity"] == "medium"
    body = await provider(verbosity="high").get_azure_responses_body("x")
    assert body["text"] == {"format": {"type": "text"}}


async def test_service_tier():
    body = await provider("gpt-5.6-sol", service_tier="priority").get_azure_responses_body("x")
    assert body["service_tier"] == "priority"
    p = provider("gpt-5.6-sol", service_tier="priority", passthrough={"service_tier": "default"})
    assert (await p.get_azure_responses_body("x"))["service_tier"] == "default"
    p = provider("gpt-5.6-sol", service_tier=None, passthrough={"service_tier": "priority"})
    assert (await p.get_azure_responses_body("x"))["service_tier"] == "priority"
    body = await provider("gpt-5.6-sol", service_tier=None).get_azure_responses_body("x")
    assert "service_tier" not in body


async def test_tools_and_tool_choice_are_flattened():
    tool = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    p = provider(
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        parallel_tool_calls=0,
        store=1,
        instructions="Be brief",
        top_p=0.5,
        max_tool_calls=3,
    )
    body = await p.get_azure_responses_body("x")
    assert body["tools"] == [{"type": "function", "name": "get_weather", "parameters": {}}]
    assert body["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert body["parallel_tool_calls"] is False
    assert body["store"] is True
    assert body["instructions"] == "Be brief"
    assert body["top_p"] == 0.5
    assert body["max_tool_calls"] == 3
    body = await provider(tool_choice="auto").get_azure_responses_body("x")
    assert body["tool_choice"] == "auto"


# --- callApi -----------------------------------------------------------------------------------


@respx.mock
async def test_successful_call_url_and_headers():
    route = respx.post(URL).mock(
        return_value=ok(
            {
                "output": [message("Hello! How can I help you?")],
                "usage": {"input_tokens": 10, "output_tokens": 8},
            }
        )
    )
    result = await provider().call_api("Hello")
    assert result["output"] == "Hello! How can I help you?"
    assert result["cached"] is False
    request = route.calls[0].request
    assert str(request.url) == f"{URL}?api-version=preview"
    assert request.headers["api-key"] == "test-key"
    assert request.headers["Content-Type"] == "application/json"
    assert "Authorization" not in request.headers
    assert sent_json(route)["model"] == "gpt-4.1-test"


@respx.mock
async def test_api_version_and_custom_headers():
    route = respx.post(URL).mock(return_value=ok({"output": [message("ok")]}))
    p = provider(apiVersion="2025-04-01-preview", headers={"X-Custom": "1"})
    await p.call_api("Hello")
    request = route.calls[0].request
    assert request.url.params["api-version"] == "2025-04-01-preview"
    assert request.headers["X-Custom"] == "1"


@respx.mock
async def test_partial_output_keeps_incomplete_finish_reason():
    response = {
        "id": "resp_azure_incomplete",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{**message("Partial Azure answer"), "status": "incomplete"}],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    respx.post(URL).mock(return_value=ok(response))
    result = await provider().call_api("Hello")
    assert "error" not in result
    assert result["output"] == "Partial Azure answer"
    assert result["raw"] == response
    assert result["finishReason"] == "length"
    assert result["metadata"]["responseStatus"] == "incomplete"
    assert result["metadata"]["incompleteReason"] == "max_output_tokens"
    assert result["tokenUsage"] == {"prompt": 10, "completion": 20, "total": 30, "numRequests": 1}


async def test_missing_api_host(azure_env):
    azure_env.delenv("AZURE_API_HOST")
    with pytest.raises(ValueError, match=r"Azure API configuration missing.*AZURE_API_HOST"):
        await provider().call_api("test")


async def test_missing_authentication(monkeypatch):
    p = provider()

    async def empty_headers() -> None:
        p.auth_headers = {}

    monkeypatch.setattr(p, "ensure_initialized", empty_headers)
    with pytest.raises(ValueError, match=r"Azure API authentication failed.*AZURE_API_KEY"):
        await p.call_api("test")


@respx.mock
async def test_entra_id_bearer_auth(azure_env, fake_az):
    azure_env.delenv("AZURE_API_KEY")
    fake_az.reply("test-entra-token", expires_on=4_102_444_800)
    route = respx.post(URL).mock(
        return_value=ok(
            {
                "output": [message("Hello from Entra ID auth!")],
                "usage": {"input_tokens": 10, "output_tokens": 8},
            }
        )
    )
    result = await provider().call_api("Hello")
    assert result["output"] == "Hello from Entra ID auth!"
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer test-entra-token"
    assert "api-key" not in request.headers


@respx.mock
async def test_cost_from_responses_usage():
    respx.post(URL).mock(
        return_value=ok(
            {"output": [message("4")], "usage": {"input_tokens": 1000, "output_tokens": 500}}
        )
    )
    result = await provider("gpt-4.1").call_api("What is 2+2?")
    assert result["cost"] == pytest.approx((1000 * 2 + 500 * 8) / 1e6, abs=1e-12)
    assert result["tokenUsage"]["prompt"] == 1000
    assert result["tokenUsage"]["completion"] == 500


@respx.mock
async def test_cached_input_rate():
    respx.post(URL).mock(return_value=ok(cached_token_response()))
    result = await provider("gpt-5.6").call_api("What is 2+2?")
    assert result["cost"] == pytest.approx(BASE_COST, abs=1e-12)
    assert result["tokenUsage"]["prompt"] == 2_000
    assert result["tokenUsage"]["completion"] == 1_000
    assert result["tokenUsage"]["cached"] == 500


@respx.mock
async def test_disk_cached_result_keeps_full_cached_usage():
    response = cached_token_response()
    response["usage"]["total_tokens"] = 3_000
    route = respx.post(URL).mock(return_value=ok(response))
    p = provider("gpt-5.6")
    await p.call_api("What is 2+2?")
    result = await p.call_api("What is 2+2?")
    assert route.call_count == 1
    assert result["cached"] is True
    assert result["tokenUsage"]["cached"] == 3_000
    assert result["tokenUsage"]["total"] == 3_000


@pytest.mark.parametrize(
    ("deployment", "config", "served", "sent", "multiplier"),
    [
        ("gpt-5.6-sol", {"passthrough": {"service_tier": "priority"}}, None, "priority", 2),
        ("gpt-5.6-sol", {"service_tier": "priority"}, None, "priority", 2),
        ("gpt-5.6-sol", {"service_tier": "priority", "passthrough": {"service_tier": "default"}},
         None, "default", 1),
        ("gpt-5.6-sol", {"service_tier": "priority"}, "default", "priority", 1),
        ("gpt-5.6-sol", {"service_tier": "auto"}, "priority", "auto", 2),
        ("my-custom-deployment",
         {"service_tier": "default", "passthrough": {"model": "gpt-5.6-sol"}},
         "priority", "default", 2),
    ],
)  # fmt: skip
@respx.mock
async def test_service_tier_pricing(deployment, config, served, sent, multiplier):
    response = cached_token_response()
    if served:
        response = {"service_tier": served, **response}
    route = respx.post(URL).mock(return_value=ok(response))
    result = await provider(deployment, **config).call_api("What is 2+2?")
    assert sent_json(route)["service_tier"] == sent
    assert result["cost"] == pytest.approx(multiplier * BASE_COST, abs=1e-12)


@respx.mock
async def test_null_service_tier_omitted_from_request_and_cost(monkeypatch):
    seen: list[Any] = []
    real = azure_responses.calculate_azure_cost

    def spy(model, config, *rest):
        seen.append(config)
        return real(model, config, *rest)

    monkeypatch.setattr(azure_responses, "calculate_azure_cost", spy)
    route = respx.post(URL).mock(return_value=ok(cached_token_response()))
    await provider("gpt-5.6-sol", service_tier=None).call_api("What is 2+2?")
    assert "service_tier" not in sent_json(route)
    assert "service_tier" not in seen[0]
    assert "service_tier" not in seen[0]["passthrough"]


@respx.mock
async def test_image_token_pricing():
    respx.post(URL).mock(
        return_value=ok(
            {
                "output": [message("ok")],
                "usage": {
                    "input_tokens": 1_000,
                    "input_tokens_details": {"image_tokens": 400},
                    "output_tokens": 0,
                },
            }
        )
    )
    result = await provider("gpt-image-1").call_api("Describe the image")
    assert result["cost"] == pytest.approx((600 * 5 + 400 * 10) / 1e6, abs=1e-12)


@respx.mock
async def test_missing_response_format_file_raises(tmp_path):
    route = respx.post(URL)
    missing = tmp_path / "missing.json"
    p = provider(response_format=f"file://{missing}")
    with pytest.raises(ValueError, match=r"Failed to load response_format file.*missing\.json"):
        await p.call_api("test")
    assert route.call_count == 0


@respx.mock
async def test_api_error():
    respx.post(URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
    )
    result = await provider(maxRetries=0).call_api("Hello")
    assert result["error"] == (
        'API error: 401 Unauthorized\n{"error":{"message":"Invalid API key"}}'
    )


@respx.mock
async def test_network_error():
    respx.post(URL).mock(side_effect=httpx.ConnectError("boom"))
    result = await provider(maxRetries=0).call_api("Hello")
    assert result["error"].startswith("API call error: ")
    assert "boom" in result["error"]


@respx.mock
async def test_error_in_body_is_formatted():
    respx.post(URL).mock(return_value=ok({"error": {"message": "bad", "type": "invalid"}}))
    result = await provider().call_api("Hello")
    assert "bad" in result["error"]


@respx.mock
async def test_deep_research_timeout(monkeypatch):
    seen: list[Any] = []
    real = azure_responses.fetch.fetch_with_cache

    async def spy(url, options, timeout, *rest):
        seen.append(timeout)
        return await real(url, options, timeout, *rest)

    monkeypatch.setattr(azure_responses.fetch, "fetch_with_cache", spy)
    respx.post(URL).mock(return_value=ok({"output": [message("Research complete")]}))
    await provider("o3-deep-research-test").call_api("Research question")
    assert seen == [600_000]


async def test_stream_rejected():
    with pytest.raises(ValueError, match=r"stream: true\) are not supported by promptphoo"):
        await provider(stream=True).call_api("Hello")


@respx.mock
async def test_background_in_progress_rejected_and_evicted():
    route = respx.post(URL).mock(return_value=ok({"id": "resp_1", "status": "queued"}))
    p = provider(passthrough={"background": True})
    result = await p.call_api("Hello")
    assert result["error"] == (
        "Background response resp_1 is queued; polling background responses is not supported "
        "by promptphoo. Remove `background` from the provider config."
    )
    await p.call_api("Hello")
    assert route.call_count == 2


# --- nested schema loading (responses-nested-schema.integration.test.ts) -----------------------

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_name": {"type": "string"},
        "date": {"type": "string"},
        "location": {"type": "string"},
    },
    "required": ["event_name", "date", "location"],
    "additionalProperties": False,
}
FLAT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


@pytest.fixture
def schema_dir(tmp_path):
    (tmp_path / "event-schema.json").write_text(json.dumps(EVENT_SCHEMA, indent=2))
    (tmp_path / "nested-format.json").write_text(
        json.dumps(
            {
                "type": "json_schema",
                "name": "event_extraction",
                "schema": f"file://{tmp_path / 'event-schema.json'}",
            }
        )
    )
    (tmp_path / "flat-format.json").write_text(
        json.dumps({"type": "json_schema", "name": "flat_schema", "schema": FLAT_SCHEMA})
    )
    (tmp_path / "invalid-format.json").write_text("{invalid json")
    (tmp_path / "templated-format.json").write_text(
        json.dumps(
            {
                "type": "json_schema",
                "name": "templated_schema",
                "schema": f"file://{tmp_path / '{{ schema_name }}.json'}",
            }
        )
    )
    return tmp_path


@pytest.fixture
def event_route():
    with respx.mock:
        text = '{"event_name": "Conference", "date": "2025-01-15", "location": "NYC"}'
        yield respx.post(URL).mock(
            return_value=ok(
                {"output": [message(text)], "usage": {"input_tokens": 10, "output_tokens": 20}}
            )
        )


async def test_nested_schema_from_file(schema_dir, event_route):
    p = provider(response_format=f"file://{schema_dir / 'nested-format.json'}")
    result = await p.call_api("Extract event info from: Conference in NYC on Jan 15")
    assert "error" not in result
    assert result["output"]
    fmt = sent_json(event_route)["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "event_extraction"
    assert fmt["schema"] == EVENT_SCHEMA


async def test_nested_schema_paths_render_vars(schema_dir, event_route):
    p = provider(response_format=f"file://{schema_dir / '{{ format_name }}.json'}")
    result = await p.call_api(
        "Extract event info",
        {
            "vars": {"format_name": "templated-format", "schema_name": "event-schema"},
            "prompt": {"raw": "Extract event info", "label": "test"},
        },
    )
    assert "error" not in result
    assert event_route.call_count == 1
    fmt = sent_json(event_route)["text"]["format"]
    assert fmt["name"] == "templated_schema"
    assert fmt["schema"]["required"] == ["event_name", "date", "location"]


async def test_only_effective_prompt_response_format_loaded(schema_dir, event_route):
    p = provider(response_format=f"file://{schema_dir / 'missing-provider.json'}")
    result = await p.call_api(
        "Extract event info",
        {
            "vars": {"format_name": "flat-format"},
            "prompt": {
                "raw": "Extract event info",
                "label": "test",
                "config": {"response_format": f"file://{schema_dir / '{{ format_name }}.json'}"},
            },
        },
    )
    assert "error" not in result
    assert sent_json(event_route)["text"]["format"]["name"] == "flat_schema"


@pytest.mark.parametrize("format_name", ["missing-format", "invalid-format"])
async def test_prompt_response_format_load_errors(schema_dir, event_route, format_name):
    p = provider(response_format={"type": "json_object"})
    response_format = f"file://{schema_dir / '{{ format_name }}.json'}"
    with pytest.raises(ValueError) as excinfo:
        await p.call_api(
            "Extract event info",
            {
                "vars": {"format_name": format_name},
                "prompt": {
                    "raw": "Extract event info",
                    "label": "test",
                    "config": {"response_format": response_format},
                },
            },
        )
    assert f"Failed to load response_format file: {response_format}" in str(excinfo.value)
    assert event_route.call_count == 0


async def test_missing_nested_schema_raises_before_request(schema_dir, event_route):
    p = provider(response_format=f"file://{schema_dir / 'templated-format.json'}")
    with pytest.raises(ValueError, match=r"missing-schema\.json"):
        await p.call_api(
            "Extract event info",
            {
                "vars": {"schema_name": "missing-schema"},
                "prompt": {"raw": "Extract event info", "label": "test"},
            },
        )
    assert event_route.call_count == 0


async def test_flat_response_format_file(schema_dir, event_route):
    p = provider(response_format=f"file://{schema_dir / 'flat-format.json'}")
    result = await p.call_api("Test prompt")
    assert "error" not in result
    fmt = sent_json(event_route)["text"]["format"]
    assert (fmt["type"], fmt["name"], fmt["schema"]) == ("json_schema", "flat_schema", FLAT_SCHEMA)


async def test_inline_shorthand_response_format(event_route):
    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "additionalProperties": False,
    }
    p = provider(response_format={"type": "json_schema", "name": "inline_test", "schema": schema})
    result = await p.call_api("Test prompt")
    assert "error" not in result
    fmt = sent_json(event_route)["text"]["format"]
    assert (fmt["name"], fmt["schema"]) == ("inline_test", schema)


async def test_nested_schema_in_inline_chat_style_format(schema_dir, event_route):
    p = provider(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "event_schema",
                "strict": True,
                "schema": f"file://{schema_dir / '{{ schema_name }}.json'}",
            },
        }
    )
    result = await p.call_api(
        "Extract event info",
        {
            "vars": {"schema_name": "event-schema"},
            "prompt": {"raw": "Extract event info", "label": "test"},
        },
    )
    assert "error" not in result
    fmt = sent_json(event_route)["text"]["format"]
    assert fmt["name"] == "event_schema"
    assert fmt["schema"]["required"] == ["event_name", "date", "location"]


_ = fake_az
