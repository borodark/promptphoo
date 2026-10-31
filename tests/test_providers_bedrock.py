"""Bedrock providers. Ports deterministic cases from test/providers/bedrock/*.test.ts.

boto3 is never called over the network: tests swap in a fake bedrock-runtime client, or use a
real client whose `before-send` hook answers locally.
"""

import asyncio
import io
import json

import pytest

from promptphoo import cache as cache_mod
from promptphoo.providers.bedrock import create_provider
from promptphoo.providers.bedrock.base import (
    create_bedrock_cache_key_hash,
    nova_output_from_message,
    nova_parse_messages,
)
from promptphoo.providers.bedrock.converse import (
    AwsBedrockConverseProvider,
    convert_tool_choice_to_converse_format,
    convert_tools_to_converse_format,
    extract_text_from_content_blocks,
    parse_converse_messages,
)
from promptphoo.providers.bedrock.index import (
    BEDROCK_MODEL,
    AwsBedrockCompletionProvider,
    AwsBedrockEmbeddingProvider,
    assert_bedrock_model_is_available,
    format_prompt_llama2_chat,
    format_prompt_llama3_instruct,
    format_prompt_llama4,
    get_handler_for_model,
)
from promptphoo.providers.bedrock.pricing import (
    calculate_bedrock_cost,
    calculate_bedrock_invoke_model_cost,
)
from promptphoo.providers.bedrock.routing import (
    get_bedrock_text_route,
    is_rejected_prefixed_grok_id,
    is_rejected_prefixed_mythos_id,
)
from promptphoo.tokens import create_empty_token_usage

AWS_ENV = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_BEDROCK_REGION",
    "AWS_BEDROCK_MAX_TOKENS",
    "AWS_BEDROCK_TEMPERATURE",
    "AWS_BEDROCK_TOP_P",
    "AWS_BEDROCK_STOP",
    "AWS_BEDROCK_MAX_GEN_LEN",
    "AWS_BEDROCK_MAX_RETRIES",
    "MISTRAL_MAX_TOKENS",
    "MISTRAL_TEMPERATURE",
    "MISTRAL_TOP_P",
    "MISTRAL_TOP_K",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in AWS_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cache_mod, "_instance", cache_mod.Cache(None, 60_000))
    cache_mod.disable_cache()
    yield
    cache_mod.enable_cache()


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, name, kwargs):
        self.calls.append((name, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def converse(self, **kwargs):
        return self._next("converse", kwargs)

    def invoke_model(self, **kwargs):
        return self._next("invoke_model", kwargs)


def invoke_body(payload):
    return {"body": io.BytesIO(json.dumps(payload).encode())}


def converse_response(text, *, reasoning=None, stop_reason="end_turn", latency=None, usage=None):
    content = []
    if reasoning:
        content.append({"reasoningContent": {"reasoningText": {"text": reasoning}}})
    if text is not None:
        content.append({"text": text})
    response = {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": usage or {"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
    }
    if latency:
        response["metrics"] = {"latencyMs": latency}
    return response


def with_client(provider, client):
    provider.bedrock = client
    return provider


def run(coro):
    return asyncio.run(coro)


# --- routing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("id", "api_mode", "model_id"),
    [
        ("bedrock:openai.gpt-5.5", "responses", "openai.gpt-5.5"),
        ("bedrock:converse:openai.gpt-5.5", "responses", "openai.gpt-5.5"),
        ("bedrock:completion:xai.grok-4.3", "responses", "xai.grok-4.3"),
        ("bedrock:responses:openai.gpt-oss-120b", "responses", "openai.gpt-oss-120b"),
        ("bedrock:openai.gpt-oss-120b-1:0", "invoke", "openai.gpt-oss-120b-1:0"),
        ("bedrock:converse:us.openai.gpt-5.6-sol", "converse", "us.openai.gpt-5.6-sol"),
        ("bedrock:converse:us.xai.grok-4.6", "converse", "us.xai.grok-4.6"),
        ("bedrock:completion:amazon.nova-pro-v1:0", "invoke", "amazon.nova-pro-v1:0"),
        ("bedrock:mantle:openai.gpt-oss-120b", "chat", "openai.gpt-oss-120b"),
        ("bedrock:mantle:xai.grok-4.3", "chat", "xai.grok-4.3"),
        (
            "bedrock:messages:us.anthropic.claude-fable-5-1",
            "messages",
            "us.anthropic.claude-fable-5-1",
        ),
        ("bedrock:anthropic.claude-mythos-5", "messages", "anthropic.claude-mythos-5"),
        ("bedrock:anthropic.claude-fable-5", "invoke", "anthropic.claude-fable-5"),
        ("bedrock:converse:anthropic.claude-mythos-5", "converse", "anthropic.claude-mythos-5"),
        ("bedrock:responses:", "responses", ""),
        ("bedrock:", "invoke", ""),
        (
            "bedrock:arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example",
            "invoke",
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example",
        ),
    ],
)
def test_get_bedrock_text_route(id, api_mode, model_id):
    assert get_bedrock_text_route(id) == {"apiMode": api_mode, "modelId": model_id}


@pytest.mark.parametrize(
    "id",
    [
        "openai:responses:gpt-5.5",
        "bedrock:kb:openai.gpt-5.5",
        "bedrock:agents:xai.grok-4.3",
        "bedrock:embeddings:openai.gpt-5.5",
        "bedrock:video:amazon.nova-reel-v1:0",
        "bedrock:toString:openai.gpt-5.5",
        "bedrock:__proto__:openai.gpt-5.5",
    ],
)
def test_get_bedrock_text_route_non_text(id):
    assert get_bedrock_text_route(id) is None


@pytest.mark.parametrize(
    ("id", "explicit", "rejected"),
    [
        ("xai.grok-4.3", False, False),
        ("xai.grok-4.3", True, False),
        ("us.xai.grok-4.6", False, False),
        ("global.xai.grok-4.6", False, False),
        ("us.xai.grok-4.6", True, True),
        ("global.xai.grok-4.6", True, True),
        ("us.xai.grok-4.3", False, True),
        ("eu.xai.grok-4.6", False, True),
        ("custom.future-model", False, False),
    ],
)
def test_rejected_prefixed_grok_id(id, explicit, rejected):
    assert is_rejected_prefixed_grok_id(id, explicit) is rejected


@pytest.mark.parametrize(
    ("id", "rejected"),
    [
        ("anthropic.claude-mythos-5", False),
        ("us.anthropic.claude-mythos-5", True),
        ("global.anthropic.claude-mythos-5", True),
        ("us.anthropic.claude-mythos-5-1", False),
    ],
)
def test_rejected_prefixed_mythos_id(id, rejected):
    assert is_rejected_prefixed_mythos_id(id) is rejected


def test_factory_routes():
    converse = create_provider("bedrock:converse:anthropic.claude-3-5-sonnet-20241022-v2:0", {})
    assert isinstance(converse, AwsBedrockConverseProvider)
    assert converse.id() == "bedrock:converse:anthropic.claude-3-5-sonnet-20241022-v2:0"
    invoke = create_provider("bedrock:amazon.nova-pro-v1:0", {})
    assert isinstance(invoke, AwsBedrockCompletionProvider)
    assert invoke.model_name == "amazon.nova-pro-v1:0"
    assert invoke.id() == "bedrock:amazon.nova-pro-v1:0"
    legacy = create_provider("bedrock:completion:meta.llama3-8b-instruct-v1:0", {})
    assert legacy.model_name == "meta.llama3-8b-instruct-v1:0"
    embed = create_provider("bedrock:embeddings:amazon.titan-embed-text-v2:0", {})
    assert isinstance(embed, AwsBedrockEmbeddingProvider)


def test_factory_rejections():
    with pytest.raises(ValueError, match="does not support geo or global inference IDs"):
        create_provider("bedrock:us.anthropic.claude-mythos-5", {})
    with pytest.raises(ValueError, match="uses the Anthropic Messages API, not Converse"):
        create_provider("bedrock:converse:anthropic.claude-mythos-5", {})
    with pytest.raises(ValueError, match="not a valid Grok mantle id"):
        create_provider("bedrock:us.xai.grok-4.3", {})
    with pytest.raises(ValueError, match="not supported by promptphoo"):
        create_provider("bedrock:kb:some-kb-id", {})
    with pytest.raises(ValueError, match="Unknown Amazon Bedrock model"):
        create_provider("bedrock:converse:anthropic.claude-v2", {})
    with pytest.raises(ValueError, match="supports only InvokeModelWithBidirectionalStream"):
        create_provider("bedrock:converse:amazon.nova-sonic-v1:0", {})


# --- pricing ---------------------------------------------------------------------------------

IN, OUT = 10_000, 5_000


def at_rates(i, o):
    return IN / 1e6 * i + OUT / 1e6 * o


@pytest.mark.parametrize(
    ("model", "i", "o", "cached"),
    [
        ("us.xai.grok-4.6", 2.2, 6.6, 0.55),
        ("global.xai.grok-4.6", 2, 6, 0.5),
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.xai.grok-4.6",
            2.2,
            6.6,
            0.55,
        ),
    ],
)
def test_grok46_pricing(model, i, o, cached):
    expected = (800 * i + 200 * cached + 500 * o) / 1e6
    assert calculate_bedrock_cost(model, 800, 500, 200) == pytest.approx(expected, abs=1e-12)
    assert calculate_bedrock_invoke_model_cost(model, 800, 500, 200) == pytest.approx(expected)


@pytest.mark.parametrize("model", ["xai.grok-4.6", "us.xai.grok-4.6-future", "global.xai.grok-4.5"])
def test_grok_no_inferred_price(model):
    assert calculate_bedrock_invoke_model_cost(model, 800, 500, 200) is None


@pytest.mark.parametrize("tier", ["priority", "flex", "reserved"])
def test_grok_unpublished_tiers(tier):
    assert (
        calculate_bedrock_cost("us.xai.grok-4.6", 800, 500, 200, 0, "us-east-1", {"type": tier})
        is None
    )


def test_bedrock_pricing_cases():
    assert calculate_bedrock_cost("global.xai.grok-4.6", 800, 500, 200, 100) is None
    assert calculate_bedrock_cost(
        "amazon.nova-lite-v1:0", 1000, 0, 10_000, 0, "us-east-1"
    ) == pytest.approx((1000 * 0.06 + 10_000 * 0.015) / 1e6, abs=1e-12)
    assert calculate_bedrock_cost(
        "amazon.nova-pro-v1:0", 1000, 0, 0, 50_000, "us-east-1"
    ) == pytest.approx(calculate_bedrock_cost("amazon.nova-pro-v1:0", 1000, 0, 0, 0, "us-east-1"))
    assert calculate_bedrock_cost(
        "meta.llama3-3-70b-instruct-v1:0", 1000, 0, 9999, 0
    ) == pytest.approx(1000 * 0.72 / 1e6, abs=1e-12)
    assert calculate_bedrock_cost("us-gov.anthropic.claude-opus-4-8", IN, OUT) == pytest.approx(
        at_rates(6, 30)
    )
    assert calculate_bedrock_cost(
        "minimax.minimax-m2", IN, OUT, 0, 0, "us-east-1", {"type": "priority"}
    ) == pytest.approx(at_rates(0.3, 1.2) * 1.75)
    assert calculate_bedrock_cost("zai.glm-4.7", IN, OUT, 0, 0, "eu-west-2") == pytest.approx(
        at_rates(0.93, 3.41)
    )
    assert calculate_bedrock_cost("openai.gpt-oss-120b-1:0", 1e6, 1e6, 0, 0, "ca-central-1") is None
    assert calculate_bedrock_cost("cohere.command-r-plus-v1:0", 1e6, 1e6) == pytest.approx(18)
    assert calculate_bedrock_cost(
        "global.anthropic.claude-sonnet-4-6", 200_001, 1_000
    ) == pytest.approx(200_001 / 1e6 * 3 + 1_000 / 1e6 * 15)


def test_sonnet4_long_context_tier():
    model = "global.anthropic.claude-sonnet-4-20250514-v1:0"
    assert calculate_bedrock_cost(model, 199_999, 1_000) == pytest.approx(
        199_999 / 1e6 * 3 + 1_000 / 1e6 * 15
    )
    assert calculate_bedrock_cost(model, 200_000, 1_000) == pytest.approx(
        200_000 / 1e6 * 6 + 1_000 / 1e6 * 22.5
    )
    assert calculate_bedrock_cost(model, 150_000, 1_000, 60_000, 0) == pytest.approx(
        150_000 / 1e6 * 6 + 60_000 / 1e6 * 6 * 0.1 + 1_000 / 1e6 * 22.5
    )


def test_opus_prefix_ordering():
    assert calculate_bedrock_cost("global.anthropic.claude-opus-5", 1_000_000, 0) == pytest.approx(
        5
    )
    assert calculate_bedrock_cost(
        "global.anthropic.claude-opus-4-5-20251101-v1:0", 1_000_000, 0
    ) == pytest.approx(5)
    assert calculate_bedrock_cost(
        "global.anthropic.claude-opus-4-1-20250805-v1:0", 1_000_000, 0
    ) == pytest.approx(15)


@pytest.mark.parametrize(
    "model",
    [
        "global.anthropic.claude-opus-5-5",
        "us.anthropic.claude-opus-5-5",
        "eu.anthropic.claude-opus-5-5",
        "jp.anthropic.claude-opus-5-5",
        "au.anthropic.claude-opus-5-5",
    ],
)
def test_opus55_cache_pricing(model):
    expected = 0.01454 * (1 if model.startswith("global.") else 1.1)
    assert calculate_bedrock_cost(model, 1000, 500, 200, 100) == pytest.approx(expected, abs=1e-8)
    assert calculate_bedrock_invoke_model_cost(model, 1000, 500, 200, 100) == pytest.approx(
        expected, abs=1e-8
    )


# --- cache key -------------------------------------------------------------------------------


def test_cache_key_hash_matches_node():
    params = {
        "modelId": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "hi"},
                    {"image": {"format": "png", "source": {"bytes": b"\x01\x02\x03"}}},
                ],
            }
        ],
    }
    # Golden values from Node: createBedrockCacheKeyHash in src/providers/bedrock/base.ts.
    assert create_bedrock_cache_key_hash({}, params, "us-east-1") == (
        "402851d39e4bc251199633f376ad775314df8c17bb1fbd927e86070cb256ffb9:"
        "558562016ac9091e9ac8516ee0f9dfcf4b1e5f46bf341c0fc260490cbf41244f"
    )
    config = {"accessKeyId": "AK", "secretAccessKey": "SK", "endpoint": "https://x"}
    assert create_bedrock_cache_key_hash(config, {"a": 1}, "eu-west-1") == (
        "193a8b2e2b737fa388d3f5e94643877d3cac99e8a4e92d4a2034dc8c8e3d3845:"
        "926ba8c3aa006ef97a21878b0b8684ec9b8d4617392a6555db25a196b51e313f"
    )


def test_cache_key_isolated_by_bearer_token(monkeypatch):
    a = create_bedrock_cache_key_hash({"apiKey": "one"}, {"x": 1}, "us-east-1")
    b = create_bedrock_cache_key_hash({"apiKey": "two"}, {"x": 1}, "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "one")
    c = create_bedrock_cache_key_hash({}, {"x": 1}, "us-east-1")
    assert len({a, b, c}) == 3


# --- base helpers ----------------------------------------------------------------------------


def test_region_resolution(monkeypatch):
    p = AwsBedrockConverseProvider("m", {"config": {"region": "eu-west-1"}})
    assert p.get_region() == "eu-west-1"
    assert AwsBedrockConverseProvider("m").get_region() == "us-east-1"
    monkeypatch.setenv("AWS_BEDROCK_REGION", "ap-south-1")
    assert AwsBedrockConverseProvider("m").get_region() == "ap-south-1"
    env_p = AwsBedrockConverseProvider("m", {"env": {"AWS_BEDROCK_REGION": "sa-east-1"}})
    assert env_p.get_region() == "sa-east-1"


def test_credentials(monkeypatch):
    p = AwsBedrockConverseProvider(
        "m", {"config": {"accessKeyId": "AK", "secretAccessKey": "SK", "sessionToken": "T"}}
    )
    assert p.get_credentials() == {
        "aws_access_key_id": "AK",
        "aws_secret_access_key": "SK",
        "aws_session_token": "T",
    }
    assert AwsBedrockConverseProvider("m", {"config": {"apiKey": "k"}}).get_credentials() is None
    assert AwsBedrockConverseProvider("m").get_credentials() is None


def test_explicit_credentials_reach_boto3_client():
    pytest.importorskip("boto3")
    p = AwsBedrockConverseProvider(
        "m", {"config": {"accessKeyId": "AK", "secretAccessKey": "SK", "region": "us-west-2"}}
    )
    client = p.get_bedrock_instance()
    assert client.meta.region_name == "us-west-2"
    creds = client._request_signer._credentials
    assert (creds.access_key, creds.secret_key) == ("AK", "SK")
    assert p.get_bedrock_instance() is client


class _Raw:
    def __init__(self, data):
        self.data = data

    def stream(self, **_):
        yield self.data


def test_bearer_token_auth_sent_on_wire(monkeypatch):
    pytest.importorskip("boto3")
    from botocore.awsrequest import AWSResponse

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    p = AwsBedrockConverseProvider("amazon.nova-micro-v1:0", {"config": {"region": "us-east-1"}})
    client = p.get_bedrock_instance()
    seen = {}

    def capture(request, **_):
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = request.url
        seen["body"] = json.loads(request.body)
        body = json.dumps(converse_response("hi")).encode()
        return AWSResponse(request.url, 200, {"content-type": "application/json"}, _Raw(body))

    client.meta.events.register("before-send.bedrock-runtime.Converse", capture)
    result = run(p.call_api("Hello"))
    assert result["output"] == "hi"
    auth = seen["auth"]
    assert (auth.decode() if isinstance(auth, bytes) else auth) == "Bearer bedrock-key"
    assert seen["url"] == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-micro-v1%3A0/converse"
    )
    assert seen["body"] == {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}


def test_nova_helpers():
    assert nova_parse_messages("system: Be nice\nuser: hi\nassistant: yo\nmore") == {
        "system": [{"text": "Be nice"}],
        "extractedMessages": [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "yo\nmore"}]},
        ],
    }
    response = {
        "output": {
            "message": {
                "content": [
                    {"text": "thinking"},
                    {"toolUse": {"toolUseId": "t1", "name": "f", "input": {"a": 1}}},
                ]
            }
        }
    }
    assert nova_output_from_message(response) == '{"toolUseId":"t1","name":"f","input":{"a":1}}'
    assert nova_output_from_message({"output": {"message": {"content": [{"text": "a"}]}}}) == "a"


# --- Converse --------------------------------------------------------------------------------


def test_converse_basic_call():
    client = FakeClient([converse_response("Hello, world!")])
    p = with_client(
        AwsBedrockConverseProvider(
            "anthropic.claude-3-5-sonnet-20241022-v2:0", {"config": {"region": "us-east-1"}}
        ),
        client,
    )
    result = run(p.call_api("Say hello"))
    assert result["output"] == "Hello, world!"
    assert result["tokenUsage"] == {"prompt": 100, "completion": 50, "total": 150, "numRequests": 1}
    assert result["cost"] == pytest.approx(0.00105, abs=1e-6)
    assert result["metadata"] == {"stopReason": "end_turn"}
    assert client.calls[0] == (
        "converse",
        {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [{"role": "user", "content": [{"text": "Say hello"}]}],
        },
    )


def test_converse_thinking_output():
    def provider(show):
        return AwsBedrockConverseProvider(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            {
                "config": {
                    "thinking": {"type": "enabled", "budget_tokens": 16000},
                    "showThinking": show,
                }
            },
        )

    shown = with_client(
        provider(True),
        FakeClient(
            [converse_response("The answer is 42.", reasoning="Let me think about this...")]
        ),
    )
    out = run(shown.call_api("q"))["output"]
    assert out == "<thinking>\nLet me think about this...\n</thinking>\n\nThe answer is 42."
    assert shown.bedrock.calls[0][1]["additionalModelRequestFields"] == {
        "thinking": {"type": "enabled", "budget_tokens": 16000}
    }
    hidden = with_client(
        provider(False), FakeClient([converse_response("The answer is 42.", reasoning="x")])
    )
    assert run(hidden.call_api("q"))["output"] == "The answer is 42."


def test_converse_empty_thinking_block_omitted():
    response = converse_response("QA OK")
    response["output"]["message"]["content"].insert(
        0, {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig-abc123"}}}
    )
    p = with_client(
        AwsBedrockConverseProvider(
            "global.anthropic.claude-fable-5", {"config": {"showThinking": True}}
        ),
        FakeClient([response]),
    )
    assert run(p.call_api("Test"))["output"] == "QA OK"


def test_converse_tool_use_and_tool_config():
    tool_use = {
        "toolUse": {"toolUseId": "t1", "name": "calculator", "input": {"expression": "2+2"}}
    }
    response = converse_response(None, stop_reason="tool_use")
    response["output"]["message"]["content"] = [tool_use]
    client = FakeClient([response])
    tools = [
        {
            "name": "calculator",
            "description": "Do math",
            "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
        }
    ]
    p = with_client(
        AwsBedrockConverseProvider(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            {"config": {"tools": tools, "toolChoice": "any"}},
        ),
        client,
    )
    result = run(p.call_api("What is 2+2?"))
    assert json.loads(result["output"]) == {
        "type": "tool_use",
        "id": "t1",
        "name": "calculator",
        "input": {"expression": "2+2"},
    }
    assert client.calls[0][1]["toolConfig"] == {
        "tools": [
            {
                "toolSpec": {
                    "name": "calculator",
                    "description": "Do math",
                    "inputSchema": {"json": tools[0]["input_schema"]},
                }
            }
        ],
        "toolChoice": {"any": {}},
    }


def test_converse_function_tool_callbacks():
    response = converse_response(None)
    response["output"]["message"]["content"] = [
        {"toolUse": {"toolUseId": "t1", "name": "add", "input": {"a": 1, "b": 2}}}
    ]

    def add(args):
        data = json.loads(args)
        return {"sum": data["a"] + data["b"]}

    p = with_client(
        AwsBedrockConverseProvider(
            "amazon.nova-pro-v1:0",
            {
                "config": {
                    "tools": [{"toolSpec": {"name": "add"}}],
                    "functionToolCallbacks": {"add": add},
                }
            },
        ),
        FakeClient([response]),
    )
    assert run(p.call_api("add"))["output"] == '{"sum":3}'


def test_converse_tool_choice_none_skips_tools():
    p = with_client(
        AwsBedrockConverseProvider(
            "amazon.nova-pro-v1:0",
            {"config": {"tools": [{"toolSpec": {"name": "x"}}], "tool_choice": "none"}},
        ),
        FakeClient([converse_response("ok")]),
    )
    run(p.call_api("hi"))
    assert "toolConfig" not in p.bedrock.calls[0][1]


def test_converse_stop_reasons_and_guardrails():
    def call(stop_reason, config=None):
        p = with_client(
            AwsBedrockConverseProvider(
                "anthropic.claude-3-5-sonnet-20241022-v2:0", {"config": config or {}}
            ),
            FakeClient(
                [converse_response("Partial output", stop_reason=stop_reason, latency=2500)]
            ),
        )
        return run(p.call_api("Test"))

    guarded = call("guardrail_intervened", {"guardrailIdentifier": "g", "guardrailVersion": 1})
    assert guarded["guardrails"] == {"flagged": True, "reason": "guardrail_intervened"}
    malformed = call("malformed_model_output")
    assert malformed["error"] == (
        "Model produced invalid output. The response could not be parsed correctly."
    )
    assert malformed["output"] == "Partial output"
    assert malformed["metadata"]["isModelError"] is True
    assert malformed["metadata"]["latencyMs"] == 2500
    bad_tool = call("malformed_tool_use")
    assert bad_tool["error"] == (
        "Model produced a malformed tool use request. Check tool configuration and input schema."
    )


def test_converse_inference_config(monkeypatch):
    p = AwsBedrockConverseProvider(
        "amazon.nova-pro-v1:0",
        {"config": {"max_tokens": 0, "temperature": 0, "top_p": 0.5, "stop": ["END"]}},
    )
    monkeypatch.setenv("AWS_BEDROCK_MAX_TOKENS", "500")
    assert p.build_inference_config() == {
        "maxTokens": 0,
        "temperature": 0,
        "topP": 0.5,
        "stopSequences": ["END"],
    }
    monkeypatch.setenv("AWS_BEDROCK_TEMPERATURE", "0.3")
    monkeypatch.setenv("AWS_BEDROCK_STOP", '["S"]')
    env_p = AwsBedrockConverseProvider("amazon.nova-pro-v1:0")
    assert env_p.build_inference_config() == {
        "maxTokens": 500,
        "temperature": 0.3,
        "stopSequences": ["S"],
    }


def test_converse_claude_sampling_rules():
    opus47 = AwsBedrockConverseProvider(
        "us.anthropic.claude-opus-4-7", {"config": {"temperature": 0.5, "topP": 0.9}}
    )
    assert opus47.build_inference_config() is None
    both = AwsBedrockConverseProvider(
        "anthropic.claude-3-5-sonnet-20241022-v2:0", {"config": {"temperature": 0.5, "topP": 0.9}}
    )
    assert both.build_inference_config() == {"topP": 0.9}


def test_converse_cache_round_trip():
    cache_mod.enable_cache()
    client = FakeClient([converse_response("cached!")])
    p = with_client(AwsBedrockConverseProvider("amazon.nova-pro-v1:0"), client)
    first = run(p.call_api("hi"))
    second = run(p.call_api("hi"))
    assert "cached" not in first
    assert second == {**first, "cached": True}
    assert len(client.calls) == 1


def test_converse_errors():
    ClientError = pytest.importorskip("botocore.exceptions").ClientError

    def call(code):
        err = ClientError({"Error": {"Code": code, "Message": "bad"}}, "Converse")
        p = with_client(AwsBedrockConverseProvider("m"), FakeClient([err]))
        return run(p.call_api("x"))["error"]

    assert call("ValidationException").startswith(
        "Bedrock Converse API validation error: An error occurred (ValidationException)"
    )
    assert call("AccessDeniedException").endswith(
        "Ensure you have bedrock:InvokeModel permission and model access is enabled."
    )
    assert call("ThrottlingException").startswith("Bedrock Converse API error: ")


def test_parse_converse_messages():
    assert parse_converse_messages("Hello") == {
        "messages": [{"role": "user", "content": [{"text": "Hello"}]}]
    }
    prompt = json.dumps(
        [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )
    assert parse_converse_messages(prompt) == {
        "system": [{"text": "Be brief"}],
        "messages": [
            {"role": "user", "content": [{"text": "Hi"}]},
            {"role": "assistant", "content": [{"text": "Hello"}]},
        ],
    }
    assert parse_converse_messages("System: sys\nUser: q\nline2\nAssistant: a") == {
        "system": [{"text": "sys"}],
        "messages": [
            {"role": "user", "content": [{"text": "q\nline2"}]},
            {"role": "assistant", "content": [{"text": "a"}]},
        ],
    }
    image = json.dumps(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpg;base64,AQID"}}
                ],
            }
        ]
    )
    assert parse_converse_messages(image)["messages"][0]["content"] == [
        {"image": {"format": "jpeg", "source": {"bytes": b"\x01\x02\x03"}}}
    ]
    tool = json.dumps(
        [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t", "content": "42"}],
            }
        ]
    )
    assert parse_converse_messages(tool)["messages"][0]["content"] == [
        {"toolResult": {"toolUseId": "t", "content": [{"text": "42"}]}}
    ]


def test_tool_conversions():
    openai_tools = [
        {
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
        }
    ]
    assert convert_tools_to_converse_format(openai_tools) == [
        {"toolSpec": {"name": "f", "description": "d", "inputSchema": {"json": {"type": "object"}}}}
    ]
    assert convert_tool_choice_to_converse_format("auto") == {"auto": {}}
    assert convert_tool_choice_to_converse_format("required") == {"any": {}}
    assert convert_tool_choice_to_converse_format("any") == {"any": {}}
    assert convert_tool_choice_to_converse_format({"tool": {"name": "f"}}) == {
        "tool": {"name": "f"}
    }
    assert convert_tool_choice_to_converse_format(
        {"type": "function", "function": {"name": "f"}}
    ) == {"tool": {"name": "f"}}
    with pytest.raises(ValueError, match="Invalid tool configuration"):
        convert_tools_to_converse_format([{"description": "x"}])


def test_extract_text_redacted():
    blocks = [{"reasoningContent": {"redactedContent": b"x"}}, {"text": "done"}]
    assert extract_text_from_content_blocks(blocks) == "<thinking>[Redacted]</thinking>\n\ndone"


# --- InvokeModel -----------------------------------------------------------------------------


def test_handler_lookup():
    assert (
        get_handler_for_model("anthropic.claude-3-haiku-20240307-v1:0")
        is BEDROCK_MODEL["CLAUDE_MESSAGES"]
    )
    assert get_handler_for_model("us.amazon.nova-2-lite-v1:0") is BEDROCK_MODEL["AMAZON_NOVA_2"]
    assert get_handler_for_model("eu.meta.llama3-1-70b-instruct-v1:0") is BEDROCK_MODEL["LLAMA3_1"]
    assert get_handler_for_model("mistral.mistral-large-2407-v1:0").name == "MISTRAL_CHAT"
    with pytest.raises(ValueError, match="requires inferenceModelType"):
        get_handler_for_model("arn:aws:bedrock:us-east-1:1:inference-profile/x")
    arn = get_handler_for_model(
        "arn:aws:bedrock:us-east-1:1:inference-profile/x", {"inferenceModelType": "nova"}
    )
    assert arn is BEDROCK_MODEL["AMAZON_NOVA"]
    with pytest.raises(ValueError, match="Unknown Amazon Bedrock model: unknown.model"):
        get_handler_for_model("unknown.model")
    with pytest.raises(ValueError, match='Load it as "bedrock:anthropic.claude-mythos-5"'):
        get_handler_for_model("us.anthropic.claude-mythos-5")
    with pytest.raises(ValueError, match="Unknown Amazon Bedrock model"):
        assert_bedrock_model_is_available("us.anthropic.claude-3-opus-20240229-v1:0")


def test_invoke_claude_messages():
    body = {
        "content": [{"type": "text", "text": "Hi there"}],
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "stop_reason": "end_turn",
    }
    client = FakeClient([invoke_body(body)])
    p = with_client(
        AwsBedrockCompletionProvider(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            {"config": {"guardrailIdentifier": 123, "guardrailVersion": 1}},
        ),
        client,
    )
    result = run(p.call_api("Hello"))
    assert result["output"] == "Hi there"
    assert result["tokenUsage"]["prompt"] == 12
    assert result["tokenUsage"]["completion"] == 3
    assert result["tokenUsage"]["numRequests"] == 1
    _, kwargs = client.calls[0]
    assert kwargs["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert kwargs["guardrailIdentifier"] == "123"
    assert kwargs["guardrailVersion"] == "1"
    assert json.loads(kwargs["body"]) == {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
    }


def test_invoke_claude_thinking_budget():
    client = FakeClient([invoke_body({"content": [{"type": "text", "text": "ok"}]})])
    p = with_client(
        AwsBedrockCompletionProvider(
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            {
                "config": {
                    "thinking": {"type": "enabled", "budget_tokens": 2048},
                    "temperature": 0.7,
                }
            },
        ),
        client,
    )
    run(p.call_api("Hello"))
    sent = json.loads(client.calls[0][1]["body"])
    assert sent["max_tokens"] == 3072
    assert "temperature" not in sent
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_invoke_nova_and_cache():
    cache_mod.enable_cache()
    body = {
        "output": {"message": {"content": [{"text": "Nova says hi"}]}},
        "usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9},
    }
    client = FakeClient([invoke_body(body)])
    p = with_client(AwsBedrockCompletionProvider("amazon.nova-micro-v1:0"), client)
    result = run(p.call_api("system: sys\nuser: hi"))
    assert result == {
        "output": "Nova says hi",
        "tokenUsage": {"prompt": 5, "completion": 4, "total": 9, "numRequests": 1},
    }
    sent = json.loads(client.calls[0][1]["body"])
    # Upstream wraps the parsed system blocks a second time.
    assert sent == {
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "system": [{"text": [{"text": "sys"}]}],
        "inferenceConfig": {"temperature": 0},
    }
    cached = run(p.call_api("system: sys\nuser: hi"))
    assert cached["cached"] is True
    assert cached["output"] == "Nova says hi"
    assert cached["tokenUsage"] == create_empty_token_usage()
    assert len(client.calls) == 1


def test_invoke_llama_and_mistral_and_titan():
    client = FakeClient(
        [
            invoke_body(
                {"generation": "llama out", "prompt_token_count": 7, "generation_token_count": 2}
            ),
            invoke_body({"outputs": [{"text": "mistral out"}]}),
            invoke_body({"results": [{"outputText": "titan out"}]}),
        ]
    )
    llama = with_client(AwsBedrockCompletionProvider("meta.llama3-8b-instruct-v1:0"), client)
    result = run(llama.call_api("Hello"))
    assert result["output"] == "llama out"
    assert result["tokenUsage"] == {"prompt": 7, "completion": 2, "total": 9, "numRequests": 1}
    assert json.loads(client.calls[0][1]["body"]) == {
        "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>",
        "temperature": 0,
        "top_p": 1,
        "max_gen_len": 1024,
    }
    mistral = with_client(AwsBedrockCompletionProvider("mistral.mistral-7b-instruct-v0:2"), client)
    assert run(mistral.call_api("Hi"))["output"] == "mistral out"
    assert json.loads(client.calls[1][1]["body"]) == {
        "prompt": "Hi",
        "stop": [],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
    }
    titan = with_client(AwsBedrockCompletionProvider("amazon.titan-text-express-v1"), client)
    with pytest.raises(ValueError, match="Unknown Amazon Bedrock model"):
        run(titan.call_api("Hi"))


def test_invoke_errors():
    ClientError = pytest.importorskip("botocore.exceptions").ClientError

    err = ClientError({"Error": {"Code": "ValidationException", "Message": "nope"}}, "InvokeModel")
    p = with_client(AwsBedrockCompletionProvider("amazon.nova-micro-v1:0"), FakeClient([err]))
    assert run(p.call_api("x")) == {
        "error": "Bedrock API invoke model error: ValidationException: nope"
    }


def test_embeddings():
    client = FakeClient([invoke_body({"embedding": [0.1, 0.2]}), invoke_body({"embedding": []})])
    p = with_client(AwsBedrockEmbeddingProvider("amazon.titan-embed-text-v2:0"), client)
    assert run(p.call_embedding_api("hi")) == {"embedding": [0.1, 0.2]}
    assert json.loads(client.calls[0][1]["body"]) == {"inputText": "hi"}
    assert run(p.call_embedding_api("hi"))["error"].startswith(
        "API response error: Error: No valid embedding found in AWS Bedrock API response"
    )


def test_llama_formatting():
    assert format_prompt_llama3_instruct(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
    ) == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    assert format_prompt_llama4(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    ) == (
        "<|begin_of_text|><|header_start|>user<|header_end|>\n\nHello<|eot|>"
        "<|header_start|>assistant<|header_end|>\n\nHi<|eot|>"
        "<|header_start|>assistant<|header_end|>"
    )
    assert (
        format_prompt_llama2_chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ]
        )
        == "<s>[INST] <<SYS>>\nsys\n<</SYS>>\n\nHi [/INST] Hello </s><s>[INST] Bye [/INST]"
    )
    with pytest.raises(ValueError, match="To:     bedrock:converse:m"):
        format_prompt_llama3_instruct(
            [{"role": "user", "content": [{"type": "image", "source": {"data": "x"}}]}], "m"
        )
