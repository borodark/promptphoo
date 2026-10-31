"""OpenAI-compatible families: litellm, docker, novita, alibaba, localai, llama, meta,
abliteration and GitHub Models. Ports deterministic cases from test/providers/{litellm,docker,
novita,alibaba,localai,llama,meta,abliteration}.test.ts and test/providers/github/, with HTTP
mocked by respx."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx
from openai_helpers import chat_response, ok, openai_env, sent_json

from promptphoo import fetch
from promptphoo.providers.compatible import (
    abliteration,
    alibaba,
    docker,
    ghmodels,
    litellm,
    llama,
    localai,
    meta,
    novita,
)

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

VENDOR_ENV = (
    "LITELLM_API_KEY", "LITELLM_API_BASE", "DOCKER_MODEL_RUNNER_BASE_URL",
    "DOCKER_MODEL_RUNNER_API_KEY", "NOVITA_API_KEY", "DASHSCOPE_API_KEY", "LOCALAI_BASE_URL",
    "LOCALAI_TEMPERATURE", "LLAMA_BASE_URL", "MODEL_API_KEY", "ABLIT_KEY", "ABLIT_API_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_MAX_TOKENS", "ANTHROPIC_TEMPERATURE",
)  # fmt: skip


@pytest.fixture(autouse=True)
def _vendor_env(monkeypatch: pytest.MonkeyPatch):
    for name in VENDOR_ENV:
        monkeypatch.delenv(name, raising=False)


def _litellm(path: str, **options):
    return litellm.create_litellm_provider(path, options)


# --- litellm ------------------------------------------------------------------------------


def test_litellm_routes_and_identity():
    chat = _litellm("litellm:gpt-4")
    assert isinstance(chat, litellm.LiteLLMProvider)
    assert chat.id() == "litellm:gpt-4"
    assert str(chat) == "[LiteLLM Provider gpt-4]"
    assert chat.to_json()["type"] == "chat"
    assert _litellm("litellm:chat:gpt-4").id() == "litellm:gpt-4"
    completion = _litellm("litellm:completion:gpt-3.5-turbo-instruct")
    assert completion.id() == "litellm:completion:gpt-3.5-turbo-instruct"
    assert str(completion) == "[LiteLLM Provider completion gpt-3.5-turbo-instruct]"
    embedding = _litellm("litellm:embeddings:text-embedding-3-small")
    assert embedding.id() == "litellm:embedding:text-embedding-3-small"
    assert str(embedding) == "[LiteLLM Provider embedding text-embedding-3-small]"
    assert embedding.to_json() == {
        "provider": "litellm",
        "model": "text-embedding-3-small",
        "type": "embedding",
        "config": embedding.config,
    }
    assert hasattr(embedding, "call_embedding_api")
    assert not hasattr(chat, "call_embedding_api")
    assert _litellm("litellm:custom:model:v1").id() == "litellm:custom:model:v1"


def test_litellm_config_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch):
    provider = _litellm("litellm:gpt-4")
    assert provider.config["apiBaseUrl"] == "http://0.0.0.0:4000"
    assert provider.config["apiKeyRequired"] is False
    assert provider.config["apiKeyEnvar"] == "LITELLM_API_KEY"
    assert provider.config["omitDefaults"] is True
    custom = _litellm(
        "litellm:gpt-4",
        config={"config": {"temperature": 0, "max_tokens": 0, "omitDefaults": False}},
    )
    assert custom.config["temperature"] == 0
    assert custom.config["max_tokens"] == 0
    assert custom.config["omitDefaults"] is False
    # null values do not override the defaults; an empty string does.
    assert (
        _litellm("litellm:gpt-4", config={"config": {"apiBaseUrl": None}}).config["apiBaseUrl"]
        == "http://0.0.0.0:4000"
    )
    assert (
        _litellm("litellm:gpt-4", config={"config": {"apiBaseUrl": ""}}).config["apiBaseUrl"] == ""
    )
    monkeypatch.setenv("LITELLM_API_BASE", "http://env-litellm.example.com")
    assert _litellm("litellm:gpt-4").config["apiBaseUrl"] == "http://env-litellm.example.com"
    scoped = _litellm(
        "litellm:gpt-4", env={"LITELLM_API_BASE": "http://provider-env-wins.example.com"}
    )
    assert scoped.config["apiBaseUrl"] == "http://provider-env-wins.example.com"
    configured = _litellm(
        "litellm:gpt-4",
        env={"LITELLM_API_BASE": "http://nope"},
        config={"config": {"apiBaseUrl": "https://config-wins.com"}},
    )
    assert configured.config["apiBaseUrl"] == "https://config-wins.com"
    monkeypatch.setenv("LITELLM_API_KEY", "test-litellm-key")
    assert _litellm("litellm:gpt-4").get_api_key() == "test-litellm-key"


@respx.mock
async def test_litellm_request_body_omits_defaults_and_openai_key():
    route = respx.post("http://0.0.0.0:4000/chat/completions").mock(
        return_value=ok(chat_response("ok"))
    )
    result = await _litellm("litellm:chat:test-model").call_api("test")
    assert result["output"] == "ok"
    body = sent_json(route)
    assert "temperature" not in body
    assert "max_tokens" not in body
    assert "authorization" not in route.calls[0].request.headers
    await _litellm("litellm:chat:test-model", config={"config": {"temperature": 0.7}}).call_api("t")
    assert sent_json(route)["temperature"] == 0.7
    await _litellm("litellm:chat:test-model", config={"config": {"omitDefaults": False}}).call_api(
        "x"
    )
    assert sent_json(route)["temperature"] == 0
    assert sent_json(route)["max_tokens"] == 1024


UNAUTHORIZED = {
    "error": {
        "message": "Authentication Error, No api key passed in.",
        "type": "auth_error",
        "code": "401",
    }
}


@pytest.mark.parametrize(
    "kind,path",
    [("chat", "chat/completions"), ("completion", "completions"), ("embedding", "embeddings")],
)
@respx.mock
async def test_litellm_explains_keyless_auth_failure(kind: str, path: str):
    respx.post(f"http://0.0.0.0:4000/{path}").mock(return_value=ok(UNAUTHORIZED, status=401))
    provider = _litellm(f"litellm:{kind}:test-model")
    result = (
        await provider.call_embedding_api("test")
        if kind == "embedding"
        else await provider.call_api("test")
    )
    assert "Authentication Error" in result["error"]
    assert "Set LITELLM_API_KEY" in result["error"]


@respx.mock
async def test_litellm_does_not_alter_unrelated_or_credentialed_errors():
    route = respx.post("http://0.0.0.0:4000/chat/completions")
    route.mock(return_value=ok({"error": {"message": "Unavailable"}}, status=503))
    unavailable = await _litellm("litellm:chat:test-model").call_api("test")
    assert "503" in unavailable["error"]
    assert "Set LITELLM_API_KEY" not in unavailable["error"]
    route.mock(return_value=ok(UNAUTHORIZED, status=401))
    keyed = _litellm("litellm:chat:test-model", env={"LITELLM_API_KEY": "scoped-key"})
    assert keyed.get_api_key() == "scoped-key"
    assert "Set LITELLM_API_KEY" not in (await keyed.call_api("test"))["error"]
    header_auth = _litellm(
        "litellm:chat:test-model",
        config={"config": {"headers": {"Authorization": "Bearer custom-key"}}},
    )
    assert "Set LITELLM_API_KEY" not in (await header_auth.call_api("test"))["error"]


def test_litellm_create_provider_uses_family_shape():
    provider = litellm.create_provider(
        "litellm:gpt-4", {"id": "custom-id", "config": {"temperature": 0.2}, "env": None}
    )
    assert provider.id() == "custom-id"
    assert provider.config["temperature"] == 0.2


# --- docker -------------------------------------------------------------------------------

DMR_MODELS = "http://localhost:12434/engines/v1/models"


def test_docker_parse_provider_path():
    assert docker.parse_provider_path("docker:chat:ai/smollm3") == {
        "type": "chat",
        "model": "ai/smollm3",
    }
    assert docker.parse_provider_path("docker:completion:ai/m")["type"] == "completion"
    assert docker.parse_provider_path("docker:embedding:ai/m") == {
        "type": "embeddings",
        "model": "ai/m",
    }
    assert docker.parse_provider_path("docker:ai/smollm3") == {
        "type": "chat",
        "model": "ai/smollm3",
    }
    assert docker.parse_provider_path("docker:hf.co/org/model:Q4_K_M")["model"] == (
        "hf.co/org/model:Q4_K_M"
    )


def test_docker_factory_types_and_env():
    chat = docker.create_provider("docker:chat:ai/m", {})
    assert isinstance(chat, docker.DMRChatCompletionProvider)
    assert chat.config["apiBaseUrl"] == "http://localhost:12434/engines/v1"
    assert chat.config["apiKey"] == "dmr"
    assert chat.get_gen_ai_system() == "docker"
    assert isinstance(
        docker.create_provider("docker:completion:ai/m", {}), docker.DMRCompletionProvider
    )
    assert isinstance(
        docker.create_provider("docker:embeddings:ai/m", {}), docker.DMREmbeddingProvider
    )
    assert isinstance(docker.create_provider("docker:ai/m", {}), docker.DMRChatCompletionProvider)
    custom = docker.create_provider(
        "docker:ai/m",
        {
            "env": {
                "DOCKER_MODEL_RUNNER_BASE_URL": "http://dmr:1",
                "DOCKER_MODEL_RUNNER_API_KEY": "k",
            }
        },
    )
    assert custom.config["apiBaseUrl"] == "http://dmr:1/engines/v1"
    assert custom.config["apiKey"] == "k"


@respx.mock
async def test_docker_has_local_model():
    respx.get(DMR_MODELS).mock(return_value=ok({"data": [{"id": "AI/SmolLM3"}]}))
    base = "http://localhost:12434/engines/v1"
    assert await docker.has_local_model("ai/smollm3", base) is True
    assert await docker.has_local_model("ai/other", base) is False


@respx.mock
async def test_docker_connection_error_is_helpful():
    respx.get(DMR_MODELS).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(RuntimeError, match="Failed to connect to Docker Model Runner"):
        await docker.fetch_local_models("http://localhost:12434/engines/v1")


@respx.mock
async def test_docker_warns_when_model_missing_but_continues(caplog: pytest.LogCaptureFixture):
    respx.get(DMR_MODELS).mock(return_value=ok({"data": []}))
    route = respx.post("http://localhost:12434/engines/v1/chat/completions").mock(
        return_value=ok(chat_response("hi"))
    )
    with caplog.at_level(logging.WARNING):
        result = await docker.create_provider("docker:ai/m", {}).call_api("hello")
    assert result["output"] == "hi"
    assert route.calls[0].request.headers["authorization"] == "Bearer dmr"
    assert "Run 'docker model pull ai/m'" in caplog.text


@respx.mock
async def test_docker_embedding_checks_model():
    respx.get(DMR_MODELS).mock(return_value=ok({"data": [{"id": "ai/embed"}]}))
    respx.post("http://localhost:12434/engines/v1/embeddings").mock(
        return_value=ok({"data": [{"embedding": [0.1, 0.2]}], "usage": {"total_tokens": 2}})
    )
    result = await docker.create_provider("docker:embedding:ai/embed", {}).call_embedding_api("x")
    assert result["embedding"] == [0.1, 0.2]


# --- novita -------------------------------------------------------------------------------


def test_novita_routes_and_errors():
    create = novita.create_novita_provider
    assert create("novita:chat:model-name").id() == "novita:chat:model-name"
    assert create("novita:completion:model-name").id() == "novita:completion:model-name"
    assert create("novita:embedding:model-name").id() == "novita:embedding:model-name"
    assert create("novita:model-name").id() == "novita:chat:model-name"
    assert isinstance(create("novita:embeddings:m"), novita.NovitaEmbeddingProvider)
    for path in ("novita:chat:", "novita:"):
        with pytest.raises(ValueError, match="Novita model name is required"):
            create(path)
    for sub in ("image", "moderation", "embeddin"):
        with pytest.raises(ValueError, match=f'Unknown Novita provider sub-type "{sub}"'):
            create(f"novita:{sub}:foo")


def test_novita_preserves_explicit_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUSTOM_NOVITA_KEY", "proxy-key")
    provider = novita.create_novita_provider(
        "novita:chat:model-name",
        {
            "config": {
                "id": "configured-id",
                "config": {
                    "apiBaseUrl": "http://localhost:15501/openai/v1",
                    "apiKeyEnvar": "CUSTOM_NOVITA_KEY",
                    "temperature": 0.4,
                },
            }
        },
    )
    assert provider.id() == "configured-id"
    assert provider.config["apiBaseUrl"] == "http://localhost:15501/openai/v1"
    assert provider.config["temperature"] == 0.4
    assert provider.get_api_key() == "proxy-key"


def test_novita_defaults_and_openai_isolation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://wrong.example.test/v1")
    chat = novita.NovitaChatCompletionProvider("deepseek/deepseek-v3.2")
    completion = novita.NovitaCompletionProvider("deepseek/deepseek-v3.2")
    embedding = novita.NovitaEmbeddingProvider("embedding-model")
    assert chat.config["apiKeyEnvar"] == "NOVITA_API_KEY"
    assert chat.config["temperature"] == 1
    assert completion.config["temperature"] == 1
    assert "temperature" not in embedding.config
    for provider in (chat, completion, embedding):
        assert provider.get_api_key() is None
        assert provider.get_organization() is None
        assert provider.get_api_url() == "https://api.novita.ai/openai/v1"


async def test_novita_fails_before_request_without_key():
    with pytest.raises(ValueError, match="Novita API key is not set"):
        await novita.NovitaChatCompletionProvider("chat-model").call_api("hello")


@respx.mock
async def test_novita_chat_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NOVITA_API_KEY", "novita-key")
    route = respx.post("https://api.novita.ai/openai/v1/chat/completions").mock(
        return_value=ok(chat_response("hello"))
    )
    result = await novita.create_provider("novita:meta/llama", {}).call_api("hi")
    assert result["output"] == "hello"
    assert route.calls[0].request.headers["authorization"] == "Bearer novita-key"
    assert sent_json(route)["temperature"] == 1


# --- alibaba ------------------------------------------------------------------------------


def test_alibaba_providers(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    provider = alibaba.create_provider("alibaba:qwen-max", {})
    assert isinstance(provider, alibaba.AlibabaChatCompletionProvider)
    assert provider.model_name == "qwen-max"
    assert provider.config["apiBaseUrl"] == alibaba.API_BASE_URL
    assert provider.config["apiKeyEnvar"] == "DASHSCOPE_API_KEY"
    assert provider.get_gen_ai_system() == "alibaba"
    custom_id = alibaba.create_provider("dashscope:chat:qwen-plus", {"id": "custom"})
    assert custom_id.model_name == "qwen-plus"
    assert custom_id.get_gen_ai_system() == "alibaba"
    embedding = alibaba.create_provider("alicloud:embedding:text-embedding-v4", {})
    assert isinstance(embedding, alibaba.AlibabaEmbeddingProvider)
    assert "qwen3.7-text-embedding" in alibaba.KNOWN_MODELS
    with pytest.raises(ValueError, match="Alibaba modelName is required"):
        alibaba.AlibabaChatCompletionProvider("")
    with pytest.raises(ValueError, match="Alibaba modelName is required"):
        alibaba.AlibabaEmbeddingProvider("")
    with caplog.at_level(logging.WARNING):
        alibaba.AlibabaChatCompletionProvider("unknown-model")
    assert "Unknown Alibaba Cloud model: unknown-model" in caplog.text
    custom = alibaba.AlibabaChatCompletionProvider(
        "qwen-max", {"config": {"apiBaseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1"}}
    )
    assert custom.config["apiBaseUrl"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    assert alibaba.AlibabaChatCompletionProvider("qwen-max").get_api_key() == "dash-key"
    scoped = alibaba.AlibabaEmbeddingProvider(
        "text-embedding-v3", {"env": {"DASHSCOPE_API_KEY": "scoped"}}
    )
    assert scoped.get_api_key() == "scoped"


# --- localai ------------------------------------------------------------------------------

LOCALAI = "http://localhost:8080/v1"


@respx.mock
async def test_localai_temperature_resolution(monkeypatch: pytest.MonkeyPatch):
    chat = respx.post(f"{LOCALAI}/chat/completions").mock(return_value=ok(chat_response("x")))
    completion = respx.post(f"{LOCALAI}/completions").mock(
        return_value=ok({"choices": [{"text": "done"}]})
    )
    result = await localai.create_provider(
        "localai:chat:m", {"config": {"temperature": 0}}
    ).call_api("p")
    assert result == {"output": "x"}
    assert sent_json(chat) == {
        "model": "m",
        "messages": [{"role": "user", "content": "p"}],
        "temperature": 0,
    }
    result = await localai.create_provider(
        "localai:completion:m", {"config": {"temperature": 0}}
    ).call_api("p")
    assert result == {"output": "done"}
    assert sent_json(completion)["temperature"] == 0
    await localai.LocalAiChatProvider("m", {"env": {"LOCALAI_TEMPERATURE": "0.3"}}).call_api("a")
    assert sent_json(chat)["temperature"] == 0.3
    await localai.LocalAiChatProvider("m", {"env": {"LOCALAI_TEMPERATURE": "0"}}).call_api("b")
    assert sent_json(chat)["temperature"] == 0
    await localai.LocalAiChatProvider(
        "m", {"config": {"temperature": 0.9}, "env": {"LOCALAI_TEMPERATURE": "0.3"}}
    ).call_api("c")
    assert sent_json(chat)["temperature"] == 0.9
    await localai.LocalAiChatProvider("m").call_api("d")
    assert sent_json(chat)["temperature"] == 0.7


@respx.mock
async def test_localai_malformed_responses():
    respx.post(f"{LOCALAI}/chat/completions").mock(return_value=ok({"choices": []}))
    respx.post(f"{LOCALAI}/completions").mock(return_value=httpx.Response(200, text="null"))
    chat = await localai.LocalAiChatProvider("m").call_api("p")
    assert chat == {"error": 'Malformed response data: {"choices":[]}'}
    completion = await localai.LocalAiCompletionProvider("m").call_api("p")
    assert completion == {"error": "Malformed response data: null"}


@respx.mock
async def test_localai_embedding_and_identity():
    respx.post(f"{LOCALAI}/embeddings").mock(return_value=ok({"data": [{"embedding": [1, 2]}]}))
    provider = localai.create_provider("localai:embeddings:e", {})
    assert provider.id() == "localai:e"
    assert str(provider) == "[LocalAI Provider e]"
    assert await provider.call_embedding_api("x") == {"embedding": [1, 2]}
    assert localai.create_provider("localai:some-model", {"id": "mine"}).id() == "mine"


# --- llama --------------------------------------------------------------------------------


def test_llama_identity():
    provider = llama.create_provider("llama:test-model", {"config": {"temperature": 0.7}})
    assert provider.model_name == "test-model"
    assert provider.config == {"temperature": 0.7}
    assert provider.id() == "llama:test-model"
    assert str(provider) == "[Llama Provider test-model]"
    assert llama.LlamaProvider("m", {"id": "test-id"}).id() == "test-id"
    assert llama.create_provider("llama", {}).id() == "llama:"


@respx.mock
async def test_llama_request_and_errors(monkeypatch: pytest.MonkeyPatch):
    route = respx.post("http://localhost:8080/completion").mock(return_value=ok({"content": "hi"}))
    provider = llama.LlamaProvider(
        "m", {"config": {"temperature": 0.7, "mirostat": 2, "logit_bias": [[15043, -1]]}}
    )
    result = await provider.call_api("Test prompt")
    assert result["output"] == "hi"
    assert result["cached"] is False
    assert sent_json(route) == {
        "prompt": "Test prompt",
        "n_predict": 512,
        "temperature": 0.7,
        "mirostat": 2,
        "logit_bias": [[15043, -1]],
    }
    await llama.LlamaProvider("m", {"config": {"n_predict": 0}}).call_api("x")
    assert sent_json(route)["n_predict"] == 0
    monkeypatch.setenv("LLAMA_BASE_URL", "http://llama.internal:9")
    respx.post("http://llama.internal:9/completion").mock(side_effect=httpx.ConnectError("boom"))
    with fetch.with_fetch_retry_context(0):
        failed = await llama.LlamaProvider("m", {"config": {}}).call_api("x")
    assert failed["error"].startswith("API call error:")


# --- abliteration -------------------------------------------------------------------------


def test_abliteration_defaults_and_routing(monkeypatch: pytest.MonkeyPatch):
    provider = abliteration.AbliterationProvider("abliterated-model")
    assert provider.id() == "abliteration:abliterated-model"
    assert str(provider) == "[Abliteration Provider abliterated-model]"
    assert provider.config["apiBaseUrl"] == "https://api.abliteration.ai/v1"
    assert provider.config["apiKeyEnvar"] == "ABLIT_KEY"
    assert provider.config["showThinking"] is False
    create = abliteration.create_abliteration_provider
    assert create("abliteration:chat:m").model_name == "m"
    assert create("abliteration:a:b").model_name == "a:b"
    for path in ("abliteration:chat:", "abliteration:"):
        with pytest.raises(ValueError, match="requires a model name"):
            create(path)
    monkeypatch.setenv("ABLIT_API_BASE_URL", "https://env.example/v1")
    assert create("abliteration:m").config["apiBaseUrl"] == "https://env.example/v1"
    scoped = abliteration.create_provider(
        "abliteration:m", {"env": {"ABLIT_API_BASE_URL": "https://scoped.example/v1"}}
    )
    assert scoped.config["apiBaseUrl"] == "https://scoped.example/v1"
    configured = abliteration.AbliterationProvider(
        "m", {"config": {"apiBaseUrl": "  "}, "env": {"ABLIT_API_BASE_URL": " "}}
    )
    assert configured.config["apiBaseUrl"] == "https://api.abliteration.ai/v1"


@pytest.mark.parametrize("model", ["abliterated-model", "abliterated-model-large-v2"])
async def test_abliteration_forwards_reasoning_effort(model: str):
    provider = abliteration.AbliterationProvider(
        model, {"config": {"reasoning_effort": "high", "max_tokens": 16384, "temperature": 0.2}}
    )
    body = (await provider.get_openai_body("Test prompt"))["body"]
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 16384
    assert body["temperature"] == 0.2
    assert "max_completion_tokens" not in body
    rendered = abliteration.AbliterationProvider(model, {"config": {"reasoning_effort": "max"}})
    body = (
        await rendered.get_openai_body(
            "p",
            {"prompt": {"raw": "p", "config": {"reasoning_effort": "{{effort}}"}},
             "vars": {"effort": "low"}},
        )
    )["body"]  # fmt: skip
    assert body["reasoning_effort"] == "low"
    assert rendered.config["reasoning_effort"] == "max"
    unset = abliteration.AbliterationProvider(model, {"config": {"reasoning_effort": None}})
    assert "reasoning_effort" not in (await unset.get_openai_body("p"))["body"]


def test_abliteration_keys_and_json(monkeypatch: pytest.MonkeyPatch):
    provider = abliteration.AbliterationProvider("m", {"config": {"apiKey": ""}})
    assert "apiKey" not in provider.to_json()["config"]
    assert provider.get_api_key() is None  # never OPENAI_API_KEY
    monkeypatch.setenv("ABLIT_KEY", "process-key")
    assert abliteration.AbliterationProvider("m").get_api_key() == "process-key"
    scoped = abliteration.AbliterationProvider("m", {"env": {"ABLIT_KEY": "scoped-key"}})
    assert scoped.get_api_key() == "scoped-key"
    assert (
        abliteration.AbliterationProvider("m", {"config": {"apiKey": "cfg"}}).get_api_key() == "cfg"
    )


@respx.mock
async def test_abliteration_call_hides_reasoning(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ABLIT_KEY", "k")
    route = respx.post("https://api.abliteration.ai/v1/chat/completions").mock(
        return_value=ok(chat_response({"content": "answer", "reasoning": "thoughts"}))
    )
    result = await abliteration.AbliterationProvider("m").call_api("q")
    assert result["output"] == "answer"
    assert route.calls[0].request.headers["authorization"] == "Bearer k"
    shown = await abliteration.AbliterationProvider(
        "m", {"config": {"showThinking": True}}
    ).call_api("q2")
    assert shown["output"] == "Thinking: thoughts\n\nanswer"


async def test_abliteration_requires_key():
    with pytest.raises(ValueError, match="ABLIT_KEY"):
        await abliteration.AbliterationProvider("m").call_api("q")


# --- github models ------------------------------------------------------------------------


async def test_github_models_retired():
    with pytest.raises(ValueError, match="GitHub Models was retired on July 30, 2026"):
        ghmodels.create_provider("github:openai/gpt-5", {})
    retired = ghmodels.RetiredGitHubProvider("openai/gpt-5", {"config": ghmodels.GITHUB_CONFIG})
    assert await retired.call_api("x") == {"error": ghmodels.GITHUB_MODELS_RETIREMENT_MESSAGE}


# --- meta ---------------------------------------------------------------------------------

META_CHAT = "https://api.meta.ai/v1/chat/completions"


def _meta(path: str = "meta:chat:muse-spark-1.1", **config):
    return meta.create_meta_provider(path, {"config": config})


def test_meta_routing():
    assert isinstance(_meta(), meta.MetaProvider)
    assert _meta().id() == "meta:chat:muse-spark-1.1"
    messages = _meta("meta:messages:muse-spark-1.1")
    assert isinstance(messages, meta.MetaMessagesProvider)
    assert messages.id() == "meta:messages:muse-spark-1.1"
    for sub in ("embedding", "image", "completion"):
        with pytest.raises(ValueError, match=f"does not expose an {sub} endpoint"):
            _meta(f"meta:{sub}:muse-spark-1.1")
    for sub in ("agents", "chatkit", "codex-sdk", "voice"):
        with pytest.raises(ValueError, match=f"Unknown Meta Model API sub-type '{sub}'"):
            _meta(f"meta:{sub}:x")
    # The Responses route needs the OpenAI Responses provider.
    try:
        import promptphoo.providers.openai.responses  # noqa: F401
    except ImportError:
        for path in ("meta:muse-spark-1.1", "meta:responses:muse-spark-1.1", "meta:"):
            with pytest.raises(ValueError, match="OpenAI Responses provider"):
                _meta(path)


def test_meta_configuration(monkeypatch: pytest.MonkeyPatch):
    provider = _meta()
    assert provider.config["apiBaseUrl"] == "https://api.meta.ai/v1"
    assert provider.config["apiKeyEnvar"] == "MODEL_API_KEY"
    assert provider.get_api_url() == "https://api.meta.ai/v1"
    assert _meta(apiBaseUrl="").get_api_url() == "https://api.meta.ai/v1"
    assert _meta(apiBaseUrl="https://proxy.example/v1/").get_api_url() == "https://proxy.example/v1"
    assert provider.get_gen_ai_system() == "meta"
    assert str(provider) == "[Meta Model API Provider muse-spark-1.1]"
    assert _meta(apiKey="secret").to_json() == {
        "provider": "meta:chat",
        "model": "muse-spark-1.1",
        "config": {"apiKeyEnvar": "MODEL_API_KEY", "apiBaseUrl": "https://api.meta.ai/v1"},
    }
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org")
    assert provider.get_api_key() is None  # OPENAI_API_KEY is set by the fixture
    assert provider.get_organization() is None
    monkeypatch.setenv("MODEL_API_KEY", "meta-key")
    assert _meta(apiKey="").get_api_key() == "meta-key"
    scoped = meta.create_meta_provider("meta:chat:m", {"env": {"MODEL_API_KEY": "scoped"}})
    assert scoped.get_api_key() == "scoped"
    monkeypatch.setenv("CUSTOM_KEY", "custom")
    assert _meta(apiKeyEnvar="CUSTOM_KEY").get_api_key() == "custom"


async def _meta_body(**config):
    return (await _meta(**config).get_openai_body("Hello"))["body"]


async def test_meta_request_body_shaping(monkeypatch: pytest.MonkeyPatch):
    body = await _meta_body()
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    assert body["temperature"] == 0
    assert (await _meta_body(reasoning_effort="xhigh"))["reasoning_effort"] == "xhigh"
    rendered = await _meta(reasoning_effort="{{effort}}").get_openai_body(
        "Hello", {"vars": {"effort": "high"}}
    )
    assert rendered["body"]["reasoning_effort"] == "high"
    for cap in (0, None, 512):
        body = await _meta_body(
            max_completion_tokens=4096,
            passthrough={"max_completion_tokens": cap, "max_tokens": 2048},
        )
        assert "max_completion_tokens" in body
        assert body["max_completion_tokens"] == cap
    body = await _meta_body(max_tokens=2048)
    assert body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body
    body = await _meta_body(max_completion_tokens=4096, passthrough={"max_tokens": 2048})
    assert body["max_completion_tokens"] == 2048
    body = await _meta_body(passthrough={"max_completion_tokens": 5000, "top_p": 0.9,
                                         "temperature": 1.5})  # fmt: skip
    assert (body["max_completion_tokens"], body["top_p"], body["temperature"]) == (5000, 0.9, 1.5)
    for name, value in (("OPENAI_TEMPERATURE", "0.9"), ("OPENAI_TOP_P", "0.5"),
                        ("OPENAI_PRESENCE_PENALTY", "0.7"), ("OPENAI_FREQUENCY_PENALTY", "0.9"),
                        ("OPENAI_MAX_COMPLETION_TOKENS", "256")):  # fmt: skip
        monkeypatch.setenv(name, value)
    body = await _meta_body()
    assert body["temperature"] == 0
    for key in ("top_p", "presence_penalty", "frequency_penalty", "max_completion_tokens"):
        assert key not in body


@pytest.mark.parametrize(
    "config,error",
    [
        ({"reasoning_effort": "none"}, "reasoning_effort 'none'"),
        ({"stop": ["\n"]}, "stop"),
        ({"passthrough": {"logprobs": True}}, "logprobs"),
        ({"passthrough": {"logit_bias": {"50256": -100}}}, "logit_bias"),
        ({"passthrough": {"n": 2}}, "n > 1"),
        ({"passthrough": {"stream": True}}, "streaming is not supported"),
        ({"stream": True}, "streaming is not supported"),
        ({"logit_bias": {"50256": -100}}, "logit_bias"),
    ],
)
async def test_meta_rejects_unsupported_options(config: dict, error: str):
    with pytest.raises(ValueError, match=error.replace(">", ">")):
        await _meta(**config).get_openai_body("Hello")


PRICING_CASES = [
    ("muse-spark-1.1", 0.002935),
    ("muse-spark-1.3", 0.002935),
    ("muse-spark-1.3-contributor", 0.0001608),
]


def test_calculate_meta_cost():
    calc = meta.calculate_meta_cost
    for model in ("muse-spark-1.1", "muse-spark-1.3"):
        assert calc(model, {}, 1000, 500) == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)
    for model, expected in PRICING_CASES:
        assert calc(model, {}, 1000, 500, 400) == pytest.approx(expected, abs=1e-12)
    assert calc("muse-unknown", {}, 1000, 500) is None
    assert calc("muse-spark-1.1", {}, None, 500) is None
    assert calc("muse-spark-1.1", {}, 1000, None) is None
    overrides = {"inputCost": 2 / 1e6, "outputCost": 8 / 1e6, "cacheReadCost": 1 / 1e6}
    assert calc("muse-spark-1.1", overrides, 1000, 500, 400) == pytest.approx(
        (600 * 2 + 400 * 1 + 500 * 8) / 1e6
    )
    assert calc("muse-unknown", {"cost": 0.000002}, 1000, 500) == pytest.approx(0.003)
    assert calc("muse-spark-1.1", {"cost": 2 / 1e6}, 1000, 500, 400) == pytest.approx(
        (1000 * 2 + 500 * 2) / 1e6
    )
    assert calc(
        "muse-spark-1.1", {"inputCost": 2 / 1e6, "outputCost": 8 / 1e6}, 1000, 500, 400
    ) == pytest.approx((1000 * 2 + 500 * 8) / 1e6)


META_USAGE = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "usage": {
        "total_tokens": 1500,
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 400},
    },
}


@pytest.mark.parametrize("model,expected", PRICING_CASES)
@respx.mock
async def test_meta_chat_cost(model: str, expected: float):
    respx.post(META_CHAT).mock(return_value=ok(META_USAGE))
    result = await meta.create_meta_provider(
        f"meta:chat:{model}", {"config": {"apiKey": "LLM|1|k"}}
    ).call_api("Say hi")
    assert result["cost"] == pytest.approx(expected, abs=1e-12)


@respx.mock
async def test_meta_chat_cost_edge_cases():
    route = respx.post(META_CHAT).mock(return_value=ok(META_USAGE))
    options = {"config": {"apiKey": "LLM|1|k"}}
    unknown = await meta.create_meta_provider("meta:chat:muse-future", options).call_api("a")
    assert "cost" not in unknown
    prompt_level = await meta.create_meta_provider("meta:chat:muse-spark-1.1", options).call_api(
        "b", {"prompt": {"raw": "b", "label": "t", "config": {"cost": 0}}, "vars": {}}
    )
    assert prompt_level["cost"] == 0
    assert route.calls[0].request.headers["authorization"] == "Bearer LLM|1|k"
    assert "openai-organization" not in route.calls[0].request.headers
    route.mock(return_value=ok({"error": {"message": "Too many requests"}}, status=429))
    no_retry = {"config": {**options["config"], "maxRetries": 0}}
    error = await meta.create_meta_provider("meta:chat:muse-spark-1.1", no_retry).call_api("c")
    assert error["error"]
    assert "cost" not in error


async def test_meta_chat_requires_meta_key():
    with pytest.raises(ValueError, match="Meta Model API key is not set. Set the MODEL_API_KEY"):
        await _meta().call_api("x")


def test_meta_messages_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    provider = _meta("meta:messages:muse-spark-1.1")
    assert provider.get_api_key() is None
    assert provider.get_api_base_url() == "https://api.meta.ai"
    assert provider.config["showThinking"] is False
    assert provider.config["max_tokens"] == 131_072
    assert provider.config["temperature"] == 0
    assert provider.config["stream"] is True
    explicit = _meta(
        "meta:messages:muse-spark-1.1",
        max_tokens=8192, temperature=0.3, stream=False, showThinking=True,
    )  # fmt: skip
    assert (explicit.config["max_tokens"], explicit.config["temperature"]) == (8192, 0.3)
    assert explicit.config["stream"] is False
    assert explicit.config["showThinking"] is True
    assert str(provider) == "[Meta Model API Messages Provider muse-spark-1.1]"


@respx.mock
async def test_meta_messages_bearer_auth_and_cost(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_API_KEY", "LLM|1|messages-key")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Gateway-Secret: anthropic-only")
    route = respx.post("https://api.meta.ai/v1/messages").mock(
        return_value=ok(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "muse-spark-1.1",
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            }
        )
    )
    result = await _meta("meta:messages:muse-spark-1.1").call_api("Hi")
    assert result["output"] == "Hello"
    headers = route.calls[0].request.headers
    assert headers["authorization"] == "Bearer LLM|1|messages-key"
    assert "x-api-key" not in headers
    assert "x-gateway-secret" not in headers
    assert result["cost"] == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)
    assert json.loads(route.calls[0].request.content)["model"] == "muse-spark-1.1"


async def test_meta_messages_requires_meta_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    with pytest.raises(ValueError, match="Meta Model API key is not set"):
        await _meta("meta:messages:muse-spark-1.1").call_api("x")
