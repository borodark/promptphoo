"""Ports test/matchers/llm-rubric.test.ts, test/assertions/llmRubric.test.ts and
test/assertions/llmRubric.audio.test.ts.

Not ported (see D25, upstream's remote grading only runs for redteam configs):
- should not call remote when rubric prompt is overridden, even if redteam is enabled
- should not call remote when a grading provider is configured, even if redteam is enabled
- should call remote when a caller prefers remote despite an injected default provider
- should include Cloud target context in remote grading requests
- should prefer filtered providers when building remote grading context
- should call remote with image outputs when multimodal grading is remote-eligible
- should replace image data URI text output for remote multimodal grading
- should use local multimodal grading when a local grading provider is configured
- should call remote when redteam is enabled and rubric prompt is not overridden ...
- should tag remote-grading transport failures with metadata.graderError
- should use local provider when redteam.provider is configured ...
Also not ported: the OpenAI Live provider audio cases (no Live provider in promptphoo). The
`.js` rubric prompt file case asserts promptphoo's not-supported error instead.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import re
from typing import Any

import pytest
import respx
from openai_helpers import CHAT_URL, ok, openai_env

from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.modelgraded import handle_llm_rubric
from promptphoo.jsvalues import json_stringify
from promptphoo.matchers import llm_grading
from promptphoo.matchers.llm_grading import matches_llm_rubric
from promptphoo.matchers.rubric import render_llm_rubric_prompt
from promptphoo.matchers.shared import try_parse
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.responses import OpenAiResponsesProvider
from promptphoo.tokens import accumulate_assertion_token_usage, create_empty_assertions

MULTIMODAL = (
    "The evaluated output includes the attached image(s). Treat the attached image(s) as primary "
    "evidence in <Output>. Inspect the visual content directly, and do not infer visual traits, "
    "demographics, safety issues, or rubric failures from the user prompt or from any "
    "base64/data URI text."
)
PLACEHOLDER = "[Image output attached. Inspect the attached image directly for visual grading.]"
EMPTY_DETAILS = {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0}
TOKENS_10 = {
    "total": 10,
    "prompt": 5,
    "completion": 5,
    "cached": 0,
    "completionDetails": EMPTY_DETAILS,
    "numRequests": 0,
}
USAGE_10 = {"total": 10, "prompt": 5, "completion": 5}
PASS_JSON = json.dumps({"pass": True, "score": 1, "reason": "image ok"})


class FakeGrader(ApiProvider):
    """createMockProvider: records calls and returns a fixed response."""

    def __init__(self, response: dict[str, Any] | None = None, id: str = "test-provider") -> None:  # noqa: A002
        super().__init__({})
        self._fake_id = id
        self.response = (
            response
            if response is not None
            else {
                "output": "Test output",
                "tokenUsage": {**USAGE_10, "cached": 0, "numRequests": 1},
            }
        )
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def id(self) -> str:
        return self._fake_id

    async def call_api(self, prompt, context=None, options=None):
        self.calls.append((prompt, context))
        return copy.deepcopy(self.response)


def grader(output: Any, **extra: Any) -> FakeGrader:
    return FakeGrader({"output": output, **extra})


def json_grader(payload: dict[str, Any], **extra: Any) -> FakeGrader:
    return grader(json.dumps(payload), **extra)


def subset(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """toMatchObject / objectContaining (one level, nested dicts recurse)."""
    for key, value in expected.items():
        assert key in actual, key
        if isinstance(value, dict) and isinstance(actual[key], dict):
            subset(actual[key], value)
        else:
            assert actual[key] == value, (key, actual[key], value)


def image_context(data: str, mime_type: str) -> dict[str, Any]:
    return {
        "providerResponse": {
            "output": "Generated image",
            "images": [{"data": data, "mimeType": mime_type}],
        }
    }


async def grade_image(provider: FakeGrader, context: dict[str, Any], rubric_prompt: Any = None):
    return await matches_llm_rubric(
        "Does the image match?",
        "Generated image",
        {"rubricPrompt": rubric_prompt or "Grade this output: {{ output }}", "provider": provider},
        {},
        None,
        context,
    )


def sent(provider: FakeGrader, index: int = 0) -> Any:
    return json.loads(provider.calls[index][0])


# --- matchesLlmRubric -------------------------------------------------------------------------


async def test_passes_when_grader_returns_passing_result():
    provider = json_grader({"pass": True, "reason": "Test grading output"}, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    subset(result, {"pass": True, "reason": "Test grading output", "score": 1})
    assert result["tokensUsed"] == TOKENS_10


async def test_preserves_component_only_usage_through_aggregation():
    provider = json_grader(
        {"pass": True, "reason": "Test grading output"}, tokenUsage={"prompt": 7, "completion": 3}
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    usage = create_empty_assertions()
    accumulate_assertion_token_usage(usage, result["tokensUsed"])
    subset(result["tokensUsed"], {"total": 10, "prompt": 7, "completion": 3})
    subset(usage, {"total": 10, "prompt": 7, "completion": 3})


async def test_reserved_output_and_rubric_vars_win_over_user_vars():
    provider = json_grader({"pass": True, "reason": "Test grading output"}, tokenUsage=USAGE_10)
    await matches_llm_rubric(
        "rubric from assertion",
        "output from provider",
        {
            "rubricPrompt": "output={{ output }}\nrubric={{ rubric }}\nextra={{ extra }}",
            "provider": provider,
        },
        {
            "output": "vars output sentinel",
            "rubric": "vars rubric sentinel",
            "extra": "kept user var",
        },
    )
    prompt, context = provider.calls[0]
    assert (
        prompt == "output=output from provider\nrubric=rubric from assertion\nextra=kept user var"
    )
    assert context is not None
    subset(
        context["vars"],
        {
            "output": "output from provider",
            "rubric": "rubric from assertion",
            "extra": "kept user var",
        },
    )


async def test_direct_object_output():
    provider = grader(
        {"pass": True, "score": 0.85, "reason": "Direct object output"}, tokenUsage=USAGE_10
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    subset(result, {"pass": True, "score": 0.85, "reason": "Direct object output"})
    assert result["tokensUsed"] == TOKENS_10


async def test_preserves_provider_completion_details():
    details = {"reasoning": 3, "acceptedPrediction": 2, "rejectedPrediction": 1}
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "ok"},
        tokenUsage={**USAGE_10, "completionDetails": details},
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result["tokensUsed"]["completionDetails"] == details


async def test_merges_provider_metadata():
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "ok"},
        metadata={"uploadId": "upload-123", "trace": {"id": "trace-456"}},
        tokenUsage=USAGE_10,
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result["metadata"] == {
        "uploadId": "upload-123",
        "trace": {"id": "trace-456"},
        "renderedGradingPrompt": "Grading prompt",
    }


async def test_preserves_cache_provenance_with_historical_usage():
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "cached grading result"},
        cached=True,
        tokenUsage={"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    subset(result["metadata"], {"cachedResponse": True, "renderedGradingPrompt": "Grading prompt"})
    assert result["tokensUsed"]["total"] == 37


async def test_does_not_trust_cache_provenance_in_fresh_metadata():
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "Fresh grading result"},
        cached=False,
        metadata={"cachedResponse": True, "uploadId": "upload-123"},
        tokenUsage={"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result["metadata"] == {
        "uploadId": "upload-123",
        "renderedGradingPrompt": "Grading prompt",
    }
    subset(result["tokensUsed"], {"total": 37, "numRequests": 1})


async def test_derives_cached_provenance_from_response():
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "Cached grading result"},
        cached=True,
        metadata={"cachedResponse": False, "uploadId": "upload-123"},
        tokenUsage={"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result["metadata"] == {
        "uploadId": "upload-123",
        "renderedGradingPrompt": "Grading prompt",
        "cachedResponse": True,
    }


async def test_rendered_grading_prompt_cannot_be_spoofed_by_metadata():
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "ok"},
        metadata={"renderedGradingPrompt": "spoofed prompt", "uploadId": "upload-123"},
        tokenUsage=USAGE_10,
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result["metadata"] == {
        "uploadId": "upload-123",
        "renderedGradingPrompt": "Grading prompt",
    }


async def test_ignores_array_metadata():
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "ok"}, metadata=["ignored"], tokenUsage=USAGE_10
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result["metadata"] == {"renderedGradingPrompt": "Grading prompt"}


async def test_sanitizes_circular_metadata():
    metadata: dict[str, Any] = {"uploadId": "upload-123", "trace": {"id": "trace-456"}}
    metadata["self"] = metadata

    class CircularGrader(FakeGrader):
        async def call_api(self, prompt, context=None, options=None):
            self.calls.append((prompt, context))
            return {
                "output": json.dumps({"pass": True, "score": 1, "reason": "ok"}),
                "metadata": metadata,
                "tokenUsage": USAGE_10,
            }

    result = await matches_llm_rubric(
        "Expected output",
        "Sample output",
        {"rubricPrompt": "Grading prompt", "provider": CircularGrader()},
    )
    assert result["metadata"] == {
        "uploadId": "upload-123",
        "trace": {"id": "trace-456"},
        "renderedGradingPrompt": "Grading prompt",
    }


# --- multimodal (images) ----------------------------------------------------------------------


async def test_attaches_image_outputs():
    provider = grader(PASS_JSON)
    result = await grade_image(provider, image_context("data:image/png;base64,abc123", "image/png"))
    prompt, context = provider.calls[0]
    assert json.loads(prompt) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Grade this output: Generated image"},
                {"type": "text", "text": MULTIMODAL},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ],
        }
    ]
    assert context is not None
    subset(context["prompt"], {"label": "llm-rubric", "raw": prompt})
    assert result["metadata"] == {
        "renderedGradingPrompt": "Grade this output: Generated image",
        "renderedGradingPromptImages": 1,
    }


async def test_replaces_image_data_uri_text_output():
    provider = grader(PASS_JSON)
    image = "data:image/png;base64,abc123"
    result = await matches_llm_rubric(
        "Does the image match?",
        image,
        {"rubricPrompt": "Grade this output: {{ output }}", "provider": provider},
        {},
        None,
        {
            "providerResponse": {
                "output": image,
                "images": [{"data": image, "mimeType": "image/png"}],
            }
        },
    )
    assert sent(provider)[0]["content"][0] == {
        "type": "text",
        "text": f"Grade this output: {PLACEHOLDER}",
    }
    assert result["metadata"] == {
        "renderedGradingPrompt": f"Grade this output: {PLACEHOLDER}",
        "renderedGradingPromptImages": 1,
    }


async def test_preserves_json_chat_rubric_prompts_with_images():
    provider = grader(PASS_JSON)
    rubric_prompt = json.dumps(
        [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Grade this output: {{ output }}"},
        ]
    )
    await grade_image(provider, image_context("abc123", "image/webp"), rubric_prompt)
    assert sent(provider) == [
        {"role": "system", "content": "Return JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Grade this output: Generated image"},
                {"type": "text", "text": MULTIMODAL},
                {"type": "image_url", "image_url": {"url": "data:image/webp;base64,abc123"}},
            ],
        },
    ]


async def test_preserves_yaml_chat_rubric_prompts_with_images():
    provider = grader(PASS_JSON)
    rubric_prompt = "\n".join(
        [
            "- role: system",
            "  content: Return JSON.",
            "- role: user",
            '  content: "Grade this output: {{ output }}"',
        ]
    )
    await grade_image(provider, image_context("abc123", "image/webp"), rubric_prompt)
    assert sent(provider) == [
        {"role": "system", "content": "Return JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Grade this output: Generated image"},
                {"type": "text", "text": MULTIMODAL},
                {"type": "image_url", "image_url": {"url": "data:image/webp;base64,abc123"}},
            ],
        },
    ]


ANTHROPIC_IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
}


@pytest.mark.parametrize(
    "provider_id",
    [
        "anthropic:messages:claude-3-5-sonnet-latest",
        "cloudflare-ai-gateway:anthropic:messages:claude-3-5-sonnet-latest",
        "cloudflare-gateway:anthropic:claude-sonnet-4-20250514",
        "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock:converse:anthropic.claude-3-5-sonnet-20241022-v2:0",
        "vertex:claude-3-5-sonnet-v2@20241022",
    ],
)
async def test_anthropic_image_parts(provider_id: str):
    provider = FakeGrader({"output": PASS_JSON}, id=provider_id)
    await grade_image(provider, image_context("data:image/png;base64,abc123", "image/png"))
    assert ANTHROPIC_IMAGE in sent(provider)[0]["content"]


@pytest.mark.parametrize("provider_id", ["google:gemini-2.5-pro", "vertex:gemini-2.5-pro"])
async def test_google_inline_data_image_parts(provider_id: str):
    provider = FakeGrader({"output": PASS_JSON}, id=provider_id)
    await grade_image(provider, image_context("data:image/png;base64,abc123", "image/png"))
    assert sent(provider) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Grade this output: Generated image"},
                {"type": "text", "text": MULTIMODAL},
                {"inlineData": {"mimeType": "image/png", "data": "abc123"}},
            ],
        }
    ]


async def test_gemini_converts_existing_chat_content_parts():
    provider = FakeGrader({"output": PASS_JSON}, id="google:gemini-2.5-pro")
    rubric_prompt = json.dumps(
        [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Grade this output: {{ output }}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,existing"}},
                ],
            },
        ]
    )
    await grade_image(provider, image_context("abc123", "image/png"), rubric_prompt)
    assert sent(provider) == [
        {"role": "system", "content": "Return JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Grade this output: Generated image"},
                {"inlineData": {"mimeType": "image/png", "data": "existing"}},
                {"type": "text", "text": MULTIMODAL},
                {"inlineData": {"mimeType": "image/png", "data": "abc123"}},
            ],
        },
    ]


async def test_gemini_normalizes_supported_existing_image_shapes():
    provider = FakeGrader({"output": PASS_JSON}, id="google:gemini-2.5-pro")
    rubric_prompt = json.dumps(
        [
            {
                "role": "user",
                "content": [
                    "Literal chat content",
                    {"type": "text", "text": "Grade this output: {{ output }}"},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,input123"},
                    {"type": "input_image", "image_url": "not-a-data-uri"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/webp",
                            "data": "anthro123",
                        },
                    },
                    {"inlineData": {"mimeType": "image/gif", "data": "inline123"}},
                    {"inline_data": {"mime_type": "image/png", "data": "snake123"}},
                    {"unexpected": "value"},
                    None,
                ],
            }
        ]
    )
    await grade_image(provider, image_context("abc123", "image/png"), rubric_prompt)
    assert sent(provider) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Literal chat content"},
                {"type": "text", "text": "Grade this output: Generated image"},
                {"inlineData": {"mimeType": "image/jpeg", "data": "input123"}},
                {"type": "text", "text": '{"type":"input_image","image_url":"not-a-data-uri"}'},
                {"inlineData": {"mimeType": "image/webp", "data": "anthro123"}},
                {"inlineData": {"mimeType": "image/gif", "data": "inline123"}},
                {"inlineData": {"mimeType": "image/png", "data": "snake123"}},
                {"type": "text", "text": '{"unexpected":"value"}'},
                {"type": "text", "text": "null"},
                {"type": "text", "text": MULTIMODAL},
                {"inlineData": {"mimeType": "image/png", "data": "abc123"}},
            ],
        }
    ]


async def test_responses_image_parts():
    provider = FakeGrader({"output": PASS_JSON}, id="openai:responses:gpt-5.4")
    await grade_image(provider, image_context("data:image/png;base64,abc123", "image/png"))
    assert sent(provider) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Grade this output: Generated image"},
                {"type": "input_text", "text": MULTIMODAL},
                {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
            ],
        }
    ]


@pytest.mark.parametrize(
    ("provider_id", "class_name"),
    [
        ("azure:my-deployment", "AzureResponsesProvider"),
        ("bedrock:openai.gpt-5.5", "BedrockOpenAiResponsesProvider"),
        ("openai:gpt-5.5", "OpenAiResponsesProvider"),
    ],
)
async def test_responses_image_parts_by_provider_class(provider_id: str, class_name: str):
    cls = type(class_name, (FakeGrader,), {})
    provider = cls({"output": PASS_JSON}, id=provider_id)
    await grade_image(provider, image_context("data:image/png;base64,abc123", "image/png"))
    assert {"type": "input_image", "image_url": "data:image/png;base64,abc123"} in sent(provider)[
        0
    ]["content"]


async def test_bedrock_completion_does_not_use_responses_parts():
    cls = type("AwsBedrockCompletionProvider", (FakeGrader,), {})
    provider = cls({"output": PASS_JSON}, id="bedrock:completion:openai.gpt-oss-120b-1:0")
    await grade_image(provider, image_context("data:image/png;base64,abc123", "image/png"))
    assert {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    } in sent(provider)[0]["content"]


async def test_responses_converts_existing_chat_content_parts():
    provider = FakeGrader({"output": PASS_JSON}, id="openai:responses:gpt-5.4")
    rubric_prompt = json.dumps(
        [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Grade this output: {{ output }}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,existing"}},
                ],
            },
        ]
    )
    await grade_image(provider, image_context("abc123", "image/png"), rubric_prompt)
    assert sent(provider) == [
        {"role": "system", "content": "Return JSON."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Grade this output: Generated image"},
                {"type": "input_image", "image_url": "data:image/png;base64,existing"},
                {"type": "input_text", "text": MULTIMODAL},
                {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
            ],
        },
    ]


async def test_appends_user_message_when_chat_rubric_has_none():
    provider = grader(PASS_JSON)
    rubric_prompt = json.dumps([{"role": "system", "content": "Return JSON."}])
    await grade_image(
        provider, image_context("data:image/png;base64,abc123", "image/png"), rubric_prompt
    )
    assert sent(provider) == [
        {"role": "system", "content": "Return JSON."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": MULTIMODAL},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ],
        },
    ]


async def test_prompt_unchanged_when_images_cannot_be_materialized():
    provider = json_grader({"pass": True, "score": 1, "reason": "text only ok"})
    result = await matches_llm_rubric(
        "Does the output match?",
        "Generated output",
        {"rubricPrompt": "Grade this output: {{ output }}", "provider": provider},
        {},
        None,
        {"providerResponse": {"output": "Generated output", "images": [{}]}},
    )
    assert provider.calls[0][0] == "Grade this output: Generated output"
    assert result["metadata"] == {"renderedGradingPrompt": "Grade this output: Generated output"}


B64_5 = base64.b64encode(bytes(5)).decode()


@pytest.mark.parametrize(
    ("images", "env", "message"),
    [
        (
            [{"data": "https://example.com/image.png", "mimeType": "image/png"}],
            {},
            "Remote image URLs are not supported for multimodal grading. Provide local image "
            "output as a data URI or raw base64 string instead.",
        ),
        (
            [
                {
                    "blobRef": {
                        "uri": f"promptfoo://blob/{'a' * 64}",
                        "hash": "a" * 64,
                        "mimeType": "image/png",
                        "sizeBytes": 11,
                        "provider": "filesystem",
                    }
                }
            ],
            {},
            "Blob-backed image outputs are not supported for multimodal grading yet. Configure the "
            "image provider to return base64 or data URI image output.",
        ),
        (
            [{"data": "   ", "mimeType": "image/png"}],
            {},
            "Image output data must contain non-empty base64 image data.",
        ),
        (
            [{"data": "abc123" + " " * 20, "mimeType": "image/png"}],
            {"PROMPTFOO_GRADING_IMAGE_MAX_RAW_CHARS": "20"},
            "Image output raw data exceeds multimodal grading size limit",
        ),
        (
            [
                {"data": "abc123", "mimeType": "image/png"},
                {"data": "def456", "mimeType": "image/png"},
            ],
            {"PROMPTFOO_GRADING_MAX_IMAGES": "1"},
            "Too many images for multimodal grading: received 2, maximum is 1.",
        ),
        (
            [{"data": B64_5, "mimeType": "image/png"}],
            {"PROMPTFOO_GRADING_IMAGE_MAX_BYTES": "4"},
            "Image output exceeds multimodal grading size limit: 5 bytes, maximum is 4.",
        ),
        (
            [{"data": B64_5, "mimeType": "image/png"}, {"data": B64_5, "mimeType": "image/png"}],
            {
                "PROMPTFOO_GRADING_IMAGE_MAX_BYTES": "10",
                "PROMPTFOO_GRADING_IMAGE_MAX_TOTAL_BYTES": "9",
            },
            "Image outputs exceed multimodal grading total size limit: 10 bytes, maximum is 9.",
        ),
    ],
)
async def test_rejects_invalid_images_before_grading(monkeypatch, images, env, message):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    provider = grader(PASS_JSON)
    with pytest.raises(Exception, match=re.escape(message)):
        await matches_llm_rubric(
            "Does the image match?",
            "Generated image",
            {"rubricPrompt": "Grade this output", "provider": provider},
            {},
            None,
            {"providerResponse": {"output": "Generated image", "images": images}},
        )
    assert provider.calls == []


async def test_normalizes_whitespace_padded_base64_image():
    provider = grader(PASS_JSON)
    await matches_llm_rubric(
        "Does the image match?",
        "Generated image",
        {"rubricPrompt": "Grade this output", "provider": provider},
        {},
        None,
        {
            "providerResponse": {
                "output": "Generated image",
                "images": [
                    {"data": "data:image/png;base64, a b c 1 2 3 ", "mimeType": "image/png"}
                ],
            }
        },
    )
    assert {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    } in sent(provider)[0]["content"]


# --- rubric rendering, malformed responses ----------------------------------------------------


async def test_renders_object_rubric():
    rubric = {"prompt": "Describe the image"}
    provider = json_grader(
        {"pass": True, "score": 1, "reason": "ok"},
        tokenUsage={"total": 1, "prompt": 1, "completion": 1},
    )
    await matches_llm_rubric(
        rubric, "Sample output", {"rubricPrompt": "Grade: {{ rubric }}", "provider": provider}
    )
    prompt, context = provider.calls[0]
    assert prompt == 'Grade: {"prompt":"Describe the image"}'
    assert context is not None
    subset(context["prompt"], {"label": "llm-rubric", "raw": prompt})
    subset(context["vars"], {"output": "Sample output", "rubric": rubric})


def _failure(reason: str, tokens: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "pass": False,
        "score": 0,
        "reason": reason,
        "metadata": {"graderError": True},
        "tokensUsed": tokens or TOKENS_10,
    }


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (
            42,
            "llm-rubric produced malformed response - output must be string or object. Output: 42",
        ),
        (None, "No output"),
        (
            [{"pass": False, "score": 0, "reason": "bad"}],
            "llm-rubric produced malformed response - output must be string or object. "
            'Output: [{"pass":false,"score":0,"reason":"bad"}]',
        ),
        (
            "This is a valid text response but contains no JSON objects",
            "Could not extract JSON from llm-rubric response",
        ),
        ("Here is the result: []", "Could not extract JSON from llm-rubric response"),
        ("Result: null", "Could not extract JSON from llm-rubric response"),
        ('"just a string"', "Could not extract JSON from llm-rubric response"),
        ("Result: 123", "Could not extract JSON from llm-rubric response"),
    ],
)
async def test_malformed_grader_responses(output: Any, reason: str):
    provider = grader(output, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result == _failure(reason)


async def test_invalid_json_string_output():
    provider = grader(
        '{ "pass": true, "reason": "Invalid JSON missing closing brace', tokenUsage=USAGE_10
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert "Could not extract JSON from llm-rubric response" in result["reason"]
    assert {k: v for k, v in result.items() if k != "reason"} == {
        k: v for k, v in _failure("").items() if k != "reason"
    }


async def test_cached_no_output_preserves_provenance():
    provider = grader(
        None,
        cached=True,
        tokenUsage={"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    subset(
        result,
        {
            "pass": False,
            "reason": "No output",
            "metadata": {"graderError": True, "cachedResponse": True},
            "tokensUsed": {"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
        },
    )


async def test_counts_fresh_failed_request_without_usage():
    result = await matches_llm_rubric(
        "Expected output",
        "Sample output",
        {"rubricPrompt": "Grading prompt", "provider": FakeGrader({"output": None})},
    )
    subset(
        result,
        {
            "pass": False,
            "reason": "No output",
            "metadata": {"graderError": True},
            "tokensUsed": {"total": 0, "numRequests": 1},
        },
    )


async def test_cached_malformed_json_preserves_provenance():
    provider = grader(
        "This cached response does not contain JSON",
        cached=True,
        tokenUsage={"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
    )
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    subset(
        result,
        {
            "pass": False,
            "metadata": {"graderError": True, "cachedResponse": True},
            "tokensUsed": {"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
        },
    )


async def test_fails_when_grader_returns_failing_result():
    provider = json_grader({"pass": False, "reason": "Grading failed"}, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Expected output",
        "Different output",
        {"rubricPrompt": "Grading prompt", "provider": provider},
    )
    subset(result, {"pass": False, "reason": "Grading failed", "score": 0})
    assert result["tokensUsed"]["numRequests"] == 0


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"error": "Provider error", "output": None, "tokenUsage": USAGE_10}, "Provider error"),
        ({"output": None, "tokenUsage": USAGE_10}, "No output"),
    ],
)
async def test_throw_on_error(response: dict[str, Any], message: str):
    grading = {"rubricPrompt": "Grading prompt", "provider": FakeGrader(response)}
    with pytest.raises(llm_grading.LlmRubricProviderError, match=message):
        await matches_llm_rubric(
            "Test rubric", "Test output", grading, {}, None, {"throwOnError": True}
        )


async def test_uses_overridden_grading_provider_config(monkeypatch):
    seen: list[Any] = []

    async def fake_call_api(self, prompt, context=None, options=None):
        seen.append((self.config.get("temperature"), self.get_api_key(), prompt, context))
        return {
            "output": json.dumps({"pass": True, "reason": "Grading passed"}),
            "tokenUsage": USAGE_10,
        }

    monkeypatch.setattr(OpenAiChatCompletionProvider, "call_api", fake_call_api)
    result = await matches_llm_rubric(
        "Expected output",
        "Sample output",
        {
            "rubricPrompt": "Grading prompt",
            "provider": {
                "id": "openai:gpt-4o-mini",
                "config": {"apiKey": "abc123", "temperature": 3.1415926},
            },
        },
    )
    subset(result, {"reason": "Grading passed", "pass": True, "score": 1})
    temperature, api_key, prompt, context = seen[0]
    assert (temperature, api_key, prompt) == (3.1415926, "abc123", "Grading prompt")
    subset(context["prompt"], {"label": "llm-rubric", "raw": "Grading prompt"})
    subset(context["vars"], {"output": "Sample output", "rubric": "Expected output"})


# --- thresholds, scores and pass values -------------------------------------------------------


def _assertion(threshold: float) -> dict[str, Any]:
    return {"type": "llm-rubric", "value": "Rubric prompt", "threshold": threshold}


async def _grade_with(payload: dict[str, Any], assertion: dict[str, Any]) -> dict[str, Any]:
    return await matches_llm_rubric(
        "Rubric prompt",
        "Sample output",
        {"rubricPrompt": "Rubric prompt", "provider": json_grader(payload)},
        {},
        assertion,
    )


@pytest.mark.parametrize(
    ("threshold", "payload", "expected"),
    [
        (0.5, {"score": 0.25, "reason": "Low score"}, {"pass": False, "score": 0.25}),
        (0.5, {"score": 0.75, "reason": "High score"}, {"pass": True, "score": 0.75}),
        (
            0.1,
            {"score": 0.25, "reason": "Low score but pass", "pass": True},
            {"pass": True, "score": 0.25},
        ),
        (
            0.8,
            {"score": 0.7, "reason": "Score below threshold", "pass": True},
            {"pass": False, "score": 0.7},
        ),
        (
            0.8,
            {"score": 0.9, "reason": "Score above threshold but explicit fail", "pass": False},
            {"pass": False, "score": 0.9},
        ),
        (0.8, {"score": 0.8, "reason": "Exactly at threshold"}, {"pass": True, "score": 0.8}),
        (0.8, {"score": 0.799, "reason": "Just below threshold"}, {"pass": False, "score": 0.799}),
        (0.8, {"pass": True, "reason": "No score provided"}, {"pass": True, "score": 1}),
        (
            0.8,
            {"score": "high", "reason": "Invalid score type", "pass": True},
            {"pass": True, "score": 1},
        ),
        (0.8, {"score": "0.9", "reason": "String score"}, {"pass": True, "score": 0.9}),
        (0.8, {"reason": "String pass", "pass": "true"}, {"pass": True}),
        (0.8, {"reason": "String fail", "pass": "false"}, {"pass": False}),
        (0, {"score": 0, "reason": "At the floor"}, {"pass": True}),
    ],
)
async def test_threshold_and_verdict_parsing(threshold, payload, expected):
    assertion = _assertion(threshold)
    result = await _grade_with(payload, assertion)
    subset(result, {"assertion": assertion, "reason": payload["reason"], **expected})


@pytest.mark.parametrize(
    "score",
    [True, False, "", " ", [], [1], {}, "invalid", "Infinity", "1e309", math.nan, math.inf],
)
async def test_rejects_invalid_score_only_verdict(score: Any):
    provider = grader({"score": score}, tokenUsage={**USAGE_10, "cached": 0, "numRequests": 1})
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    subset(
        result,
        {
            "pass": False,
            "score": 0,
            "metadata": {"graderError": True},
            "tokensUsed": USAGE_10,
        },
    )


@pytest.mark.parametrize(
    ("payload", "passed", "score"),
    [
        ({"score": 1, "reason": "Strong match"}, True, 1),
        ({"score": 0, "reason": "Did not meet the rubric"}, False, 0),
        ({"pass": True, "score": 0, "reason": "Passing per rubric override"}, True, 0),
    ],
)
async def test_score_only_and_explicit_pass(payload, passed, score):
    provider = json_grader(payload, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert (result["pass"], result["score"]) == (passed, score)


async def test_fails_when_both_pass_and_score_missing():
    provider = json_grader({"reason": "Looked fine to me"}, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Expected output", "Sample output", {"rubricPrompt": "Grading prompt", "provider": provider}
    )
    assert result["pass"] is False
    assert "neither a pass verdict nor a finite numeric score" in result["reason"]


# --- rubric prompts loaded from files ---------------------------------------------------------

PASS_TEST = {"pass": True, "score": 1, "reason": "Test passed"}


def _write(tmp_path, rel: str, content: str) -> None:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


async def test_loads_rubric_prompt_from_text_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "path/to/external/rubric.txt", "This is an external rubric prompt")
    provider = json_grader(PASS_TEST, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Test rubric",
        "Test output",
        {"rubricPrompt": "file://path/to/external/rubric.txt", "provider": provider},
    )
    prompt, context = provider.calls[0]
    assert prompt == "This is an external rubric prompt"
    assert context is not None
    subset(context["prompt"], {"label": "llm-rubric", "raw": prompt})
    subset(context["vars"], {"output": "Test output", "rubric": "Test rubric"})
    subset(result, PASS_TEST)
    assert result["tokensUsed"] == TOKENS_10


async def test_loads_json_rubric_file_with_nunjucks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(
        tmp_path,
        "path/to/rubric.json",
        """[
  {%- set system_prompt -%}
  Evaluate the response
  {%- endset -%}
  { "role": "system", "content": {{ system_prompt | dump }} },
  { "role": "user", "content": "Output: {{ output }}" }
]""",
    )
    provider = json_grader(PASS_TEST, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Test rubric",
        "Test output",
        {"rubricPrompt": "file://path/to/rubric.json", "provider": provider},
    )
    prompt = provider.calls[0][0]
    assert "Evaluate the response" in prompt
    assert "Test output" in prompt
    assert json.loads(prompt) == [
        {"role": "system", "content": "Evaluate the response"},
        {"role": "user", "content": "Output: Test output"},
    ]
    subset(result, PASS_TEST)
    assert result["tokensUsed"] == TOKENS_10


async def test_loads_yaml_rubric_file_with_nunjucks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(
        tmp_path,
        "path/to/rubric.yaml",
        """{%- set system_prompt -%}
Evaluate the response
{%- endset -%}
- role: system
  content: {{ system_prompt }}
- role: user
  content: "Output: {{ output }}\"""",
    )
    provider = json_grader(PASS_TEST, tokenUsage=USAGE_10)
    result = await matches_llm_rubric(
        "Test rubric",
        "Test output",
        {"rubricPrompt": "file://path/to/rubric.yaml", "provider": provider},
    )
    assert "Evaluate the response" in provider.calls[0][0]
    subset(result, PASS_TEST)
    assert result["tokensUsed"] == TOKENS_10


async def test_js_rubric_prompt_file_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = json_grader(PASS_TEST)
    with pytest.raises(ValueError, match="JavaScript rubric prompt files are not supported"):
        await matches_llm_rubric(
            "Test rubric",
            "Test output",
            {"rubricPrompt": "file://path/to/external/file.js", "provider": provider},
        )
    assert provider.calls == []


async def test_missing_rubric_prompt_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = json_grader(PASS_TEST)
    with pytest.raises(ValueError, match="File does not exist"):
        await matches_llm_rubric(
            "Test rubric",
            "Test output",
            {"rubricPrompt": "file://path/to/external/rubric.txt", "provider": provider},
        )
    assert provider.calls == []


# --- tryParse and renderLlmRubricPrompt -------------------------------------------------------


def test_try_parse():
    assert try_parse('{"key": "value"}') == {"key": "value"}
    assert try_parse("not json") == "not json"
    assert try_parse("") == ""
    assert try_parse(None) is None


@pytest.mark.parametrize(
    ("template", "context", "expected"),
    [
        (
            '{"role": "user", "content": "Hello {{name}}"}',
            {"name": "World"},
            {"role": "user", "content": "Hello World"},
        ),
        (
            '{"nested": {"text": "{{var}}", "number": 42}}',
            {"var": "test"},
            {"nested": {"text": "test", "number": 42}},
        ),
        (
            '{"text": "{{object}}"}',
            {"object": {"foo": "bar", "baz": [1, 2, 3]}},
            {"text": '{"foo":"bar","baz":[1,2,3]}'},
        ),
        (
            '{"items": [{"name": "{{name1}}"}, {"name": "{{name2}}"}]}',
            {"name1": "Alice", "name2": "Bob"},
            {"items": [{"name": "Alice"}, {"name": "Bob"}]},
        ),
        (
            '{"content": "Line 1\\nLine {{number}}\\nLine 3"}',
            {"number": "2"},
            {"content": "Line 1\nLine 2\nLine 3"},
        ),
        (
            '{"outer": "{{value1}}", "inner": {"value": "{{value2}}"}}',
            {"value1": "outer value", "value2": "inner value"},
            {"outer": "outer value", "inner": {"value": "inner value"}},
        ),
        (
            '{"content": "This needs \\"escaping\\" and {{var}} too"}',
            {"var": 'var with "quotes"'},
            {"content": 'This needs "escaping" and var with "quotes" too'},
        ),
        (
            json.dumps(
                {
                    "role": "system",
                    "content": "Process this: {{input}}",
                    "config": {
                        "options": [
                            {"id": 1, "label": "{{option1}}"},
                            {"id": 2, "label": "{{option2}}"},
                        ]
                    },
                }
            ),
            {"input": "test input", "option1": "First Option", "option2": "Second Option"},
            {
                "role": "system",
                "content": "Process this: test input",
                "config": {
                    "options": [
                        {"id": 1, "label": "First Option"},
                        {"id": 2, "label": "Second Option"},
                    ]
                },
            },
        ),
    ],
)
async def test_render_json_rubric_prompt(template: str, context: dict[str, Any], expected: Any):
    assert json.loads(await render_llm_rubric_prompt(template, context)) == expected


async def test_render_json_rubric_prompt_output_is_compact_json():
    result = await render_llm_rubric_prompt(
        '{"nested": {"text": "{{var}}", "number": 42}}', {"var": "test"}
    )
    assert result == '{"nested":{"text":"test","number":42}}'


async def test_render_legacy_template():
    assert await render_llm_rubric_prompt("Hello {{name}}", {"name": "World"}) == "Hello World"


async def test_render_stringifies_objects_in_arrays():
    objects = [
        {"name": "Object 1", "properties": {"color": "red", "size": "large"}},
        {"name": "Object 2", "properties": {"color": "blue", "size": "small"}},
    ]
    result = await render_llm_rubric_prompt("Source Text:\n{{input}}", {"input": objects})
    assert "[object Object]" not in result
    assert json_stringify(objects[0]) in result
    assert json_stringify(objects[1]) in result
    # Nunjucks prints an array as Array.prototype.join(",").
    assert result == "Source Text:\n" + ",".join(json_stringify(o) for o in objects)


async def test_render_mixed_arrays():
    mixed = ["string item", {"name": "Object item"}, 42, [1, 2, 3]]
    result = await render_llm_rubric_prompt("Items: {{items}}", {"items": mixed})
    assert result == 'Items: string item,{"name":"Object item"},42,[1,2,3]'


async def test_render_join_filter():
    result = await render_llm_rubric_prompt(
        'statements:\n{{statements|join("\\n")}}',
        {"statements": ["Statement 1", "Statement 2", "Statement 3"]},
    )
    assert result == "statements:\nStatement 1\nStatement 2\nStatement 3"


async def test_render_stringifies_objects_in_arrays_with_strings():
    items = [{"name": "Item 1", "price": 10}, "string item", {"name": "Item 2", "price": 20}]
    result = await render_llm_rubric_prompt("Items: {{items}}", {"items": items})
    assert "[object Object]" not in result
    assert json_stringify(items[0]) in result
    assert "string item" in result
    assert json_stringify(items[2]) in result


async def test_render_stringifies_deeply_nested_data():
    data = {
        "products": [
            {
                "name": "Item 1",
                "price": 10,
                "details": {
                    "color": "red",
                    "specs": {"weight": "2kg", "dimensions": {"width": 10, "height": 20}},
                },
            },
            "string item",
            {
                "name": "Item 2",
                "price": 20,
                "nested": [{"a": 1}, {"b": 2}],
                "metadata": {"tags": ["electronics", "gadget"]},
            },
        ]
    }
    result = await render_llm_rubric_prompt("Complex data: {{data}}", {"data": data})
    assert "[object Object]" not in result
    assert '"specs":{"weight":"2kg"' in result
    assert '"dimensions":{"width":10,"height":20}' in result
    assert '[{"a":1},{"b":2}]' in result
    assert '"tags":["electronics","gadget"]' in result
    assert "string item" in result


# --- handleLlmRubric --------------------------------------------------------------------------


def make_params(**overrides: Any) -> AssertionParams:
    base: dict[str, Any] = {
        "assertion": {"type": "llm-rubric", "value": "test rubric"},
        "base_type": "llm-rubric",
        "inverse": False,
        "output": "test output",
        "output_string": "test output string",
        "test": {"vars": {}},
        "provider_response": {},
        "assertion_value_context": {"prompt": "test prompt", "vars": {}, "test": {"vars": {}}},
    }
    base.update(overrides)
    return AssertionParams(**base)


@pytest.fixture
def mock_matcher(monkeypatch):
    state: dict[str, Any] = {"calls": [], "result": {"pass": True, "score": 1, "reason": "ok"}}

    async def fake(*args: Any) -> dict[str, Any]:
        state["calls"].append(args)
        return copy.deepcopy(state["result"])

    monkeypatch.setattr(llm_grading, "matches_llm_rubric", fake)
    return state


PASSING = {"pass": True, "score": 1, "reason": "Assertion passed"}


@pytest.mark.parametrize(
    ("rendered", "expected_rubric"),
    [
        ("test rendered value", "test rendered value"),
        ({"test": "value"}, {"test": "value"}),
        (None, ""),
        (["foo", "bar"], ["foo", "bar"]),
    ],
)
async def test_handler_rendered_values(mock_matcher, rendered, expected_rubric):
    mock_matcher["result"] = PASSING
    params = make_params(rendered_value=rendered)
    assert await handle_llm_rubric(params) == PASSING
    assert mock_matcher["calls"] == [
        (expected_rubric, "test output string", None, {}, params.assertion, None, None)
    ]


async def test_handler_passes_provider_response_images(mock_matcher):
    mock_matcher["result"] = PASSING
    provider_response = {
        "output": "test output",
        "images": [{"data": "data:image/png;base64,abc123", "mimeType": "image/png"}],
    }
    params = make_params(rendered_value="test rubric", provider_response=provider_response)
    assert await handle_llm_rubric(params) == PASSING
    assert mock_matcher["calls"][0][5] == {"providerResponse": provider_response}


async def test_handler_does_not_pass_images_after_transform(mock_matcher):
    mock_matcher["result"] = PASSING
    params = make_params(
        assertion={"type": "llm-rubric", "value": "test rubric", "transform": "output.text"},
        rendered_value="test rubric",
        output_string="transformed output",
        provider_response={
            "output": {"text": "transformed output"},
            "images": [{"data": "data:image/png;base64,abc123", "mimeType": "image/png"}],
        },
    )
    await handle_llm_rubric(params)
    assert mock_matcher["calls"][0] == (
        "test rubric",
        "transformed output",
        None,
        {},
        params.assertion,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("matcher_result", "expected"),
    [
        (
            {"pass": True, "score": 0.75, "reason": "criterion matched"},
            {"pass": False, "score": 0.25, "reason": "criterion matched"},
        ),
        (
            {"pass": False, "score": 0.25, "reason": "criterion did not match"},
            {"pass": True, "score": 0.75, "reason": "criterion did not match"},
        ),
        (
            {"pass": True, "score": 1, "reason": "matched"},
            {"pass": False, "score": 0, "reason": "matched"},
        ),
        (
            {"pass": False, "score": 0, "reason": "no match"},
            {"pass": True, "score": 1, "reason": "no match"},
        ),
    ],
)
async def test_handler_inverse(mock_matcher, matcher_result, expected):
    mock_matcher["result"] = matcher_result
    result = await handle_llm_rubric(make_params(inverse=True, rendered_value="test rubric"))
    assert result == pytest.approx(expected)


async def test_handler_does_not_invert_grader_failures(mock_matcher):
    failure = {
        "pass": False,
        "score": 0,
        "reason": "grader failed",
        "metadata": {"graderError": True},
    }
    mock_matcher["result"] = failure
    params = make_params(inverse=True, rendered_value="test rubric")
    result = await handle_llm_rubric(params)
    assert result == {**failure, "assertion": params.assertion}
    assert result["assertion"] is params.assertion


async def test_handler_clamps_out_of_range_inverted_score(mock_matcher):
    mock_matcher["result"] = {"pass": True, "score": 5, "reason": "matched"}
    result = await handle_llm_rubric(make_params(inverse=True, rendered_value="test rubric"))
    assert (result["pass"], result["score"]) == (False, 0)


async def test_handler_treats_nan_score_as_zero_when_inverting(mock_matcher):
    mock_matcher["result"] = {"pass": False, "score": math.nan, "reason": "malformed"}
    result = await handle_llm_rubric(make_params(inverse=True, rendered_value="test rubric"))
    assert (result["pass"], result["score"]) == (True, 1)


@pytest.mark.parametrize(
    "rubric_prompt",
    [
        [{"role": "system", "content": "test prompt"}],
        [{"role": "user", "content": "bar"}],
        [],
        {"foo": "bar", "baz": 3},
    ],
)
async def test_handler_stringifies_object_rubric_prompt(mock_matcher, rubric_prompt):
    params = make_params(test={"vars": {}, "options": {"rubricPrompt": rubric_prompt}})
    await handle_llm_rubric(params)
    assert params.test["options"]["rubricPrompt"] == json_stringify(rubric_prompt)


async def test_handler_does_not_restringify_string_rubric_prompt(mock_matcher):
    prompt = '[{"role":"system","content":"already stringified"}]'
    params = make_params(test={"vars": {}, "options": {"rubricPrompt": prompt}})
    await handle_llm_rubric(params)
    assert params.test["options"]["rubricPrompt"] == prompt


async def test_handler_uses_options_rubric_prompt_as_assertion_value(mock_matcher):
    mock_matcher["result"] = {"pass": True, "score": 2, "reason": "used options rubric"}
    params = make_params(
        assertion={"type": "llm-rubric", "value": None},
        test={"vars": {}, "options": {"rubricPrompt": "rubric from options"}},
    )
    assert await handle_llm_rubric(params) == mock_matcher["result"]
    assert params.assertion["value"] == "rubric from options"


async def test_handler_does_not_overwrite_assertion_value(mock_matcher):
    params = make_params(
        assertion={"type": "llm-rubric", "value": "already set"},
        test={"vars": {}, "options": {"rubricPrompt": "rubric from options"}},
    )
    await handle_llm_rubric(params)
    assert params.assertion["value"] == "already set"


async def test_handler_without_options(mock_matcher):
    mock_matcher["result"] = {"pass": True, "score": 3, "reason": "no options"}
    params = make_params(assertion={"type": "llm-rubric", "value": None})
    assert await handle_llm_rubric(params) == mock_matcher["result"]
    assert params.assertion.get("value") is None


async def test_handler_rejects_invalid_rendered_value(mock_matcher):
    with pytest.raises(
        AssertionValueError,
        match='"llm-rubric" assertion type must have a string or object value',
    ):
        await handle_llm_rubric(make_params(rendered_value=123))


async def test_handler_passes_through_rendered_grading_prompt(mock_matcher):
    mock_matcher["result"] = {
        "pass": True,
        "score": 1,
        "reason": "test reason",
        "metadata": {
            "renderedGradingPrompt": '[{"role":"system","content":"grading instructions"}]'
        },
    }
    result = await handle_llm_rubric(make_params(rendered_value="test rubric"))
    assert (
        result["metadata"]["renderedGradingPrompt"]
        == '[{"role":"system","content":"grading instructions"}]'
    )


# --- llm-rubric audio grading -----------------------------------------------------------------

AUDIO = {"data": "UklGRgAAAABXQVZF", "format": "wav", "transcript": "Hello."}
GATEWAY_URL = "https://gateway.example.com/v1/chat/completions"
CALM = {
    "choices": [
        {"message": {"content": json.dumps({"pass": True, "score": 1, "reason": "Calm."})}}
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


async def grade_audio(provider: Any, target_audio: Any = AUDIO, **overrides: Any):
    params: dict[str, Any] = {
        "assertion": {"type": "llm-rubric", "value": "The speaker sounds calm."},
        "base_type": "llm-rubric",
        "assertion_value_context": {"prompt": "Say hello", "vars": {}, "test": {}},
        "inverse": False,
        "rendered_value": "The speaker sounds calm.",
        "output": "Hello.",
        "output_string": "Hello.",
        "test": {
            "options": {
                "provider": provider,
                "rubricPrompt": [
                    {"role": "system", "content": "Return a JSON grade."},
                    {"role": "user", "content": "Grade {{output}} for {{rubric}}"},
                ],
            }
        },
        "provider_response": {"output": "Hello.", "audio": target_audio},
    }
    params.update(overrides)
    return await handle_llm_rubric(AssertionParams(**params))


class RecordingCall:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def __call__(self, prompt, context=None, options=None):
        self.prompts.append(prompt)
        return copy.deepcopy(self.response)


def spy(provider: Any, response: dict[str, Any]) -> RecordingCall:
    call = RecordingCall(response)
    provider.call_api = call
    return call


@pytest.mark.usefixtures(openai_env.__name__)
class TestAudioGrading:
    @pytest.mark.parametrize(
        ("model", "provider_id"),
        [
            ("gpt-audio-1.5", "voice-quality-judge"),
            ("gpt-4o-audio-preview", "voice-quality-judge"),
            ("gpt-4o-mini-audio-preview", "voice-quality-judge"),
            ("gpt-audio-1.5", "responses:voice-quality-judge"),
            ("gpt-audio-1.5", "google:voice-quality-judge"),
        ],
    )
    @respx.mock
    async def test_sends_native_audio_through_chat_transport(self, model, provider_id):
        config: dict[str, Any] = {"modalities": ["text"]}
        if model.endswith("-preview"):
            config["apiBaseUrl"] = "https://gateway.example.com/v1"
        route = respx.post(GATEWAY_URL if model.endswith("-preview") else CHAT_URL).mock(
            return_value=ok(CALM)
        )
        provider = OpenAiChatCompletionProvider(model, {"id": provider_id, "config": config})
        result = await grade_audio(provider)
        body = json.loads(route.calls[0].request.content)
        assert body["model"] == model
        assert body["modalities"] == ["text"]
        assert body["messages"][0] == {"role": "system", "content": "Return a JSON grade."}
        assert body["messages"][1]["content"][0] == {
            "type": "text",
            "text": "Grade Hello. for The speaker sounds calm.",
        }
        assert {
            "type": "input_audio",
            "input_audio": {"data": AUDIO["data"], "format": "wav"},
        } in body["messages"][1]["content"]
        subset(result, {"pass": True, "score": 1, "tokensUsed": {"total": 30}})
        assert result["metadata"]["renderedGradingPromptAudio"] is True
        assert AUDIO["data"] not in result["metadata"]["renderedGradingPrompt"]

    async def test_accepts_mp3_and_preserves_failing_grade(self):
        provider = OpenAiChatCompletionProvider("gpt-audio-1.5")
        call = spy(
            provider,
            {"output": json.dumps({"pass": False, "score": 0.2, "reason": "Speech is clipped."})},
        )
        subset(
            await grade_audio(provider, {**AUDIO, "format": "mp3"}), {"pass": False, "score": 0.2}
        )
        assert {
            "type": "input_audio",
            "input_audio": {"data": AUDIO["data"], "format": "mp3"},
        } in json.loads(call.prompts[0])[1]["content"]

    @pytest.mark.parametrize("transcript", [None, "Hello."])
    @respx.mock
    async def test_removes_duplicated_audio_from_grading_text(self, transcript):
        route = respx.post(CHAT_URL).mock(return_value=ok(CALM))
        provider = OpenAiChatCompletionProvider("gpt-audio-1.5")
        target = {**AUDIO, "transcript": transcript}
        result = await grade_audio(
            provider,
            target,
            output=AUDIO["data"],
            output_string=AUDIO["data"],
            provider_response={"output": AUDIO["data"], "isBase64": True, "audio": target},
        )
        body = json.loads(route.calls[0].request.content)
        assert body["messages"][1]["content"][0] == {
            "type": "text",
            "text": f"Grade {transcript or '[Audio output]'} for The speaker sounds calm.",
        }
        assert {
            "type": "input_audio",
            "input_audio": {"data": AUDIO["data"], "format": "wav"},
        } in body["messages"][1]["content"]
        assert AUDIO["data"] not in result["metadata"]["renderedGradingPrompt"]

    async def test_accepts_line_wrapped_base64_at_size_limit(self):
        provider = OpenAiChatCompletionProvider("gpt-audio-1.5")
        data = base64.b64encode(bytes(20 * 1024 * 1024)).decode()
        call = spy(provider, {"output": '{"pass":true,"score":1}'})
        wrapped = re.sub(r"(.{76})", "\\1\r\n", data)
        result = await grade_audio(provider, {**AUDIO, "data": wrapped})
        assert result["pass"] is True
        parts = json.loads(call.prompts[0])[1]["content"]
        audio_part = next(p for p in parts if p.get("type") == "input_audio")
        assert audio_part["input_audio"]["data"] == data

    @pytest.mark.parametrize(
        ("output", "data"),
        [
            (AUDIO["data"], re.sub(r"(.{4})", "\\1\r\n", AUDIO["data"])),
            (re.sub(r"(.{4})", "\\1\r\n", AUDIO["data"]), AUDIO["data"]),
            (f"data:audio/wav;base64,{AUDIO['data']}", AUDIO["data"]),
        ],
    )
    @respx.mock
    async def test_recognizes_equivalent_base64_output(self, output, data):
        route = respx.post(CHAT_URL).mock(return_value=ok(CALM))
        provider = OpenAiChatCompletionProvider("gpt-audio-1.5")
        target = {**AUDIO, "data": data}
        result = await grade_audio(
            provider,
            target,
            output=output,
            output_string=output,
            provider_response={"output": output, "audio": target},
        )
        body = json.loads(route.calls[0].request.content)
        assert body["messages"][1]["content"][0] == {
            "type": "text",
            "text": "Grade Hello. for The speaker sounds calm.",
        }
        assert AUDIO["data"] not in result["metadata"]["renderedGradingPrompt"]

    @pytest.mark.parametrize(
        "make_provider",
        [
            lambda: OpenAiChatCompletionProvider("gpt-4.1"),
            lambda: OpenAiResponsesProvider("gpt-5.6"),
        ],
    )
    async def test_keeps_transcript_grading_for_text_grader(self, make_provider):
        provider = make_provider()
        call = spy(provider, {"output": '{"pass":true,"score":1}'})
        result = await grade_audio(provider, {"transcript": "Hello."})
        assert json.loads(call.prompts[0])[1]["content"] == (
            "Grade Hello. for The speaker sounds calm."
        )
        assert "renderedGradingPromptAudio" not in result["metadata"]

    @respx.mock
    async def test_does_not_attach_audio_after_transform(self):
        route = respx.post(CHAT_URL).mock(return_value=ok(CALM))
        provider = OpenAiChatCompletionProvider("gpt-audio-1.5")
        result = await grade_audio(
            provider,
            AUDIO,
            assertion={
                "type": "llm-rubric",
                "value": "Contains hello",
                "transform": "output.toLowerCase()",
            },
        )
        body = json.loads(route.calls[0].request.content)
        assert body["messages"][1]["content"] == "Grade Hello. for The speaker sounds calm."
        assert "renderedGradingPromptAudio" not in result["metadata"]

    @pytest.mark.parametrize(
        ("target", "message"),
        [
            ({"format": "wav"}, "valid base64"),
            ({**AUDIO, "data": "not base64!"}, "valid base64"),
            ({**AUDIO, "format": "pcm16"}, "WAV or MP3"),
            ({**AUDIO, "format": None}, "WAV or MP3"),
            ({**AUDIO, "data": "A" * (28 * 1024 * 1024)}, "20 MiB"),
            ({**AUDIO, "data": f"promptfoo://blob/{'a' * 64}"}, "blob references"),
        ],
    )
    @respx.mock
    async def test_rejects_unsupported_audio_before_contacting_grader(self, target, message):
        route = respx.post(CHAT_URL).mock(return_value=ok(CALM))
        provider = OpenAiChatCompletionProvider("gpt-audio-1.5")
        with pytest.raises(Exception, match=message):
            await grade_audio(provider, target, inverse=True)
        assert route.call_count == 0

    async def test_preserves_cached_metadata_without_audio_bytes(self):
        provider = OpenAiChatCompletionProvider("gpt-audio-1.5")
        spy(
            provider,
            {
                "output": '{"pass":true,"score":1}',
                "cached": True,
                "tokenUsage": {"total": 30, "cached": 30},
            },
        )
        result = await grade_audio(provider)
        subset(result["metadata"], {"cachedResponse": True, "renderedGradingPromptAudio": True})
        assert result["tokensUsed"]["cached"] == 30
        assert AUDIO["data"] not in json_stringify(result)
