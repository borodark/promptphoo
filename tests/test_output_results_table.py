"""Ports test/util/convertEvalResultsToTable.test.ts (convertResultsToTable)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from promptphoo.output import convert_results_to_table

PROMPT = {"raw": "test prompt", "display": "test prompt", "id": "prompt1", "label": "test prompt"}


def results_file(results: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    return {"version": 4, "prompts": [PROMPT], "results": {"results": results}, **overrides}


def result(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "test1",
        "testIdx": 0,
        "promptIdx": 0,
        "prompt": {"raw": "test prompt", "label": "Test Prompt"},
        "provider": {"id": "test-provider"},
        "success": True,
        "promptId": "prompt1",
        "testCase": {},
        "score": 1,
        "latencyMs": 100,
        "namedScores": {},
        **overrides,
    }


def test_converts_results_to_table_format() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={"var1": "value1", "var2": "value2"},
                    response={"output": "test output"},
                    provider={"id": "test-provider", "label": "Test Provider"},
                    failureReason="none",
                )
            ]
        )
    )
    assert table == {
        "head": {"prompts": [PROMPT], "vars": ["var1", "var2"]},
        "body": [
            {
                "outputs": [
                    {
                        "id": "test1",
                        "testIdx": 0,
                        "promptIdx": 0,
                        "prompt": "test prompt",
                        "provider": "Test Provider",
                        "success": True,
                        "promptId": "prompt1",
                        "testCase": {},
                        "score": 1,
                        "latencyMs": 100,
                        "namedScores": {},
                        "vars": {"var1": "value1", "var2": "value2"},
                        "response": {"output": "test output"},
                        "failureReason": "none",
                        "text": "test output",
                        "pass": True,
                        "cost": 0,
                        "tokenUsage": None,
                    }
                ],
                "vars": ["value1", "value2"],
                "test": {},
                "testIdx": 0,
            }
        ],
    }


def test_error_responses() -> None:
    table = convert_results_to_table(
        results_file([result(error="Test error", success=False, vars={}, score=0)])
    )
    assert table["body"][0]["outputs"][0]["text"] == "Test error"


def test_null_output_falls_back_to_error() -> None:
    table = convert_results_to_table(
        results_file(
            [result(vars={}, response={"output": None}, error="Some error", success=False)]
        )
    )
    assert table["body"][0]["outputs"][0]["text"] == "Some error"


def test_object_output_is_json() -> None:
    table = convert_results_to_table(results_file([result(response={"output": {"a": [1, 2.0]}})]))
    assert table["body"][0]["outputs"][0]["text"] == '{"a":[1,2]}'


def test_assertion_failure_shows_output_not_error() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    success=False,
                    error="Expected output to contain foo",
                    response={"output": "bar"},
                    testCase={"assert": [{"type": "contains", "value": "foo"}]},
                )
            ]
        )
    )
    assert table["body"][0]["outputs"][0]["text"] == "bar"


def test_redteam_final_prompt_replaces_single_var() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={"query": "original"},
                    response={"output": "o", "metadata": {"redteamFinalPrompt": "final prompt"}},
                )
            ]
        )
    )
    assert table["body"][0]["vars"] == ["final prompt"]


def test_provider_reported_prompt_replaces_prompt_var_among_many() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={"prompt": "orig", "harmCategory": "x"},
                    response={"output": "o", "prompt": [{"role": "user", "content": "hi"}]},
                )
            ]
        )
    )
    assert table["head"]["vars"] == ["harmCategory", "prompt"]
    assert table["body"][0]["vars"] == ["x", '[{"role":"user","content":"hi"}]']


def test_audio_and_images_media() -> None:
    audio = {"id": "a1", "data": "base64", "format": "wav", "extra": "dropped"}
    images = [{"data": "img", "mimeType": "image/png", "other": 1}]
    table = convert_results_to_table(
        results_file([result(response={"output": "x", "audio": audio, "images": images})])
    )
    cell = table["body"][0]["outputs"][0]
    assert cell["audio"] == {"id": "a1", "data": "base64", "format": "wav"}
    assert cell["images"] == [{"data": "img", "mimeType": "image/png"}]


def test_object_vars_are_pretty_printed() -> None:
    conversation = [{"user": "How do I work with you?"}, {"assistant": "I can help."}]
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={
                        "conversation": conversation,
                        "simpleVar": "This is a simple string",
                        "objectVar": {"key1": "value1", "key2": "value2"},
                    },
                    response={"output": "test output"},
                )
            ],
            vars=["conversation", "simpleVar", "objectVar"],
        )
    )
    assert table["head"]["vars"] == ["conversation", "simpleVar", "objectVar"]
    assert table["body"][0]["vars"][0] == json.dumps(conversation, indent=2)
    assert table["body"][0]["vars"][1] == "This is a simple string"
    assert table["body"][0]["vars"][2] == json.dumps({"key1": "value1", "key2": "value2"}, indent=2)


def test_session_id_copied_from_metadata() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={"question": "What is AI?"},
                    metadata={"sessionId": "session-123"},
                    response={"output": "o"},
                )
            ]
        )
    )
    assert table["head"]["vars"] == ["question", "sessionId"]
    assert table["body"][0]["vars"] == ["What is AI?", "session-123"]


def test_user_session_id_not_overwritten() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={"question": "q", "sessionId": "user-session"},
                    metadata={"sessionId": "metadata-session"},
                    response={"output": "o"},
                )
            ]
        )
    )
    assert table["body"][0]["vars"][table["head"]["vars"].index("sessionId")] == "user-session"


def test_missing_vars_created_for_session_id() -> None:
    r = result(metadata={"sessionId": "s"}, response={"output": "o"})
    table = convert_results_to_table(results_file([r]))
    assert table["head"]["vars"] == ["sessionId"]
    assert table["body"][0]["vars"] == ["s"]


def test_session_ids_array_filtered_and_stringified() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={"question": "What is AI?"},
                    metadata={
                        "sessionIds": ["session-1", None, "", "session-2", 123, "session-3"],
                        "sessionId": "ignored",
                    },
                    response={"output": "o"},
                )
            ]
        )
    )
    idx = table["head"]["vars"].index("sessionId")
    assert table["body"][0]["vars"][idx] == "session-1\nsession-2\n123\nsession-3"


def test_non_array_session_ids_fall_back_to_session_id() -> None:
    table = convert_results_to_table(
        results_file(
            [result(vars={}, metadata={"sessionIds": "nope", "sessionId": "s1"}, response={})]
        )
    )
    assert table["body"][0]["vars"] == ["s1"]


def test_preserves_persisted_variable_order() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    id="test2",
                    testIdx=1,
                    vars={"zebra": "z-row2", "apple": "a-row2", "mango": "m-row2", "banana": "b2"},
                    response={"output": "test output"},
                ),
                result(
                    vars={"zebra": "z-row1", "apple": "a-row1", "mango": "m-row1"},
                    response={"output": "test output"},
                ),
            ],
            vars=["zebra", "apple", "mango", "banana"],
        )
    )
    assert table["head"]["vars"] == ["zebra", "apple", "mango", "banana"]
    assert table["body"][0]["vars"] == ["z-row1", "a-row1", "m-row1", ""]
    assert table["body"][1]["vars"] == ["z-row2", "a-row2", "m-row2", "b2"]


def test_metadata_only_display_columns_sorted_after_prefix() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    id="test2",
                    testIdx=1,
                    vars={"prompt": "row2"},
                    response={"output": "o", "metadata": {"transformDisplayVars": {"zebra": "z"}}},
                ),
                result(
                    vars={"prompt": "row1"},
                    response={"output": "o", "metadata": {"transformDisplayVars": {"apple": "a"}}},
                ),
            ],
            vars=["prompt"],
        )
    )
    assert table["head"]["vars"] == ["prompt", "apple", "zebra"]


def test_alphabetical_fallback_without_persisted_vars() -> None:
    table = convert_results_to_table(
        results_file([result(vars={"zebra": "z", "apple": "a"}, response={"output": "o"})])
    )
    assert table["head"]["vars"] == ["apple", "zebra"]


def test_runtime_only_vars_appended_sorted() -> None:
    table = convert_results_to_table(
        results_file(
            [result(vars={"prompt": "p1", "zeta": "z1", "alpha": "a1"}, response={"output": "o"})],
            vars=["prompt"],
        )
    )
    assert table["head"]["vars"] == ["prompt", "alpha", "zeta"]
    assert table["body"][0]["vars"] == ["p1", "a1", "z1"]


@pytest.mark.parametrize(("falsy", "rendered"), [("", ""), (0, "0"), (False, "false")])
def test_falsy_var_not_overwritten_by_transform_display_vars(falsy: Any, rendered: str) -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(
                    vars={"prompt": falsy},
                    response={
                        "output": "o",
                        "metadata": {"transformDisplayVars": {"prompt": "from-transform"}},
                    },
                )
            ],
            vars=["prompt"],
        )
    )
    assert table["head"]["vars"] == ["prompt"]
    assert table["body"][0]["vars"] == [rendered]


def test_malformed_vars_field(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="promptphoo"):
        table = convert_results_to_table(
            results_file(
                [result(vars={"zebra": "z", "apple": "a"}, response={"output": "o"})],
                vars="not-an-array",
            )
        )
    assert table["head"]["vars"] == ["apple", "zebra"]
    assert "not string[]" in caplog.text


def test_collision_debug_breadcrumb(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="promptphoo"):
        table = convert_results_to_table(
            results_file(
                [
                    result(
                        vars={"prompt": "original"},
                        response={
                            "output": "o",
                            "metadata": {"transformDisplayVars": {"prompt": "shadowed"}},
                        },
                    )
                ],
                vars=["prompt"],
            )
        )
    assert table["body"][0]["vars"] == ["original"]
    assert "key 'prompt' collides" in caplog.text


def test_dedups_corrupt_persisted_vars() -> None:
    table = convert_results_to_table(
        results_file(
            [result(vars={"prompt": "p", "foo": "f"}, response={"output": "o"})],
            vars=["prompt", "prompt", "foo"],
        )
    )
    assert table["head"]["vars"] == ["prompt", "foo"]
    assert table["body"][0]["vars"] == ["p", "f"]


def test_requires_prompts() -> None:
    with pytest.raises(ValueError, match="Prompts are required"):
        convert_results_to_table({"version": 3, "results": {"results": []}})


def test_rows_ordered_by_test_idx_and_outputs_by_prompt_idx() -> None:
    table = convert_results_to_table(
        results_file(
            [
                result(id="b", testIdx=1, promptIdx=1, response={"output": "1-1"}),
                result(id="a", testIdx=1, promptIdx=0, response={"output": "1-0"}),
                result(id="c", testIdx=0, promptIdx=0, response={"output": "0-0"}),
            ]
        )
    )
    assert [r["testIdx"] for r in table["body"]] == [0, 1]
    assert [o["text"] for o in table["body"][1]["outputs"]] == ["1-0", "1-1"]
