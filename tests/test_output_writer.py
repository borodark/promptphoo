"""Ports test/util/output.test.ts, test/util/jsonlOutput.test.ts and test/util/jsonExport.test.ts
(writeOutput / writeMultipleOutputs / createOutputMetadata / createJunitXml) against an
in-memory `Eval` and real files in a temp directory.
"""

from __future__ import annotations

import json
import os
import re
import stat
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from promptphoo.models.eval import Eval
from promptphoo.output import (
    OutputWriteError,
    create_output_metadata,
    get_table,
    write_multiple_outputs,
    write_output,
)
from promptphoo.output.junit import create_junit_xml
from promptphoo.yamlload import load_yaml

NONE, ASSERT, ERROR = 0, 1, 2


def make_result(**overrides: Any) -> dict[str, Any]:
    """test/factories createEvaluateResult."""
    return {
        "success": True,
        "failureReason": NONE,
        "score": 1,
        "namedScores": {},
        "latencyMs": 100,
        "provider": {"id": "echo"},
        "prompt": {"raw": "Hello {{name}}", "label": "Hello prompt"},
        "response": {"output": "Hello"},
        "vars": {},
        "promptIdx": 0,
        "testIdx": 0,
        "testCase": {},
        "promptId": "prompt-1",
        **overrides,
    }


async def make_eval(config: dict[str, Any] | None = None, *results: dict[str, Any], **kw: Any):
    ev = Eval(config or {}, created_at=1746284400000, **kw)
    for r in results:
        await ev.append_result(r)
    return ev


class FakeEval:
    """Minimal record for legacy / persisted code paths."""

    def __init__(self, **kw: Any) -> None:
        self.id = "eval-fake"
        self.config: dict[str, Any] = {}
        self.created_at: Any = 1746284400000
        self.author = None
        self.vars: list[str] = []
        self.prompts: list[dict[str, Any]] = []
        self.results: list[Any] = []
        self.persisted = False
        self.old_results: dict[str, Any] | None = None
        self._version = 4
        self.loaded: list[Any] = []
        self.__dict__.update(kw)

    def version(self) -> int:
        return self._version

    def load_results(self) -> list[Any]:
        return self.loaded


def junit(xml: str) -> ET.Element:
    return ET.fromstring(xml.encode("utf-8"))


# --- basic formats --------------------------------------------------------------------------


async def test_csv_output(tmp_path: Path) -> None:
    ev = await make_eval(
        {},
        make_result(
            vars={"var1": "value1", "var2": "value2"},
            testCase={"vars": {"var1": "value1", "var2": "value2"}, "description": "first"},
            response={"output": "Test output"},
            namedScores={"accuracy": 0.756},
        ),
        make_result(
            testIdx=1,
            success=False,
            failureReason=ASSERT,
            score=0.25,
            testCase={"vars": {"var1": "=cmd", "var2": "-x"}},
            response={"output": "@SUM(A1)"},
            gradingResult={"pass": False, "score": 0.25, "reason": 'Has "quote", comma'},
        ),
        vars=["var1", "var2"],
        prompts=[
            {
                "raw": "Hello",
                "label": "Hello prompt",
                "provider": "echo",
                "metrics": {"namedScoresCount": {"accuracy": 1}},
            }
        ],
    )
    path = tmp_path / "out" / "output.csv"
    write_output(str(path), ev, None)
    assert path.read_text("utf-8") == (
        "Description,var1,var2,[echo] Hello prompt,Status,Score,Named Scores,Metric: accuracy,"
        "Grader Reason,Comment\n"
        'first,value1,value2,Test output,PASS,1.00,"{""accuracy"":0.76}",0.76,,\n'
        ',\'=cmd,\'-x,\'@SUM(A1),FAIL,0.25,,,"Has ""quote"", comma",\n'
    )


async def test_csv_headers_written_without_results(tmp_path: Path) -> None:
    ev = await make_eval({})
    path = tmp_path / "output.csv"
    write_output(str(path), ev, None)
    assert path.read_text("utf-8") == "\n"


async def test_csv_formula_escaping_can_be_disabled(tmp_path: Path) -> None:
    ev = await make_eval(
        {"env": {"PROMPTFOO_DISABLE_CSV_FORMULA_ESCAPING": "true"}},
        make_result(response={"output": "=1+1"}),
        prompts=[{"raw": "p", "label": "p", "metrics": {"namedScoresCount": {}}}],
    )
    path = tmp_path / "output.csv"
    write_output(str(path), ev, None)
    assert path.read_text("utf-8").splitlines()[1] == "=1+1,PASS,1.00,,,"


async def test_csv_redteam_columns(tmp_path: Path) -> None:
    ev = await make_eval(
        {"redteam": {}},
        make_result(metadata={"pluginId": "harmful", "messages": [{"role": "user"}]}),
        prompts=[{"raw": "p", "label": "p", "metrics": {"namedScoresCount": {}}}],
    )
    path = tmp_path / "output.csv"
    write_output(str(path), ev, None)
    header, row = path.read_text("utf-8").splitlines()
    assert header.endswith(
        "Messages,RedteamHistory,RedteamTreeHistory,pluginId,strategyId,sessionId,sessionIds"
    )
    assert row.endswith(',"[{""role"":""user""}]",,,harmful,basic,,')


async def test_json_output(tmp_path: Path) -> None:
    ev = await make_eval(
        {
            "description": "Eval",
            "env": {
                "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
                "ANTHROPIC_API_KEY": "anthropic-token",
                "AWS_REGION": "us-east-1",
            },
            "defaultTest": {"options": {"apiKey": "another-secret", "temperature": 0.1}},
        },
        make_result(score=1.0, latencyMs=12.0),
        vars=["name"],
        author="me@example.com",
    )
    path = tmp_path / "output.json"
    write_output(str(path), ev, "https://share/x")
    text = path.read_text("utf-8")
    data = json.loads(text)
    assert list(data) == ["evalId", "results", "config", "shareableUrl", "metadata", "vars"]
    assert data["evalId"] == ev.id
    assert data["shareableUrl"] == "https://share/x"
    assert data["config"]["env"]["AWS_BEARER_TOKEN_BEDROCK"] == "[REDACTED]"
    assert data["config"]["env"]["ANTHROPIC_API_KEY"] == "[REDACTED]"
    assert data["config"]["env"]["AWS_REGION"] == "us-east-1"
    assert data["config"]["defaultTest"]["options"]["apiKey"] == "[REDACTED]"
    assert data["config"]["defaultTest"]["options"]["temperature"] == 0.1
    assert data["results"]["version"] == 3
    assert data["results"]["results"][0]["latencyMs"] == 12
    assert '"latencyMs": 12,' in text  # integral floats print like JS numbers
    assert text.startswith('{\n  "evalId": ')
    assert data["vars"] == ["name"]
    meta = data["metadata"]
    assert list(meta) == [
        "promptfooVersion", "nodeVersion", "platform", "arch", "exportedAt",
        "evaluationCreatedAt", "author",
    ]  # fmt: skip
    assert meta["evaluationCreatedAt"] == "2025-05-03T15:00:00.000Z"
    assert meta["author"] == "me@example.com"


async def test_json_null_shareable_url_and_runtime_options(tmp_path: Path) -> None:
    ev = await make_eval(
        {}, runtime_options={"cache": False, "abortSignal": object(), "progressCallback": print}
    )
    path = tmp_path / "output.json"
    write_output(str(path), ev, None)
    data = json.loads(path.read_text("utf-8"))
    assert data["shareableUrl"] is None
    assert data["runtimeOptions"] == {"cache": False}
    assert "vars" not in data


@pytest.mark.parametrize("ext", ["yaml", "yml", "txt"])
async def test_yaml_output_sanitizes_runtime_options(tmp_path: Path, ext: str) -> None:
    ev = await make_eval(
        {}, runtime_options={"cache": False, "abortSignal": object(), "progressCallback": print}
    )
    path = tmp_path / f"output.{ext}"
    write_output(str(path), ev, None)
    parsed = load_yaml(path.read_text("utf-8"))
    assert parsed["runtimeOptions"] == {"cache": False}


async def test_yaml_output_redacts_secrets(tmp_path: Path) -> None:
    ev = await make_eval(
        {
            "env": {"ANTHROPIC_API_KEY": "anthropic-token", "AWS_REGION": "us-east-1"},
            "defaultTest": {"options": {"apiKey": "another-secret", "temperature": 0.1}},
        },
        make_result(),
    )
    path = tmp_path / "output.yaml"
    write_output(str(path), ev, None)
    text = path.read_text("utf-8")
    parsed = load_yaml(text)
    assert parsed["config"]["env"]["ANTHROPIC_API_KEY"] == "[REDACTED]"
    assert parsed["config"]["env"]["AWS_REGION"] == "us-east-1"
    assert parsed["config"]["defaultTest"]["options"]["apiKey"] == "[REDACTED]"
    assert parsed["shareableUrl"] is None
    assert text.startswith(f"evalId: {ev.id}\nresults:\n  version: 3\n")
    assert "anthropic-token" not in text


async def test_xml_output(tmp_path: Path) -> None:
    ev = await make_eval({"description": "x <&>"}, make_result())
    path = tmp_path / "output.xml"
    write_output(str(path), ev, None)
    text = path.read_text("utf-8")
    root = ET.fromstring(text)
    assert root.tag == "promptfoo"
    assert root.findtext("evalId") == ev.id
    assert root.findtext("shareableUrl") == ""  # empty element
    assert root.findtext("config/description") == "x <&>"
    assert root.findtext("results/results/success") == "true"
    assert "<gradingResult></gradingResult>" in text


async def test_promptfoo_xml_separate_from_junit(tmp_path: Path) -> None:
    ev = await make_eval({})
    write_output(str(tmp_path / "output.xml"), ev, None)
    write_output(str(tmp_path / "output.junit.xml"), ev, None)
    assert "<promptfoo>" in (tmp_path / "output.xml").read_text("utf-8")
    assert "<testsuites" in (tmp_path / "output.junit.xml").read_text("utf-8")


async def test_unsupported_format(tmp_path: Path) -> None:
    ev = await make_eval({})
    with pytest.raises(ValueError, match=r"Unsupported output file format foo\. Please use one of"):
        write_output(str(tmp_path / "out.foo"), ev, None)


async def test_google_sheets_not_supported() -> None:
    ev = await make_eval({})
    with pytest.raises(RuntimeError, match="Google Sheets"):
        write_output("https://docs.google.com/spreadsheets/d/abc", ev, None)


async def test_write_multiple_outputs_attempts_all_before_raising(tmp_path: Path) -> None:
    ev = await make_eval({}, make_result())
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    good = tmp_path / "output.txt"
    with pytest.raises(OutputWriteError, match="One or more output writes failed") as info:
        write_multiple_outputs([str(blocker / "output.json"), str(good)], ev, None)
    assert len(info.value.errors) == 1
    assert good.exists()


async def test_write_multiple_outputs_all_formats(tmp_path: Path) -> None:
    ev = await make_eval({}, make_result())
    names = ["o.json", "o.jsonl", "o.csv", "o.yaml", "o.txt", "o.html", "o.xml", "o.junit.xml"]
    write_multiple_outputs([str(tmp_path / n) for n in names], ev, None)
    assert all((tmp_path / n).stat().st_size > 0 for n in names)


# --- HTML -----------------------------------------------------------------------------------


async def test_html_template_escapes_user_content() -> None:
    from importlib import resources

    template = resources.files("promptphoo").joinpath("_data", "tableOutput.html").read_text()
    for needle in (
        "{{ config.description | default('Eval Output') | escape }}",
        "{{ header | escape }}",
        "{{ cell.text | escape }}",
        "{{ cell.reason | escape }}",
        "{{ cell.error | escape }}",
        "{{ cell.description | escape }}",
        'data-search="{{ cell.statusLabel | escape }}',
        'data-status="{{ cell.status | escape }}"',
        'data-variable-name="{{ cell.name | escape }}"',
    ):
        assert needle in template


async def test_html_report_summary_values(tmp_path: Path) -> None:
    ev = await make_eval({"description": "HTML facelift report"})
    await ev.append_prompts([{"raw": "Prompt", "label": "Prompt", "provider": "provider"}])
    ev.set_vars(["input"])
    base = {"provider": {"id": "provider"}, "prompt": {"raw": "Prompt", "label": "Prompt"}}
    await ev.append_result(
        make_result(
            **base,
            response={"output": "Passing output"},
            vars={"input": "one"},
            testCase={"vars": {"input": "one"}, "description": "Passing <test> description"},
            promptId="prompt",
            gradingResult={"pass": True, "score": 1, "reason": "Passing reason"},
        )
    )
    await ev.append_result(
        make_result(
            **base,
            success=False,
            failureReason=ASSERT,
            score=0,
            response={"output": "Failing output"},
            vars={"input": "two"},
            testIdx=1,
            testCase={"vars": {"input": "two"}},
            promptId="prompt",
            gradingResult={"pass": False, "score": 0, "reason": "Failing reason"},
        )
    )
    path = tmp_path / "output.html"
    write_output(str(path), ev, None)
    html = path.read_text("utf-8")
    for needle in (
        "HTML facelift report",
        '<p class="metric-label">Total Results</p>',
        '<p class="metric-value">2</p>',
        "50.0%",
        ">PASS<",
        ">FAIL<",
        "Score 1.00",
        "Score 0.00",
        "Passing output",
        "Failing output",
        "Passing &lt;test&gt; description",
        "Passing reason",
        "Failing reason",
        "View detail",
        "Result detail - row 1, prompt 1",
        "Variables",
        "data-report-search",
        "No rows match the current search and status filters.",
    ):
        assert needle in html, needle
    assert "Passing <test> description" not in html


async def test_html_classifies_non_error_failures_as_failures(tmp_path: Path) -> None:
    ev = await make_eval({"description": "HTML failure classification"})
    await ev.append_prompts([{"raw": "Prompt", "label": "Prompt", "provider": "provider"}])
    ev.set_vars(["input"])
    await ev.append_result(
        make_result(
            success=False,
            score=0,
            response={"output": ""},
            testCase={"vars": {"input": "missing-output"}},
            vars={"input": "missing-output"},
        )
    )
    path = tmp_path / "output.html"
    write_output(str(path), ev, None)
    html = path.read_text("utf-8")
    assert 'data-status="fail"' in html
    assert ">FAIL<" in html


async def test_html_runtime_errors_searchable(tmp_path: Path) -> None:
    ev = await make_eval({"description": "HTML error search"})
    await ev.append_prompts([{"raw": "Prompt", "label": "Prompt", "provider": "provider"}])
    ev.set_vars(["input"])
    await ev.append_result(
        make_result(
            success=False,
            failureReason=ERROR,
            score=0,
            response={"output": "Visible output despite failure"},
            error="provider timed out",
            vars={"input": "shared value"},
            testCase={"vars": {"input": "shared value"}},
            gradingResult={"pass": False, "score": 0, "reason": "Evaluation failed"},
        )
    )
    path = tmp_path / "output.html"
    write_output(str(path), ev, None)
    html = path.read_text("utf-8")
    assert 'data-search="ERROR 0.00 provider timed out Evaluation failed provider timed out"' in (
        html
    )
    assert 'data-variable-name="input"' in html


# --- JUnit ----------------------------------------------------------------------------------


async def test_junit_grouped_by_provider_and_prompt(tmp_path: Path) -> None:
    ev = await make_eval(
        {},
        make_result(
            latencyMs=1250,
            response={"output": "Hello Alice"},
            vars={"name": "Alice"},
            testCase={"description": "passing test"},
        ),
        make_result(
            success=False,
            failureReason=ASSERT,
            score=0.5,
            latencyMs=2500,
            response={"output": "SECRET MODEL OUTPUT"},
            vars={"name": "Bob"},
            testIdx=1,
            testCase={"description": "failing test"},
            gradingResult={
                "pass": False,
                "score": 0.5,
                "reason": "Expected output to contain Bob",
                "componentResults": [
                    {"pass": True, "score": 1, "assertion": {"type": "contains", "value": "Hello"}},
                    {"pass": False, "score": 0, "assertion": {"type": "contains", "value": "Bob"}},
                ],
            },
        ),
        make_result(
            success=False,
            failureReason=ERROR,
            score=0,
            latencyMs=500,
            response={"output": ""},
            error="provider request failed",
            vars={"name": "Carol"},
            testIdx=2,
        ),
    )
    path = tmp_path / "promptfoo.junit.xml"
    write_output(str(path), ev, None)
    xml = path.read_text("utf-8")
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>\n')
    root = junit(xml)
    assert root.attrib == {
        "errors": "1", "failures": "1", "name": "promptfoo", "skipped": "0", "tests": "3",
        "time": "4.25",
    }  # fmt: skip
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.attrib == {
        "errors": "1", "failures": "1", "name": "[echo] prompt 1", "skipped": "0", "tests": "3",
        "time": "4.25", "timestamp": "2025-05-03T15:00:00.000Z",
    }  # fmt: skip
    cases = suite.findall("testcase")
    assert [c.attrib for c in cases] == [
        {"name": "test 1: passing test", "time": "1.25", "classname": "[echo] prompt 1"},
        {"name": "test 2: failing test", "time": "2.5", "classname": "[echo] prompt 1"},
        {"name": "test 3", "time": "0.5", "classname": "[echo] prompt 1"},
    ]
    failure = cases[1].find("failure")
    assert failure is not None
    assert failure.text == "Score: 0.5\nReason: Assertion failed\nFailed assertions:\n- contains"
    assert failure.attrib == {"message": "Assertion failed", "type": "assertion"}
    error = cases[2].find("error")
    assert error is not None and error.text == "Reason: Evaluation error"
    for secret in ("SECRET MODEL OUTPUT", "Alice", "Carol", "provider request failed"):
        assert secret not in xml


def test_junit_legacy_results() -> None:
    legacy = make_result(
        success=False,
        failureReason=ASSERT,
        score=0,
        latencyMs=0,
        provider={"label": "legacy provider"},
        prompt={"raw": "Legacy raw prompt", "label": ""},
        promptId="legacy-prompt",
    )
    ev = FakeEval(created_at="not-a-date", _version=3, old_results={"results": [legacy]})
    suite = junit(create_junit_xml(ev)).find("testsuite")
    assert suite is not None
    assert suite.attrib["name"] == "[legacy provider] prompt 1"
    assert "timestamp" not in suite.attrib
    failure = suite.find("testcase/failure")
    assert failure is not None and failure.text == "Score: 0\nReason: Assertion failed"


def test_junit_legacy_prompts_without_ids_in_separate_suites() -> None:
    results = [
        make_result(promptIdx=i, prompt={"raw": raw, "label": ""}, promptId="")
        for i, raw in enumerate(["Prompt A", "Prompt B"])
    ]
    ev = FakeEval(_version=3, old_results={"results": results})
    names = [s.attrib["name"] for s in junit(create_junit_xml(ev)).findall("testsuite")]
    assert names == ["[echo] prompt 1", "[echo] prompt 2"]


async def test_junit_does_not_use_prompt_text_or_labels() -> None:
    ev = await make_eval(
        {}, make_result(prompt={"raw": "Sensitive system prompt", "label": "Sensitive label"})
    )
    xml = create_junit_xml(ev)
    assert "[echo] prompt 1" in xml
    assert "Sensitive system prompt" not in xml
    assert "Sensitive label" not in xml


def test_junit_persisted_results() -> None:
    persisted = make_result(
        success=False,
        failureReason=ERROR,
        score=0,
        latencyMs=float("nan"),
        provider={"id": "", "label": ""},
        gradingResult={"pass": False, "score": 0, "reason": "grader unavailable"},
        promptId="persisted-prompt",
    )
    ev = FakeEval(persisted=True, loaded=[persisted])
    suite = junit(create_junit_xml(ev)).find("testsuite")
    assert suite is not None
    assert suite.attrib["name"] == "[unknown provider] prompt 1"
    assert suite.attrib["time"] == "0"
    assert suite.attrib["timestamp"] == "2025-05-03T15:00:00.000Z"
    error = suite.find("testcase/error")
    assert error is not None and error.text == "Reason: Evaluation error"


async def test_junit_forbidden_characters() -> None:
    invalid = "\u0000\u0001\u0008\u000e\u001b\u001f\ud800\ufffe\udfff\uffff"
    ev = await make_eval(
        {},
        make_result(
            provider={"id": "echo", "label": f"target{invalid} & <model>"},
            testCase={"description": f"case{invalid} \U0001f680 \u4e2d\u6587"},
        ),
    )
    suite = junit(create_junit_xml(ev)).find("testsuite")
    assert suite is not None
    assert re.fullmatch(r"\[target & <model>\] prompt 1 \([a-f0-9]{16}\)", suite.attrib["name"])
    case = suite.find("testcase")
    assert case is not None and case.attrib["name"] == "test 1: case \U0001f680 \u4e2d\u6587"


async def test_junit_long_reasons_stay_compact() -> None:
    ev = await make_eval(
        {},
        make_result(
            success=False,
            failureReason=ASSERT,
            score=0,
            gradingResult={"pass": False, "score": 0, "reason": "x" * 600},
        ),
    )
    xml = create_junit_xml(ev)
    assert "x" * 600 not in xml
    assert "Failed assertions:\n- assertion" in xml


async def test_junit_removes_forbidden_characters_before_limits() -> None:
    cases = [
        ("\u0000", "\u0000", "unknown provider", "test 1"),
        (
            "\ud800" * 510 + "VISIBLE_PROVIDER",
            "\u0000" * 510 + "VISIBLE_TEST",
            "VISIBLE_PROVIDER",
            "test 2: VISIBLE_TEST",
        ),
        (
            "y" * 508 + "\U0001f680tailmore",
            "y" * 508 + "\U0001f680tailmore",
            "y" * 508 + "\U0001f680...",
            "test 3: " + "y" * 508 + "\U0001f680...",
        ),
    ]
    ev = await make_eval({})
    for idx, (label, description, _, _) in enumerate(cases):
        await ev.append_result(
            make_result(
                provider={"id": "echo", "label": label},
                testCase={"description": description},
                testIdx=idx,
            )
        )
    suites = junit(create_junit_xml(ev)).findall("testsuite")
    for suite, (_, _, provider, testcase) in zip(suites, cases, strict=True):
        assert f"[{provider}] prompt 1" in suite.attrib["name"]
        tc = suite.find("testcase")
        assert tc is not None and tc.attrib["name"] == testcase


async def test_junit_distinguishes_providers_with_matching_labels() -> None:
    ev = await make_eval(
        {},
        *(
            make_result(provider={"id": pid, "label": "shared"}, prompt={"raw": pid, "label": pid})
            for pid in ("first", "second")
        ),
    )
    names = [s.attrib["name"] for s in junit(create_junit_xml(ev)).findall("testsuite")]
    assert len(names) == 2 and len(set(names)) == 2
    assert all(re.fullmatch(r"\[shared\] prompt 1 \([a-f0-9]{16}\)", n) for n in names)


def test_junit_classnames_consistent_when_prompt_idx_varies() -> None:
    results = [
        make_result(promptIdx=0, testIdx=0, promptId="shared"),
        make_result(promptIdx=2, testIdx=1, promptId="shared"),
    ]
    ev = FakeEval(_version=3, old_results={"results": results})
    suite = junit(create_junit_xml(ev)).find("testsuite")
    assert suite is not None
    assert suite.attrib["name"] == "[echo] prompt 1"
    assert suite.attrib["tests"] == "2"
    assert {c.attrib["classname"] for c in suite.findall("testcase")} == {"[echo] prompt 1"}


@pytest.mark.parametrize(
    ("error", "secret"),
    [
        ("request failed: 500", "request failed: 500"),
        ('API error: 401\n{"code":"invalid_api_key"}', "invalid_api_key"),
    ],
)
async def test_junit_omits_raw_error_text(error: str, secret: str) -> None:
    ev = await make_eval({}, make_result(success=False, failureReason=ERROR, error=error))
    xml = create_junit_xml(ev)
    node = junit(xml).find("testsuite/testcase/error")
    assert node is not None and node.text == "Reason: Evaluation error"
    assert secret not in xml


# --- JSONL ----------------------------------------------------------------------------------


async def test_jsonl_one_sanitized_result_per_line(tmp_path: Path) -> None:
    ev = await make_eval(
        {},
        make_result(
            response={
                "output": "a",
                "metadata": {"http": {"headers": {"authorization": "Bearer secret-token"}}},
            }
        ),
        make_result(testIdx=1, response={"output": "b"}),
    )
    path = tmp_path / "out.jsonl"
    write_output(str(path), ev, None)
    text = path.read_text("utf-8")
    assert text.endswith(os.linesep)
    lines = text.splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert [r["testIdx"] for r in rows] == [0, 1]
    assert rows[1]["response"] == {"output": "b"}
    assert "secret-token" not in text
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


async def test_jsonl_strip_flags(tmp_path: Path) -> None:
    ev = await make_eval(
        {"env": {"PROMPTFOO_STRIP_RESPONSE_OUTPUT": "true", "PROMPTFOO_STRIP_TEST_VARS": "true"}},
        make_result(
            response={"output": "TOP SECRET"}, vars={"x": "v"}, testCase={"vars": {"x": "v"}}
        ),
    )
    path = tmp_path / "out.jsonl"
    write_output(str(path), ev, None)
    row = json.loads(path.read_text("utf-8"))
    assert row["response"]["output"] == "[output stripped]"
    assert row["vars"] == {}
    assert "TOP SECRET" not in path.read_text("utf-8")


async def test_jsonl_empty_eval(tmp_path: Path) -> None:
    ev = await make_eval({})
    path = tmp_path / "out.jsonl"
    write_output(str(path), ev, None)
    assert path.read_text("utf-8") == ""


async def test_jsonl_preserves_existing_permissions(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text("old\n")
    os.chmod(path, 0o640)
    ev = await make_eval({}, make_result())
    write_output(str(path), ev, None)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert "old" not in path.read_text("utf-8")


async def test_jsonl_preserves_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real.jsonl"
    target.write_text("old\n")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    ev = await make_eval({}, make_result())
    write_output(str(link), ev, None)
    assert link.is_symlink()
    assert json.loads(target.read_text("utf-8"))["testIdx"] == 0


async def test_jsonl_dangling_symlink_creates_target(tmp_path: Path) -> None:
    link = tmp_path / "link.jsonl"
    link.symlink_to(tmp_path / "nested" / "target.jsonl")
    ev = await make_eval({}, make_result())
    write_output(str(link), ev, None)
    assert link.is_symlink()
    assert (tmp_path / "nested" / "target.jsonl").read_text("utf-8").count("\n") == 1


# --- createOutputMetadata / tables ----------------------------------------------------------


def test_output_metadata_fields() -> None:
    meta = create_output_metadata(FakeEval(created_at=1746284400000, author="test@example.com"))
    assert meta["promptfooVersion"] == "0.125.1"
    assert meta["evaluationCreatedAt"] == "2025-05-03T15:00:00.000Z"
    assert meta["author"] == "test@example.com"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", meta["exportedAt"])
    assert meta["platform"] in ("linux", "darwin", "win32", "freebsd") or meta["platform"]
    assert meta["arch"]


@pytest.mark.parametrize("created_at", [None, 0, "not-a-date"])
def test_output_metadata_without_valid_created_at(created_at: Any) -> None:
    meta = create_output_metadata(FakeEval(created_at=created_at))
    assert "evaluationCreatedAt" not in meta
    assert "author" not in meta


async def test_get_table_from_eval_does_not_mutate_results() -> None:
    ev = await make_eval(
        {},
        make_result(
            vars={"q": "orig"},
            testCase={"vars": {"q": "orig"}},
            response={"output": "o", "prompt": "provider prompt"},
        ),
        vars=["q"],
        prompts=[{"raw": "p", "label": "p", "provider": "echo"}],
    )
    table = get_table(ev)
    assert table["head"]["vars"] == ["q"]
    assert table["body"][0]["vars"] == ["provider prompt"]
    assert ev.results[0].test_case["vars"] == {"q": "orig"}


def test_get_table_legacy() -> None:
    legacy_table = {"head": {"prompts": [], "vars": ["a"]}, "body": []}
    assert get_table(FakeEval(_version=3, old_results={"table": legacy_table})) == legacy_table
    assert get_table(FakeEval(_version=3, old_results={})) == {
        "head": {"prompts": [], "vars": []},
        "body": [],
    }
