"""End-to-end: `promptfoo eval` with llm-rubric assertions graded by the default grading
provider (openai gpt-6-sol, JSON mode) against a respx-mocked OpenAI API."""

from __future__ import annotations

import json

import pytest
import respx
from click.testing import CliRunner
from openai_helpers import CHAT_URL, ok, openai_env

from promptphoo.cli.main import build_cli
from promptphoo.prompts.grading import DEFAULT_GRADING_PROMPT

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

CONFIG = """\
description: llm-rubric e2e
prompts:
  - "Greet {{name}}"
providers:
  - openai:chat:gpt-4o-mini
tests:
  - vars: {name: Alice}
    assert:
      - type: llm-rubric
        value: Greets {{name}} politely
        metric: politeness
  - vars: {name: Bob}
    assert:
      - type: not-llm-rubric
        value: Is rude
      - type: llm-rubric
        value: Mentions the weather
        threshold: 0.8
"""


def _target_or_grader(request):
    body = json.loads(request.content)
    content = body["messages"][-1]["content"]
    if body["model"] == "gpt-4o-mini":
        name = content.removeprefix("Greet ")
        reply = f"Hello {name}, nice to meet you!"
        usage = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    else:
        assert body["model"] == "gpt-6-sol"
        assert body["response_format"] == {"type": "json_object"}
        if "Mentions the weather" in content:
            verdict = {"pass": True, "score": 0.5, "reason": "Barely mentions weather"}
        elif "Is rude" in content:
            verdict = {"pass": False, "score": 0, "reason": "Not rude at all"}
        else:
            verdict = {"pass": True, "score": 1, "reason": "Polite greeting"}
        reply = json.dumps(verdict)
        usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    return ok(
        {
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": usage,
        }
    )


@respx.mock
def test_eval_llm_rubric_with_default_grader(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    route = respx.post(CHAT_URL).mock(side_effect=_target_or_grader)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-write", "--no-table"]
    result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)

    bodies = [json.loads(c.request.content) for c in route.calls]
    assert sorted(b["model"] for b in bodies) == ["gpt-4o-mini"] * 2 + ["gpt-6-sol"] * 3
    assert result.exit_code == 100, result.output

    # The grader receives the rendered default rubric prompt as chat messages.
    alice = next(b for b in bodies if b["model"] == "gpt-6-sol" and "Greets Alice" in json.dumps(b))
    expected_messages = json.loads(DEFAULT_GRADING_PROMPT)
    assert alice["messages"][0] == expected_messages[0]
    assert alice["messages"][1]["role"] == "user"
    assert alice["messages"][1]["content"] == (
        "<Output>\nHello Alice, nice to meet you!\n</Output>\n"
        "<Rubric>\nGreets Alice politely\n</Rubric>"
    )

    data = json.loads((tmp_path / "out.json").read_text())
    rows = sorted(data["results"]["results"], key=lambda r: r["vars"]["name"])
    assert [r["success"] for r in rows] == [True, False]

    alice_grading = rows[0]["gradingResult"]
    assert alice_grading["namedScores"] == {"politeness": 1}
    component = alice_grading["componentResults"][0]
    assert component["reason"] == "Polite greeting"
    assert component["metadata"]["renderedGradingPrompt"].startswith('[{"role":"system"')
    assert component["tokensUsed"]["total"] == 60

    bob_components = rows[1]["gradingResult"]["componentResults"]
    assert [c["pass"] for c in bob_components] == [True, False]
    assert bob_components[0]["score"] == 1  # not-llm-rubric inverts the score
    assert bob_components[1]["score"] == 0.5
    assert bob_components[1]["reason"] == "Barely mentions weather"
