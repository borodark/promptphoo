"""End-to-end: `promptfoo eval` with select-best graded by a Python provider, and max-score."""

from __future__ import annotations

import json

from click.testing import CliRunner

from promptphoo import db
from promptphoo.cli.main import build_cli
from promptphoo.models.eval import Eval

GRADER = """\
import json


def call_api(prompt, options, context):
    content = json.loads(prompt)[0]["content"]
    for block in content.split('<Text index="')[1:]:
        index, _, body = block.partition('">')
        if "Paris" in body.split("</Text>")[0]:
            return {"output": index, "tokenUsage": {"total": 5, "prompt": 4, "completion": 1}}
    return {"output": "none"}
"""

SELECT_BEST = """\
description: select-best e2e
prompts:
  - "Lyon is a city in {{country}}"
  - "Paris is the capital of {{country}}"
providers:
  - echo
tests:
  - vars: {country: France}
    assert:
      - type: contains
        value: "{{country}}"
      - type: select-best
        value: names the capital
        provider: file://grader.py
"""

MAX_SCORE = """\
description: max-score e2e
prompts:
  - "short"
  - "a much longer answer about Paris"
  - "Paris"
providers:
  - echo
tests:
  - assert:
      - type: contains
        value: Paris
      - type: python
        value: "len(output) > 10"
        metric: long
      - type: max-score
        value:
          weights: {python: 2}
"""


def _eval(tmp_path, monkeypatch, config: str, *extra: str):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "grader.py").write_text(GRADER)
    (tmp_path / "promptfooconfig.yaml").write_text(config)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-table", *extra]
    try:
        result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
        data = json.loads((tmp_path / "out.json").read_text())
    finally:
        db.close_db()
    return result, data


def _by_prompt(data):
    return sorted(data["results"]["results"], key=lambda r: r["promptIdx"])


def test_select_best_end_to_end_with_python_grader(tmp_path, monkeypatch):
    result, data = _eval(tmp_path, monkeypatch, SELECT_BEST)
    assert result.exit_code == 100, result.output
    rows = _by_prompt(data)
    assert [r["success"] for r in rows] == [False, True]
    assert rows[0]["failureReason"] == 1
    assert rows[0]["error"] == "Output not selected: names the capital"
    assert rows[0]["score"] == 0
    winner = rows[1]["gradingResult"]
    component = dict(winner["componentResults"][-1])
    tokens = component.pop("tokensUsed")
    assert component == {
        "pass": True,
        "score": 1,
        "reason": "Output selected as the best: names the capital",
        "assertion": {"type": "select-best", "value": "names the capital"},
    }
    assert (tokens["total"], tokens["prompt"], tokens["completion"]) == (5, 4, 1)
    assert rows[1]["tokenUsage"]["assertions"]["total"] == 5
    stats = data["results"]["stats"]
    assert (stats["successes"], stats["failures"], stats["errors"]) == (1, 1, 0)
    prompts = data["results"]["prompts"]
    assert [p["metrics"]["testPassCount"] for p in prompts] == [0, 1]
    assert [p["metrics"]["assertPassCount"] for p in prompts] == [1, 2]
    assert [p["metrics"]["assertFailCount"] for p in prompts] == [1, 0]
    assert [p["metrics"]["score"] for p in prompts] == [0, 1]
    # the stored grader reference is the provider path
    stored = rows[0]["testCase"]["assert"][1]
    assert stored["provider"] == "file://grader.py"

    # the persisted rows were updated in place
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    record = Eval.latest()
    assert record is not None
    rows_db = sorted(record.load_results(), key=lambda r: r.prompt_idx)
    assert [r.success for r in rows_db] == [False, True]
    assert rows_db[0].error == "Output not selected: names the capital"
    db.close_db()


def test_select_best_end_to_end_without_writing(tmp_path, monkeypatch):
    result, data = _eval(tmp_path, monkeypatch, SELECT_BEST, "--no-write")
    assert result.exit_code == 100, result.output
    assert [r["success"] for r in _by_prompt(data)] == [False, True]


def test_max_score_end_to_end(tmp_path, monkeypatch):
    result, data = _eval(tmp_path, monkeypatch, MAX_SCORE)
    assert result.exit_code == 100, result.output
    rows = _by_prompt(data)
    # "short": 0/3; the long Paris answer: (1 + 2) / 3 = 1; "Paris": 1/3
    assert [r["success"] for r in rows] == [False, True, False]
    assert rows[1]["gradingResult"]["componentResults"][-1]["reason"] == (
        "Selected as highest scoring output (score: 1.000)"
    )
    # "Paris" already failed, so its error and reason keep the earlier assertion's verdict
    assert rows[2]["error"] == "Python code returned false\nlen(output) > 10"
    assert rows[2]["gradingResult"]["componentResults"][-1]["reason"] == (
        "Not selected (score: 0.333, max: 1.000)"
    )
    assert rows[2]["gradingResult"]["namedScores"]["maxScore"] == 1 / 3
    assert rows[0]["error"] == "Python code returned false\nlen(output) > 10"
    assert rows[1]["namedScores"] == {"long": 1}
    stats = data["results"]["stats"]
    assert (stats["successes"], stats["failures"], stats["errors"]) == (1, 2, 0)
