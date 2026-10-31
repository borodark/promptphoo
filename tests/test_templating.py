import json
from pathlib import Path

import pytest

from promptphoo import cli_state, templating

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "templating_cases.json").read_text())
GOLDEN = json.loads((FIXTURES / "templating_golden.json").read_text())
PAIRS = list(zip(CASES, GOLDEN, strict=True))


@pytest.mark.parametrize(("case", "expected"), PAIRS, ids=lambda x: "")
def test_matches_nunjucks(case, expected):
    if "error" in expected:
        with pytest.raises(Exception):  # noqa: B017
            templating.render_string(case["t"], case["c"])
    else:
        assert templating.render_string(case["t"], case["c"]) == expected["ok"]


def test_env_global_layers(monkeypatch):
    monkeypatch.setenv("PF_TEST_VAR", "from-process")
    assert templating.render_string("{{ env.PF_TEST_VAR }}") == "from-process"
    with cli_state.with_env({"PF_TEST_VAR": "from-suite"}):
        assert templating.render_string("{{ env.PF_TEST_VAR }}") == "from-suite"
    monkeypatch.setenv("PROMPTFOO_DISABLE_TEMPLATE_ENV_VARS", "true")
    assert templating.render_string("{{ env.PF_TEST_VAR }}") == ""


def test_disable_templating(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_DISABLE_TEMPLATING", "1")
    assert templating.render_string("{{ x }}", {"x": 1}) == "{{ x }}"
    assert templating.render_string("{{ x }}", {"x": 1}, is_grader=True) == "1"


def test_render_env_only_preserves_runtime_templates(monkeypatch):
    monkeypatch.setenv("PF_KEY", "secret")
    obj = {
        "a": "{{ env.PF_KEY }}-{{ vars.x }}",
        "b": ["{{ env.PF_UNSET }}", "{{ env.PF_UNSET | default('d') }}"],
        "_conversation": "{{ env.PF_KEY }}",
    }
    assert templating.render_env_only_in_object(obj) == {
        "a": "secret-{{ vars.x }}",
        "b": ["{{ env.PF_UNSET }}", "d"],
        "_conversation": "{{ env.PF_KEY }}",
    }


def test_extract_variables():
    t = "{{ a }} {{ b.c | upper }} {% if d %}{% endif %} {% for x in items %}{{ x }}{% endfor %}"
    assert sorted(templating.extract_variables_from_template(t)) == ["a", "b.c", "d", "items"]


def test_template_references_variable():
    assert templating.template_references_variable("{{ _conversation }}", "_conversation")
    assert not templating.template_references_variable("'_conversation'", "_conversation")
    assert not templating.template_references_variable(
        "{% set _conversation = 1 %}{{ _conversation }}", "_conversation"
    )


def test_python_filters(tmp_path):
    (tmp_path / "f.py").write_text("def shout(s):\n    return s.upper() + '!'\n")
    filters = templating.read_filters({"shout": "f.py"}, str(tmp_path))
    assert templating.render_string("{{ x | shout }}", {"x": "hi"}, filters=filters) == "HI!"
