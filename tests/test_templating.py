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


# --- sandbox (D61) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "{{ ''.__class__.__mro__[1].__subclasses__() }}",
        "{{ (1).__class__.__base__.__subclasses__() }}",
        "{{ x.upper.__globals__ }}",
    ],
)
def test_templates_cannot_reach_python_internals(template):
    from jinja2.exceptions import SecurityError

    try:
        out = templating.render_string(template, {"x": "s"})
    except SecurityError:
        return
    assert "class" not in out and "function" not in out and out in ("", "undefined")


def test_var_values_rendered_as_templates_are_sandboxed():
    # Var values are themselves rendered as templates (as upstream does), so a dataset row is
    # template code; the sandbox is what keeps an untrusted CSV from running Python.
    import asyncio

    from jinja2.exceptions import SecurityError

    from promptphoo.evaluator.render import render_prompt

    prompt = {"raw": "Q: {{ q }}", "label": "p"}
    assert asyncio.run(render_prompt(prompt, {"q": "{{ 7 * 6 }}"})) == "Q: 42"
    # Unsandboxed, this renders "Q: 399" (the number of classes reachable from str). Sandboxed,
    # the unsafe attribute is undefined, or raises when the undefined is used.
    hostile = "{{ ''.__class__.__mro__[1].__subclasses__()|length }}"
    try:
        out = asyncio.run(render_prompt(prompt, {"q": hostile}))
    except SecurityError:
        return
    assert out == "Q: "


def test_engine_compiles_each_template_once(monkeypatch):
    engine = templating.Engine(templating.NunjucksCompatEnvironment(), is_grader=False)
    calls = []
    original = engine.env.from_string
    monkeypatch.setattr(engine.env, "from_string", lambda t: calls.append(t) or original(t))
    rendered = [engine.render_string("Hi {{ n }}", {"n": i}) for i in range(3)]
    assert rendered == ["Hi 0", "Hi 1", "Hi 2"]
    assert calls == ["Hi {{ n }}"]


def test_engine_compiled_cache_is_bounded(monkeypatch):
    engine = templating.Engine(templating.NunjucksCompatEnvironment(), is_grader=False)
    monkeypatch.setattr(templating.Engine, "MAX_COMPILED", 4)
    for i in range(10):
        assert engine.render_string(f"{i} {{{{ x }}}}", {"x": "y"}) == f"{i} y"
    assert len(engine._compiled) <= 4


def test_template_errors_are_not_cached():
    engine = templating.Engine(templating.NunjucksCompatEnvironment(), is_grader=False)
    for _ in range(2):
        with pytest.raises(Exception, match="unexpected"):
            engine.render_string("{{ x ")
    assert engine._compiled == {}
