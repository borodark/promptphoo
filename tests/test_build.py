import pytest

from promptphoo import cli_state
from promptphoo.evaluator.build import build_tests_from_suite, generate_var_combinations


def test_var_combinations_cartesian():
    combos = generate_var_combinations({"a": ["x", "y"], "b": "z", "c": [1, 2], "d": []})
    assert combos == [
        {"a": "x", "b": "z", "c": [1, 2], "d": []},
        {"a": "y", "b": "z", "c": [1, 2], "d": []},
    ]


def test_var_combinations_file_glob(tmp_path):
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    with cli_state.with_base_path(str(tmp_path)):
        combos = generate_var_combinations({"doc": "file://*.txt"})
        assert combos == [{"doc": "file://a.txt"}, {"doc": "file://b.txt"}]
        with pytest.raises(FileNotFoundError):
            generate_var_combinations({"doc": "file://missing.txt"})


def test_scenarios_merge_with_default_test():
    suite = {
        "tests": [],
        "defaultTest": {"vars": {"d": 1}, "options": {"o": 1}, "metadata": {"m": 1}},
        "scenarios": [
            {
                "config": [{"vars": {"lang": "en"}}, {"vars": {"lang": "fr"}}],
                "tests": [{"vars": {"q": "hi"}, "assert": [{"type": "contains", "value": "h"}]}],
            }
        ],
    }
    tests = build_tests_from_suite(suite)
    assert [t["vars"] for t in tests] == [
        {"d": 1, "lang": "en", "q": "hi"},
        {"d": 1, "lang": "fr", "q": "hi"},
    ]
    assert tests[1]["metadata"] == {"m": 1, "conversationId": "__scenario_1__"}
    assert build_tests_from_suite({"tests": []}) == [{}]
