"""The library entry point (promptphoo.evaluate; ports src/evaluate.ts)."""

import asyncio
import json
import os
import threading

import pytest

import promptphoo
from promptphoo import cli_state, db, envars
from promptphoo.api import evaluate, evaluate_async, evaluate_in_thread
from promptphoo.models.eval import Eval
from promptphoo.scheduler.cancellation import AbortController, AbortError


def _suite(tmp_path, **extra):
    return {
        "basePath": str(tmp_path),
        "prompts": ["say {{word}}"],
        "providers": ["echo"],
        "tests": [
            {"vars": {"word": "yes"}, "assert": [{"type": "contains", "value": "yes"}]},
            {"vars": {"word": "no"}, "assert": [{"type": "contains", "value": "yes"}]},
        ],
        **extra,
    }


def test_root_exports_resolve_lazily():
    assert promptphoo.evaluate is evaluate
    assert promptphoo.AbortController is AbortController
    assert promptphoo.Eval is Eval
    assert "evaluate_async" in dir(promptphoo)
    with pytest.raises(AttributeError):
        promptphoo.not_a_thing  # noqa: B018


def test_evaluate_runs_in_process_and_returns_the_record(tmp_path):
    record = evaluate(_suite(tmp_path), {"cache": False})
    assert isinstance(record, Eval) and not record.persisted
    summary = record.to_evaluate_summary()
    assert (summary["stats"]["successes"], summary["stats"]["failures"]) == (1, 2 - 1)
    by_word = {r["vars"]["word"]: r for r in summary["results"]}
    assert by_word["yes"]["success"] and not by_word["no"]["success"]
    assert by_word["no"]["gradingResult"]["componentResults"][0]["assertion"]["type"] == "contains"
    db.run_db_migrations()
    assert Eval.latest() is None  # nothing written to the database by default


def test_file_provider_generator_assertion_and_hook_run_in_this_process(tmp_path):
    (tmp_path / "provider.py").write_text(
        "import os\n"
        "def call_api(prompt, options, context):\n"
        "    return {'output': f\"{prompt}|{options['config']['tag']}|{os.getpid()}\","
        " 'metadata': {'intent': 'x'}}\n"
    )
    (tmp_path / "tests.py").write_text(
        "def make(config):\n"
        "    return [{'vars': {'word': w}, 'assert': [{'type': 'python', 'value': 'file://check.py',"
        " 'metric': 'm::' + w}]} for w in config['words']]\n"
    )
    (tmp_path / "check.py").write_text(
        "def get_assert(output, context):\n"
        "    return {'pass': True, 'score': 0.5, 'reason': context['vars']['word']}\n"
    )
    (tmp_path / "hooks.py").write_text(
        "import json, os\n"
        "def afterEach(context, hook=None):\n"
        "    r = context['result']\n"
        "    with open(os.environ['HOOK_OUT'], 'a') as f:\n"
        "        f.write(json.dumps({'word': r['vars']['word'], 'score': r['score'],"
        " 'intent': r['response']['metadata']['intent'], 'pid': os.getpid()}) + '\\n')\n"
        "    return context\n"
    )
    hook_out = tmp_path / "hook.jsonl"
    os.environ["HOOK_OUT"] = str(hook_out)
    try:
        record = evaluate(
            {
                "basePath": str(tmp_path),
                "prompts": ["{{word}}"],
                "providers": [{"id": "file://provider.py", "config": {"tag": "T"}}],
                "tests": [{"path": "file://tests.py:make", "config": {"words": ["a", "b"]}}],
                "extensions": ["file://hooks.py:afterEach"],
            },
            {"cache": False, "maxConcurrency": 2},
        )
    finally:
        del os.environ["HOOK_OUT"]
    results = record.to_evaluate_summary()["results"]
    outputs = {r["vars"]["word"]: r["response"]["output"] for r in results}
    assert outputs["a"].startswith("a|T|") and outputs["a"].endswith(str(os.getpid()))
    assert {r["gradingResult"]["componentResults"][0]["assertion"]["metric"] for r in results} == {
        "m::a",
        "m::b",
    }
    hooks = [json.loads(line) for line in hook_out.read_text().splitlines()]
    assert sorted((h["word"], h["score"], h["intent"]) for h in hooks) == [
        ("a", 0.5, "x"),
        ("b", 0.5, "x"),
    ]
    assert {h["pid"] for h in hooks} == {os.getpid()}


def test_progress_callback_and_output_path(tmp_path):
    calls = []
    out = tmp_path / "out" / "results.json"
    record = evaluate(
        _suite(tmp_path, outputPath=str(out)),
        {
            "cache": False,
            "progressCallback": lambda done, total, idx, step, metrics: calls.append((done, total)),
        },
    )
    assert sorted(calls) == [(1, 2), (2, 2)]
    written = json.loads(out.read_text())
    assert written["evalId"] == record.id and len(written["results"]["results"]) == 2
    assert written["results"]["stats"]["successes"] == 1


def test_suite_evaluate_options_apply_under_explicit_options(tmp_path):
    seen = {}
    (tmp_path / "p.py").write_text(
        "import threading, time\n"
        "live = set()\n"
        "def call_api(prompt, options, context):\n"
        "    live.add(threading.get_ident()); time.sleep(0.05)\n"
        "    return {'output': str(len(live))}\n"
    )
    suite = {
        "basePath": str(tmp_path),
        "prompts": ["{{w}}"],
        "providers": ["file://p.py"],
        "tests": [{"vars": {"w": str(i)}} for i in range(4)],
        "evaluateOptions": {"repeat": 2, "maxConcurrency": 1},
    }
    record = evaluate(suite, {"cache": False})
    assert len(record.to_evaluate_summary()["results"]) == 8  # repeat from the suite
    record = evaluate(suite, {"cache": False, "repeat": 1})
    assert len(record.to_evaluate_summary()["results"]) == 4  # explicit option wins
    assert seen == {}


def test_env_is_scoped_to_the_call_and_never_written_to_the_process(tmp_path):
    (tmp_path / "p.py").write_text(
        "from promptphoo import envars\n"
        "def call_api(prompt, options, context):\n"
        "    return {'output': envars.get_env_string('EVAL_API_TOKEN') or 'unset'}\n"
    )
    suite = {
        "basePath": str(tmp_path),
        "prompts": ["x"],
        "providers": ["file://p.py"],
        "tests": [{"vars": {"x": "1"}}],
        "env": {"EVAL_API_TOKEN": "from-suite"},
    }
    record = evaluate(suite, {"cache": False})
    assert record.to_evaluate_summary()["results"][0]["response"]["output"] == "from-suite"
    assert "EVAL_API_TOKEN" not in os.environ
    assert cli_state.env() in (None, {}) and cli_state.config() is None
    assert envars.get_env_string("EVAL_API_TOKEN") is None


def test_write_latest_results_persists_to_the_shared_database(tmp_path):
    record = evaluate(
        _suite(tmp_path, writeLatestResults=True, description="lib run"), {"cache": False}
    )
    assert record.persisted
    latest = Eval.latest()
    assert latest is not None and latest.id == record.id
    assert latest.config["description"] == "lib run"
    assert len(latest.to_evaluate_summary()["results"]) == 2


def test_abort_signal_stops_the_run(tmp_path):
    (tmp_path / "slow.py").write_text(
        "import time\n"
        "def call_api(prompt, options, context):\n"
        "    time.sleep(0.2)\n"
        "    return {'output': prompt}\n"
    )
    controller = AbortController()
    suite = {
        "basePath": str(tmp_path),
        "prompts": ["{{i}}"],
        "providers": ["file://slow.py"],
        "tests": [{"vars": {"i": str(i)}} for i in range(20)],
    }
    threading.Timer(0.3, controller.abort).start()
    record = evaluate(
        suite, {"cache": False, "maxConcurrency": 1, "abortSignal": controller.signal}
    )
    # As upstream: the run stops early and the partial record comes back; the signal says why.
    assert controller.signal.aborted
    assert 0 < len(record.to_evaluate_summary()["results"]) < 20
    already = AbortController()
    already.abort()
    with pytest.raises(AbortError):
        evaluate(suite, {"cache": False, "abortSignal": already.signal})


def test_sync_entry_refuses_a_running_loop_and_thread_entry_works_there(tmp_path):
    async def inside_loop():
        with pytest.raises(RuntimeError, match="evaluate_async"):
            evaluate(_suite(tmp_path), {"cache": False})
        record = evaluate_in_thread(_suite(tmp_path), {"cache": False})
        direct = await evaluate_async(_suite(tmp_path), {"cache": False})
        return record, direct

    record, direct = asyncio.run(inside_loop())
    assert record.to_evaluate_summary()["stats"]["successes"] == 1
    assert direct.to_evaluate_summary()["stats"]["successes"] == 1


def test_errors_are_exceptions_not_exit_codes(tmp_path):
    with pytest.raises(TypeError):
        evaluate("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(Exception) as err:
        evaluate(
            {
                "basePath": str(tmp_path),
                "prompts": ["x"],
                "providers": ["nonexistent:provider:x"],
                "tests": [{"vars": {"x": "1"}}],
            }
        )
    assert "nonexistent" in str(err.value)
