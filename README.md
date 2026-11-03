<div align="center">

# THE PROMPTPHOO

**Vol. 0 · No. 1 · Alpha Edition · Detroit, October 31, 2026 · Free of charge (MIT)**

*All the evals that are fit to print*

**This is not promptfoo.** It is an unofficial Python port, not affiliated with or supported by the promptfoo team.
Report problems here, not to the promptfoo project.

</div>

---

## PYTHON PORT OF PROMPTFOO REACHES ALPHA; NODE AND PYTHON AGREE ON EVERY GRADE

### Same configs, same flags, same database — and, in a fully local test, the same scores to the third decimal

*By I.O.*

**DETROIT** — promptphoo, a Python 3 port of the [promptfoo](https://github.com/promptfoo/promptfoo) eval core, is now available as an alpha. It reads the same `promptfooconfig.yaml`, accepts the same `eval` flags, and writes to the same `~/.promptfoo` database as promptfoo 0.125.1, so an evaluation can start in Node and finish in Python, or the other way round.

In a test run entirely on one graphics card, both runtimes evaluated two Ollama models with a local grader and local embeddings. The model outputs were identical, and so was every one of the 18 assertion scores, from `llm-rubric` and `g-eval` to embedding similarities such as 0.819. On a repeat run, 17 of 18 matched; the eighteenth differed only because one model worded its answer differently that time.

Against the live Anthropic API, every one of the 69 requests promptphoo sent matched a request Node sent for the same tests, headers included, and both runtimes reached the same verdicts on nine configurations, from tool use and extended thinking to a bad key. The whole live test cost about a dollar. *Full results: the [verification report](docs/VERIFICATION.md).*

The port follows its original closely and makes no claim to improve on it. Where it departs, the departure is recorded. *Editorial: the [Translator's Foreword](#editorial-translators-foreword), below.*

---

| **AT A GLANCE** | |
|---|---|
| Tracks upstream | promptfoo **0.125.1** (MIT) |
| Python | 3.11+ |
| Tests | about 6,500 passing, including Node-interop suites that run when `NODE_PROMPTFOO` is set |
| Node parity | 63 upstream smoke fixtures byte-identical after normalizing ids, timestamps and latency |
| Shared state | the same SQLite schema and drizzle migrations; Python and Node read each other's evals |
| Live check | Anthropic: 69 of 69 requests identical to Node's on the wire; same verdicts on 9 live configs ([report](docs/VERIFICATION.md)) |
| Not yet checked | Python 3.11 and 3.13, macOS, Windows, live OpenAI/Azure/Google/Bedrock (mocked tests only) |
| Status | **alpha**: expect rough edges, and please report them |

**IN THIS EDITION** · [Getting started](#getting-started) · [Migrating from promptfoo](docs/MIGRATING.md) · [What's covered](#whats-covered) · [What's not](#not-in-this-edition) · [Corrections & departures](#corrections-and-departures) · [Advisory](#advisory) · [Editorial](#editorial-translators-foreword) · [The back page](#the-back-page)

---

## Getting started

There is no PyPI release yet, so install from a clone:

```bash
git clone <this repository> promptphoo && cd promptphoo
python3 -m venv .venv && .venv/bin/pip install .
.venv/bin/promptphoo init --no-interactive      # writes promptfooconfig.yaml + README.md
.venv/bin/promptphoo eval
```

The package installs two commands, `promptphoo` and its short form `ph`. It does not install `promptfoo` or `pf`, so it never shadows Node promptfoo, and you always know which tool you are running. Moving an existing promptfoo project over? Read the [migration guide](docs/MIGRATING.md).

### From Python

The same engine is a library. `evaluate` takes the mapping a `promptfooconfig.yaml` would hold
and returns the eval record; nothing is persisted unless the suite says `writeLatestResults: true`.

```python
from promptphoo import evaluate

record = evaluate(
    {
        "prompts": ["{{question}}"],
        "providers": ["openai:gpt-4.1-mini"],            # or file://my_provider.py
        "tests": [
            {"vars": {"question": "What is 2+2?"},
             "assert": [{"type": "llm-rubric", "value": "Says four"}]},
        ],
    },
    {"cache": False, "maxConcurrency": 4,
     "progressCallback": lambda done, total, idx, step, metrics: print(f"{done}/{total}")},
)
summary = record.to_evaluate_summary()        # the `results` object of a results.json
print(summary["stats"], summary["results"][0]["gradingResult"]["reason"])
```

`file://` providers, test generators, `python` assertions and extension hooks run in the calling
process. State is scoped to the call: nothing is read from `.env` or written to `os.environ`, and
errors are exceptions, never exit codes. Cancel with `AbortController` (`abortSignal`); the partial
record comes back and `signal.aborted` says why. Inside a running event loop await
`evaluate_async`, or use `evaluate_in_thread` from a Flask view or a worker. The full surface is
the docstring of `promptphoo.api` (D79).

A wheel for an internal index or a `requirements.txt` comes from `python -m build --wheel`
(`dist/promptphoo-<version>-py3-none-any.whl`); `pip install promptphoo @ git+<repo>@<ref>` works too.

**No API key?** `examples/local-ollama/` runs the whole path locally: models under test, a model-graded rubric and embeddings. Pull the models it lists with `ollama pull` and run `promptphoo eval -c examples/local-ollama/promptfooconfig.yaml`.

## What's covered

- **Library:** `promptphoo.evaluate(suite, options)` and `evaluate_async`, upstream's `promptfoo.evaluate()`; `load_api_provider(s)`, `AbortController`, `Eval` and the error types at the package root.
- **Commands** (as `promptphoo <command>` or `ph <command>`): `eval`, `init`, `validate` (config and target), `list`, `show`, `delete`, `export` (evals and logs), `import`, `retry`, `cache clear`, `config`, `logs`, `debug`, `feedback`, `generate dataset`, `generate assertions`. `share` is registered but always fails; results stay local.
- **Eval flags:** the full set, including filters, `--repeat`, `--resume`, `--retry-errors`, `--filter-failing`, `--watch`, `-x` extension hooks (Python), `--grader` and `--suggest-prompts`. `--share` only warns, and `--remote` is accepted and ignored.
- **Prompts and tests:** text, chat JSON, YAML, Markdown, Jinja and CSV prompts; Python prompt functions; test files in YAML, JSON, JSONL and CSV, plus Python `generate_tests`; `defaultTest`; scenarios; `$ref`s.
- **Providers:**
  - OpenAI (chat, completion, Responses, embeddings, moderation), Anthropic, Azure OpenAI, Google AI Studio and Vertex, Bedrock (InvokeModel, Converse, Messages and embeddings), Ollama, HTTP.
  - Around forty OpenAI-compatible families (DeepSeek, Groq, OpenRouter, xAI, Perplexity, LiteLLM, llama.cpp and more).
  - `echo`, `sequence`, `exec:`, `webhook:` and Python (`file://provider.py`).
- **Assertions:**
  - Deterministic: string, regex, JSON, XML, HTML, SQL, text-similarity metrics (levenshtein, rouge, bleu, gleu, meteor), cost and latency, finish reason, perplexity, refusals, word count, guardrails, tool calls, skill use, trajectories, plus `python` and `webhook`.
  - Model-graded: `llm-rubric`, `factuality`, `model-graded-factuality`, `model-graded-closedqa`, `g-eval`, `search-rubric`, `answer-relevance`, `context-*`, `conversation-relevance`, `similar`, `classifier`, `moderation`, `agent-rubric` (with an explicitly configured grader), `trajectory:goal-success`, `select-best`, `max-score`.
  - Trace-based: `trace-span-count`, `trace-span-duration`, `trace-error-spans`.
- **Output formats:** console table, JSON, JSONL, CSV, YAML, TXT, HTML, XML and JUnit, matching Node's output byte for byte.
- **Media blobs and tracing:** blobs stored in Node's layout; an OTLP receiver (JSON and protobuf) and promptfoo's own spans.

## Not in this edition

- The web viewer and `promptfoo view`.
- Red-teaming, code scanning, the optimizer, MCP, model audit.
- Cloud features such as `share` and remote grading.
- **Hooks in JavaScript, TypeScript or Ruby.** Use `file://*.py` hooks, which keep upstream's signatures: `call_api`, `get_assert`, `get_transform`, and so on. A JS hook fails with a clear message rather than silently.
- Mistral and AI21 providers, which upstream implements with their own request code. They are planned.
- Other providers: HuggingFace, Replicate, Cohere, Voyage, watsonx, SageMaker, Bedrock agents and knowledge bases, Claude on Vertex, WebSocket and browser targets, and the OpenAI and Google image, audio, video and live routes.
- Test cases from Excel, Google Sheets, Hugging Face datasets, Azure Blob or SharePoint; `derivedMetrics`; streaming responses.
- The `pi` assertion, which needs Promptfoo's remote scoring API.
- The full inventory, item by item against upstream, is in [`docs/SCOPE.md`](docs/SCOPE.md).

## Corrections and departures

Every place where promptphoo deliberately behaves differently from Node is logged, one numbered decision per choice, in [`docs/DECISIONS.md`](docs/DECISIONS.md). Examples include in-house replacements for the OpenTelemetry SDK and for protobufjs, a separate response cache file (`cache-py.json`), and red-team-only branches that grade locally. The porting plan is in [`docs/PLAN_PORT.md`](docs/PLAN_PORT.md).

**Where you see a difference from Node that isn't logged there, that's a bug. Please file it.**

## Advisory

- **Safer defaults than upstream** (decision D61): TLS certificates are verified unless `PROMPTFOO_INSECURE_SSL=true`; templates are sandboxed; `promptphoo feedback` sends nothing anywhere. See [SECURITY.md](SECURITY.md).
- **Hooks are code.** Python hooks, `exec:` providers and transforms run with your user's permissions, exactly as they do in Node. Only evaluate configs you trust.
- **Test data is templated, but sandboxed.** Prompts and var values are rendered as templates, as upstream does. The sandbox stops them reaching Python internals, but they can still read environment variables through `{{ env.X }}` and send them to a provider. `PROMPTFOO_DISABLE_TEMPLATING=true` turns templating off.
- **Tracing's OTLP receiver** listens on localhost only while an eval with tracing enabled is running.

---

## Editorial: Translator's Foreword

Some tools change what a team believes it can check. Before them, a prompt was something one tweaked, read a few outputs from, and hoped about. After them, a prompt was something one tested. promptfoo is one of those tools, and this repository is an attempt to say so in another language.

It began modestly, as the good ones do. The first commit landed on the last day of March 2023 — one day short of April Fools', which is either a coincidence or the first test case. It held a README, an evaluator of fifty-seven lines, and a module called `apiCaller.ts`, and it described itself as "a CLI tool that facilitates prompt editing and evaluation." Its author, Ian Webster, had been leading LLM engineering at Discord, home of the chatbot Clyde, where AI products reached two hundred million users. It is hard to change a prompt for two hundred million people by squinting at a few replies. Somebody had to write the test runner.

Michael D'Angelo, who had scaled machine learning to a hundred million people at Smile Identity, joined as co-founder and, by the plain count of the history, became its most prolific committer. Around them gathered nearly four hundred contributors, and between them they gave evaluation the shape of a test suite: prompts, providers and cases written down in a file; assertions that pass or fail; results that can be stored, compared and run again tomorrow. None of this was obvious when they began. It looks obvious now, which is what good design looks like afterwards.

The code keeps its sense of humour in the examples, where models are asked, with complete seriousness, to tweet about bananas, and in one case to write about "time-traveling characters who only eat historical versions of bananas." It keeps its realism elsewhere. Somewhere in the source is a list that begins `'I am not able to', 'I am really sorry', 'I am sorry'` — the collected ways a language model declines. It is the most honest data structure in the field. It crossed into Python intact, apologies and all.

promptphoo is a port, and it means to be one. The name is pronounced exactly like the original, which is the point: the same sound, a different spelling. Where Python lets hooks and providers run inside the evaluator's own process, that is the only freedom taken. Everything else is borrowed, and every difference that remains is written down, because a port that hides its differences is worse than none.

In March 2026 promptfoo agreed to join OpenAI, and its authors wrote that it would remain open source and MIT-licensed. That sentence is the reason this translation can exist at all.

If promptfoo serves your needs, use promptfoo. It is the original, and it is maintained by the people who understand it best. promptphoo exists for the narrower case where Python is the place one has to stand. The original was worth translating, its authors deserve the credit for what works here, and whatever errors the translation contains are mine alone.

*— I.O., Detroit, 2026. The full foreword, including the debts to RAGAS, DeepEval, Nunjucks, drizzle and Ollama, is in [FOREWORD.md](FOREWORD.md).*

---

## The back page

**For contributors:**

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,all]'
.venv/bin/pytest -q -n auto && .venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src tests tools && .venv/bin/pyright
NODE_PROMPTFOO="node /path/to/promptfoo/dist/src/main.js" .venv/bin/python tools/parity.py --all
```

- `tools/parity.py` runs upstream's smoke fixtures through both CLIs and diffs the normalized JSON. Its outputs land in `.parity/`.
- `tools/live/run_anthropic.sh` repeats the live Anthropic comparison (it spends about $1 of API credit). What was verified, how, and what wasn't is in [`docs/VERIFICATION.md`](docs/VERIFICATION.md).
- With `NODE_PROMPTFOO` set, the interop suites also run. They check that each CLI reads the other's databases, exports, imports, blobs and traces.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) has the porting rule, the decision-log format and the commit conventions.
- `src/promptphoo/_data/` holds upstream assets that are vendored unchanged, or generated from upstream by Node: migrations, the config schema, grading prompts and init templates. The scripts under `tools/` regenerate them.

**Credits.** promptfoo is © Promptfoo and its contributors, released under the MIT license. promptphoo is an independent port. It is not affiliated with or endorsed by Promptfoo or OpenAI.
