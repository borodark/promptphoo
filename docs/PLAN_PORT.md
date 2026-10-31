<!-- Snapshot of the approved plan (~/.claude/plans/reactive-sprouting-tower.md, 2026-10-30).
     Phases and layout live here; the reasons behind choices live in DECISIONS.md. -->

# Plan: Port promptfoo eval core to Python 3 (`promptphoo`)

## Context
`../promptfoo` (v0.125.1) is a ~250k-line TypeScript LLM-eval toolkit plus a 200k-line React UI. We want a Python 3 implementation in `/home/io/projects/promptphoo` (currently empty, not a git repo).

Decisions already made: scope D1, drop-in compatibility D2, Python-only hooks D3, Jinja2 for Nunjucks D4 (see [DECISIONS.md](DECISIONS.md)).

Success means two things:
1. The deterministic smoke fixtures in `../promptfoo/test/smoke/fixtures/configs/` give the same normalized JSON output under Node and Python.
2. Evals written by Python appear correctly in Node's `promptfoo list` and `promptfoo show`, and the reverse also works.

## Tech stack
- Python ≥3.11. Package managed with `uv`, src layout.
- CLI: `click`. A custom option class supports commander-style variadic options (`-c a.yaml b.yaml`). Console scripts are `promptfoo`, `pf` and `promptphoo`.
- Schemas: plain camelCase dicts validated against the vendored JSON schema (D7; the original plan named pydantic, see D6).
- Async: `asyncio` with `httpx.AsyncClient`. Concurrency comes from a semaphore, standing in for `async.forEachOfLimit`.
- Templating: `jinja2` through a `NunjucksCompatEnvironment` (see below).
- Other libraries: `pyyaml`, `jsonschema`, `rapidfuzz`, `rich` (table and progress), `sqlite3` from the stdlib, `python-dotenv`.
- Optional extras: `nltk` (meteor), `sqlglot` (is-sql), `boto3` (bedrock), `google-auth` (vertex), `pypdf`.
- Tooling: `ruff`, `pyright`, `pytest`, `pytest-asyncio`, `respx` (HTTP mocking).

## Package layout (`src/promptphoo/`)
Each module mirrors its TypeScript source:

| Module | Ports from | Contents |
|---|---|---|
| `cli/` | `src/main.ts`, `src/commands/*.ts`, `src/node/doEval.ts` | `main.py` plus one module per command |
| `types/` | `src/types/index.ts`, `src/types/providers.ts`, `src/contracts/providers.ts` | pydantic models |
| `config/` | `src/util/config/load.ts` | `$ref` dereferencing, env-template rendering, combining multiple configs, defaultTest and scenarios |
| `files.py` | `src/util/file.ts`, `src/util/fileReference.ts` | file:// loading, globs |
| `testcases/` | `src/util/testCaseReader.ts`, `src/csv.ts`, `src/googleSheets.ts` | test loaders, including `__expected` columns |
| `prompts/` | `src/prompts/` | prompt processors and verbatim grading templates (`grading.py` from `src/prompts/grading.ts`, `ragas.py` from `src/external/prompts/ragas.ts`) |
| `templating.py` | `src/util/templates.ts`, `src/util/render.ts` | Jinja2 shim |
| `evaluator/` | `src/evaluator.ts`, `src/evaluatorHelpers.ts` | `build.py` (var cartesian product, scenarios, run-option fan-out), `run.py` (runEval), `render.py` (`renderPrompt` and var loading), `stats.py`, `comparison.py` (select-best, max-score) |
| `scheduler/` | `src/scheduler/` | retry policy, rate-limit registry, header parser |
| `providers/` | `src/providers/` | see Phase 3 |
| `assertions/` | `src/assertions/` | one module per handler, a dispatcher (`index.py`) and `result.py` (from `assertionsResult.ts`) |
| `matchers/` | `src/matchers/` | llm-rubric, factuality, closed-qa, g-eval, similarity, RAG, `get_grading_provider` |
| `python_hooks.py` | `src/python/` | Python hook runner (see below) |
| `db/` | `src/database/`, `src/migrate.ts` | `connection.py` (WAL pragmas, `evalLastWritten` signal), `migrate.py` (drizzle-compatible), `tables.py` |
| `models/` | `src/models/eval.ts`, `src/models/evalResult.ts`, `src/models/prompt.ts` | eval, result and prompt models |
| `cache.py` | `src/cache.ts` | response cache |
| `output/` | `src/table.ts`, `src/util/output.ts` | console table, summary, and writers for json, csv, yaml, txt, html, jsonl, xml and junit |
| `globalconfig.py` | `src/globalConfig/` | `promptfoo.yaml` |
| `envars.py` | `src/envars.ts` | typed `PROMPTFOO_*` accessors |

`drizzle/*.sql`, `drizzle/meta/_journal.json`, `src/tableOutput.html` and `src/python/wrapper.py` / `persistent_wrapper.py` are copied in as package data.

## Key design points

**DB interop:** this is the hardest drop-in constraint.
- Ship drizzle's `.sql` files and `_journal.json` unchanged.
- `migrate.py` copies drizzle's libsql migrator:
  - It uses the table `__drizzle_migrations(id, hash, created_at)`, where `hash` is the sha256 of the file text and `created_at` is the journal `when`.
  - It applies the entries newer than `max(created_at)`, splitting on `--> statement-breakpoint`, in one transaction.
  - If the DB is newer than the bundled journal, it refuses to write and tells the user to upgrade.
- Connection pragmas follow `src/database/index.ts:150-190`.
- The eval id format is `eval-XXX-YYYY-MM-DDTHH:MM:SS` (`src/models/eval.ts:99`).
- Python writes the v4 layout: per-row `eval_results` rows with UUID ids, and `evals.results` holding only durations. It must also read v3 evals.
- The prompt id is the sha256 used in `src/models/prompt.ts:24`, and the dataset id is `sha256(JSON.stringify(tests))`. Port JS `JSON.stringify` exactly with `json.dumps(separators=(",",":"), ensure_ascii=False)` and test it against Node.
- Python touches `evalLastWritten` after every write.

**Cache:** Node's `cache.json` (KeyvFile) is rewritten whole with no locking, so sharing it is unsafe. Python gets its own default file, `<configDir>/cache/cache-py.json`. It keeps the same `{value, expires}` envelope and TTL env vars, and `PROMPTFOO_CACHE_PATH` still overrides it. Key formats follow the Node providers where that is cheap, but cross-runtime cache hits are not a goal.

**Templating shim (`NunjucksCompatEnvironment`):**
- Settings: autoescape off; `env` global; filters `load` (json.loads), `dump` (JSON.stringify semantics), `safe`, `default`, `join`, `length`, `upper`, `lower`, `trim` and `replace`.
- Set `undefined=ChainableUndefined`, or `StrictUndefined` when `throwOnUndefined` is set.
- Proxy JS idioms inside templates: `x.length`, `.toUpperCase()`, and `for k, v in dict` without `.items()`. Do this with a custom `getattr` and a for-loop preprocessor.
- Support `PROMPTFOO_DISABLE_TEMPLATING` and the other template env vars.
- `nunjucksFilters` in config must point to `.py` files that export a callable.
- Implement variable extraction (`extractVariablesFromTemplate`) with `jinja2.meta`.

**Python hooks:**
- By default hooks run in-process: `importlib` loads `file://x.py:fn`, and `async` functions are awaited.
- The signatures stay the same:
  - providers: `call_api(prompt, options, context)`, `call_embedding_api` and `call_classification_api`
  - assertions: `get_assert(output, context)`, with inline code wrapped as `main(output, context)`
  - transforms: `get_transform(output, context)`
  - vars: `get_var(var_name, prompt, vars)`
  - test generation: `generate_tests`
  - config refs: `get_config()`
  - prompt functions: `fn(context)`
- If `PROMPTFOO_PYTHON` or a provider's `config.pythonExecutable` names a different interpreter, the runner falls back to a subprocess using the bundled `persistent_wrapper.py` protocol. That protocol is `CALL|fn|req|resp` and `READY` / `DONE|resp` (`src/python/workerPool.ts`), with worker-count priority taken from `src/providers/pythonCompletion.ts:168`.
- Results are normalized the way `normalizeScriptResult` does it: bool, number or GradingResult.

**JS-expression strings** (HTTP `transformResponse: "json.choices[0].message.content"`, provider `transform`, `transformVars`):
- These are evaluated as Python expressions with `json`, `text`, `context`, `output` and `vars` in scope.
- Wrap dicts in an attribute-access proxy so dotted paths work unchanged.
- Anything else, such as arrow functions or `=>`, gives a clear "JS transforms are not supported; use file://x.py" error.

**Providers:**
- A registry of `(prefix-match, factory)` pairs ports `src/providers/registry.ts`, along with the id grammar `prefix:modelType:model` and the five config shapes (`src/util/providerRef.ts`).
- Load-time `{{ env.X }}` rendering follows `src/providers/index.ts:108-314`.
- `ApiProvider` is a Protocol with `id()` and `async call_api(prompt, context, options)`, mirroring `src/types/providers.ts:139`.
- The 42 OpenAI-compatible wrappers become a single `OpenAICompatibleProvider` plus a data table of `{prefix, base_url, key_env}`. As built (see D23), the table covers only the pure routing wrappers; families with their own behaviour keep a small subclass each under `providers/compatible/`.

## Phases
Each phase ends with its parity-harness subset passing.

**Phase 0: scaffolding and parity harness**
- Run `uv init`, then set up pyproject, ruff, pyright and pytest.
- Write the parity harness: `tools/parity.py <config>` runs `npx promptfoo eval -c X -o node.json --no-write` (this needs `npm ci` in `../promptfoo`) and `promptphoo eval -c X -o py.json --no-write`. It normalizes ids, timestamps, latency and durations, then diffs the two files.
- Feed it the fixtures from `test/smoke/fixtures/configs/`.

**Phase 1: config, types, prompts, templating, tests**
- Pydantic schemas.
- `readConfig`, dereferencing, combining configs, env vars.
- Prompt processors: txt with `---` separators, json/jsonl chat, yaml, md, j2, csv, py functions, exec.
- Test readers: yaml, json, jsonl, csv with `__` columns, py `generate_tests`, globs. Google Sheets is optional.
- `defaultTest`, scenarios, the var cartesian product (`evaluator.ts:2142`) and the Jinja shim.
- Unit tests port the relevant `test/util/*` and `test/prompts/*` specs.

**Phase 2: evaluator, deterministic path and DB**
- `Evaluator` and `runEval`: concurrency, `repeat` with the `repeat:i` cache namespace, delay, abort, timeouts (`PROMPTFOO_EVAL_TIMEOUT_MS`, `PROMPTFOO_MAX_EVAL_TIME_MS`), `transformVars` and `options.transform`. Serial mode is forced for `_conversation` and `storeOutputAs`.
- Providers: `echo`, `sequence`, `python:` / `file://*.py`, `exec:`.
- Deterministic assertions: equals, contains/icontains (and the -all/-any variants), starts-with, regex (translate JS flags), is-json and contains-json (jsonschema), is-xml, contains-xml, is-html, contains-html, is-sql, contains-sql, levenshtein, rouge-n/l/s (hand-port `js-rouge` tokenization), bleu, gleu, meteor, word-count, cost, latency, finish-reason, perplexity, perplexity-score, is-refusal, is-valid-function-call, is-valid-openai-function-call, is-valid-openai-tools-call, tool-call-f1, trajectory:*, python and webhook.
- Assertion mechanics: the `not-` prefix, weight/threshold/metric, assert-set, the weighted mean and `namedScores` (`src/assertions/assertionsResult.ts`).
- Stats and token usage.
- Outputs: the console table and all file writers.
- DB writes and migrations, and the exit code from `PROMPTFOO_PASS_RATE_THRESHOLD`.
- Exit criterion: at least 90% of the echo/python/exec smoke fixtures match Node.

**Phase 3: network providers, cache and scheduler**
- The cache module.
- Retry policy: 3 retries with exponential backoff, `PROMPTFOO_RETRY_5XX`, and adaptive concurrency from rate-limit headers.
- Providers, in order:
  1. openai: chat, responses, embedding, completion, moderation
  2. anthropic messages
  3. the OpenAI-compatible table
  4. http: body/raw templating, transformRequest/Response, sessionParser, auth, tls, multipart; signatureAuth comes later
  5. ollama
  6. azure
  7. google: ai.studio and vertex
  8. bedrock: converse first
- Cost tables come from the per-provider pricing files.
- Tests use `respx` to replay recorded responses.

**Phase 4: model-graded assertions**
- Default grader selection, ported faithfully from `src/providers/defaults.ts:73-259`.
- `get_grading_provider` from `src/matchers/providers.ts:205`.
- Assertions: llm-rubric, model-graded-closedqa, factuality, model-graded-factuality, g-eval, answer-relevance, context-*, similar (cosine, dot, euclidean), classifier, moderation, select-best and max-score (`evaluator.ts:4962-5263`), and search-rubric.
- Grading prompts are copied verbatim. Rendering parity is tested by comparing Jinja output with the Nunjucks output captured from Node.

**Phase 5: rest of the CLI and eval flags**
- Filters: first-n, pattern, range, prompts, providers, sample/seed, failing, errors-only, metadata.
- `--resume` and `--retry-errors`, `--watch`, `--model-outputs`, `-x` extension hooks (Python only: beforeAll/afterAll/beforeEach/afterEach), and `--var`/`--tag`.
- Commands:
  - `list`, `show`, `delete`, `export` and `import`, which must read and write v3 and v4
  - `retry`, `cache clear`, `init`, `validate`, `config`, `share`
  - `generate dataset`/`assertions` are optional
- Telemetry is a no-op.

**Phase 6: blobs and tracing (optional)**
- Filesystem blob store with the same layout (`<configDir>/blobs/ab/cd/<sha>`) and the `blob_assets` / `blob_references` rows. Until then, behave as if `PROMPTFOO_INLINE_MEDIA=true` but still resolve `promptfoo://blob/` URIs when reading.
- An OTLP trace receiver and the trace-* assertions.

**Out of scope:** redteam, web UI and server, code-scan, optimizer, MCP, model-audit, JS and Ruby hooks, golang and ruby providers, codex/realtime/agents providers, cloud (`promptfoo://provider`).

## Risks to watch
- **Nunjucks vs Jinja semantics:** truthiness of `[]` and `{}`, JS method calls, `|dump` spacing. The parity harness fixtures `nunjucks-filters.yaml` and `nunjucks-conditionals.yaml` cover this.
- **JS `JSON.stringify` vs `json.dumps`:** key order, number formatting (`1.0` vs `1`), unicode. These feed into hashes and ids.
- **JS regex flags and syntax** in `regex` assertions and `__expected` strings.
- **Drizzle migrations:** if Node adds a new one, Python must ship it before it can write to that DB.

## Verification
1. `uv run pytest`: unit tests ported from `../promptfoo/test/{assertions,matchers,prompts,util}`, focused on the deterministic parts.
2. `uv run tools/parity.py --all`: every fixture in `test/smoke/fixtures/configs/` gives a matching normalized JSON diff between Node and Python. A list of expected divergences is checked in (JS hooks, etc.).
3. DB round-trip, with `PROMPTFOO_CONFIG_DIR` pointing at a temp dir:
   - Run `promptphoo eval -c basic.yaml`, then confirm the eval appears in Node's `npx promptfoo list` and `npx promptfoo show <id>`.
   - Run a Node eval, then `promptphoo list` / `show` / `export`.
   - Run Node migrations on a DB created by Python, and the reverse, and confirm both are no-ops.
4. Live smoke test, gated on API keys: `examples/getting-started` and `examples/eval-g-eval` against OpenAI and Anthropic. Compare pass/fail with Node.
