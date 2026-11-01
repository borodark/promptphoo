# Changelog

## 0.1.1 — unreleased

The first public alpha: a Python 3 port of the promptfoo 0.125.1 eval core. It reads the same
configs, accepts the same `eval` flags and shares the same `~/.promptfoo` database as Node. What
is and is not covered is in [`docs/SCOPE.md`](docs/SCOPE.md); every deliberate difference from
Node is in [`docs/DECISIONS.md`](docs/DECISIONS.md). The porting plan these phases follow is in
[`docs/PLAN_PORT.md`](docs/PLAN_PORT.md).

- **Scaffolding and parity harness.** `tools/parity.py` runs upstream's smoke fixtures under Node
  and Python and diffs the normalized output (D11, D12).
- **Config, prompts and tests.** `promptfooconfig.yaml` loading with `$ref`s, env rendering,
  `defaultTest` and scenarios; text, chat, YAML, Markdown, Jinja, CSV and Python prompts; YAML,
  JSON, JSONL, CSV and Python test sources. Templates run on Jinja2 made to behave like Nunjucks
  (D4), and runtime data stays plain camelCase dicts (D7).
- **Evaluator and database.** The eval loop with concurrency, repeats, delays and timeouts; the
  deterministic assertions; every output format; and drizzle-compatible migrations on the shared
  SQLite schema (D9, D10).
- **Network providers.** OpenAI, Anthropic, Azure OpenAI, Google AI Studio and Vertex, Bedrock,
  Ollama, HTTP and about forty OpenAI-compatible families, on a shared fetch layer with cache,
  retries and rate-limit scheduling (D18–D24, D31).
- **Model-graded assertions.** `llm-rubric`, `factuality`, `g-eval`, the RAG metrics,
  `conversation-relevance`, `similar`, `select-best`, `max-score` and more, with Node's default
  grader selection. Grading is always local (D25–D28, D32–D35).
- **CLI.** `list`, `show`, `delete`, `export`, `import`, `retry`, `cache clear`, `init`,
  `validate`, `config`, `logs`, `debug`, `feedback` and `generate`; every eval flag, including
  `--resume`, `--retry-errors`, `--filter-failing`, `--watch` and Python extension hooks
  (D36–D47). `share` is registered but always fails (D44).
- **Media blobs and tracing.** A blob store in Node's layout, an OTLP receiver, a trace store,
  the trace assertions and promptfoo's own spans (D48, D49, D51–D57).
- **Packaging.** MIT licensing with upstream's notice kept (D58); optional extras for the
  libraries some features need (D59).
- **Safer defaults than upstream.** Templates and inline expressions are sandboxed, TLS
  certificates are verified by default, and `promptphoo feedback` sends nothing (D61). See
  [`SECURITY.md`](SECURITY.md).
- **A name of its own.** Only `promptphoo` and `ph` are installed, and every message names
  promptphoo, so it is never mistaken for Node promptfoo (D62). Moving a project over is covered
  in [`docs/MIGRATING.md`](docs/MIGRATING.md).
- **Fixed by live verification.** A `?` inside a YAML flow mapping (`{ q: What is it? }`) now
  parses as js-yaml does; Anthropic requests no longer carry an `x-promptfoo-version` header
  (D64); float sums follow JavaScript's left-to-right order, so cosine similarities and averages
  match Node to the last bit (D63).
- **Malformed assertions and dangling references fail at load time, as in Node.**
  `validateAssertions` is ported: an assertion without a `type` (often a stray `-` before `value:`
  in YAML), a non-object, a bad `assert-set` or a schema violation now stops the run with
  `AssertValidationError` and Node's message, instead of reaching the evaluator. The two checks
  that follow it upstream are ported too: a test, `defaultTest` or scenario whose `providers`
  names no configured provider, and a test or `defaultTest` whose `prompts` names no prompt, stop
  the run; the prompt error points at the `path:line:column` of the reference in the YAML config.
  Found by `tools/port_map.py`, which generates [`docs/PORT_MAP.md`](docs/PORT_MAP.md) from the
  module docstrings.
- **Config mistakes are reported, not dumped.** An assertion without a `type`, an unknown
  provider or prompt reference, a YAML syntax error, an unknown provider id or a missing file
  now prints one message and exits 1 in `eval` and `generate`, where Node prints a stack trace;
  `--watch` keeps watching after a bad edit (D66, D67).
- **Closed QA and select-best grade with Anthropic.** Both grading prompts are a single system
  message, which the Anthropic API rejects, so in Node they always fail with an Anthropic grader.
  promptphoo sends such a prompt as the user turn (D68).
- **Silent prompt fallbacks now warn.** A JSON, YAML or CSV prompt file that fails to parse, or a
  nested `file://` reference inside one that cannot be loaded, is still used as raw text, as in
  Node, but promptphoo says so (D69).
- **Results are listed in test order.** Outputs, exports and JUnit reports of a stored eval list
  rows by test, then prompt, where Node lists them in whatever order a concurrent run stored them
  (D70).
- **Id prefixes work in `show` and `delete`.** Node matches the prefix and then fails; promptphoo
  acts on the one matching eval and lists the matches of an ambiguous prefix without deleting.
  Legacy v3 evals now list their prompts (D71).
- **Grader config reuse is warned about.** A grader named by the id of a configured provider
  reuses that provider's config, as in Node; promptphoo now says so once per grader and shows how
  to give the grader its own settings (D72).
- **Security.** A model output kept with `storeOutputAs` is inserted as plain text, so it can
  no longer pull `{{ env.* }}` secrets or files into the next prompt (D73). `--safe-mode` now
  treats test files as data: code, `file://` references and script providers in them stop the
  load, and inline Python assertions are refused (D74). The database, cache, logs and blobs are
  owner-only (D75). SECURITY.md now says that keys written inline in a config are stored in the
  database.
- **Breaking, for safety.** Var values no longer read environment variables: `{{ env.* }}` in a
  CSV cell or test-file var renders empty, with a warning (`PROMPTPHOO_VARS_READ_ENV=true`
  restores it; D76). CSV and JSONL datasets can no longer carry code: Python assertions, script
  providers and code-file references in them stop the load (`PROMPTPHOO_TRUST_TEST_DATA=true`
  restores it; D77).
- **Verification.** 6,543 tests, 63 byte-identical parity fixtures, a clean-wheel and a
  base-install run, two fully local Ollama runs, and a live Anthropic run in which all 69 requests
  matched Node's on the wire. Results, method and what was not checked are in
  [`docs/VERIFICATION.md`](docs/VERIFICATION.md); `tools/live/` reruns the Anthropic part.
