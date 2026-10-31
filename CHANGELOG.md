# Changelog

## 0.1.0a1 — unreleased

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
