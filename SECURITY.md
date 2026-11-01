# Security

promptphoo runs evaluations: it reads configuration, renders prompts, calls model APIs with your
keys, and runs any hooks your configuration names. This page explains what it trusts, what it
protects you from, and how to report a problem.

## Reporting a vulnerability

Please report security problems **privately**, using GitHub's "Report a vulnerability" button on
this repository's Security tab. Do not open a public issue. You should get a reply within a week.

promptphoo is an independent port of promptfoo. If the problem is also in promptfoo itself, please
report it to the promptfoo project as well; a fix here does not reach its users.

## The trust model

**Configuration is code.** A `promptfooconfig.yaml` can name Python hooks (`file://*.py`),
`exec:` providers, transforms and extension hooks. They run with your user's permissions, inside
the promptphoo process (Python hooks) or as child processes that inherit your whole environment,
API keys included (`exec:`). There is no sandbox around them, and there cannot be one inside a
single Python process. **Only run configurations you trust**, as you would only run scripts you
trust.

**Data should not be code.** Test cases, CSV files, spreadsheets and datasets are often shared or
downloaded. promptphoo treats them as data, with these protections (decision D61 in
[docs/DECISIONS.md](docs/DECISIONS.md)):

- **Templates are sandboxed.** Prompts and var values are rendered as templates, as upstream
  promptfoo does. The template engine is Jinja2's `SandboxedEnvironment`, so a template cannot
  reach Python internals (`__class__`, `__subclasses__`, `__globals__`) to run code. Inline
  expressions such as `transformResponse: "json.choices[0].message.content"` use the same sandbox.
- **Var values cannot read the environment.** In upstream a var value such as
  `{{ env.OPENAI_API_KEY }}` renders the key into the prompt sent to the provider under test.
  promptphoo renders var values with an empty `env` and warns (D76).
  `PROMPTPHOO_VARS_READ_ENV=true` restores upstream's behaviour. Prompts and config values still
  read the environment; config-authored `{{ env.* }}` in inline vars is rendered at load.
- **Model output is never a template.** An output kept with `storeOutputAs` goes into the next
  prompt as plain text. Upstream renders it, so a model steered by injected content could answer
  `{{ env.OPENAI_API_KEY }}` or `file:///path` and have the key or the file's contents sent in
  the next prompt (D73).

**CSV and JSONL datasets cannot carry code** (D77). In upstream a CSV `__expected` cell such as
`python: ...` runs Python in the promptfoo process. promptphoo refuses code assertions, script
providers and `file://` references to code files in CSV and JSONL files, unless
`PROMPTPHOO_TRUST_TEST_DATA=true`. Plain `file://` references in them (a document, an image)
still load, from any path. YAML and JSON test files are trusted like the config outside safe
mode. **For data you did not write, use `--safe-mode`** (or `commandLineOptions: {safeMode:
true}`), which makes every test file data (D74):

- Tests read from test files (CSV, JSON, JSONL, YAML, scenario test files) are checked before
  anything in them is loaded. The eval stops with a message if one has a `python`, `javascript`
  or `ruby` assertion (including `not-` forms), names a script provider (`exec:`, `python:`,
  `file://`, ...), refers to any `file://` path or `$ref` file, loads its vars from files, or
  lists another test file.
- Inline Python assertions and inline transforms are refused everywhere, as upstream refuses
  inline JavaScript. A `file://` hook named in the config still runs: the config is code.
- Safe mode does not refuse `webhook` assertions or HTTP graders in test data, which can send
  outputs to a URL.

## Network defaults

- **TLS certificates are verified.** This departs from upstream, which skips verification unless
  `PROMPTFOO_INSECURE_SSL=false`. Behind a proxy that intercepts TLS, point
  `PROMPTFOO_CA_CERT_PATH` at its CA certificate. `PROMPTFOO_INSECURE_SSL=true` turns
  verification off; use it only on networks you control.
- **No telemetry, and no feedback upload.** promptphoo sends nothing to promptfoo's servers.
  `promptphoo feedback` only tells you where to open an issue. The network calls promptphoo makes
  are to the providers, graders and URLs your configuration names.
- **The tracing receiver** (OTLP, used only when tracing is enabled) listens on `127.0.0.1` while
  an eval runs, without authentication, with a 10 MB request limit. Setting its host to `0.0.0.0`
  exposes it to your network; don't do that on a shared network.

## Local state

Results are stored in `~/.promptfoo/promptfoo.db` (or `PROMPTFOO_CONFIG_DIR`), shared with Node
promptfoo, and responses are cached in `~/.promptfoo/cache/cache-py.json`. Both can contain your
prompts and model outputs in plain text. Treat that directory like any other file holding work data.

- **A key supplied through the environment is not written there** (checked below). **A key
  written inline in a config** (`apiKey: sk-...`, an `Authorization` header) **is**: the config
  is stored verbatim in the `evals` table, as upstream stores it. Output files, `export` and the
  logs scrub it. Keep keys in the environment.
- **Owner-only files.** promptphoo creates the state directory 0700 and its database, cache,
  logs, blobs and `promptfoo.yaml` 0600, and narrows an existing world-readable database to 0600
  (D75). Node creates them with your umask, usually readable by every local user, and promptphoo
  leaves an existing directory's mode alone. To close an existing one:
  `chmod 700 ~/.promptfoo`.

## How key handling was checked

The live Anthropic verification (`tools/live/`, results in
[docs/VERIFICATION.md](docs/VERIFICATION.md)) reads the key from a file only inside the commands
that use it. Its recording proxy drops `x-api-key` and `authorization` before writing anything.
After the run, every output, log, recording, cache and database it produced was searched for the
key, and it appeared in none of them.
