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
- **What the sandbox does not stop:** a template can read environment variables
  (`{{ env.OPENAI_API_KEY }}`) and place them in a prompt, which is then sent to the provider under
  test. If you evaluate datasets you did not write against providers you do not control, set
  `PROMPTFOO_DISABLE_TEMPLATING=true`, or run with only the keys that eval needs.
- Inline transforms are refused entirely in safe mode.

## Network defaults

- **TLS certificates are verified.** This departs from upstream, which skips verification unless
  `PROMPTFOO_INSECURE_SSL=false`. Behind a proxy that intercepts TLS, point
  `PROMPTFOO_CA_CERT_PATH` at its CA certificate. `PROMPTFOO_INSECURE_SSL=true` turns
  verification off; use it only on networks you control.
- **No telemetry, and no feedback upload.** promptphoo sends nothing to promptfoo's servers.
  `promptfoo feedback` only tells you where to open an issue. The network calls promptphoo makes
  are to the providers, graders and URLs your configuration names.
- **The tracing receiver** (OTLP, used only when tracing is enabled) listens on `127.0.0.1` while
  an eval runs, without authentication, with a 10 MB request limit. Setting its host to `0.0.0.0`
  exposes it to your network; don't do that on a shared network.

## Local state

Results are stored in `~/.promptfoo/promptfoo.db` (or `PROMPTFOO_CONFIG_DIR`), shared with Node
promptfoo, and responses are cached in `~/.promptfoo/cache/cache-py.json`. Both can contain your
prompts and model outputs in plain text. Treat that directory like any other file holding work data.
