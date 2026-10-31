---
name: port-promptfoo
description: Port a module of upstream promptfoo (TypeScript, ../promptfoo, v0.125.1) to promptphoo (Python 3). Use when adding or extending any promptphoo feature that exists upstream — providers, assertions, evaluator, DB, CLI commands, output writers — or when checking parity with Node behaviour.
---

# Porting promptfoo → promptphoo

The goal is a **drop-in replacement**: the same YAML configs, CLI flags, `~/.promptfoo` SQLite schema and drizzle
migrations, and output JSON. Scope is the eval core. Redteam, the web UI, code-scan, the optimizer, MCP and
JS/Ruby hooks are all out. The full plan is in `/home/io/.claude/plans/reactive-sprouting-tower.md`.

## Workflow

1. **Read upstream first, then port it faithfully.** Strip the imports while reading:
   `grep -v "^import" ../promptfoo/src/<file>.ts`, or use `sed -n '/export function X/,/^}/p'` for one function.
   Keep Node's names (snake_cased), its branch order and its error messages, because users and the parity
   diffs both see them. Put the source file in the module docstring ("Ports src/...ts").
2. **Put the module where the plan's layout table says** (`src/promptphoo/<area>/`). Import modules from
   later phases lazily, inside the function, so that earlier phases still import cleanly.
3. **Write tests** in `tests/test_<area>.py`. Use the upstream fixtures (`../promptfoo/test/...`,
   `../promptfoo/test/smoke/fixtures/configs/`) and the expected values from upstream's own specs.
   When it's practical, generate golden output from real Node. One example is `tools/nunjucks_golden.js`,
   which needs nunjucks installed under a scratch `NODE_PATH`.
4. **Verify** with:
   `.venv/bin/pytest -q && .venv/bin/ruff check src tests tools && .venv/bin/ruff format --check src tests tools && .venv/bin/pyright`
   Pyright errors about modules that don't exist yet are expected until the phase that adds them.
5. **Check parity** once `eval` exists: `.venv/bin/python tools/parity.py <config|--all>`. It needs a
   reference Node promptfoo, either `npx -y promptfoo@0.125.1` (ask before running it) or
   `NODE_PROMPTFOO="node ../promptfoo/dist/src/main.js"` after `npm ci && npm run build` upstream.
6. **Commit** each coherent piece on `main` with the message `Phase N: <what>`. If the commit makes or
   changes a design decision, add or amend its entry in `docs/DECISIONS.md` in the same commit, and end
   the message with a `Decision: D<n>` trailer (several: `Decision: D<n>, D<m>`). Before logging,
   re-read the Assumption lines of entries whose Modules you touched; if one no longer holds, write a
   new entry that replaces it.

## Conventions (keep these consistent)

- **Runtime data is plain dicts** with Node's camelCase keys (configs, tests, prompts, provider responses,
  token usage, grading results). Don't use pydantic models at runtime. Schema checks only warn, through the
  vendored `_data/config-schema.json`.
- **JS `undefined` means the key is omitted.** Use `defined(**kw)` from `prompts/processors.py`, or
  conditional spreads. Use `jsvalues.UNDEFINED` only when the difference really matters.
- **JS semantics live in `jsvalues.py`:**
  - `json_stringify` is `JSON.stringify` (compact; integral floats print as ints; undefined and callables
    are dropped). Use it for **every hash, id and cache key**.
  - `js_string` is `String()`; `is_truthy` is JS truthiness, where `[]` and `{}` are truthy.
  - `json.dumps` is never correct for ids or hashes.
- **Paths:** use `files.node_join` (normalized, like `path.join`). `os.path.join` keeps `..` segments and
  gives labels that differ from Node's.
- **YAML:** always use `yamlload.load_yaml` (js-yaml CORE/YAML 1.2 rules), never a bare `yaml.safe_load`.
- **Templating:** use `templating.get_engine()` / `render_string`. This is Jinja2 behaving like Nunjucks.
  Don't create a raw `jinja2.Environment`.
- **Global CLI state** (basePath, env, maxConcurrency) goes through the ContextVar accessors in `cli_state.py`
  and the `with_*` context managers. Env lookups go through `envars.get_env_*`, which applies the layered
  overrides.
- **Python hooks** run in-process through `python_hooks.load_function` and `call_async`. Keep upstream's
  hook signatures: `call_api(prompt, options, context)`, `get_assert(output, context)`,
  `get_transform(output, context)`, `get_var(name, prompt, vars)`, `generate_tests(config)`. JSON-round-trip
  arguments and results so scripts see exactly what they would see under Node.
- **JS hook references** (`file://*.js`, `javascript` assertions, arrow-function strings) fail with a
  clear "not supported, use Python" error. They must not fail silently.
- **Vendored upstream assets** go in `src/promptphoo/_data/` unchanged (drizzle SQL and journal, python
  wrappers, tableOutput.html, config-schema.json). Ruff and pyright exclude that directory.
- **Python's cache file** is `<configDir>/cache/cache-py.json`, kept separate from Node's `cache.json`.
  It uses the same `{value, expires}` envelope.
- Code style: line length 100 and py311. Use `# fmt: skip` only on dense literal blocks.

## Decision log

`docs/DECISIONS.md` is the record of *why*. The local knowledge graph indexes it as
`decision:promptphoo:D<n>`. Each entry is `## N. Title` followed by bullets: Decision, Rationale,
Assumption, Implication, `Modules:` (repo-relative paths, comma-separated) and `Commits:` (backticked
shas). The extractor finds replacements only from the exact wording `- Note: Supersedes D<n>.` on the
new entry and `- Note: Superseded by D<n>.` on the old one, so keep each to its own bullet with no
colon after the verb. Elsewhere (this skill, PLAN_PORT.md, docstrings), write "see D<n>", and keep
the words supersede/retire/obsolete off any line that also has a D-number, or the graph links that
line to the decision as if it replaced it.
Reasons in `tools/parity.py` EXPECTED are indexed too, as `divergence:promptphoo:<fixture>`.

## Known parity traps

- Nunjucks vs Jinja: `+` string concatenation, loose `==`, and member calls such as `.length`.
- JS regex syntax and flags, including in `regex` assertions and `__expected` strings.
- node-glob result order. The Python port sorts glob results.
- Operator precedence when porting `a ?? b ? c : d`. Parenthesize explicitly.
- Integral floats (`1.0`) must serialize as `1`.
