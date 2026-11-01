# Migrating from promptfoo (Node) to promptphoo

This guide is for teams already running [promptfoo](https://github.com/promptfoo/promptfoo) who
want to run the same evaluations from Python. promptphoo tracks **promptfoo 0.125.1**.

promptphoo is an unofficial port. It is not promptfoo, and the promptfoo team does not support it.
Report problems with promptphoo with `promptphoo feedback` or in this repository's issue
tracker, not to the promptfoo project.

## 1. Check that promptphoo covers what you use

promptphoo ports the eval core. **Before you migrate, check that you don't need any of these**,
which stay in Node promptfoo:

| Not in promptphoo | What to do |
|---|---|
| Web viewer (`promptfoo view`) | Keep Node installed; it reads promptphoo's results (step 5) |
| Red-teaming, code scanning, model audit, MCP, the optimizer | Keep using Node promptfoo for these |
| Sharing and cloud features (`share`, `auth`, remote grading) | Export instead: `promptphoo export eval <id> -o results.json` |
| JavaScript, TypeScript and Ruby hooks | Rewrite them in Python (step 3) |
| Remote test sources (Google Sheets, Hugging Face datasets, `https://` test files) | Download the file and reference it locally |

The full list, with the reason for each gap, is in [docs/SCOPE.md](SCOPE.md). Every deliberate
difference from Node is recorded in [docs/DECISIONS.md](DECISIONS.md).

## 2. Install side by side

```bash
pip install promptphoo        # or, until there is a release: pip install . from a clone
promptphoo --version          # promptphoo 0.1.1 (compatible with promptfoo 0.125.1)
```

promptphoo installs two commands, `promptphoo` and its short form `ph`. It deliberately does
**not** install `promptfoo` or `pf`, so it never shadows Node promptfoo on your `PATH`. Both tools
can be installed at once, and you always know which one you are running.

If you really want `promptfoo` to mean promptphoo in your own shell, add an alias yourself:

```bash
alias promptfoo=promptphoo    # your choice; remove it to get Node promptfoo back
```

Don't put that alias on shared machines or in CI images other people use. They will report
promptphoo's behaviour as promptfoo bugs.

## 3. Run your existing configuration

Your `promptfooconfig.yaml`, prompt files and test files do not need to change unless they use
JavaScript. Start by validating:

```bash
promptphoo validate -c promptfooconfig.yaml
promptphoo eval -c promptfooconfig.yaml
```

The flags are the same as Node's `promptfoo eval`: filters, `--repeat`, `--resume`,
`--retry-errors`, `-o`, `--grader`, and so on.

### Rewriting JavaScript hooks in Python

promptphoo runs Python hooks with the same function names and arguments as promptfoo's Python
support. A JavaScript reference fails with a clear "not supported by promptphoo" message, never
silently.

| Node promptfoo | promptphoo |
|---|---|
| `id: file://provider.js` | `id: file://provider.py`, with `def call_api(prompt, options, context)` |
| `type: javascript` assertion | `type: python`, with inline code or `file://check.py` with `def get_assert(output, context)` |
| `transform: file://t.js` | `transform: file://t.py`, with `def get_transform(output, context)` |
| `vars: { x: file://v.js }` | `file://v.py`, with `def get_var(var_name, prompt, other_vars)` |
| `tests: file://gen.js` | `file://gen.py`, with `def generate_tests(config)` |
| `nunjucksFilters: { f: path/to/filter.js }` | `path/to/filter.py`, exporting a function |
| `-x file://hooks.js:fn` (extensions) | `-x file://hooks.py:fn`, called with `(hook_name, context)` |

**Short inline expressions still work.** Examples are
`transformResponse: "json.choices[0].message.content"`, `output.trim()`, `a === b ? x : y` and
`{output: x}`. Arrow functions, statements and multi-line code need a Python file.

Python hooks run inside the promptphoo process, so they are faster than under Node, which starts a
Python worker. The flip side is that a hook that hangs or crashes affects the whole eval.

### Templates

promptfoo uses Nunjucks; promptphoo uses Jinja2, made to behave like Nunjucks. These all work as
they do in Node:
- `{{ var }}`, filters such as `dump`, `load`, `join` and `default`, and `for k, v in obj`;
- common JavaScript methods: `.length`, `.toUpperCase()`, `.toLowerCase()`, `.trim()`,
  `.includes()`, `.startsWith()`, `.endsWith()`, `.split()`, `.slice()`, `.replace()`,
  `.join()`, `.indexOf()`.

Two differences:
- Templates are **sandboxed**. Attributes that start with an underscore (`__class__` and the
  like) are unavailable. Ordinary templates never use them.
- A JavaScript method not listed above will fail. Move that logic into a Python hook, or a Python
  `nunjucksFilters` entry.

## 4. Check what behaves differently

| Area | Node promptfoo | promptphoo | What to do |
|---|---|---|---|
| TLS certificates | Not verified unless `PROMPTFOO_INSECURE_SSL=false` | **Verified** unless `PROMPTFOO_INSECURE_SSL=true` | Behind a TLS-intercepting proxy, set `PROMPTFOO_CA_CERT_PATH` |
| Response cache | `~/.promptfoo/cache/cache.json` | `~/.promptfoo/cache/cache-py.json` | Your first promptphoo runs call providers again, so expect fresh API costs once |
| `feedback` command | Posts to promptfoo's servers | Sends nothing; points to promptphoo's issue tracker | — |
| Commands in messages | `promptfoo eval`, `promptfoo view` | `promptphoo eval`; `npx promptfoo@latest view` for the Node-only viewer | — |
| `model-graded-closedqa` and `select-best` with an Anthropic grader | Always fail (`400 at least one message is required`) | **Grade**: the system-only grading prompt is sent as the user turn (D68) | Expect these assertions to start passing or failing on their merits |

Environment variables (`OPENAI_API_KEY`, `PROMPTFOO_*`), exit codes and
`PROMPTFOO_PASS_RATE_THRESHOLD` work the same way.

**A good migration check:** run one eval in both tools with caching off, then compare pass/fail
for each test:

```bash
npx promptfoo@0.125.1 eval -c promptfooconfig.yaml --no-cache -o node.json
promptphoo eval -c promptfooconfig.yaml --no-cache -o py.json
```

With deterministic assertions, results should match exactly. Model-graded ones match when the
models are deterministic (temperature 0 and a fixed seed). That is how promptphoo itself was
checked; the results are in [`VERIFICATION.md`](VERIFICATION.md). If you find a difference that
`docs/DECISIONS.md` doesn't explain, please report it as a bug.

## 5. Shared results database

Both tools use the same `~/.promptfoo/promptfoo.db`, with the same schema and migrations:
- `promptphoo list` and `promptphoo show` see evals written by Node, and Node's `promptfoo list`,
  `promptfoo show` and `promptfoo view` see evals written by promptphoo.
- `promptphoo eval --resume` and `promptphoo retry` work on evals stored there.

**Keep the two versions in step.** If a newer Node promptfoo migrates the database beyond what
your promptphoo release knows, promptphoo stops with "migrated by a newer promptfoo version" and
won't write to it. Either upgrade promptphoo, or point the two tools at different directories with
`PROMPTFOO_CONFIG_DIR`.

## 6. Update CI

Replace

```yaml
- run: npx promptfoo@latest eval -c promptfooconfig.yaml
```

with

```yaml
- run: pip install promptphoo
- run: promptphoo eval -c promptfooconfig.yaml
```

Pin the promptphoo version, as you would pin promptfoo, so that results can be compared across
runs.

## 7. Going back

Nothing in a migration is one-way. Your configs stay valid for Node promptfoo, except hooks you
rewrote in Python: Node can run Python hooks too, through `file://*.py`. The database is shared.
To go back, run `npx promptfoo` again; the results promptphoo wrote are still there.
