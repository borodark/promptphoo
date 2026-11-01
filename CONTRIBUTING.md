# Contributing to promptphoo

promptphoo is a port. Its job is to behave like promptfoo 0.125.1, so the Node implementation is
the specification. A change is correct when Python and Node agree, or when the difference is
deliberate and written down in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Setting up

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,all]'
```

The optional extras (`nlp`, `sql`, `bedrock`, `vertex`, `pdf`, `http-signing`, or `all`) pull in
the libraries some assertions and providers need. Tests for a missing extra are skipped.

## Verifying a change

Run all four checks before you commit:

```bash
.venv/bin/pytest -q -n auto && .venv/bin/ruff check src tests tools \
  && .venv/bin/ruff format --check src tests tools && .venv/bin/pyright
```

The test suite never touches your real `~/.promptfoo`: an autouse fixture in
`tests/conftest.py` points the config directory at a temporary one.

`tools/check_refs.py` checks that the file paths and `D<n>` references in the docs, docstrings
and comments resolve. Pass `--upstream <path>` to also check the upstream TypeScript paths.

`tools/port_map.py` regenerates [`docs/PORT_MAP.md`](docs/PORT_MAP.md) from the module
docstrings, so name the upstream files a module ports in its docstring. It reads upstream's
0.125.1 tag from `../promptfoo` (or `--upstream <path>`); `--check` fails when the map is stale.

## Checking against Node

Parity and interop checks need a reference Node promptfoo 0.125.1. Install the published package
and point `NODE_PROMPTFOO` at its entry point (D11):

```bash
export NODE_PROMPTFOO="node /path/to/promptfoo/dist/src/main.js"
.venv/bin/python tools/parity.py --all          # or one config: tools/parity.py path/to/config.yaml
```

- `tools/parity.py` runs upstream's smoke fixtures through both CLIs, normalizes ids, timestamps
  and latencies, and diffs the JSON output. Fixtures that are expected to differ are listed in its
  `EXPECTED` table, each with its reason (D12). Outputs land in `.parity/`.
- With `NODE_PROMPTFOO` set, `pytest` also runs the interop suites, which check that each CLI
  reads the other's databases, exports, imports, blobs and traces. Without it they are skipped.
- Several golden files under `tests/fixtures/` were recorded from Node by the scripts in
  `tools/`. Regenerate them with those scripts after an upstream bump.

**Live checks.** Unit tests never touch the network. `tools/live/run_anthropic.sh` compares the
two CLIs against the real Anthropic API, spending about $1 (D65). Phase 1 records Node's
exchanges through a local proxy and replays them to promptphoo, so every request and result row
can be diffed exactly; phase 2 runs both live and compares outcomes. Keep the key in a file and
pass its path:

```bash
ANTHROPIC_KEY_FILE=~/path/to/key tools/live/run_anthropic.sh    # results in .live/
```

It needs Ollama with `nomic-embed-text` for the embedding assertions. Record what you verified in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## Porting rule: read upstream first

1. Read the upstream TypeScript before writing any Python, and port it faithfully: keep Node's
   names (snake_cased), its branch order and its error messages, because users and the parity
   diffs both see them.
2. Name the source in the module docstring: `Ports src/<path>.ts`.
3. Port upstream's own tests and fixtures where they exist, and record goldens from real Node
   where practical.
4. Where the port has to differ, make the difference loud (a clear error rather than a silent
   change), and log it as a decision.

The conventions that keep the port consistent (plain camelCase dicts at runtime, `json_stringify`
for every hash and id, `yamlload.load_yaml`, the templating engine, `files.node_join`, and so on)
are listed in [`.claude/skills/port-promptfoo/SKILL.md`](.claude/skills/port-promptfoo/SKILL.md).

## The decision log

Every deliberate departure from Node, and every design choice a later reader would ask about,
is an entry in [`docs/DECISIONS.md`](docs/DECISIONS.md). The format is described at the top of
that file. In short:

- Each entry is `## N. Title` followed by `- Decision:`, `- Rationale:`, `- Alternatives:`,
  `- Assumption:`, `- Implication:`, `- Note:`, `- Modules:` and `- Commits:` bullets.
- The log is append-only. An out-of-date entry gets a `- Note:` bullet; a reversal is a new entry,
  linked with the exact bullets `- Note: Supersedes D<n>.` and `- Note: Superseded by D<n>.`
- Numbers are never reused or renumbered.

Before you add an entry, re-read the Assumption lines of the entries whose Modules you touched.
If one no longer holds, the change needs a new entry.

## Commits

- One coherent change per commit, with a short imperative subject. Commits from the porting
  phases are prefixed `Phase N:`.
- A commit that adds or changes a decision includes the `docs/DECISIONS.md` edit and ends with a
  trailer naming it:

  ```
  Decision: D<n>
  ```

  For several, `Decision: D<n>, D<m>`. Then list the commit's sha in the entry's Commits bullet.
- History is kept as it is: fixes go forward in new commits.

## How this code was written

promptphoo was developed with the help of AI coding agents, which read the TypeScript and wrote
much of the Python. Every change, whoever or whatever wrote it, is held to the same bar: it must
agree with the Node implementation, through ported upstream tests, goldens recorded from Node,
the parity harness or the interop suites, or else its difference must be logged as a decision.
Reviews should ask for that evidence.

## Reporting bugs

A difference from Node that is not logged in `docs/DECISIONS.md` is a bug. Please report it with
the config, the command, and both outputs.
