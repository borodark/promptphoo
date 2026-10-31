---
description: promptphoo architectural decisions
---

# promptphoo decisions

The decision log for the Python port of promptfoo v0.125.1. The local knowledge graph
(`~/exmc-scratch/kg_ladybug`) indexes it as `decision:promptphoo:D<n>`.

Format rules (the extractor parses them):
- Each entry is `## N. Title` followed by `- Field:` bullets.
- `- Modules:` lists repo-relative paths, comma-separated.
- `- Commits:` lists backticked short shas.
- When one decision replaces another, write `- Note: Supersedes D<n>.` on the new entry and
  `- Note: Superseded by D<n>.` on the old one. Keep that sentence to just those words.
- Commits that make or change a decision carry a `Decision: D<n>` trailer.

The phases and module layout are in [PLAN_PORT.md](PLAN_PORT.md). How to port is in
`.claude/skills/port-promptfoo/SKILL.md`.

## 1. Eval core first; web UI, redteam, code-scan, optimizer, MCP and model-audit are out
- Decision: Port the eval pipeline (config, prompts, tests, evaluator, assertions, providers, DB, CLI) and leave out the web UI, redteam, code-scan, the optimizer, MCP and model-audit.
- Rationale: The eval core is what CLI users run day to day. The UI alone is about 200k lines of React.
- Assumption: Users drive promptfoo from the CLI and CI, not the web viewer.
- Implication: There is no `view` command and no `eval setup`. Upstream's UI can still read promptphoo's evals, because the DB is shared (D2).
- Modules: README.md
- Commits: `c11539d`

## 2. Drop-in compatibility with promptfoo 0.125.1
- Decision: Read the same `promptfooconfig.yaml`, accept the same `eval` flags, and share the `~/.promptfoo/promptfoo.db` schema and drizzle migrations.
- Rationale: Existing configs, CI scripts and eval history keep working, and Node and Python can run against one state directory.
- Assumption: Upstream stays on drizzle's SQLite layout, and new migrations are shipped before promptphoo writes to a DB that has them.
- Implication: Every output (JSON results, CSV, YAML, XML, console table) is diffed against Node by the parity harness (D12).
- Modules: src/promptphoo/config/load.py, src/promptphoo/cli/eval.py, src/promptphoo/_data/drizzle
- Commits: `2df350e` `9ce42b2` `9187072`

## 3. Python-only hooks; JS, Ruby and Go hooks fail with a clear error
- Decision: Run Python providers, assertions, transforms and test generators in-process. Reject `file://*.js`, the `javascript` and `ruby` assertions, and arrow-function strings with a "not supported, use Python" error.
- Rationale: Embedding a JS runtime would bring back the Node dependency the port removes.
- Assumption: Hook users can rewrite a few lines of JS as Python.
- Implication: Fixtures that use JS hooks are skipped in parity; inline transform expressions are evaluated as Python.
- Modules: src/promptphoo/python_hooks.py, src/promptphoo/transform.py, src/promptphoo/providers/registry.py
- Commits: `953f0ac` `6d075ff`

## 4. Jinja2 with a Nunjucks compatibility shim instead of Nunjucks
- Decision: Render templates with Jinja2 configured to behave like Nunjucks: autoescape off, chainable undefined, JS-style truthiness, `.length` and other member calls, and the `dump` filter with JSON.stringify spacing.
- Rationale: There is no maintained Nunjucks for Python, and the two template languages are close.
- Assumption: Configs use the common subset of Nunjucks.
- Implication: Template behaviour is checked against goldens recorded from real Nunjucks (`tools/nunjucks_golden.js`).
- Modules: src/promptphoo/templating.py
- Commits: `c382975`

## 5. Python keeps its own cache file, cache-py.json
- Decision: Store cached responses in `<configDir>/cache/cache-py.json`, using Node's `{value, expires}` envelope, instead of sharing Node's `cache.json`.
- Rationale: Node's KeyvFile cache rewrites the whole file with no locking, so two runtimes sharing it would lose each other's writes.
- Assumption: Cache hits shared between Node and Python are not worth that risk.
- Implication: `PROMPTFOO_CACHE_PATH` still moves the cache directory. Cache keys follow upstream's semantics but need not match byte for byte.
- Modules: src/promptphoo/cache.py
- Commits: `953f0ac` `2a03eed`

## 6. Runtime types as pydantic v2 models mirroring the Zod schemas
- Decision: The approved tech stack listed pydantic models for configs, tests and results.
- Rationale: Pydantic gives typed access and validation.
- Note: Superseded by D7.
- Modules: pyproject.toml

## 7. Runtime data is plain camelCase dicts; schema checks only warn
- Decision: Configs, tests, prompts, provider responses and grading results stay plain dicts with Node's camelCase keys. Validation runs the vendored `config-schema.json` and only warns.
- Rationale: Upstream's Zod schemas are passthrough in many places, and parity diffs compare serialized JSON key by key, so models would add a conversion layer that each output would have to undo.
- Assumption: Hooks and output writers expect the same shapes Node uses.
- Implication: pydantic is still listed in pyproject.toml, but nothing imports it.
- Note: Supersedes D6.
- Modules: src/promptphoo/ajv.py, src/promptphoo/_data/config-schema.json
- Commits: `6d075ff`

## 8. JS undefined is an omitted key, or the UNDEFINED sentinel where Node emits the key
- Decision: Model JS `undefined` by leaving the key out. Where Node still emits the key (empty XML tags, the `config` key on prompts, the duration fields in stats), use `jsvalues.UNDEFINED`.
- Rationale: JSON output drops undefined keys, but the XML builder and key order do not.
- Assumption: The sentinel is falsy and is never written to JSON.
- Modules: src/promptphoo/jsvalues.py, src/promptphoo/output/writer.py, src/promptphoo/evaluator/run.py, src/promptphoo/models/eval.py
- Commits: `ecab33d`

## 9. json_stringify for every hash, id and cache key
- Decision: Compute prompt ids, dataset ids, cache keys and hashes over `jsvalues.json_stringify`, which reproduces JS `JSON.stringify`, and never over `json.dumps`.
- Rationale: Ids must match Node's for shared DB rows (D2). `json.dumps` writes `1.0` and spaced separators.
- Assumption: JSON.stringify's number formatting and key order are reproduced exactly.
- Modules: src/promptphoo/jsvalues.py, src/promptphoo/models/eval.py
- Commits: `602ea28` `9d47e04`

## 10. Drizzle-compatible migrator that refuses a newer database
- Decision: Reimplement drizzle's libsql migrator. It uses the `__drizzle_migrations` table, takes the sha256 of each migration file, applies newer journal entries in one transaction, and refuses to run if the DB already has migrations promptphoo does not ship.
- Rationale: Node and Python must each see the other's migrations as already applied.
- Assumption: The vendored journal is the one for promptfoo 0.125.1.
- Implication: If Node upgrades the DB, promptphoo refuses to write to it until it ships the new migrations.
- Modules: src/promptphoo/db/__init__.py, src/promptphoo/_data/drizzle
- Commits: `9ce42b2`

## 11. Reference Node promptfoo is an npm install run as dist/src/main.js
- Decision: Install promptfoo 0.125.1 with npm in a scratch directory and run `node .../dist/src/main.js`, which bypasses `entrypoint.js`. Pass the command to the harness in `NODE_PROMPTFOO`.
- Rationale: `entrypoint.js` requires Node 22.22 or later, and the container has 22.21.1. The user authorised the npm install in the container.
- Assumption: Running on Node 22.21 doesn't change promptfoo's output for these fixtures.
- Modules: tools/parity.py
- Commits: `400e703`

## 12. Parity harness: parallel, normalized, with a list of expected divergences
- Decision:
  - `tools/parity.py --all` runs every smoke fixture under Node and Python in parallel, one thread per core.
  - It masks eval ids, timestamps and latencies, and sorts result rows before diffing, because Node loads results without ORDER BY.
  - Fixtures in its EXPECTED dict are reported as XFAIL/XERROR, so the harness exits 0.
  - The GPU is not used, because this workload has nothing for it.
- Rationale: The user asked to use all the CPUs. A full run takes about 10 seconds.
- Assumption: Every EXPECTED entry names the phase or the upstream reason that explains it.
- Implication: The Phase 2 exit criterion was met with 61 PASS and 0 unexpected failures.
- Modules: tools/parity.py
- Commits: `400e703`

## 13. pytest runs serially by default
- Decision: pytest-xdist is installed in the venv but is opt-in (`pytest -n auto`). It is not in pyproject.toml.
- Rationale: The suite takes about 8 to 11 seconds either way, because worker startup costs about as much as it saves.
- Assumption: The suite stays fast. Revisit if it grows past about 30 seconds.

## 14. Console table colour follows TTY detection
- Decision: Colour the console table only when stdout is a TTY, using the supports-color rules.
- Note: Superseded by D15.
- Modules: src/promptphoo/output/ansi.py
- Commits: `9187072`

## 15. Table borders are coloured even when piped
- Decision: Colour table borders and headers unless `FORCE_COLOR` parses to 0 or `--no-color` is given. `NO_COLOR` does not turn them off. Chalk-coloured text keeps the TTY rule.
- Rationale: Measured against Node 0.125.1, which colours piped table output. Something in promptfoo enables @colors/colors globally.
- Assumption: Upstream keeps that global enable. Re-check if Node output changes.
- Implication: The `ansi.py` module docstring describes the TTY rule, which applies only to chalk text.
- Note: Supersedes D14.
- Modules: src/promptphoo/output/ansi.py
- Commits: `ecab33d`

## 16. Config pick() uses Python truthiness
- Decision: Merge file config and defaults with Python's `or`.
- Note: Superseded by D17.
- Modules: src/promptphoo/config/load.py
- Commits: `2df350e`

## 17. Config pick() uses JS truthiness
- Decision: `pick()` uses `jsvalues.is_truthy`, so an empty `{}` or `[]` in the file config is kept, as with Node's `fileConfig.x || defaultConfig.x`.
- Rationale: Python's `or` dropped empty `env`, `tags`, `metadata` and `evaluateOptions`, which made the parity diffs fail.
- Note: Supersedes D16.
- Modules: src/promptphoo/config/load.py, src/promptphoo/jsvalues.py
- Commits: `400e703`

## 18. Phase 3 starts with a fixed fetch contract and a lazy provider registry
- Decision:
  - Before splitting the work, commit a small working `fetch.py` with stable signatures (`fetch_with_cache`, `fetch_with_retries`).
  - Commit a `NETWORK_FACTORIES` table that routes provider-id prefixes to `module:create_provider`, imported on first use.
- Rationale: Provider ports could then run in parallel against an interface that already exists, without editing a shared registry file.
- Assumption: The faithful fetch port only adds keyword arguments.
- Modules: src/promptphoo/fetch, src/promptphoo/providers/registry.py
- Commits: `ba71ed9`

## 19. Phase 3 wave 1 runs as six worktree agents; duplicated helpers are merged when branches land
- Decision:
  - Port six areas in parallel git worktrees: fetch/cache, scheduler, OpenAI core, Anthropic/Bedrock, HTTP, and Google/Ollama.
  - When two branches port the same upstream helper, keep one copy and add the functions only the other has.
- Rationale: The areas are mostly independent, and the user asked to use all the CPUs.
- Implication: `providers/shared.py` kept the Google branch's version plus four functions from the Anthropic branch. `finish_reason.py` and `providers/credentials.py` still duplicate helpers in `shared.py`, to be folded together.
- Modules: src/promptphoo/providers/shared.py, src/promptphoo/cache.py
- Commits: `cef9bb0` `c6eec72`

## 20. Wave 2 waits for the OpenAI core
- Decision: Start the OpenAI Responses API, Azure and the table of about 40 OpenAI-compatible wrappers only after the OpenAI chat provider lands.
- Rationale: All three subclass or reuse the OpenAI chat provider.
- Modules: src/promptphoo/providers

## 21. TLS verification follows upstream's PROMPTFOO_INSECURE_SSL, which defaults to true
- Decision: Port `rejectUnauthorized: !getEnvBool('PROMPTFOO_INSECURE_SSL', true)` as written, so certificates are not verified unless `PROMPTFOO_INSECURE_SSL=false`.
- Rationale: Drop-in behaviour (D2). Upstream `src/util/fetch/index.ts:289`.
- Assumption: The user accepts upstream's default. This is open for review: verifying by default would be safer.
- Modules: src/promptphoo/fetch/__init__.py
- Commits: `b27751e`
- Note: Superseded by D61.

## 22. One retry context, owned by the fetch layer
- Decision: The scheduler sets the provider's `maxRetries` scope through `promptphoo.fetch.with_fetch_retry_context`. `fetch_context.py` re-exports fetch's retry context and error predicates rather than defining its own.
- Rationale: The fetch and scheduler branches each ported `retryContext.ts`, so a `maxRetries: 0` set by the scheduler never reached `fetch_with_retries`.
- Modules: src/promptphoo/fetch_context.py, src/promptphoo/fetch/__init__.py, src/promptphoo/scheduler/rate_limit_registry.py
- Commits: `d13755e`

## 23. The Responses provider rejects streaming and background polling
- Decision:
  - `openai:responses` raises a clear error for `stream: true`. A `background: true` request that comes back queued or in progress returns an error instead of polling.
  - Background responses that complete synchronously, or come back cancelled or failed, are handled with upstream's messages and evicted from the cache.
  - `config.mcp` is not rejected, because upstream's Responses provider ignores it too. Hosted `type: mcp` tools pass through to the API.
- Rationale: Upstream's SSE reader and its background polling, coalescing and cancellation code is about 700 lines that eval runs rarely need. A loud error is better than silently changing the request.
- Assumption: Eval configs seldom set `stream` or `background` on Responses providers.
- Implication: Azure Responses inherits these limits until streaming and polling are ported.
- Modules: src/promptphoo/providers/openai/responses.py, src/promptphoo/providers/responses/stream.py
- Commits: `0b0c44b`

## 24. Azure Entra ID auth reimplements two @azure/identity credentials; auth starts on the first request
- Decision:
  - Upstream picks `ClientSecretCredential` (when the client id, secret and tenant are all set) or else `AzureCliCredential`. `providers/azure/identity.py` reimplements both: the client-credentials grant as a form POST to `<authorityHost>/<tenant>/oauth2/v2.0/token` through the fetch layer, and `az account get-access-token --output json --resource <resource>` as a subprocess. Both return `{token, expiresOnTimestamp}` and use @azure/identity's error texts. No other credential type exists, because upstream builds no other.
  - The client-secret credential caches tokens per scope until five minutes before expiry, as MSAL does. The provider's own refresh window (also five minutes) is ported unchanged.
  - Authentication starts when the first request calls `ensure_initialized`, not eagerly in the constructor. Concurrent requests share one in-flight attempt, and a failure is retried by the next request, as upstream does.
  - Credential creation is synchronous and cached on the provider. Upstream's "install @azure/identity" error has no equivalent here, and construction errors propagate as the "Azure Authentication failed" error.
  - The Azure pricing tables (`AZURE_MODELS` and the cache-read and priority maps) are generated by Node from upstream by `tools/gen_azure_tables.mjs` into `providers/azure/tables.py`, as for OpenAI. Cost is `calculateAzureCost` over that table, not the OpenAI billing helpers, as upstream does.
  - Supported: `azure:chat`, `azure:completion` and `azure:embedding(s)`. `azure:responses` raises NotImplementedError until the Responses port lands for Azure (wave 2b). Assistant, Foundry agent, image, video, moderation (Content Safety) and realtime raise "not supported by promptphoo". MCP (`config.mcp.enabled`) and tracing spans are not ported.
- Rationale: No pip installs, and the eval core needs only API-key auth plus the two credentials upstream actually constructs. A Python constructor has no running event loop to start authentication on.
- Assumption: Users who rely on managed identity or workload identity get those only through Azure CLI login, exactly as upstream (which never builds DefaultAzureCredential).
- Implication: `FunctionCallbackHandler` (without MCP) now lives in `providers/function_callbacks.py`. The evaluator calls `maybe_emit_azure_openai_warning` after appending prompts.
  - `azure:responses:<deployment>` is now supported (`providers/azure/responses.py`). As upstream, it extends the Azure base and composes the shared ResponsesProcessor, with cost from `calculateAzureCost`. It keeps D23's limits: `stream: true` raises the OpenAI Responses error, and a queued or in-progress background response returns an error and is evicted from the cache.
- Modules: src/promptphoo/providers/azure, src/promptphoo/providers/function_callbacks.py, src/promptphoo/evaluator/run.py, tools/gen_azure_tables.mjs
- Commits: `48aa106`

## 25. LLM-judge matchers always grade locally; Pi and JS rubric prompts raise
- Decision:
  - `matches_llm_rubric` never takes upstream's remote-grading branch. Upstream only uses it when the config has a `redteam` section, and redteam is out of scope. The grader is the configured provider or the default `llmRubricProvider`/`gradingJsonProvider`.
  - The `pi` assertion (`matches_pi_score`) raises NotImplementedError, because it needs Promptfoo's remote Pi scoring API.
  - A `rubricPrompt: file://*.js` raises a clear not-supported error. Other file types are read as raw text and rendered, as upstream does.
  - `GEVAL_PROMPT_STEPS` and `GEVAL_PROMPT_EVALUATE` live in `matchers/llm_grading.py`. Upstream defines them in `src/prompts/index.ts`, which the vendored grading prompts JSON does not cover. The text was copied from the upstream template literals, and a test compares it with the upstream file.
  - Rendered grading prompts are compared with real Nunjucks output: `tools/gen_grading_render_cases.py` writes the cases, and `tools/grading_render_golden.js` renders them through a copy of upstream's `renderLlmRubricPrompt`.
  - `finalize_graded_assertion` (upstream `ragDefaults.ts`) is in `assertions/modelgraded.py`. `trajectory:goal-success` stays in `assertions/trajectory.py`, next to the other trajectory assertions.
- Rationale: Drop-in behaviour for every non-redteam config, and loud errors instead of silently different grading.
- Assumption: Configs with a `redteam` section are not run through promptphoo.
- Implication: When the RAG branch lands, it may carry its own copy of `ragDefaults.ts`. Keep one copy and import it from both modules.
- Modules: src/promptphoo/matchers/rubric.py, src/promptphoo/matchers/llm_grading.py, src/promptphoo/assertions/modelgraded.py, src/promptphoo/assertions/trajectory.py, tools/grading_render_golden.js, tools/gen_grading_render_cases.py
- Commits: `3a9c8f9`

## 26. agent-rubric needs an explicitly configured agentic grader
- Decision:
  - `matches_agent_rubric` uses only `grading.provider`. When none is set, upstream falls back to the Codex SDK default (`getCodexDefaultProviders().llmRubricProvider`). promptphoo has no Codex provider, so it raises upstream's "requires an agentic grading provider" error instead.
  - `is_agentic_provider` is ported as upstream's id-prefix list. `is_agent_workspace` always returns false, because promptphoo never creates copy_working_dir workspaces, so the grader is never pointed at the target's `workingDir`.
- Rationale: Coding-agent providers are out of scope (plan). An error is safer than grading with a plain model, which upstream also refuses to do.
- Assumption: Users who need agent-rubric bring a Python provider whose id names an agentic runtime, or wait for those providers to be ported.
- Modules: src/promptphoo/matchers/agent.py, src/promptphoo/assertions/modelgraded.py
- Commits: `3a9c8f9`

## 27. RAG and similarity assertions run against shimmed grading infrastructure
- Decision:
  - `matchers/providers.py`, `matchers/shared.py`, `providers/defaults.py` and the package `matchers/__init__.py` are minimal shims with the shared-contract names, each headed `# SHIM: replaced by the grading-infrastructure branch at merge`. The defaults shim builds only upstream's OpenAI branch (`text-embedding-3-large`, `gpt-6-sol`, `omni-moderation-latest`, a Responses web-search grader). `call_grading_provider` only checks the abort signal; it does no rate limiting, queueing or tracing spans.
  - `matchers/rubric.py` is shimmed the same way with only `load_rubric_prompt` and `render_llm_rubric_prompt`, because the RAG and search-rubric matchers need them and rubric.ts belongs to the llm-rubric branch.
  - `assertions/rag_defaults.py` (ragDefaults.ts) is owned here and also carries `is_grader_failure` from llmGrading.ts, so that `finalize_graded_assertion` and `apply_rag_inverse` have no dependency on the llm-rubric branch. `assertions/context_utils.py` ports contextUtils.ts, with `get_transform_error_message` reading the `__cause__` that `transform()` chains.
  - Handler and matcher invariants raise with Node's `Invariant failed: ` prefix, which is the text upstream users see. `assertions/base.invariant` omits that prefix and is left alone.
  - Grader prompts are checked byte for byte against real Nunjucks renders (`tools/rag_prompts_golden.js` writes `tests/fixtures/rag_prompts_golden.json`).
- Rationale: The grading-infrastructure branch runs in parallel. Shims with identical names let this branch test end to end, and at merge the other branch's files replace them.
- Assumption: The other branch exports these exact names and signatures, and ports `isGraderFailure` with the same `metadata.graderError === true` rule. If its `rubric.py` uses different names, the RAG and search matchers' imports must follow it.
- Implication: Once the shims are replaced, drop the duplicate `is_grader_failure` in rag_defaults.py if llm_grading exports one.
- Modules: src/promptphoo/matchers, src/promptphoo/providers/defaults.py, src/promptphoo/assertions/rag_defaults.py, src/promptphoo/assertions/context_utils.py, src/promptphoo/assertions/modelgraded_rag.py, tools/rag_prompts_golden.js
- Note: at merge the shims were replaced by the grading-infrastructure branch's modules (D32–D34); the RAG matchers now call the real `call_grading_provider`, so their `invoke` callbacks accept the optional `on_response_headers` argument.
- Commits: `336d4e7`

## 28. Similarity, moderation and classifier paths that depend on unported upstream code
- Decision:
  - `matches_similarity` has no remote-grading branch. Upstream sends cosine `similar` to promptfoo's cloud grader under a `redteam` config; redteam is out of scope, so grading is always local.
  - `matches_moderation` keeps upstream's Replicate fallback (`replicate:moderation:meta/llama-guard-4-12b` when only `REPLICATE_API_KEY`/`REPLICATE_API_TOKEN` is set). There is no Replicate provider, so that path fails loudly with the provider loader's error rather than switching to another moderator.
  - `classifier` works with any provider that has `call_classification_api(text)`, such as a Python provider (`file://x.py` defining `call_classification_api`). The HuggingFace classification provider is not required. Classification and moderation calls pass only upstream's positional text arguments; the abort-signal `options` argument upstream adds under an evaluator context is not forwarded, because the Python provider methods do not take it.
  - search-rubric's last-resort `load_web_search_provider` keeps upstream's candidate order and model ids. Candidates whose provider family is not ported (perplexity, xai) are skipped like any load failure.
- Rationale: Port upstream's branches as written, and fail loudly where a dependency is missing instead of changing which provider grades.
- Assumption: Eval users of moderation have an OpenAI key or configure the grader explicitly.
- Modules: src/promptphoo/matchers/similarity.py, src/promptphoo/matchers/moderation.py, src/promptphoo/matchers/classification.py, src/promptphoo/providers/web_search_utils.py
- Commits: `336d4e7`

## 29. The select-best / max-score pass mutates stored EvalResult records; resume and cancellation paths wait for Phase 5
- Decision:
  - The comparison pass (upstream `processComparisonAssertions` and the helpers around it in `src/evaluator.ts`) lives in `evaluator/comparison.py` as a `ComparisonPass` bound to the `Evaluator`. `run.py` calls `prepare_row` before each `append_result` (upstream `persistEvalRow`: snapshot the live grader, store its sanitized form in `testCase`) and `process` after every step has run, before prompts are appended for the last time.
  - The pass works on the store's own `EvalResult` objects, as upstream works on its `EvalResult` models: `store.results` when not persisted, `read_results_by_test_idx` (no ORDER BY) when persisted, and `save_result` writes the mutated row back. `EvaluationStore` gained those three members, and `EvalResult.save()` ports upstream's update-by-id (insert when not yet persisted).
  - Prompt metrics and evaluator stats are adjusted by the ported `updateComparisonStats` / `updateComparisonResultCounts` and `updateAssertionMetrics`, including the cached-target `numRequests` probe count.
  - `runCompareAssertion` is in `assertions/index.py` like upstream. Validation errors are upstream's: a select-best failure (one prompt, non-string value, grader exception) turns every comparable row into an ERROR with the generic "Error grading select-best (...)" reason and a saved `__promptfoo.comparisonError`; a max-score failure (one output, no other assertion) aborts the eval. select-best inside an assert-set is skipped, as upstream.
  - Not ported yet: the `cliState.resume` skip of completed comparisons, retry-errors filtering, rows that failed to persist (`readFailedResultsByTestIdx`), JSONL final-result tracking, grading cancellation (`markComparisonRowsAborted`) and progress-bar updates. The resume-time grader restoration (`restoreComparisonProvider`, redaction checks) is ported but only runs for results of the current run.
  - Shims marked `# SHIM: replaced by the grading-infrastructure branch at merge`: `matchers/__init__.py`, `matchers/shared.py`, `matchers/providers.py`, `matchers/rubric.py` and `providers/defaults.py`, holding only what `matchers/comparison.py` needs.
- Rationale: Mutating the stored records keeps upstream's control flow and its persisted-update path line for line, so a later resume port only fills in the skipped branches.
- Assumption: Persisted rows reach the comparison in SQLite's order for `WHERE eval_id = ? AND test_idx = ?`, which uses the `(eval_id, test_idx, success)` index, so failing rows come before passing ones rather than in prompt order. Node runs the same query on the same schema and was observed to agree (parity PASS on `tests/fixtures/parity_comparison/`). Only the order of texts in the grader prompt depends on it.
- Implication: The grading-infrastructure branch's versions of the shim modules win at merge; `matchers/comparison.py` imports them lazily by the contract names.
- Modules: src/promptphoo/evaluator/comparison.py, src/promptphoo/evaluator/run.py, src/promptphoo/models/eval.py, src/promptphoo/matchers/comparison.py, src/promptphoo/assertions/index.py
- Commits: `9af5f6e`
- Note: Superseded by D36.

## 31. OpenAI-compatible families: a data table for pure routing wrappers, a small subclass for the rest
- Decision:
  - The `providers/compatible/` package ports the upstream providers that subclass or wrap the OpenAI chat, completion and embedding providers.
  - `table.py` is a data table of `{prefix, apiBaseUrl, apiKeyEnvar, override rules, passthrough mode, unsupported types}`. It covers togetherai, aimlapi, cometapi and nscale, whose upstream factories only route `<prefix>:[chat|completion|embedding(s):]<model>`.
  - Every family with its own behaviour keeps a module named after the upstream file, with upstream's class names, ids, `toString`, `toJSON`, missing-key text and cost tables. That covers cost tables, request rewriting, URL building, its own `callApi`, or id/toJSON overrides.
  - Those modules are deepseek, cerebras, openrouter, orcarouter, truefoundry, xai, perplexity, groq, fireworks, hyperbolic, nvidia, moonshot, minimax, litellm, docker, novita, alibaba, localai, llama, meta, abliteration, cloudflare-ai, cloudflare-gateway, databricks, mlflow-gateway, portkey and snowflake. The small ones (atlascloud, llamaapi, cloudera, jfrog/qwak, helicone, envoy) share `simple.py`.
  - Registry entries follow upstream providerMap order. The cerebras, envoy, litellm, novita, nscale and togetherai prefixes are checked before the script and JavaScript-file factories, as upstream's `providerFamilies` are.
  - Not ported, and failing with a clear error:
    - `vercel:` needs the Vercel AI SDK.
    - `mistral:` and `ai21:` have their own request path upstream, and the error points to `openai:chat` with their base URL.
    - The image, audio, video and voice routes of xai, nscale, cometapi and hyperbolic.
    - `github:` raises upstream's own GitHub Models retirement error.
  - `groq:responses:`, `meta:<model>`, `meta:responses:` and `xai:responses:` need the OpenAI Responses provider. groq and meta import `promptphoo.providers.openai.responses` lazily and subclass `OpenAiResponsesProvider` once it exists. xai's Responses route is rejected outright.
- Rationale:
  - About two thirds of the upstream wrappers carry real logic, so one generic class would have needed a quirk field per family and lost upstream's names and error texts.
  - A module per upstream file keeps parity diffs and future upstream syncs one-to-one.
- Assumption:
  - The Responses provider will expose `OpenAiResponsesProvider` in `promptphoo.providers.openai.responses`, with the same `get_openai_body` hook as chat.
  - `configuredProviderFactory` families (deepseek, openrouter, orcarouter, perplexity, truefoundry, minimax, llamaapi, atlascloud, xai) take the `{config: providerOptions, env}` shape, so a provider-level `id` is dropped for most of them, as it is upstream.
- Implication: Abort signals, tracing spans and response-header observers are left out of the providers that override `callApi` (openrouter, snowflake, xai), as they are in the OpenAI chat port.
- Modules: src/promptphoo/providers/compatible, src/promptphoo/providers/registry.py

## 32. Graders always run locally; grader calls skip the provider resource registry
- Decision:
  - `matchers/providers.py` ports `src/matchers/providers.ts`. `should_use_remote_grading` always returns False and `get_remote_grading_context` returns `{}`, so no grader ever calls the promptfoo cloud.
  - `call_grading_provider` keeps upstream's scheduling: the provider call queue from the execution context, then the context's rate-limit registry unless the provider `is_rate_limit_wrapped`, then the tracing context's `withProviderSpan`, all under `run_provider_call_with_abort`. Upstream's `providerRegistry.withProvider` wrapper is dropped, because promptphoo has no provider resource registry.
  - `get_grading_provider` loads through `promptphoo.providers.load_api_provider`, looked up at call time, with upstream's argument shapes: string ids get `base_path` only, typed-map entries also get `env=cli_state.env()`.
  - Provider type checks (`embedding`, `classification`, `moderation`) test for the snake_case methods `call_embedding_api` / `call_similarity_api`, `call_classification_api` and `call_moderation_api`.
- Rationale: Cloud features are out of scope (PLAN_PORT.md). The registry only tracks the lifetime of long-lived provider resources, which no ported provider opens.
- Assumption: No eval config relies on remote grading, which upstream enables only for redteam and cloud-linked runs.
- Modules: src/promptphoo/matchers/providers.py, src/promptphoo/matchers/shared.py
- Commits: `956e36a`

## 33. Default graders: unported families are skipped, Codex never selected, no env-scope bundle cache
- Decision:
  - `providers/defaults.py` ports `src/providers/defaults.ts` with its selection order and credential checks: Azure, Anthropic, Google AI Studio, Vertex (ADC), Mistral, xAI, Codex, OpenAI, then the Azure Content Safety moderation swap and the `set_default_*_providers` overrides. Credentials go through `resolve_provider_api_key` (so an explicit empty value masks the ambient one), and `env?.X ?? getEnvString('X')` becomes `_env_string`.
  - Mistral, xAI, Voyage and Azure Content Safety are imported lazily (`promptphoo.providers.mistral.defaults`, `promptphoo.providers.xai.defaults`, `promptphoo.providers.voyage`, `promptphoo.providers.azure.moderation`). While a module is missing, its branch is skipped as if its credentials were absent, so a Mistral- or xAI-only setup falls through to the OpenAI defaults. Each has a TODO at its loader.
  - `has_codex_default_credentials` always returns False, so the Codex branch is never taken.
  - `get_anthropic_providers` builds a fresh bundle on every call. Upstream caches one bundle per `cliState.envScope`, and promptphoo has no scope object. Upstream builds fresh bundles outside a scope too.
  - `providers/openai/defaults.py` and `providers/anthropic/defaults.py` hold the family defaults. Google's already live in `google/ai_studio.py` and `google/vertex.py`, Azure's `defaults.ts` constants are already in `azure/tables.py`, and GitHub's retired default providers are not used by the selection, so they are left to the OpenAI-compatible branch's `ghmodels` module.
  - `set_default_completion_providers` / `set_default_embedding_providers` keep upstream's process-wide module state. Tests reset it with monkeypatch.
- Rationale: The selection order is user-visible: it decides which model grades and who is billed. Lazy imports let the defaults land before the provider families they name.
- Assumption: When Mistral, xAI, Voyage or Content Safety are ported, they expose `get_mistral_providers(env)`, `MistralEmbeddingProvider({env})`, `get_xai_providers(env)`, `VoyageEmbeddingProvider(model, config, env)` and `AzureModerationProvider(model, {env})` at those paths, mirroring upstream. A port at another path has to update the loaders.
- Implication: Until then, a user with only `MISTRAL_API_KEY` or `XAI_API_KEY` gets OpenAI graders (which fail without `OPENAI_API_KEY`), where upstream would use Mistral or xAI.
- Modules: src/promptphoo/providers/defaults.py, src/promptphoo/providers/openai/defaults.py, src/promptphoo/providers/anthropic/defaults.py
- Commits: `956e36a`

## 34. Grader references reuse configured providers; the final test shares provider instances
- Decision:
  - `grading_provider.py` ports `src/util/gradingProvider.ts`. The evaluator calls `resolve_runtime_grading_provider_references` after preparing each test, as upstream's `buildRunEvalOptions` does, so a grader written as the id or label of a configured provider reuses that loaded instance.
  - `get_final_test` copies the test with `_clone_test_value`, which deep-copies plain data but shares `ApiProvider` instances and callables, and applies JS truthiness to `assertion.provider || test.options.provider` and to `rubricPrompt`.
  - `--grader` and `defaultTest.options.provider` were already forwarded by `config/load.py` and `cli/eval.py`, and `get_grading_provider` reads the `defaultTest` fallback from `cli_state.config()`.
- Rationale: Resolved graders are provider instances inside `options` and `assert`, and they hold clients and locks that `copy.deepcopy` cannot copy. Upstream likewise reattaches `test.provider` after its clone, because the clone loses functions.
- Modules: src/promptphoo/grading_provider.py, src/promptphoo/assertions/index.py, src/promptphoo/evaluator/run.py
- Commits: `956e36a`

## 35. Parity sets PROMPTFOO_CAPTURE_PATH and compares captured grader prompts byte for byte
- Decision:
  - `tools/parity.py` gives each Node and Python run its own `PROMPTFOO_CAPTURE_PATH` (`.parity/<fixture>/{node,py}.capture.txt`), as upstream's smoke tests do for capture-script graders. When either run writes a capture, the two files must be identical, or the fixture FAILs before the JSON diff.
  - The `conversation-relevance` template (`matchers/deepeval.py`) is kept as Python string literals, not vendored through `_data/grading_prompts.json`, because upstream builds it in functions from the window. `tools/conversation_relevance_golden.mjs` runs upstream's own `conversationRelevancyTemplate.ts` under `node --experimental-strip-types` (and real Nunjucks for custom `rubricPrompt`s) to write `tests/fixtures/conversation_relevance_golden.json`.
- Rationale: Without the variable, `capture-conversation-relevance-grader.js` throws in both CLIs, and the fixture then only compares exec-provider error text, which tests nothing about the grader prompt. With it, the SSTI fixture checks the actual prompt sent to the grader.
- Assumption: Fixtures that do not use a capture script ignore `PROMPTFOO_CAPTURE_PATH`. Upstream's smoke configs read it only from capture scripts.
- Modules: tools/parity.py, tools/conversation_relevance_golden.mjs, src/promptphoo/matchers/deepeval.py, src/promptphoo/assertions/deepeval.py
- Commits: `a3a1ffc`

## 36. Resume, retry-errors and Ctrl+C pause replay the stored eval; the comparison pass finishes its Phase 5 branches
- Note: Supersedes D29.
- Decision:
  - `eval --resume [evalId]`, `eval --retry-errors` and `retry <evalId>` port `src/commands/eval.ts`, `src/node/doEval.ts`, `src/commands/retry.ts` and the resume branches of `src/evaluator.ts`. They read and write the same `evals` / `eval_results` rows as Node, so either CLI can resume or retry an eval written by the other. This was checked by hand in both directions for pause then resume, retry-errors and `retry`.
  - A resume replays the stored eval:
    - It uses the saved config and the persisted `runtimeOptions`: repeat, cache, maxConcurrency, delay, filterRange and the provider filter. CLI filters, `--grader`, `--var` and `--tag` are ignored with upstream's warnings or errors.
    - Steps whose `(testIdx, promptIdx)` pair is already stored are skipped (`Eval.read_completed_index_pairs`). Retry mode excludes ERROR rows from that set.
    - Saved prompt columns and metrics are restored (`restorePromptColumns`, `buildCompletedPrompts`, `backfillNamedScoreWeights`). If the columns no longer match, the run raises upstream's "Cannot resume evaluation because the saved provider/prompt columns differ" invariant.
  - Retry deletes the old ERROR rows only after the re-run succeeds (chunked by 500), then calls `recalculate_prompt_metrics` and rewrites any JSONL `outputPath`. If a result failed to persist, the old rows are kept and the retry fails with the recovery hint.
  - Pause: when the CLI writes to the DB, the first SIGINT aborts a `pauseSignal` (`scheduler/cancellation.py`). In-flight steps are cancelled through `await_with_abort`, completed rows are kept, and the eval is saved with upstream's "Evaluation paused. ID: …" and resume hint, exit 0. A second SIGINT, or 10 s without finishing, calls `os._exit(130)`.
  - The comparison pass now has the branches D29 left out:
    - It skips completed comparisons on resume.
    - It filters `cliState._retryErrorResultIds` (`cli_state.retry_error_result_ids`) out of the rows it compares.
    - Rows that failed to persist join the comparison through `readFailedResultsByTestIdx` and are never saved.
    - Grading cancellation calls `markComparisonRowsAborted`, which saves `Aborted: <reason>` and a `__promptfoo.comparisonError` so a later resume restores and regrades the verdict.
- Rationale: Node's resume state lives entirely in the database, so porting the DB-backed paths line for line is what lets the two CLIs resume each other's runs.
- Assumption: `asyncio` task cancellation is an acceptable stand-in for upstream's AbortSignal plumbing: provider calls stop at the next await, as Node's do when the signal reaches them.
- Implication: Not ported:
  - Extension-hook prompt recovery (`preparePromptRecovery`).
  - `--share` after retry.
  - Streaming JSONL writers that append during a resume. Python writes JSONL once, at the end.
  - Progress-bar updates and the repeat-stability summary.
  - Deferring in-flight grading until the pause completes. Python cancels it too, and the rows are marked aborted.
  - The `isEvalTimedOut` continue-after-timeout path. On `maxEvalTimeMs`, Python marks the comparison rows aborted and re-raises.
- Modules: src/promptphoo/cli/eval.py, src/promptphoo/cli/retry.py, src/promptphoo/cli_state.py, src/promptphoo/evaluator/run.py, src/promptphoo/evaluator/comparison.py, src/promptphoo/models/eval.py
- Commits: `c65714b`

## 37. --filter-failing reads either a results JSON file or a stored eval id and matches by vars and provider
- Decision:
  - `filters.filter_tests_by_results` ports `src/util/eval/filterTests.ts` and `src/util/comparison.ts`:
    - A path ending in `.json` is read as a results file. Anything else is looked up as an eval id and summarized with `to_evaluate_summary`.
    - Results match tests by `js_deep_equal` on vars after the runtime vars (`_`-prefixed keys and `sessionId`) are dropped, plus the provider identifier when the test names one. `defaultTest.vars` are merged first, and the unmerged vars are tried as a fallback.
    - Runtime vars from the matching result are restored onto the test. A result with runtime vars is preferred.
    - Runtime-generated tests that match no suite test are extracted from `testCase`, without its provider, and deduplicated.
  - `--filter-failing` selects `!success`, `--filter-failing-only` selects ASSERT failures and `--filter-errors-only` selects ERROR rows. Failing-only and errors-only together take the union. The metadata filter also applies to extracted tests, and the empty-result warning is upstream's.
- Rationale: The same rules as Node mean the same selection from the same eval. This was checked against Node for all three flags and their union, from both a DB id and a JSON file.
- Modules: src/promptphoo/filters.py, src/promptphoo/cli/eval.py
- Commits: `c65714b`

## 38. Extension hooks run in-process through transform; JavaScript extensions are rejected before the eval starts
- Decision:
  - `evaluator/extensions.py` ports `runExtensionHook`, `getExtensionHookName` and `restorePromptFunctions` (src/evaluatorHelpers.ts) and `runBeforeAllExtensions` / `synchronizeLegacyTransportHeaders` (src/evaluator.ts). Hooks go through `transform.transform(ext, ..., validate_return=False)`, so `file://hooks.py:fn` runs in-process with JSON-round-tripped arguments, as Node's `runPython` does, and a hook with no function name calls `get_transform`.
  - Calling conventions, hook filtering and mutation rules are upstream's: a known hook name (`:beforeAll`) runs only for that hook as `fn(context, {hookName})`; any other name runs for every hook as `fn(hookName, context)`. beforeAll may replace the suite's prompts, providerPromptMap, tests, scenarios, defaultTest, nunjucksFilters, derivedMetrics and redteam (a key missing from the returned suite becomes undefined); beforeEach replaces the test; afterEach merges finite namedScores, metadata and response.metadata into the row before it is persisted, and a failing afterEach is logged and ignored; afterAll's return value is ignored.
  - `run.py` calls beforeAll before the prompt columns are built, beforeEach per eval step (inside the per-step timeout), afterEach before `append_result`, and afterAll after `set_vars`, with results read back through `load_results()`.
  - `check_extensions_supported` rejects `.js/.ts/.mjs/...` extensions when the eval starts, so an `:afterAll`-only JS hook does not fail after the whole eval has run. The CLI logs `ExtensionHookError` on one line and exits 1.
  - The seeded providerPromptMap check compares by value only. Upstream also compares array identity, which a hook's JSON copy never shares, so the outcome is the same.
- Rationale: Upstream's own Python hooks already see JSON copies, so reusing the transform path keeps the payloads, error wrapping (`Extension hook "<hook>" failed for <ext>: Transform failed ...`) and function-name parsing identical.
- Assumption: Every hook context is JSON-serializable through `json_stringify` (providers through `to_json`). A hook that needs a live provider object cannot get one, as under Node's Python runner.
- Implication: `python-prompt-with-extension.yaml` now passes parity, so its EXPECTED entry is gone.
- Modules: src/promptphoo/evaluator/extensions.py, src/promptphoo/evaluator/run.py, src/promptphoo/cli/eval.py, tools/parity.py
- Commits: `597e573`

## 39. --watch polls file stamps instead of using chokidar or a watch library
- Decision:
  - `cli/watch.py` ports the doEval watch block and `resolveTestsWatchPaths` with its helpers (src/util/testCaseReader.ts). The watch list is built after the first run from the config paths, `file://` prompts and providers, and the tests (CLI `--tests` against the CWD, `--vars` against the config directory, else the config's tests and test sources), exactly as upstream builds it.
  - The loop polls each file's `(mtime_ns, size)` every 0.5 s. A change, or a file that reappears after an atomic save, triggers a rerun; a deletion does not (chokidar's `change` with `atomic: true`). Paths matching chokidar's `ignored: /^\./` are skipped. Ctrl-C logs `Stopping watch mode...`.
  - Each rerun is a fresh `asyncio.run(do_eval(...))` on a copy of the original command options. Upstream's recoverable errors (EvalRunError, ConfigResolutionError, rejected prompt suggestions) plus `ExtensionHookError` are logged and watching continues. Watch mode skips the pass-rate exit code, as upstream does.
  - `do_eval` takes an optional `watch_state` dict that receives the resolved config, config paths and test sources.
- Rationale: `watchfiles` and `watchdog` are not dependencies. Polling a handful of explicit files is cheap, and the list is fixed after the first run upstream too.
- Assumption: Users watch a small set of files, so a 0.5 s poll is fast enough and costs nothing noticeable.
- Implication: Upstream's quirk of watching a `file://prompt.py:fn` prompt as the literal path `prompt.py:fn` (which never changes) is kept.
- Modules: src/promptphoo/cli/watch.py, src/promptphoo/cli/eval.py
- Commits: `597e573`

## 40. list/show/delete queries: upstream's prefix lookups and in-memory grouping, ported as is
- Decision:
  - `db/queries.py` ports the read and delete helpers of `src/util/database.ts` (`getPrompts`, `getTestCases`, `getPromptFromHash`, `getDatasetFromHash`, `getEvalFromId`, `deleteEval`, `deleteEvals`, `deleteAllEvals`), `deleteTraceRecordsForEvals` and `EvalQueries.getVarsFromEvals`. `models/eval.py` gained `Eval.get_many` (upstream `Eval.getMany`) and `Eval.get_table`, appended without touching the existing methods.
  - Upstream's quirks are kept: prompts and datasets are grouped in memory over the newest 100 evals; `getEvalFromId` matches an id prefix, but `show <prefix>` and `delete <prefix>` then look the prefix up as an exact id and fail with upstream's messages; `Eval.getMany` does not read the `results` column, so a legacy v3 eval lists with no prompts and never appears under `list prompts`.
  - `localeCompare` sorts use `locale_compare_key` (case-folded, then lower case first), which agrees with ICU root collation on eval ids and hex hashes.
  - Deleting an eval removes spans, traces, join rows and results in upstream's order. `blob_references` rows go through their `ON DELETE CASCADE` foreign key, and `blob_assets` stay, as upstream.
  - The delete signal is written as `{"type":"delete","deletedEvalIds":[...],"timestamp":...}` (no ids for "all"), without upstream's coalescing with a signal pending from the last few hundred milliseconds.
  - Query helpers open their own cursor with `row_factory = None`, because `Eval.find_by_id` leaves a dict row factory on the shared connection.
  - Not ported: `getEvalSummaries` and the standalone-eval cache, which only the web server and MCP use.
- Rationale: Users and scripts see the prefix behaviour and the sort orders, and the goldens recorded from Node pin them.
- Assumption: Eval ids contain only ASCII letters, digits, `-` and `:`, so the casefold key orders them as ICU does.
- Implication: A promptphoo `delete` followed within a moment by another write does not merge both into one signal, so a running Node viewer may miss the delete until its next refresh.
- Modules: src/promptphoo/db/queries.py, src/promptphoo/models/eval.py
- Commits: `89b55d0`

## 41. list/show/delete output is checked byte for byte against goldens recorded from Node
- Decision:
  - `cli/list_.py`, `cli/show.py` and `cli/delete.py` port `src/commands/list.ts`, `show.ts` and `delete.ts` on top of the existing `wrap_table` / `generate_table` ports. Commander's `show [id]` and `delete <id>`, which also have subcommands, are click groups (`IdOrSubcommandGroup`) that send a first argument naming no subcommand to a hidden default command. Every command runs the startup migrations, as `main.ts` does, and exits with upstream's `process.exitCode`.
  - `tools/list_show_golden.py` runs real Node on three echo evals plus one imported legacy v3 eval, and writes the DB rows of each stage and Node's output for 37 commands to `tests/fixtures/list_show/golden.json`. `tests/test_cli_list_show_delete.py` loads those rows into a migrated DB and compares promptphoo's output byte for byte, ANSI table colours included, and the table row counts after Node's delete sequence.
  - Node's logger writes errors to stdout, while promptphoo's handler sends them to stderr, so the comparison uses the merged stream. (Since the Phase 5 merge, promptphoo routes every level to stdout like Node, or all to stderr with PROMPTFOO_LOG_TO_STDERR=true; the merged-stream comparison still holds.) Node's "There were some errors during the operation. See logs for more details." footer, with its per-run log file paths, is cut from the goldens, because promptphoo writes no log files.
  - `delete eval all` asks `? <message> (Y/n)` on stdout and reads one line with @inquirer/confirm's rules: an empty answer is yes (its default), otherwise only an answer starting with `y` is. Inquirer's cursor redraws and `✔` summary line are not reproduced. At end of input nothing is deleted and the exit code is 1, as Node exits on its unsettled prompt. `-y/--yes` skips the prompt; it is a promptphoo addition that Node does not accept.
  - Interop tests that run real Node (both writers, both readers, and deletes on copies of one DB) are skipped unless `NODE_PROMPTFOO` is set.
- Rationale: Recorded goldens keep the parity check in the default test run without Node, while the gated tests catch drift against a live CLI.
- Assumption: Node 0.125.1 is the reference. Regenerate the goldens with the tool after an upstream bump.
- Implication: Scripts that pipe an answer into `delete eval all` get the same result and exit code under both CLIs; only the terminal rendering of the prompt differs.
- Modules: src/promptphoo/cli/list_.py, src/promptphoo/cli/show.py, src/promptphoo/cli/delete.py, tools/list_show_golden.py, tests/test_cli_list_show_delete.py, tests/fixtures/list_show/golden.json
- Commits: `89b55d0`

## 42. export and import write and read Node's exact files and rows; only three fields are nondeterministic
- Decision:
  - `cli/export.py` ports `src/commands/export.ts`. `export eval <id|latest>` goes through `output.writer` (`-o` takes every `eval -o` format) or prints `createOutputData` as two-space JSON. `export logs` (`-n/--count`, `-o/--output`) rebuilds upstream's hand-rolled tar (same 512-byte headers, padding and two zero blocks) inside one zlib gzip stream at level 9. `getLogDirectory`/`getLogFiles` from `src/util/logs.ts` are private to that module.
  - `cli/import_.py` ports `src/commands/import.ts` with upstream's branch order: the `--force` preflight runs before the DB lookup, and the collision, `--new-id` and `--force` paths follow it, including the "was deleted for a --force replacement" disclosure. v3 summaries go through `db/import_export.create_eval_record`, a full port of `Eval.create` with every option (completed prompts, vars, runtime options, durations, prompts/datasets/tags rows), and then `create_many_from_evaluate_result`. Legacy v2 summaries get the bare `evals` insert. `importers/` ports `src/importers/` (format sniffing and the OpenAI Evals JSONL converter).
  - Traces in an export are validated and written to the `traces`/`spans` tables, and trace ids are remapped as upstream remaps them (`--new-id`, duplicates, ids already in the store). Export emits `traces` too, see D53.
  - Error text copies Node where users see it: `ENOENT: no such file or directory, open '<file>'`, V8's `Unexpected token 'x', "<text>" is not valid JSON` (approximated for long inputs), and `Failed to export eval: Error: <message>`. `JSON.parse` rejects NaN/Infinity, and JSONL lines are cut with JS `trim` semantics, which strip a BOM.
  - `output/export_to_file.py` ports `src/util/exportToFile/` (`convertEvalResultToTableCell`, `convertTestResultsToTableRow`, `JsonlFileWriter`) for reuse by table pages. `EvalResult` objects are first projected to a Node instance's own properties.
  - Three parity fixes outside this area came from the byte-for-byte comparison. `sanitize_config_for_output` reassigns an empty tracing `auth`/`headers` object, because `{}` is truthy in JS. The template engine gains Nunjucks's `escape`/`e` filter (`&quot;`, `&#39;`, `&#92;`, where markupsafe writes `&#34;`). `db.migrate` reads with a tuple cursor, so it survives a dict `row_factory` that `Eval.find_by_id` installs on the shared connection.
- Rationale: The requirement is interop. A Python export of a DB eval is byte-identical to Node's apart from `metadata.exportedAt` (wall clock) and `metadata.nodeVersion` (`v22.x` vs `python-3.x`). After a re-import, each result's `id` also differs, because import mints new eval_results uuids in both runtimes. An OpenAI Evals import also stamps `evals.created_at` with the wall clock. Normalising those four fields, Node and Python exports, imports and re-exports agree for v4 evals, legacy table-backed evals, traced exports and both OpenAI fixtures, in every output format and in every table row.
- Assumption: Node writes no other nondeterministic export field. `platform`/`arch` are equal on one machine, and the eval_results read order is SQLite's for the same query on the same file.
- Implication: `tests/test_import_export_node.py` checks all of this when `NODE_PROMPTFOO` is set. `tests/test_import_export.py` checks it offline against `tests/fixtures/import_export/node-v4-export.json`, which Node wrote. Traced evals keep `traces` on export and re-export, see D53.
- Modules: src/promptphoo/cli/export.py, src/promptphoo/cli/import_.py, src/promptphoo/db/import_export.py, src/promptphoo/importers/parse.py, src/promptphoo/importers/openai_evals.py, src/promptphoo/output/export_to_file.py, src/promptphoo/sanitizer.py, src/promptphoo/templating.py, src/promptphoo/db/__init__.py, tests/test_import_export.py, tests/test_import_export_node.py
- Commits: `a4dfc5e`

## 43. Embedded blob assets: validated like upstream, then dropped until the blob store exists
- Decision: `import` checks `blobAssets` exactly as upstream checks them: the shape check, the `collectBlobHashes` reference scan (depth 64, 100 000-char strings), the 50 MB limit and the base64 length bound, Node-lenient base64 decoding, size and sha256 checks, and the same error messages. The checks run before the `--force` delete. Valid, referenced assets are then skipped with one warning instead of being stored, and `blob_assets`/`blob_references` stay empty. `export eval --include-media` is accepted and does nothing, so no `blobAssets` are emitted.
- Rationale: Phase 6 owns the filesystem blob store (docs/PLAN_PORT.md). Validating now means a corrupt portable export fails the same way in both runtimes and cannot delete the eval it was meant to replace.
- Assumption: Until Phase 6, users who need media round-trips import with Node. The eval's `promptfoo://blob/` references are kept verbatim, so a later Phase 6 store (or Node) can resolve them.
- Implication: Phase 6 replaces `import_blob_assets` with `storeBlob` plus `recordBlobReference(location='import')`, and fills `blobAssets` in `create_output_data` when `include_media` is set.
- Note: Superseded by D49.
- Modules: src/promptphoo/cli/import_.py, src/promptphoo/cli/export.py
- Commits: `a4dfc5e`

## 44. `share` is registered but always fails; results stay local
- Decision:
  - `cli/share.py` registers `share [id]` with upstream's `--show-auth` and `-y/--yes` options, logs a not-supported error that points to `promptfoo export`, and exits 1. Nothing is uploaded.
  - `eval --share` keeps its existing warning-only behaviour in `cli/eval.py`: the eval still runs and is stored locally.
- Rationale: Sharing uploads results to promptfoo.app or a self-hosted server, and cloud features are out of scope (PLAN_PORT.md). A registered command with a clear error is better than click's "No such command", and failing the eval because of `--share` would break configs that set it routinely.
- Assumption: Users who need sharing run Node promptfoo against the same `~/.promptfoo` database, which promptphoo keeps compatible.
- Modules: src/promptphoo/cli/share.py
- Commits: `317cbb9`

## 45. Utility commands: zod-style validation messages, local-only target tests, Node-layout log files
- Decision:
  - `validate` and the config loader's schema warning render jsonschema errors as `z.prettifyError` would (`config/zod_errors.py`): zod's English messages, `toDotPath` paths, issues sorted by path length, the "exactly one of targets/providers" refine added only when no aborting issue exists, and upstream's `dedent` applied to the validate message. Upstream's second pass (TestSuiteSchema over the loaded suite) is not ported.
  - `validate target` behaves as upstream does with PROMPTFOO_DISABLE_REMOTE_GENERATION set: HTTP connectivity runs a one-test eval and never calls the cloud analyzer, and the session test makes both requests and then reports "remote grading is disabled". Cloud UUID targets fail.
  - `init --no-interactive` writes byte-identical files from templates vendored in `_data/init_templates.json`, regenerated by `tools/gen_init_templates.mjs`, which evaluates upstream's template literals with Node. The wizard uses numbered click prompts, and its redteam choice fails as not supported. `--example` downloads from GitHub through the shared fetch layer.
  - `cli.main.main` attaches per-run debug and error log files (`logfiles.py`) with Node's names, directory (PROMPTFOO_LOG_DIR), line format, 50-file pruning and the closing "There were some errors" notice. The debug file gets what the root logger passes (info and up, or debug with `--verbose`). Node's debug file always holds debug records.
  - `config` reads and writes `<configDir>/promptfoo.yaml` through `globalconfig.py` in js-yaml's dump layout. The file is only created by commands that read it, not at every startup as Node does.
  - `debug` keeps upstream's report shape, but `nodeVersion` becomes `pythonVersion` and loaded providers serialize as id, label and config.
- Rationale: These messages and files are what users and scripts see, and the log and config files are shared with Node. Lowering the root logger to DEBUG for the file would change console-dependent behaviour (`eval --table` checks the level) and the cost of debug logging everywhere.
- Assumption: config-schema.json stays generated from upstream's zod schemas, so a schema error maps one-to-one to a zod issue. Where they diverge (issue order inside one object, unions whose branches fail in different ways, refines the JSON schema cannot express), the text is approximate.
- Implication: Every log level now goes to stdout, as Node's winston console does, or all of it to stderr when PROMPTFOO_LOG_TO_STDERR=true (fixed at the Phase 5 merge). Text and stream both match.
- Modules: src/promptphoo/cli/validate.py, src/promptphoo/config/zod_errors.py, src/promptphoo/config/load.py, src/promptphoo/cli/init.py, src/promptphoo/onboarding.py, src/promptphoo/logfiles.py, src/promptphoo/cli/logs.py, src/promptphoo/cli/main.py, src/promptphoo/globalconfig.py, src/promptphoo/cli/config.py, src/promptphoo/cli/debug.py, src/promptphoo/cli/cache.py, src/promptphoo/cli/feedback.py, tools/gen_init_templates.mjs
- Commits: `317cbb9`

## 46. --suggest-prompts is ported; a rejected suggestion run exits 1 without the empty summary
- Decision:
  - `suggestions.py` ports `generatePrompts` (src/suggestions.ts), `maybeAddGeneratedPrompts` and `PromptSuggestionsRejectedError` (src/evaluator.ts) and `promptYesNo`. It uses the default `suggestionsProvider`, upstream's `SUGGEST_PROMPTS_SYSTEM_MESSAGE` payload, four concurrent calls, and keeps successful variants when others fail. `cli/eval.py` ports `resolveSuggestionOptions`, so `--suggest-prompts N` (1-50) enables suggestions by itself, and `evaluateOptions.generateSuggestions` / `suggestionsCount` from the config still work.
  - When the user accepts no variant, the eval raises `PromptSuggestionsRejectedError`, which the CLI logs and exits 1. Upstream sets `process.exitCode = 1` and goes on to print the summary of the empty eval.
  - The eval-flag audit against commander's options also changed: `--tag` now merges `commandLineOptions.tags` under the CLI tags (upstream `runtimeTagsForEval`), `--verbose` is accepted on `eval`, and `-w`, `-x` and `--suggest-prompts` have their help texts. Out-of-range numeric flags fail with click's usage error (exit 2) rather than upstream's "Invalid command options" (exit 1).
- Rationale: The interactive question makes suggestions a terminal feature, so the empty summary after a refusal carries no information.
- Modules: src/promptphoo/suggestions.py, src/promptphoo/cli/eval.py, src/promptphoo/evaluator/run.py
- Commits: `597e573`

## 47. generate dataset/assertions keep upstream's prompts by porting dedent over the raw template segments
- Decision:
  - `cli/generate.py` ports src/commands/generate/{dataset,assertions}.ts. `synthesis/dataset.py` and `synthesis/assertions.py` port src/testCase/synthesis.ts and src/assertions/synthesis.ts. The default provider is `get_default_providers()["synthesizeProvider"]`; `--provider` loads through `load_api_provider`.
  - `synthesis.dedent` ports the npm dedent 1.7.2 tag. `tools/gen_synthesis_templates.py` copies each upstream ``dedent`...` `` literal's raw segments (escapes unprocessed, as `strings.raw`) into `synthesis/templates.py`, and the Python code supplies the interpolated values. dedent's quirks therefore carry over, such as turning the raw `\\b` of the Python examples into `\` plus a backspace.
  - `tools/synthesis_golden.mjs` runs upstream's own prompt builders, cut out of the TS sources, with the real dedent and writes `tests/fixtures/synthesis_golden.json`. A manual run of both CLIs against the same fake `file://` Python provider gave byte-identical prompts, YAML and CSV output, and written configs.
  - `resolve_configs` gained upstream's `type` argument (`config_type`), so dataset and assertion configs need no providers.
  - The cache is always disabled, as upstream's `.option('--no-cache', ..., false)` makes commander default `cache` to false. No progress bar is drawn.
- Rationale: Retyping the prompt texts by hand would drift from upstream in whitespace and escapes. Generated raw segments plus a dedent port stay exact and can be regenerated on an upstream sync.
- Assumption: Upstream keeps writing these prompts as `dedent` template literals. A switch to plain strings would need the generator changed.
- Modules: src/promptphoo/cli/generate.py, src/promptphoo/synthesis/__init__.py, src/promptphoo/synthesis/dataset.py, src/promptphoo/synthesis/assertions.py, src/promptphoo/synthesis/templates.py, src/promptphoo/config/load.py, tools/gen_synthesis_templates.py, tools/synthesis_golden.mjs
- Commits: `597e573`

## 48. The blob store is a synchronous port of src/blobs, hooked in at upstream's call sites
- Decision:
  - `src/promptphoo/blobs/` ports constants.ts, types.ts, index.ts (`store_blob`, `get_blob_by_hash`, `record_blob_reference`, the share-authorization checks, `is_safe_inline_blob_mime_type`, the provider get/set/reset functions), filesystemProvider.ts, blobRefs.ts and extractor.ts. Files go to `<configDir>/blobs/ab/cd/<sha256>` with the same pretty-printed `.meta.json` sidecar, staged in a `mkdtemp` directory and published data first. Rows go to `blob_assets` and `blob_references` with drizzle's SQL, including `ON CONFLICT DO NOTHING` and the `CURRENT_TIMESTAMP` defaults.
  - The API is synchronous, because the filesystem and sqlite3 are, and the store-once cache needs no pending-promise bookkeeping. Upstream runs `Promise.all` over `images[]`, `turns[]` and metadata arrays. Each callback reaches its cache entry or its store before the first await, so sequential iteration gives the same stores, locations and reference rows.
  - The extractor runs where upstream runs it: in `_apply_response_outcome` after the transforms (the grader still sees the raw `audio`), in `EvalResult.create_from_evaluate_result`, and in `create_many_from_evaluate_result` unless `PROMPTFOO_INLINE_MEDIA` is set. `is_blob_storage_enabled` reads `PROMPTFOO_INLINE_MEDIA` through `get_env_bool`, so the config's `env:` layer applies as in Node. `PROMPTFOO_SHARE_INLINE_BLOBS` is share-only and is not read.
  - Upstream's failure modes are kept. A store whose `blob_references` insert fails its foreign key (as with `eval --no-write`, where the eval row never exists) raises, and the test case becomes an error row. Node 0.125.1 does the same; promptphoo's message is sqlite3's `FOREIGN KEY constraint failed` instead of drizzle's `Failed query: ...`. The `b64_json` branch swallows every error, storage errors included, as upstream's try/catch does.
  - The raw-SVG preview check replaces fast-xml-parser's `XMLValidator` with a non-namespaced expat parse plus the ported fxp parser's `svg` root check. The two agree on ordinary SVG, but an SVG using an undeclared entity such as `&nbsp;` passes fxp and fails expat, so it gets no preview blob.
  - `remoteUpload.ts`, `shareUpload.ts` and `src/storage/` (the provider-side media store used by the video providers) are cloud or provider features and are not ported.
  - `tests/conftest.py` resets the provider singleton after every test, because it caches its base path.
- Rationale: A Python eval and a Node eval over the same config must leave byte-identical blob files and the same rows (tests/test_blobs_node.py checks this, along with the persisted responses). Keeping upstream's call sites and error paths keeps that true for the failure cases too.
- Assumption: Node keeps creating the eval row before it runs. If upstream started persisting `blob_references` without an eval row, `--no-write` would stop failing in Node and this port would need the same change.
- Implication: `promptfoo://blob/<hash>` URIs in Python-written results now resolve through the same store in both runtimes. The parity fixtures carry no media, so the parity summary is unchanged.
- Modules: src/promptphoo/blobs/__init__.py, src/promptphoo/blobs/constants.py, src/promptphoo/blobs/types.py, src/promptphoo/blobs/blob_refs.py, src/promptphoo/blobs/filesystem_provider.py, src/promptphoo/blobs/extractor.py, src/promptphoo/evaluator/run.py, src/promptphoo/models/eval.py, src/promptphoo/db/import_export.py, tests/conftest.py, tests/test_blobs.py, tests/test_blobs_node.py, tests/fixtures/blobs/provider.py
- Commits: `f21cee5`

## 49. `import` stores embedded blobs and `export --include-media` embeds them, as upstream
- Decision:
  - `import` stores each validated, referenced `blobAssets` entry with `store_blob` before the `--force` delete. Its MIME type is sanitized to the inline-safe set, or `application/octet-stream` otherwise. The stored hash is re-checked. After the results are written, `record_blob_reference(location='import')` links each asset to the new eval and upgrades any reference the extractor recorded while it scanned the imported results. The validation and the error messages are as in D43.
  - `create_output_data(..., include_media=True)` ports `exportBlobAssets`. It scans the results (without `metadata.blobUris` when PROMPTFOO_STRIP_RESPONSE_OUTPUT is on) and the traces with `collect_blob_hashes`, which returns an insertion-ordered dict so the assets come out in Node's order. It reads only share-authorized blobs (a reference with a kind, or one from an import), skips missing and oversized ones with a warning, and writes `blobAssets`. `write_output` passes the flag to the json and yaml/yml/txt writers only, as upstream does.
- Rationale: These were the follow-ups D43 deferred to the blob store. With them, a media eval can move between the runtimes in either direction.
- Assumption: The export format stays `{hash, mimeType, sizeBytes, data}` with standard base64.
- Implication: An `--include-media` export written by either runtime from either runtime's DB is identical apart from exportedAt and nodeVersion. Importing it into either runtime gives the same blob files and rows. SVG previews come back as `application/octet-stream` in both runtimes, by upstream's design.
- Note: Supersedes D43.
- Modules: src/promptphoo/cli/import_.py, src/promptphoo/cli/export.py, src/promptphoo/output/writer.py, src/promptphoo/db/import_export.py, tests/test_import_export.py, tests/test_blobs_node.py
- Commits: `f21cee5`

## 51. The OTLP receiver is an asyncio port of upstream's express app, byte for byte at the HTTP level
- Decision:
  - `tracing/otlp_receiver.py` ports `src/tracing/otlpReceiver.ts`: `POST /v1/traces` (OTLP/JSON and OTLP/protobuf, gated by `acceptFormats`), `GET /v1/traces` (service info), `POST /v1/logs` (OTLP/JSON logs become 1 ms child spans), `GET /health` and `GET /debug/status`. Each handler is a straight port, covering id conversion (hex, base64 via Node-lenient decoding, zero ids), attribute parsing, the receiver-created trace rows for `evaluation.id` spans, ingest-time redaction (span name and status message echoes too), registered per-evaluation trace policies, and the log body/name limits.
  - There is no web framework. A small HTTP/1.1 server on `asyncio.start_server` (keep-alive, chunked bodies, `Expect: 100-continue`) stands in for express 5 and body-parser 2:
    - routing is case-insensitive with an optional trailing slash, and HEAD is answered as GET;
    - the JSON parser is strict and has a 10 MB limit, and `{}` stands in for an empty body;
    - gzip, deflate and identity bodies are accepted;
    - `X-Powered-By: Express`, weak ETags, finalhandler's 404 page, and body-parser's status codes are copied. Invalid JSON is a 400 `{"error":"Invalid JSON"}`; other parser errors go to the global handler as a 500 `{"error":"Internal server error"}`.
  - `listen` rejects with Node's message (`listen EADDRINUSE: address already in use <host>:<port>`). `start_otlp_receiver` drops the singleton after a failed listen, so the next start makes a new receiver.
  - `tracing/protobuf.py` is a pure-Python decoder for ExportTraceServiceRequest that follows protobufjs 8's generated decoders:
    - a field with the wrong wire type is skipped;
    - a repeated singular message merges into the earlier one, and the last oneof member wins;
    - strings must be valid UTF-8;
    - nesting is limited to depth 100, and an overflowing length prefix is rejected;
    - errors carry protobufjs's text, prefixed `Invalid protobuf data:`.
  - `toObject` is applied with `defaults: true`, so 64-bit values become JS numbers.
  - `tools/otlp_protobuf_golden.py` fuzzes the decoder against protobufjs through `tools/otlp_node.mjs`. Its output, `tests/fixtures/tracing/protobuf_golden.jsonl`, holds 240 valid and mutated payloads with Node's results, and the tests replay it offline.
  - `jsvalues.number_to_string` now prints integral doubles of 2^53 and above with JS's shortest round-trip digits (`2**60` -> `1152921504606847000`), as the decoded 64-bit fields need.
- Rationale: Providers export to the receiver with stock OTLP SDKs, so what they see (status codes, bodies and headers) must be Node's. `protobuf` and `opentelemetry-proto` are not dependencies, and the request schema is small and fixed.
- Assumption: Exporters send `application/json` or `application/x-protobuf` with identity, gzip or deflate encoding. Node also decodes `br` and `zstd`, which this receiver answers with a 500. A JSON array where an id string is expected is converted by the JS string coercion upstream applies, which is not reproduced for every shape.
- Implication: Run against Node's receiver (via `tools/otlp_node.mjs serve`), 36 hand-picked requests (routing, content types, encodings, malformed bodies, the size limit, logs) get the same status, headers and body from both, apart from `/debug/status` uptime. `tests/test_tracing_otlp_receiver.py` ports upstream's receiver tests, and `tests/test_tracing_protobuf.py` ports the protobuf tests and replays the golden file.
- Modules: src/promptphoo/tracing/otlp_receiver.py, src/promptphoo/tracing/protobuf.py, src/promptphoo/jsvalues.py, src/promptphoo/cli_state.py, tools/otlp_node.mjs, tools/otlp_protobuf_golden.py, tests/fixtures/tracing/protobuf_golden.jsonl, tests/test_tracing_otlp_receiver.py, tests/test_tracing_protobuf.py
- Commits: `3b43b7d`

## 52. One TraceStore writes Node's rows; the evaluator leases the shared receiver as upstream does
- Decision:
  - `tracing/store.py` replaces D54's shim and is the only writer of the `traces`/`spans` tables. `get_trace_store()` returns a singleton TraceStore with upstream's async methods: `create_trace`, `add_spans`, `get_traces_by_evaluation`, `get_trace`, `get_trace_metadata`, `delete_old_traces` and `get_spans`. Each has a `*_sync` twin, which `import`, `export` and the output writer call.
  - Row writes match drizzle's:
    - uuid4 ids and `created_at` in epoch ms;
    - `JSON.stringify` text for metadata and attributes, with NULL for undefined;
    - `ON CONFLICT DO NOTHING` on `trace_id` and on `(trace_id, span_id)`;
    - a span batch goes in one transaction.
  - `deleteOldTraces` also matches rows whose `created_at` is a SQLite timestamp string, as upstream's cutoff SQL does.
  - Reads also follow upstream. Span attributes are sanitized by default (`tracing/sanitize_attributes.py`: credential keys become `<redacted>`, configured patterns become `[REDACTED]`, and values are truncated at 400 UTF-16 units). `get_spans` ports the earliest-start, depth, internal-span, span-filter and maxSpans selection.
  - `cli/import_.py` writes traces through the store. The duplicate trace writers in `db/import_export.py` are gone.
  - `tracing/evaluator_tracing.py` replaces the shims of D54/D57. It ports the receiver lifecycle in full:
    - a refcounted lease, so overlapping evals share one receiver and the first eval's endpoint and formats are kept;
    - in-flight starts and stops are serialized, and a failed stop keeps the lease;
    - each eval's trace policy is registered and dropped when it releases;
    - retention pruning, and `tracing.failOnReceiverStartFailure`;
    - `cliState.activeOtlpReceiver`.
  - D57's `promptfoo.test_case` root span through the GenAI tracer is kept.
  - `Evaluator.evaluate` follows upstream's order: acquire the receiver, initialize OTEL, run; then flush OTEL, wait `SPAN_EXPORT_GRACE_SECONDS` (3 s) if a receiver is running, and release the lease.
- Rationale: The coordinator's contract kept the shim's method names and camelCase dict shapes, so the trace assertions, the local span exporter and import all share one store. sqlite3 is synchronous, so each async method wraps its sync twin and both kinds of caller can use it.
- Assumption: Python file providers run in a worker thread (`python_hooks.call_async`), so a provider that POSTs to the in-loop receiver cannot deadlock it. Spans are read back only after the grace wait.
- Implication: `tests/test_tracing_receiver_node.py` runs the same traced eval under both CLIs, skipped without NODE_PROMPTFOO. Its stdlib provider (`tests/fixtures/tracing/otlp_posting_provider.py`) exports OTLP/JSON and OTLP/protobuf spans to the eval's receiver. The provider-posted span rows and the trace rows match Node's. Tracing telemetry (`record('feature_used')`) is not ported.
- Modules: src/promptphoo/tracing/store.py, src/promptphoo/tracing/sanitize_attributes.py, src/promptphoo/tracing/evaluator_tracing.py, src/promptphoo/evaluator/run.py, src/promptphoo/cli/import_.py, src/promptphoo/db/import_export.py, tests/test_tracing_store.py, tests/test_tracing_sanitize_attributes.py, tests/test_tracing_evaluator_tracing.py, tests/test_tracing_receiver_node.py, tests/fixtures/tracing/otlp_posting_provider.py
- Commits: `3b43b7d`

## 53. `export` emits `traces` as upstream does, and trace parity compares them
- Decision:
  - `create_output_data` adds `traces` right after `runtimeOptions`. The traces come from `get_trace_store().get_traces_by_evaluation(evalId)` with attributes sanitized, and the key is left out when there are none.
  - `project_traces_for_output` ports projectTracesForOutput:
    - PROMPTFOO_STRIP_METADATA drops trace metadata;
    - PROMPTFOO_STRIP_TEST_VARS drops `metadata.vars`, and the metadata too if nothing is left;
    - stripping prompt text or response output removes the `promptfoo.request.body` / `promptfoo.response.body` span attributes, and an emptied `attributes` goes with them.
  - A failed fetch logs upstream's warning ("Failed to fetch traces for output <id>; traces omitted from export: <err>") and writes the file without traces. Blob export scans these traces, as upstream's does.
  - `tools/parity.py` no longer ignores `traces` for the trace-assertions fixture. Exported spans are compared by the name path from the root, without span ids or times, and traces and spans are sorted. Node's and Python's 2 traces and 23 spans then match.
- Rationale: This closes the gap D42 left open: a traced eval no longer loses its traces on a Python export or re-export.
- Assumption: Within one trace, two spans with the same name path have the same attributes and status, so sorting them gives a stable order.
- Implication: `tests/test_tracing_receiver_node.py` checks that Node's `export` of the Python-written eval equals Python's (and the reverse), apart from `exportedAt` and `nodeVersion`. It also checks that import followed by export in either runtime keeps `traces`. `tests/test_output_traces.py` ports upstream's trace cases from the output and JSON-export tests.
- Modules: src/promptphoo/output/writer.py, tools/parity.py, tests/test_output_traces.py
- Commits: `3b43b7d`

## 54. Trace assertions read the trace store through the assertion context; the evaluator gives each traced step a W3C trace context
- Decision:
  - `assertions/trace.py` ports traceUtils.ts and the `trace-span-count`, `trace-span-duration` and `trace-error-spans` handlers with upstream's error strings and thresholds. As upstream, they read the raw `assertion.value` and ignore a `not-` prefix. The attribute filter compares with JS `===`, so `1` never matches `true`.
  - `run_assertion` / `run_assertions` port loadTraceData: they poll `get_trace_store().get_trace(traceId, {sanitizeAttributes: false})` until the span count is stable, with the `PROMPTFOO_TRACE_FETCH_*` env bounds, preload once per test, and set `context["trace"]` = `{traceId, evaluationId, testCaseId, metadata, spans}` for every trace-aware assertion (`trace-*`, `trajectory:*`, `python`, `file://` / `package:` values). The trajectory assertions already read `context["trace"]`, so they now grade real spans. Grader calls get a child `traceparent` in their call context.
  - `evaluator/run.py` ports createRunEvalTest's `tracingEnabled`/`evaluationId` metadata stamp, generateTraceContextIfNeeded before the target call (`traceparent`, `evaluationId`, `testCaseId` on the call context), collectExternalTraceAfterProviderCall when `tracing.provider` is external, `traceId` for the assertions, and getTraceId/getTraceLinkage (`traceId`, `evaluationId`) on success and error rows.
  - `tracing/trace_context.py` ports fetchTraceContext (external provider, else the local store), extractTraceIdFromTraceparent and the TraceContextData shape. In-flight external fetches are shared per config object (`id()` of the dict, as upstream's WeakMap key). `tracing/span_filter.py`, `tracing/tool_attributes.py` and `tracing/span_roles.py` port their upstream files; `trajectory.py` now imports the tool-attribute keys and `matches_pattern` from them instead of keeping copies.
  - Shims (since replaced, see D52), each headed `# SHIM: replaced by the trace-store branch at merge`: `tracing/store.py` (a faithful TraceStore over the drizzle `traces`/`spans` tables), `tracing/sanitize_attributes.py` and `tracing/evaluator_tracing.py`. The last one has only the trace-context half of evaluatorTracing.ts, with no OTLP receiver and no OpenTelemetry root span, so its trace and span ids are random as when Node's root span is invalid.
  - `span_roles.py` keeps the role in a ContextVar. `get_active_traceparent` reads `ACTIVE_SPAN_CONTEXT`, which nothing sets yet, so grader calls always get a fresh child traceparent. Upstream's `withGraderSpan` around each assertion is not ported (an xfail in `tests/test_assertions_trace.py` records it).
- Rationale: The assertions only need the stored spans, and upstream reads them from the store in the assertion step, not from fetchTraceContext. Porting that path exactly lets a target that writes spans by any route (OTLP, an external backend, or straight into the store) be graded the same way under Node and Python.
- Assumption: Spans reach the `spans` table before grading or during the stable-poll window. promptfoo's own spans (test case, provider, grader) come from the spans branch; until it lands, a traced Python eval stores only the target's spans, so unfiltered `trace-*` patterns count fewer spans than Node does.
- Implication: With tracing enabled, every test with a `python` or `file://` assertion polls the store (6 x 250 ms by default) when no spans arrive, as upstream does.
- Modules: src/promptphoo/assertions/trace.py, src/promptphoo/assertions/index.py, src/promptphoo/assertions/trajectory.py, src/promptphoo/evaluator/run.py, src/promptphoo/tracing/trace_context.py, src/promptphoo/tracing/span_filter.py, src/promptphoo/tracing/span_roles.py, src/promptphoo/tracing/tool_attributes.py, src/promptphoo/tracing/store.py, src/promptphoo/tracing/sanitize_attributes.py, src/promptphoo/tracing/evaluator_tracing.py
- Commits: `b646b5b`

## 55. External trace providers fetch through fetch_with_proxy; trace parity runs a stdlib sqlite3 span writer under both CLIs
- Decision:
  - `tracing/providers/` ports index, types, fetch, tempo, langfuse and braintrust. They are tested only with respx (`tests/test_tracing_providers.py`).
  - promptphoo's fetch has no AbortSignal and no `redirect: 'error'`, so `fetch_trace_response` stands in for upstream's fetch options:
    - The request runs under an asyncio deadline of `timeout` ms (default 10000) and raises a `TimeoutError`-named error when it expires.
    - A signal that is already aborted stops the request with `AbortError`. A signal that fires mid-request does not interrupt it; task cancellation does.
    - A response whose final URL differs from the requested one is rejected with "fetch failed", which is the error undici raises for a redirect.
  - Response bodies are read eagerly, and the byte limit is checked after the read. The Content-Length check and the Langfuse per-page byte budget are kept.
  - Tool spans whose input is JSON null get `tool.arguments: "null"`, which keeps Node's `JSON.stringify(null)`, because `get_normalized_tool_attributes` treats None as undefined.
  - `Date.parse` is approximated by `fetch.header_parser.date_parse`.
  - The parity fixture `tests/fixtures/parity_tracing/trace-assertions.yaml` runs `span_writer.py`, a Python provider that inserts its spans with stdlib sqlite3 into `$PROMPTFOO_CONFIG_DIR/promptfoo.db` under its traceparent's trace id. Node's Python worker and promptphoo run the same code, and the gradings match byte for byte. `tools/parity.py` treats `traceId` as volatile and leaves out the top-level `traces` export for this fixture only (`IGNORED_TOP_KEYS`). That export holds promptfoo's own spans with random span ids. The ignore is gone; `traces` are compared without span ids and times, see D53.
  - `tests/test_trace_assertions_e2e.py` runs the in-process variant, writing through `get_trace_store().add_spans`.
- Rationale: A provider that writes through promptphoo's store cannot run under Node, but both CLIs read spans from the same SQLite table. Writing the rows directly gives a real cross-implementation check of the assertion path.
- Assumption: Node and promptphoo both keep the database at `<configDir>/promptfoo.db` in WAL mode, and Node's spans for the test case, provider call and graders never match the fixture's `tool.*` / `llm.*` / `agent.*` patterns.
- Modules: src/promptphoo/tracing/providers/__init__.py, src/promptphoo/tracing/providers/types.py, src/promptphoo/tracing/providers/fetch.py, src/promptphoo/tracing/providers/tempo.py, src/promptphoo/tracing/providers/langfuse.py, src/promptphoo/tracing/providers/braintrust.py, tools/parity.py, tests/fixtures/parity_tracing/trace-assertions.yaml, tests/fixtures/parity_tracing/span_writer.py
- Commits: `b646b5b`

## 56. promptfoo's own spans come from an in-house tracer that stands in for the OpenTelemetry JS SDK
- Decision:
  - `opentelemetry` is not a dependency. `tracing/otel_api.py` ports the slice of `@opentelemetry/api` that promptfoo uses:
    - SpanKind, SpanStatusCode and TraceFlags.
    - SpanContext, plus an immutable Context held in a ContextVar. `context.with_` awaits under that context, and asyncio tasks inherit it.
    - The `trace` facade: a ProxyTracer and a NoopTracer, which passes a valid parent through.
    - The W3C traceparent propagator. Like upstream's, it accepts lowercase hex only.
  - `tracing/otel_trace_sdk.py` ports the parts of sdk-trace-base 2.x that promptfoo uses:
    - The ParentBased(AlwaysOn) sampler, random W3C ids, and HrTime times (start from `Date.now()` ms, end from a monotonic clock).
    - setStatus: UNSET is ignored, OK is final, and only ERROR keeps a message.
    - setAttribute drops None and invalid values and keeps at most 128 attributes. recordException records an `exception` event.
    - Simple and Batch span processors (512/2048/5000 ms/30000 ms, overridable by `OTEL_BSP_*`), the in-memory exporter, and BasicTracerProvider.register().
  - `tracing/local_span_exporter.py` ports LocalSpanExporter. It writes through `get_trace_store().add_spans` with `{skipTraceCheck: false, warnIfMissingTrace: false}`, retries once after 50 ms, skips FOREIGN KEY failures and fails the export on any other error.
  - `tracing/otlp_http_exporter.py` is the OTLP/HTTP exporter, sending JSON rather than upstream's protobuf. It encodes like otlp-transformer's JSON serializer: hex ids, nanosecond times as strings, `kind + 1`, intValue/doubleValue. It reads `OTEL_EXPORTER_OTLP_*` headers and timeout, and retries 429/502/503/504 up to 5 times with jittered backoff and Retry-After.
  - `otel_sdk.py`, `otel_config.py`, `resource_attributes.py`, `genai_tracer.py`, `target_tracer.py` and `span_roles.py` port their upstream files. The resource is `service.name`/`service.version` = 0.125.1 and merges into every stored span's attributes, as Node's exporter does.
  - Known divergences from Node:
    - recordException uses the Python exception class name as `exception.type` unless the error carries a string `code`.
    - There are no SIGINT/SIGTERM flush handlers.
    - `shutdown_otel` unregisters the global provider and propagator, so a second eval in the same process re-initializes. Node's global stays set.
    - BatchSpanProcessor.force_flush also waits for exports already started by the timer.
    - The `OTEL_TRACES_SAMPLER` env and span-limit env vars are not read.
    - The OTLP User-Agent is `OTel-OTLP-Exporter-Python-promptphoo/0.125.1`.
- Rationale: promptfoo needs only a small, synchronous-API slice of OpenTelemetry: start a span, set attributes, propagate traceparent, batch-export to the local store. Porting that slice keeps span names, attributes, ids, status and parentage the same as Node's, adds no dependency, and lets tests inspect spans in memory.
- Assumption: Nothing in promptphoo needs OTel auto-instrumentation, metrics or logs. An OTLP collector that accepts protobuf also accepts `application/json` on `/v1/traces`.
- Implication: A user who installs the real OpenTelemetry Python SDK does not get promptfoo's spans through it. Spans go to the local store and, when `PROMPTFOO_OTEL_ENDPOINT` / `OTEL_EXPORTER_OTLP_ENDPOINT` is set, to that endpoint.
- Modules: src/promptphoo/tracing/otel_api.py, src/promptphoo/tracing/otel_trace_sdk.py, src/promptphoo/tracing/otel_sdk.py, src/promptphoo/tracing/otel_config.py, src/promptphoo/tracing/local_span_exporter.py, src/promptphoo/tracing/otlp_http_exporter.py, src/promptphoo/tracing/resource_attributes.py, src/promptphoo/tracing/genai_tracer.py, src/promptphoo/tracing/target_tracer.py, src/promptphoo/tracing/span_roles.py
- Commits: `57ba6af`

## 57. Span call sites follow upstream: test-case root, target, grader and GenAI provider spans
- Decision:
  - `Evaluator.evaluate` initializes the tracer when env, suite, defaultTest or any test enables tracing, and flushes and shuts it down in `finally`.
  - `generate_trace_context_if_needed` (in `tracing/evaluator_tracing.py`, see D52) starts the `promptfoo.test_case` root span, with upstream's attributes, and derives the traceparent from it. A traced step runs inside `with_test_case_span`, which sets status ERROR with the row's error as the message, under a provider-call tracing context (getActiveTraceparent, testIndex, withGraderSpan, withProviderSpan).
  - The target call goes through `with_traced_provider_call`:
    - The span is named after the label, else the provider id, has CLIENT kind and role `target`.
    - The call context gets a child traceparent.
    - Embedding calls get a GenAI `embeddings` span instead.
  - `run_assertion` wraps each assertion of a traced step in `grader <type>` (withGraderSpan). Grader calls get the active traceparent, so `matchers/providers.py`'s existing withProviderSpan hook makes `grader provider <label>` spans under it.
  - `get_active_traceparent` reads the active span from the tracer's context. This replaces D54's `ACTIVE_SPAN_CONTEXT` ContextVar. The xfail for withGraderSpan in `tests/test_assertions_trace.py` is removed.
  - withGenAISpan is added to every promptphoo provider whose upstream calls it: openai chat/completion/responses, anthropic messages, azure chat, bedrock converse, google vertex, ollama completion/chat and openrouter. Each uses upstream's span context and result extractor. A body-building failure in openai chat and openrouter still gets a span.
  - `tests/test_tracing_node_parity.py` runs two traced configs under both CLIs, skipped unless NODE_PROMPTFOO is set:
    - echo, a Python file provider and llm-rubric/contains/equals graders;
    - the six HTTP providers against a canned local LLM server (`tests/tracing_fake_llm.py`).
  - It compares the stored spans with ids and times dropped: name path from the root, attributes with the eval id normalized, and status code and message. Both configs match Node.
  - Not ported:
    - flushOtel before trace-aware assertions. Batch export runs after grading, as under Node's 5 s delay, so assertions see only the target's own spans.
    - withGenAIToolSpan wiring in function callbacks.
    - Tracing in comparison.py.
    - Spans for providers promptphoo lacks.
- Rationale: Matching upstream's call sites means the stored traces, the web UI's trace view and trace assertions with `includeInternalSpans` see the same tree under Node and Python.
- Assumption: Upstream keeps the BatchSpanProcessor delay above the time a test takes to grade. If it ever flushes before trace-aware assertions, `tests/test_trace_assertions_e2e.py` and the trace-assertions parity fixture will diverge.
- Modules: src/promptphoo/evaluator/run.py, src/promptphoo/assertions/index.py, src/promptphoo/tracing/evaluator_tracing.py, src/promptphoo/providers/openai/chat.py, src/promptphoo/providers/openai/completion.py, src/promptphoo/providers/openai/responses.py, src/promptphoo/providers/anthropic/messages.py, src/promptphoo/providers/azure/chat.py, src/promptphoo/providers/bedrock/converse.py, src/promptphoo/providers/google/vertex.py, src/promptphoo/providers/ollama.py, src/promptphoo/providers/compatible/openrouter.py, tests/test_tracing_node_parity.py
- Commits: `57ba6af`

## 58. Licensing: MIT for the port, upstream's notice kept, third-party notices in NOTICE
- Decision: promptphoo is MIT-licensed. LICENSE keeps promptfoo's "Copyright (c) Promptfoo 2025" line beside "The promptphoo authors", because substantial portions are translated from or copied out of promptfoo. NOTICE lists every other source with its license: RAGAS and DeepEval (Apache-2.0, text in LICENSES/Apache-2.0.txt), the ported npm algorithms (fastest-levenshtein, js-rouge, natural, dedent: MIT), and fdlibm (Sun's permissive notice, preserved). `src/promptphoo/_data/NOTICE` marks the vendored files. pyproject declares `license = "MIT AND Apache-2.0"` and ships all of them through `license-files`.
- Rationale: The project should be easy to adopt in companies. Every component is permissive, none copyleft, and an SPDX expression plus a single NOTICE is what license scanners and legal reviews look for. Keeping upstream's notice is what its MIT license requires.
- Alternatives: Putting a header on every source file (REUSE-style) was rejected for now as noise; NOTICE plus module docstrings ("Ports src/...ts", "Apache-2.0; see NOTICE") identify provenance.
- Implication: New vendored files or ported third-party algorithms need a NOTICE entry. The NOTICE also states that promptphoo is unofficial and not endorsed by Promptfoo, Inc. or OpenAI.
- Modules: LICENSE, NOTICE, LICENSES/Apache-2.0.txt, src/promptphoo/_data/NOTICE, pyproject.toml

## 61. Safer defaults than upstream: sandboxed templates, TLS verified, no feedback upload
- Decision: Three deliberate departures from upstream, made before the public alpha.
    - Templates render in Jinja2's `SandboxedEnvironment`. Attributes such as `__class__`, `__subclasses__` and `__globals__` are unreachable, so a template, or a var value rendered as one, cannot run Python. Inline expressions (`transformResponse`, `transform`, `transformVars`) compile through the same environment.
    - TLS certificates are verified unless `PROMPTFOO_INSECURE_SSL=true`. Upstream defaults that variable to true and skips verification.
    - `promptphoo feedback` sends nothing. It points to promptphoo's issue tracker, read from the package's `Issues` project URL. Upstream posts the message, and optionally the user's email, to api.promptfoo.dev.
- Rationale: Prompts, tests and var values are routinely loaded from shared files, spreadsheets and datasets, and upstream renders var values as templates. Unsandboxed, any of them could execute code on the machine running the eval. Unverified TLS lets anyone on the network path read API keys. Feedback about an independent port should not go to a company that does not maintain it.
- Alternatives: Keep upstream's behaviour and document it (D21's original choice). Rejected: a new project should not ship insecure defaults it can cheaply avoid. Sandboxing Python hooks was also considered and rejected; hooks are code by design, and Python cannot be sandboxed reliably inside one process.
- Assumption: Real templates never use underscore attributes. The full suite (6,512 tests) and all 63 parity fixtures gave identical results with the sandbox on. If a template legitimately needs a blocked attribute, it should move to a Python hook rather than weaken the sandbox. A user behind an intercepting proxy sets `PROMPTFOO_CA_CERT_PATH`, or `PROMPTFOO_INSECURE_SSL=true` as a last resort.
- Implication: Configs that relied on unverified TLS (self-signed endpoints) now fail until a CA is configured. SECURITY.md describes the trust model: configs and hooks are code; datasets are data.
- Note: Supersedes D21.
- Modules: src/promptphoo/templating.py, src/promptphoo/fetch/__init__.py, src/promptphoo/cli/feedback.py, SECURITY.md, tests/test_templating.py, tests/test_fetch.py, tests/test_cli_misc.py

## 62. promptphoo names itself: only `promptphoo` and `ph` are installed, and messages say promptphoo
- Decision: The package installs the console scripts `promptphoo` and `ph`, not `promptfoo` or `pf`. Every command the CLI suggests is spelled `promptphoo <sub>` (`--version` prints `promptphoo X (compatible with promptfoo 0.125.1)`, and click's program name is `promptphoo`). Commands only Node promptfoo has (`view`, `redteam`, `auth`) are spelled `npx promptfoo@latest <sub>` and labelled as Node's. The eval summary drops upstream's share and promptfoo.app lines, points the viewer line at Node's viewer of the shared database, and gives `promptphoo feedback` for feedback. Docs links to promptfoo.dev configuration pages stay, because the configuration format is the same.
- Rationale: The names sound alike. Installing a `promptfoo` script let promptphoo shadow Node promptfoo on PATH, so users could run the port without knowing it and report its behaviour to the promptfoo project. Naming the tool everywhere keeps the two apart and keeps bug reports with the right project.
- Alternatives: Keep the aliases as the drop-in default (rejected for the reason above). Install the aliases through an extra (rejected: extras cannot add console scripts). Users who want the drop-in add a shell alias themselves; docs/MIGRATING.md shows how and says not to do so on shared machines.
- Implication: Node golden outputs (list/show, init files) differ only in command spelling; tests/naming.py maps Node's spelling to promptphoo's so everything else is still compared byte for byte. Marking Python-written evals in the shared database was considered and left out: the only free-form column, `runtime_options`, feeds Node's `--resume`.
- Modules: pyproject.toml, src/promptphoo/promptfoo_command.py, src/promptphoo/cli/main.py, src/promptphoo/output/summary.py, src/promptphoo/onboarding.py, src/promptphoo/cli/*.py, tests/naming.py, docs/MIGRATING.md, README.md
