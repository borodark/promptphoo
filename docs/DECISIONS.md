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
