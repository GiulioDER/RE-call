# Supported API

This page names the public integration surface. Anything outside these paths may support
benchmarks, migrations, or experiments, and can change more freely.

Compatibility and upgrade rules for this surface are in [COMPATIBILITY.md](COMPATIBILITY.md).

## Python

| Surface | Import | Purpose |
|---|---|---|
| Trust search | `recall.trust.trusted_search` | Return verdicts, confidence, provenance, and abstention state. |
| Reasoning | `recall.reasoning.reason` | Run explicit opt-in reasoning from trusted retrieval, bounded provider ports, graph projections, and citation validation. |
| Reasoning graph | `recall.reasoning_graph.build_reasoning_graph` | Derive immutable, generation-bound authored and semantic graph projections for reasoning and proposal inspection. |
| Embeddings | `recall.embeddings.resolve_embedder` | Construct supported embedding backends from configuration. |
| Multimodal provenance | `recall.multimodal.build_media_ref`, `project_media_evidence` | Validate bounded media references and authorize original object references in the isolated multimodal tenant. |
| Generation store | `recall.generation_store.GenerationStore` | Serve immutable, tenant-scoped generations. |
| pgvector store | `recall.store.PgVectorStore` | Local indexing and retrieval over PostgreSQL plus pgvector. |
| Related evidence | `recall.related.trusted_related` | Opt in, independently trusted source, ordinal, or supersession related evidence, bounded to 50 candidates. |
| Source security | `recall.security_policy.SourceSecurityPolicy` | Enforce tenant bound source authorization, operation purposes, classification, egress, and redaction at indexing and serving boundaries. |
| Current state | `recall.current_state.project_current_state` | Pure, generation-bound authored state projection. |
| Query routing | `recall.query_class.classify_query` and `route_query` | Versioned deterministic query classes and shadow routing decisions. |
| LangChain | `recall.integrations.langchain.RecallRetriever` | Use RE-call as a LangChain retriever. |
| LlamaIndex | `recall.integrations.llamaindex.RecallRetriever` | Use RE-call as a LlamaIndex retriever. |
| Claude Agent SDK | `recall_agent.RecallAgentMemory` | In-process SDK tools, SessionStart digest, and `ClaudeAgentOptions` assembly for Claude Agent SDK apps ([USING_WITH_AGENT_SDK.md](USING_WITH_AGENT_SDK.md)). |
| Serving JSON | `recall_mcp.service.serving_json` | The one renderer every serving surface (MCP server, Agent SDK tools) uses, so results are byte-identical across transports. |
| Errors | `recall.errors.RecallError` | Common base of every deliberate recall/recall_mcp exception. Each family also keeps its historical built-in base (`RuntimeError` or `ValueError`), so existing handlers keep working. |

The expected application pattern is to call `trusted_search`, check `result.abstained`, and answer
only from returned hits whose verdict and provenance satisfy the caller's policy.

## Unknown-term vocabulary check

`trusted_search` checks entity-like query tokens when the store supports a global vocabulary
lookup. Lite supports it through FTS5, and a PostgreSQL generation store through the active
generation's full-text index (`tsv`, its own configuration); the legacy PostgreSQL route does not.
The heuristic selects tokens of at
least three characters containing a letter and a digit, all-capital names, or an inner capital.
Plain words are not checked. An absent token demotes otherwise `ok` hits to `unknown_term` and
the result abstains. Other verdicts and raw candidate scores retain their existing meaning.

Python results expose `diagnostics.unknown_term_check` and `diagnostics.unknown_terms`. Search
and evidence JSON, including in-process agent responses, expose those two fields directly:

| State | Meaning |
|---|---|
| `checked` | The vocabulary check completed. `unknown_terms` lists absent query tokens. |
| `unavailable` | The store lacks this capability, or a constrained request cannot safely use its global vocabulary. No lookup of excluded sources occurs and verdicts are unchanged. |
| `disabled` | `RECALL_UNKNOWN_TERM_GATE` disabled the optional check. |
| `not_run` | No check was reached or an older payload omitted these additive fields. |

An unavailable vocabulary check is not evidence that a term was absent. Calibration and
`trust_state` still govern trust; this check does not replace them. Source, folder, facet and
authorization constraints report `unavailable` until scoped vocabulary checking is supported.
`RECALL_UNKNOWN_TERM_GATE` defaults to `1`; `0`, `false`, `no`, or `off` disables it, ignoring case
and surrounding whitespace. An explicitly supplied `env` mapping controls the request even when
the process environment differs. See [ENVIRONMENT.md](ENVIRONMENT.md).

## Command Line

Every `recall` subcommand, in `recall --help` order. `tests/test_api_doc_drift.py` diffs this
table against the registered parsers, so a command cannot ship undocumented or linger here after
removal.

| Command | Purpose |
|---|---|
| `recall setup` | Guided local setup: embedder and reranker choices, optional per-corpus calibration, and optional CLAUDE.md/memory scaffolding. |
| `recall wizard` | The same install as a scriptable pipeline: `--headless --config` drives every corpus to a calibrated, promoted generation ([WIZARD.md](WIZARD.md)). |
| `recall uninstall` | Remove what setup installed: MCP registrations, hooks, and optionally the database stack. |
| `recall doctor` | Diagnose an install end to end and change nothing: interpreter, package, console scripts on PATH, embedder backend, Docker, database, pgvector, schema, whether the configured table and tenant actually hold chunks, calibration, and the Claude Code registration. Prints the repair command for each problem. `--json` for machines. Exits non-zero only when something is blocked, so a missing calibration does not fail a script. |
| `recall quickstart` | From a fresh `pip install` to a real answer: start a throwaway PostgreSQL, index the bundled 22-document demo corpus into `quickstart_chunks`/`quickstart`, answer three queries, and print the values the Claude Code plugin asks for. `--remove` destroys it. Calibrates nothing and registers nothing. |
| `recall schema` | Apply, inspect, and plan PostgreSQL schema migrations (`status`, `plan`, `apply`, `grants`). |
| `recall manifest` | Build and verify index manifests (`create`, `inventory`, `verify`). |
| `recall lite` | The one-file local store (`recall setup --lite`). `refresh` indexes what changed in the memo folder under the store's write lane and recalibrates; `--check` reports what a refresh would find and writes nothing. The MCP server runs the same refresh on its own when `RECALL_INDEX_ROOT` names the folder: at start, and before a search at most every `RECALL_LITE_REFRESH_INTERVAL` seconds (30), waiting up to `RECALL_LITE_REFRESH_WAIT` (3); `RECALL_LITE_AUTO_REFRESH=0` turns it off. |
| `recall generation` | Immutable generation lifecycle (`build`, `validate`, `promote`, `abandon`, `rollback`, `list`, `gc`, `stated-facts`: record the facts a generation's memos state, the active one by default). |
| `recall graph` | Inspect or rebuild the deterministic Evidence Graph V1 (`rebuild`) without changing chunks or generation identity. |
| `recall index` | Index a folder of supported documents or code. |
| `recall forget` | Permanently erase indexed sources; the right-to-erasure path. Facts whose only evidence they were are retracted, not erased: their values stay in the append-only fact ledger and evidence cards. |
| `recall search` | Query an indexed corpus through the trust layer. |
| `recall scopes` | List the folders or facets a search can be filtered by, with their sizes. |
| `recall reasoning` | Inspect projections (`projection`), proposals (`proposals`), queries (`query`), traces (`trace`), audits (`audit`), and opt-in reasoning without changing ordinary retrieval behavior. Query and trace accept `--graph-expansion auto|off|one-hop`; `auto` uses bounded global one-hop expansion for nonempty queries. |
| `recall extract` | Extract structured truth claims from memo prose (`run`, `show`). Reads only; writes nothing. Off unless `RECALL_TRUTH_EXTRACTION=1`. |
| `recall rewrite` | Review extracted claims and model-judged supersession pairs (`plan`, `apply`, `reject`, `verify`) and declare accepted ones in corpus frontmatter. Dry run by default; `--reviewer` and `--note` are required. The supersession arbiter is off unless `RECALL_SUPERSESSION_ARBITER=1`. |
| `recall dashboard` | Open a local review page (127.0.0.1 only) for supersession proposals: agent reports from `recall_report_stale` (the local sidecar, and with a connected corpus database also the `stale_report` rows of the tenants named by `--reports-tenant`, defaulting to the SQLite file's tenant or `memory` for PostgreSQL, mapped to this folder's memos and their quotes recovered from them) and cached arbiter judgements. Accepting writes `supersedes:` into the newer memo exactly as `recall rewrite apply` does; rejecting records the claim so it is not proposed again. Also a reader that renders each memo with what replaced it and what links to it, a 3D memory graph, a person-only editor for `supersedes`, validity dates and status, recent activity and memory health, all from the memo files. Without a DSN file it automatically reads a SQLite store from `RECALL_SERVING_DSN`, then `RECALL_DSN`. `--db-dsn-file` takes precedence and is required for PostgreSQL through a SELECT-only role (optionally with `--tunnel HOST`; SQLite rejects a tunnel). The corpus database is read over a read-only session: tenants with their embedder, size and calibration, generations, the searches recorded by the decision ledger (`RECALL_DECISION_LEDGER=1`), and a Control page counting, per memory, how often search returned it beside what agents reported with `recall_report_use`. A home page lists what needs attention; a top bar finds memos and switches corpus. With the database it also draws any corpus's graph from the database (`/graph?corpus=TENANT`: supersessions, authored links and Python imports between files, plus files that share an uncommon name) and opens one file's page from it (`/corpus/source`): its state, what replaces it, its relations, imports and chunks. A Configuration page shows what RE-call runs with on this machine, in each RE-call MCP server registered for the project, and (with `--config-host HOST`) on a remote host, each report produced by `python -m recall_mcp.config_report` where the values live, so every secret is withheld before it reaches the page. Calls no model. |
| `recall demo` | Index the sample corpus and run example searches. |
| `recall code` | Index RE-call source code and run example code searches. |
| `recall lint` | Validate memo frontmatter and corpus shape. |
| `recall check` | Validate one memo, optionally in strict mode. |
| `recall calibrate` | Fit an abstention threshold from a labeled query file (legacy single-shot form). |
| `recall calibration` | Calibration artifact lifecycle (`calibrate`, `carry-forward`, `drift`, `auto`, `list`, `show`, `export`, `import`), the reported near-miss certificate (`near-miss`), and the served versus exact maximum diagnostic (`served-gap`). |
| `recall provenance` | Apply structured facts through the deterministic provenance controller, or inspect the current fact projection (`apply`, `current`). |
| `recall backup` | Inspect, create, verify, and restore encrypted Aurora backups through the explicit `status`, `create`, `verify`, and `restore` workflows. |
| `recall secret` | `verify`: verify that every running ECS task reports the intended nonsecret secret version identifiers. |
| `recall idempotency` | `reconcile`: record an operator verified mutation result after a replay result was lost. |
| `recall-enterprise` | Manage generation routing and readiness for production deployments. |

## MCP

### Health endpoints

The HTTP deployment exposes `GET /livez`, `GET /readyz`, and `GET /startupz`. `livez` is
dependency free and returns `200` while the process is running. `startupz` returns `200` after
startup completed, otherwise `503`. `readyz` returns `200` only when the database, schema, pgvector,
RLS, active generation, and calibration checks pass. Redis limiter state is reported but does not
make reads unready, because reads have a bounded local fallback and mutations fail closed.

Mutating MCP tools require an idempotency key on HTTP deployments for writes, forget, and admin
operations. Most tools expose it as `idempotency_key`; `recall_apply_fact` exposes the same value
as `request_id`. PostgreSQL records the completed response before Redis stores its replay cache,
so a repeated mutation key returns the original response even when Redis lost the response write.
The receipt is bound to the idempotency key, tool, and canonical request arguments, so reusing one
key for a different operation or different arguments is rejected as `idempotency_conflict`. The
`recall_apply_fact` tool calls this public key `request_id`; other mutating tools call it
`idempotency_key`. If neither receipt is available, the server returns
`reconciliation_required` and does not execute the mutation again.

An operator resolves that state only after checking the mutation's side effect in its owning
system. Save the exact JSON response, then run:

```bash
recall --tenant TENANT idempotency reconcile \
  --key KEY --operation TOOL --fingerprint FINGERPRINT \
  --result-file verified-result.json --confirm RECONCILE_IDEMPOTENCY
```

The command writes only the PostgreSQL durable receipt. It does not clear the Redis reservation and
does not execute the mutation, so a later retry replays the verified response without risking a
second side effect. Without the confirmation value it prints a dry run.

`readyz` runs one shared control-plane check and a deterministic, bounded sample of tenant stores.
The JSON response includes `checks.control_plane`, `checks.tenant_probes` (the number of stores
probed), `checks.rate_limiter`, and `failures`. Tenant inventory counts are kept out of this
unauthenticated endpoint; the full configured count and probe limit are emitted in startup logs.
The probe count is not the configured tenant count, so a `200` response is representative process
readiness only and does not prove that every configured tenant is correctly provisioned. Perform a
separate exhaustive tenant validation when that guarantee is required.
Durable receipts are operational replay records rather than long-term audit history. Retain them
for the replay window and prune older rows with scheduled database maintenance, for example:

```sql
DELETE FROM recall_idempotency_receipts
WHERE expires_at <= now();
```

The MCP server is `python -m recall_mcp.server`. Every registered tool, in `tools/list` order;
the same drift test diffs this table against the `@mcp.tool` registrations:

| Tool | Purpose |
|---|---|
| `recall_search` | Search trusted memory. |
| `recall_evidence` | Return evidence for a query. |
| `recall_current_facts` | Read the current structured fact projection. Each fact names its memos in `sources`; values that disagree with no supersession between their memos are listed under `contested`. `served_state` is `ok`, `degraded` (a check against the served memos failed; `served_lookups_failed` names it, and stated facts may be missing) or `unchecked` (a store with no such lookup). A lite store records the facts its memos state in `recall_graph` frontmatter whenever it indexes them, from the folders it refreshes, or the prefixes `RECALL_STATED_FACT_SOURCES` names (`none` turns it off). A PostgreSQL tenant records them only where that variable names places (a URI such as `s3://bucket/corpus/memory`, an absolute path, or a path under `RECALL_INDEX_ROOT`, matched against each memo's source), when a generation is promoted or rolled back, when the legacy table is indexed, and on `recall generation stated-facts`. Set the variable in the process that promotes and in the one that serves, since the rule is applied again when facts are read; with it on, a memo it admits that supersedes another memo also retires that memo's applied facts. Uploads never state. |
| `recall_apply_fact` | Apply a reviewed structured fact through the provenance controller. |
| `recall_related` | Retrieve independently trusted structural related evidence. |
| `recall_current_state` | Inspect a deterministic authored current state projection. |
| `recall_reasoning_query` | Run an explicit opt-in reasoning query over trusted retrieval. `graph_expansion` defaults to `auto`, which uses bounded global one-hop expansion for every nonempty query. `off` and `one_hop` remain explicit overrides. Precision admission diagnostics and a policy fingerprint are additive response fields. Legacy `expand_retrieval` remains available when configured. |
| `recall_query_construction_challenge` | Start or continue bounded query construction with an original-model challenge, deterministic candidate controls, and generation-bound trusted retrieval. |
| `recall_reasoning_projection` | Inspect the generation-bound reasoning graph projection. |
| `recall_reasoning_proposals` | Inspect inference proposals as review candidates. |
| `recall_rewrite_plan` | Report which key a proposal would declare, in which file. Writes nothing. |
| `recall_report_stale` | Report that one memory used in a task is replaced by another, with a verbatim quote from each. Queues the claim for a person to review (`recall dashboard`, `recall rewrite`); edits no memo and changes no verdict. When both memos are files under `RECALL_INDEX_ROOT` it goes to the `.recall` sidecar there; when the store is database-backed and both are in the served generation, with the quotes verbatim in the served text where the store can show it, it is (also, or only) one `stale_report` row per claim per day in `recall_audit_events`, which a dashboard on another machine reads. That row keeps a fingerprint of each quote, never the memo text. The result's `recorded_in` says where it went. Needs `recall:write`. |
| `recall_report_use` | After a task that searched memory, report what the memories did for it: `effect` (helped, no_difference, misled), the memories `used` and any found `wrong`, and optionally whether the task succeeded. Every memory named must exist in the served generation. Appends one `use_report` row to the audit ledger, counted as agent-reported on the dashboard's Control page; changes no memo, verdict or ranking. Needs `recall:write`. |
| `recall_reasoning_audit` | Report reasoning integration state and diagnostics. |
| `recall_index` | Index allowed files beneath `RECALL_INDEX_ROOT`. |
| `recall_tenants` | Return the tenant scopes visible to this caller (the full inventory needs `recall:admin`). |
| `recall_ingest` | Upload bounded source files and index them, debiting the tenant's byte quota. |
| `recall_job_status` | Return the state of one ingest job, scoped to the caller's tenant. |
| `recall_calibration_status` | Return the latest calibration artifact bound to the caller's generation. |
| `recall_calibration_run` | Fit corpus calibration. PostgreSQL creates a draft for the active generation. Lite activates completed fits immediately: a certified fit governs serving, and a completed uncertified fit replaces the prior calibration and blocks strict answers. Attempts that cannot complete retain the prior artifact. Lite has no separate publish step. |
| `recall_calibration_publish` | Publish one certified calibration artifact. Requires `recall:admin`: publication changes what the whole tenant serves. |
| `recall_forget` | Erase indexed source material, including its staged upload files, and retract the facts whose only evidence it was; a retracted fact's value stays in the append-only fact ledger. |
| `recall_inventory` | List every source in the caller's memory with the digest of its bytes, so a sync client can diff instead of re-uploading. Not in the `read` or `search` presets: a file listing answers no question an agent asks. |
| `recall_stats` | Report counters and operational state. |
| `recall_remember` | Assistant surface, lite only (`RECALL_ASSISTANT_MEMOS` names a memo folder inside `RECALL_INDEX_ROOT`): write ONE fact as one memo, with `valid_from`, `valid_until`, `supersedes` (only a memo the assistant wrote) and an optional structured `fact`, and index it before returning `{memo_id}`. Optional `request_id`, `expected_revision` and `include_receipt` provide recoverable corrections as described below. With the surface on, `recall_search`, `recall_evidence`, `recall_current_facts` and `recall_apply_fact` answer in a compact view naming each hit by its memo id; the `assistant` preset serves exactly these six tools. Unset, none of this is registered. |
| `recall_forget_memo` | Assistant surface, lite only: forget one memo of the memo folder by its id, retracting the facts whose only evidence it was, then delete its file; a retracted fact's value stays in the append-only fact ledger. `{deleted: 0}` for an id that names no memo there. |

For an evaluation that replays simulated days, `RECALL_EVAL_CLOCK=<file>` (lite only) makes the
server read "now" from that file, one ISO 8601 time with an offset, re-read on every call: search
verdicts, row write times, fact application and the current facts follow it, while a caller that
passes its own time keeps it. A missing or offset-less time fails the call rather than falling back
to the wall clock (`recall.eval_clock`).

Search's `abstained`, `decision_state`, and `evidence.citable` describe present-state support.
For a question about an earlier time, the optional `historical` field lists individually relevant,
certified superseded or expired chunk identifiers in `historical.candidates`. Its scope is
`historical_only`, and `requires_validity_check` is always true. Before citing a candidate as a
fact at the requested time, match that time to the hit's `valid_from` and `valid_until` and its
text. If they do not establish that time, describe only the archived assertion, without claiming
it held then. These candidates never expand present-state citations. Disabling
`RECALL_SEARCH_EVIDENCE` removes both evidence surfaces and their citation advice.

`recall_search` and `recall_evidence` also accept an optional `locale` argument for presentation
localization. When supplied, the response gains an additive `localized` object containing display
text keyed by `chunk_id`. Canonical hit text, provenance, evidence items, `system_prompt`, and
`user_message` are never translated in place. Localization is disabled unless
`RECALL_TRANSLATION_ENABLED=1` configures a validated HTTPS text endpoint. Provider failures are
fail soft and return canonical values with a fixed warning. Enabling the provider sends selected
retrieved passage text to that endpoint, so deployments with sensitive corpora should use a
self-hosted endpoint and should treat localized values as display data only. The explicit
`RECALL_TRANSLATION_ALLOW_HTTP=1` override permits cleartext HTTP for a deliberately controlled
endpoint and must not be used across an untrusted network.

Both tools also accept optional `scope`, `modality`, and `route_id` fields for request aware
retrieval planning. `RECALL_RETRIEVAL_PLANS_JSON` enables the versioned planner. It selects one
primary physical tenant and, only for an ambiguous request, at most one bounded rescue tenant.
The response carries `retrieval_plan` with the route id, selection reason, allowed tenant set,
per leg limits, latency bound, and selected tenant identities. The planner never compares scores
between tenants. When bounded federation is explicitly enabled, the serving adapter executes the
selected certified legs through isolated stores and adds federation diagnostics without changing
the compatibility route. An unset plan keeps the existing single tenant search behavior. Invalid
route configuration, unknown route ids, unallowed tenants,
retired generations, and runtime embedding-profile mismatches fail closed; calibration and trust
remain bound to the existing serving trust boundary.

The static README viewer uses these provider locale identifiers: `english`, `italian`, `spanish`,
`french`, `german`, `portuguese`, `chinese_simplified`, `japanese`, `korean`, `russian`, `arabic`,
`hindi`, and `turkish`. Other provider identifiers may be passed to the MCP or CLI presentation
surfaces. An unsupported identifier or provider failure leaves canonical text unchanged and marks
the localized object as a fallback.

Related expansion and structured retrieval explanations are disabled by default. With
`explain=true`, `recall_search` and `recall_evidence` also return a count only memory audit under
`explanation.details.memory_audit`; see [MEMORY_OBSERVABILITY.md](MEMORY_OBSERVABILITY.md). Graph expansion in
reasoning uses bounded global one-hop activation for nonempty queries. Set
`RECALL_ROUTING_MODE=active` only for a preregistered routing experiment. The default `shadow`
mode records the deterministic decision without changing retrieval behavior. See
[Active routing promotion gates](ROUTING_GATES.md) before enabling active mode.
Bounded specialist federation is a separate opt in layer. See
[FEDERATED_RETRIEVAL.md](FEDERATED_RETRIEVAL.md). Its default is `RECALL_FEDERATION_MODE=off`,
which executes only the primary tenant leg. `shadow` executes a bounded plan without changing
the served control result; `active` is required before a caller may serve the merged result.
`recall_current_state`
defaults to a fail closed maximum of 1000 source records and accepts an explicit `max_records`
bound; use `source` to project one authored lineage when a tenant is larger.

`recall_ingest` accepts a JSON object with `files`, `category`, optional `tenant`, and optional
`idempotency_key` fields. Each `files` entry has `name` and base64 encoded `content_b64` fields.
`category` is one of `documents`, `code`, or `memory` and defaults to `memory`. Upload count,
decoded byte, filename, path, and staging limits are enforced before indexing. HTTP mutations must
provide an idempotency key, and reusing a key with a different category or file payload returns
`idempotency_conflict`; a completed retry returns the original JSON response. The response is the
same bounded indexing result used by `recall_index`, with the staged job id included when the
ingest path creates an asynchronous job. See `docs/ENVIRONMENT.md` for the upload limits.

The CLI accepts the same additive presentation option, for example:

```console
recall search "deployment notes" --locale italian
```

**There is deliberately no `recall_rewrite_apply`.** Nothing reaches corpus metadata without a
named human, and the MCP client is the model: letting it supply a reviewer id and an audit note
would make that gate a formality it satisfies by typing a string, so the gate becomes a field
rather than a person. This surface proposes; a human declares at `recall rewrite apply`.
`recall_mcp/` makes no file write call of any kind, and two tests hold that line, one for a write
call and one for a write import.

`recall_rewrite_plan` hands off a **claim key**, not a proposal id. Its proposals come from the
deterministic rules over the store graph while `recall rewrite apply` resolves ids against the
filesystem extractor, and provider, tenant, generation and pipeline are all hashed into a
proposal id, so those two id spaces are disjoint. Claim keys are generation independent, which is
also why the rejection ledger is keyed by them.

Authentication, tenant isolation, and transport modes are documented in [AUTH.md](AUTH.md).
Reasoning policy and operational behavior are documented in
[REASONING_OPERATIONS.md](REASONING_OPERATIONS.md).

## Stability Boundary

Benchmark harnesses under `benchmarks/`, result builders under `results/`, and experimental code
under `benchmarks/finetune/` are not the library API. They are retained for reproducibility and
evidence review.

## Retrieval eligibility and answerability

Reasoning response decoding preserves these diagnostics and accepts older payloads with the
same unchecked and unmeasured defaults. Retrieval expansion and proof repair retain identical
coverage reports. When merged results disagree about coverage, the aggregate is unmeasured
without a report; its calibration ID is retained only when all merged diagnostics name the same
fit. Stored reports with inconsistent counts, rates, measurement states, or provisional warnings
are unavailable diagnostics and do not change evidence admission.

Search and evidence responses add `query_support` and `calibration_coverage`. Existing
`decision_state`, `decision`, `abstained`, verdicts, confidence and citation eligibility retain
their behavior. `supported`, `answer` and a citable chunk mean that retrieval trust checks admitted
evidence; they do not establish that the requested detail is stated or that a generated claim is
correct. Callers must establish support for each claim from the eligible passages.

`query_support` is `{ "status": "not_checked", "scope": "returned_evidence", "method": null }`.
No answerability verifier runs. It remains not checked on certified, degraded and empty results.
`validate_answer` still checks structure and citation identity only.

`calibration_coverage` contains the selected `calibration_id`, a `status`, a `method` and a
`report`. When available, `method` is `synthetic_corpus_probes` and `report` preserves the stored
`unknown_entity` and `in_vocabulary` counts, rejection rates, per-class measurement states and
`provisional_in_vocabulary` warning. These are synthetic diagnostics, not audited labels or
per-query semantic support. `measured` means both probe classes were measured, not that the gate
passed them: a measured report can still be provisional.

Missing measurements, a backend without this optional reader, or a report for another calibration
produce `unmeasured` and a null report. Reader failures produce `unavailable`. A carried calibration
does not inherit near-miss measurements from the parent fit. Coverage is read in the search's
snapshot for the calibration actually selected, without refitting or changing admission.

The Python trusted result and evidence bundle carry the same dataclasses. MCP, the in-process
agent tools, the compact assistant view and CLI report the distinction. Older MCP payloads without
these additive fields remain readable, defaulting to not checked support and unmeasured coverage.

## Assistant memo corrections

See [Memo corrections](MEMO_CORRECTIONS.md) for optional request identities, revision preconditions,
write receipts, conflict behavior and legacy compatibility.
