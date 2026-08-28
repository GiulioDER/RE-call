# Supported API

This page names the public integration surface. Anything outside these paths may support
benchmarks, migrations, or experiments, and can change more freely.

## Python

| Surface | Import | Purpose |
|---|---|---|
| Trust search | `recall.trust.trusted_search` | Return verdicts, confidence, provenance, and abstention state. |
| Reasoning | `recall.reasoning.reason` | Run explicit opt-in reasoning from trusted retrieval, bounded provider ports, graph projections, and citation validation. |
| Reasoning graph | `recall.reasoning_graph.build_reasoning_graph` | Derive immutable, generation-bound authored and semantic graph projections for reasoning and proposal inspection. |
| Embeddings | `recall.embeddings.make_embedder` | Construct supported embedding backends from configuration. |
| Generation store | `recall.generation_store.GenerationStore` | Serve immutable, tenant-scoped generations. |
| pgvector store | `recall.store.PgVectorStore` | Local indexing and retrieval over PostgreSQL plus pgvector. |
| Related evidence | `recall.related.trusted_related` | Opt in, independently trusted source, ordinal, or supersession related evidence, bounded to 50 candidates. |
| Current state | `recall.current_state.project_current_state` | Pure, generation-bound authored state projection. |
| Query routing | `recall.query_class.classify_query` and `route_query` | Versioned deterministic query classes and shadow routing decisions. |
| LangChain | `recall.integrations.langchain.RecallRetriever` | Use RE-call as a LangChain retriever. |
| LlamaIndex | `recall.integrations.llamaindex.RecallRetriever` | Use RE-call as a LlamaIndex retriever. |
| Claude Agent SDK | `recall_agent.RecallAgentMemory` | In-process SDK tools, SessionStart digest, and `ClaudeAgentOptions` assembly for Claude Agent SDK apps ([USING_WITH_AGENT_SDK.md](USING_WITH_AGENT_SDK.md)). |
| Serving JSON | `recall_mcp.service.serving_json` | The one renderer every serving surface (MCP server, Agent SDK tools) uses, so results are byte-identical across transports. |
| Errors | `recall.errors.RecallError` | Common base of every deliberate recall/recall_mcp exception. Each family also keeps its historical built-in base (`RuntimeError` or `ValueError`), so existing handlers keep working. |

The expected application pattern is to call `trusted_search`, check `result.abstained`, and answer
only from returned hits whose verdict and provenance satisfy the caller's policy.

## Command Line

Every `recall` subcommand, in `recall --help` order. `tests/test_api_doc_drift.py` diffs this
table against the registered parsers, so a command cannot ship undocumented or linger here after
removal.

| Command | Purpose |
|---|---|
| `recall setup` | Guided local setup: embedder/reranker/entailment choices, optional per-corpus calibration, and optional CLAUDE.md/memory scaffolding. |
| `recall wizard` | The same install as a scriptable pipeline: `--headless --config` drives every corpus to a calibrated, promoted generation ([WIZARD.md](WIZARD.md)). |
| `recall uninstall` | Remove what setup installed: MCP registrations, hooks, and optionally the database stack. |
| `recall doctor` | Diagnose an install end to end and change nothing: interpreter, package, console scripts on PATH, embedder backend, Docker, database, pgvector, schema, whether the configured table and tenant actually hold chunks, calibration, and the Claude Code registration. Prints the repair command for each problem. `--json` for machines. Exits non-zero only when something is blocked, so a missing calibration does not fail a script. |
| `recall schema` | Apply, inspect, and plan PostgreSQL schema migrations (`status`, `plan`, `apply`, `grants`). |
| `recall manifest` | Build and verify index manifests (`create`, `inventory`, `verify`). |
| `recall generation` | Immutable generation lifecycle (`build`, `validate`, `promote`, `abandon`, `rollback`, `list`, `gc`). |
| `recall index` | Index a markdown corpus. |
| `recall forget` | Permanently erase indexed sources; the right-to-erasure path. |
| `recall search` | Query an indexed corpus through the trust layer. |
| `recall reasoning` | Inspect projections (`projection`), proposals (`proposals`), queries (`query`), traces (`trace`), audits (`audit`), and opt-in reasoning without changing ordinary retrieval behavior. |
| `recall graph` | Inspect or rebuild the deterministic Evidence Graph V1 (`rebuild`) without changing chunks or generation identity. |
| `recall extract` | Extract structured truth claims from memo prose (`run`, `show`). Reads only; writes nothing. Off unless `RECALL_TRUTH_EXTRACTION=1`. |
| `recall rewrite` | Review extracted claims (`plan`, `apply`, `reject`, `verify`) and declare accepted ones in corpus frontmatter. Dry run by default; `--reviewer` and `--note` are required. |
| `recall quickstart` | From a fresh `pip install` to a real answer: start a throwaway PostgreSQL, index the bundled 22-document demo corpus into `quickstart_chunks`/`quickstart`, answer three queries, and print the values the Claude Code plugin asks for. `--remove` destroys it. Calibrates nothing and registers nothing. |
| `recall demo` | Index the sample corpus and run example searches. |
| `recall code` | Index RE-call source code and run example code searches. |
| `recall lint` | Validate memo frontmatter and corpus shape. |
| `recall check` | Validate one memo, optionally in strict mode. |
| `recall calibrate` | Fit an abstention threshold from a labeled query file (legacy single-shot form). |
| `recall calibration` | Calibration artifact lifecycle (`calibrate`, `carry-forward`, `drift`, `auto`, `list`, `show`, `export`, `import`). |
| `recall-enterprise` | Manage generation routing and readiness for production deployments. |

## MCP

The MCP server is `python -m recall_mcp.server`. Every registered tool, in `tools/list` order;
the same drift test diffs this table against the `@mcp.tool` registrations:

| Tool | Purpose |
|---|---|
| `recall_search` | Search trusted memory. |
| `recall_evidence` | Return evidence for a query. |
| `recall_related` | Retrieve independently trusted structural related evidence. |
| `recall_current_state` | Inspect a deterministic authored current state projection. |
| `recall_reasoning_query` | Run an explicit opt-in reasoning query over trusted retrieval. Set `graph_expansion` to `one_hop` to enable Evidence Graph V1. Precision admission diagnostics and a policy fingerprint are additive response fields. Legacy `expand_retrieval` remains available when configured. |
| `recall_query_construction_challenge` | Start or continue bounded query construction with an original-model challenge, deterministic candidate controls, and generation-bound trusted retrieval. |
| `recall_reasoning_projection` | Inspect the generation-bound reasoning graph projection. |
| `recall_graph_first_retrieval` | Probe deterministic graph-derived query seeds before ordinary trusted retrieval; graph output remains proposal data. |
| `recall_reasoning_proposals` | Inspect inference proposals as review candidates. |
| `recall_rewrite_plan` | Report which key a proposal would declare, in which file. Writes nothing. |
| `recall_reasoning_audit` | Report reasoning integration state and diagnostics. |
| `recall_index` | Index allowed files beneath `RECALL_INDEX_ROOT`. |
| `recall_tenants` | Return the tenant scopes visible to this caller (the full inventory needs `recall:admin`). |
| `recall_ingest` | Upload bounded source files and index them, debiting the tenant's byte quota. |
| `recall_job_status` | Return the state of one ingest job, scoped to the caller's tenant. |
| `recall_calibration_status` | Return the latest calibration artifact bound to the caller's generation. |
| `recall_calibration_run` | Create a draft calibration artifact for the active generation. |
| `recall_calibration_publish` | Publish one certified calibration artifact. Requires `recall:admin`: publication changes what the whole tenant serves. |
| `recall_forget` | Erase indexed source material, including its staged upload files. |
| `recall_stats` | Report counters and operational state. |

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

The static README viewer uses these provider locale identifiers: `english`, `italian`, `spanish`,
`french`, `german`, `portuguese`, `chinese_simplified`, `japanese`, `korean`, `russian`, `arabic`,
`hindi`, and `turkish`. Other provider identifiers may be passed to the MCP or CLI presentation
surfaces. An unsupported identifier or provider failure leaves canonical text unchanged and marks
the localized object as a fallback.

Related expansion and structured retrieval explanations are disabled by default. Set
`RECALL_ROUTING_MODE=active` only for a preregistered routing experiment. The default `shadow`
mode records the deterministic decision without changing retrieval behavior. `recall_current_state`
defaults to a fail closed maximum of 1000 source records and accepts an explicit `max_records`
bound; use `source` to project one authored lineage when a tenant is larger.

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
