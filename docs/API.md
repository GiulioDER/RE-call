# Supported API

This page names the public integration surface. Anything outside these paths may support
benchmarks, migrations, or experiments, and can change more freely.

## Python

| Surface | Import | Purpose |
|---|---|---|
| Trust search | `recall.trust.trusted_search` | Return verdicts, confidence, provenance, and abstention state. |
| Reasoning | `recall.reasoning.reason` | Run explicit opt-in reasoning from trusted retrieval, bounded provider ports, graph projections, and citation validation. |
| Reasoning graph | `recall.reasoning_graph.build_reasoning_graph` | Derive immutable, generation-bound graph projections for reasoning and proposal inspection. |
| Embeddings | `recall.embeddings.make_embedder` | Construct supported embedding backends from configuration. |
| Generation store | `recall.generation_store.GenerationStore` | Serve immutable, tenant-scoped generations. |
| pgvector store | `recall.store.PgVectorStore` | Local indexing and retrieval over PostgreSQL plus pgvector. |
| LangChain | `recall.integrations.langchain.RecallRetriever` | Use RE-call as a LangChain retriever. |
| LlamaIndex | `recall.integrations.llamaindex.RecallRetriever` | Use RE-call as a LlamaIndex retriever. |

The expected application pattern is to call `trusted_search`, check `result.abstained`, and answer
only from returned hits whose verdict and provenance satisfy the caller's policy.

## Command Line

| Command | Purpose |
|---|---|
| `recall setup` | Guided local setup: embedder/reranker/entailment choices, optional per-corpus calibration, and optional CLAUDE.md/memory scaffolding. |
| `recall schema` | Apply, inspect, and plan PostgreSQL schema migrations. |
| `recall index` | Index a markdown corpus. |
| `recall search` | Query an indexed corpus through the trust layer. |
| `recall reasoning` | Inspect projections, proposals, traces, audits, and opt-in reasoning queries without changing ordinary retrieval behavior. |
| `recall extract` | Extract structured truth claims from memo prose. Reads only; writes nothing. Off unless `RECALL_TRUTH_EXTRACTION=1`. |
| `recall rewrite` | Review extracted claims and declare accepted ones in corpus frontmatter. Dry run by default; `--reviewer` and `--note` are required. |
| `recall lint` | Validate memo frontmatter and corpus shape. |
| `recall check` | Validate one memo, optionally in strict mode. |
| `recall demo` | Run the bundled five-minute product example. |
| `recall-enterprise` | Manage generation routing and readiness for production deployments. |

## MCP

The MCP server is `python -m recall_mcp.server`. Its supported tools are:

| Tool | Purpose |
|---|---|
| `recall_search` | Search trusted memory. |
| `recall_evidence` | Return evidence for a query. |
| `recall_index` | Index allowed files beneath `RECALL_INDEX_ROOT`. |
| `recall_forget` | Erase indexed source material. |
| `recall_stats` | Report counters and operational state. |
| `recall_reasoning_query` | Run an explicit opt-in reasoning query over trusted retrieval. |
| `recall_reasoning_projection` | Inspect the generation-bound reasoning graph projection. |
| `recall_reasoning_proposals` | Inspect inference proposals as review candidates. |
| `recall_reasoning_audit` | Report reasoning integration state and diagnostics. |
| `recall_rewrite_plan` | Report which key a proposal would declare, in which file. Writes nothing. |

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
