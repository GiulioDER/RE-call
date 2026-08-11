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
| `recall setup` | Guided local setup, including optional per-corpus calibration. |
| `recall schema` | Apply, inspect, and plan PostgreSQL schema migrations. |
| `recall index` | Index a markdown corpus. |
| `recall search` | Query an indexed corpus through the trust layer. |
| `recall reasoning` | Inspect projections, proposals, traces, audits, and opt-in reasoning queries without changing ordinary retrieval behavior. |
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

Authentication, tenant isolation, and transport modes are documented in [AUTH.md](AUTH.md).
Reasoning policy and operational behavior are documented in
[REASONING_OPERATIONS.md](REASONING_OPERATIONS.md).

## Stability Boundary

Benchmark harnesses under `benchmarks/`, result builders under `results/`, and experimental code
under `benchmarks/finetune/` are not the library API. They are retained for reproducibility and
evidence review.
