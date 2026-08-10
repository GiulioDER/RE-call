# Supported API

This page names the public integration surface. Anything outside these paths may support
benchmarks, migrations, or experiments, and can change more freely.

## Python

| Surface | Import | Purpose |
|---|---|---|
| Trust search | `recall.trust.trusted_search` | Return verdicts, confidence, provenance, and abstention state. |
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

Authentication, tenant isolation, and transport modes are documented in [AUTH.md](AUTH.md).

## Stability Boundary

Benchmark harnesses under `benchmarks/`, result builders under `results/`, and experimental code
under `benchmarks/finetune/` are not the library API. They are retained for reproducibility and
evidence review.
