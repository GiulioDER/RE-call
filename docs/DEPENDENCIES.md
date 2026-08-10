# Dependency Policy

This file keeps the package metadata readable while preserving the rationale behind pinned floors,
ceilings, and optional extras.

## Distribution Name

The distribution name is `recall-rag`; the import name is `recall`. The plain `recall` name on PyPI
belongs to an unrelated Python 2 era RPC framework, so the package cannot use that distribution
name. The intended `re-call` name is also rejected by PyPI's similarity guard because separators are
collapsed during normalization.

Install `recall-rag`, not `recall`. Installing both into one environment is unsafe because both
provide a top-level `recall` module and the last installed distribution wins the import path.

## Runtime Floors

`pgvector>=0.4` is required because `from pgvector import Vector` is a top-level export starting in
0.4. On 0.3.x, importing `recall.store` fails.

`psycopg-pool` is optional because CLI use can run on a single connection. Server processes should
install the `pool` extra or an extra that includes it.

## MCP Extra

The `mcp` floor is 1.27.2. Earlier versions do not carry all authenticated server fields RE-call
uses:

| Version boundary | Why it matters |
|---|---|
| `>=1.10.0` | Adds the resource-server auth split needed by `recall_mcp.auth`. |
| `>=1.27.2` | Adds `AccessToken.subject` and `.claims`, used to carry tenant identity. |
| `<2` | `mcp` 2.0 moves or renames APIs used by `recall_mcp.server`; raising this cap is a port. |

`PyJWT[crypto]` is declared directly rather than inherited transitively from `mcp`, because
`recall_mcp.oidc` imports RSA support at module import time.

Keep the `mcp` extra and the `dev` extra in step so CI exercises the same API surface users install.

## Sparse, Rerank, Entailment, and Fine-tuning

Learned sparse retrieval uses `transformers` directly rather than `sentence-transformers`. That
keeps the SPLADE experiment from raising the reranker, entailment, and fine-tuning floor as a side
effect. Model weights are downloaded or provisioned by the user, never vendored into the package.

`transformers` is capped below 6 because this project has already been broken by a major release
walking in through an unbounded floor. The current range is a compatibility claim for the APIs used
here; raising the cap should be treated as a port and tested separately.

Reranking, entailment, and fine-tuning share the same `sentence-transformers` floor. Bump them
together.

## Framework Adapters

The LangChain and LlamaIndex extras depend on their core packages, not the framework meta-packages:

| Extra | Dependency | Reason |
|---|---|---|
| `langchain` | `langchain-core` | The adapter needs `BaseRetriever` and `Document`. |
| `llamaindex` | `llama-index-core` | The adapter needs `BaseRetriever`, `TextNode`, and `NodeWithScore`. |

Both are mirrored in the `dev` extra so adapter tests and type checks run in CI.

## Benchmark Extra

The `bench` extra is for deliberate local benchmark runs, not normal development or CI. It includes
heavy or competitor-specific dependencies.

`mem0ai` is pinned exactly because it is the competitor being measured. Its extraction prompt,
retrieval defaults, and API move between releases, so a range would let a fresh install silently
change a published comparison. Bumping it requires a deliberate re-run.

`pyarrow` is required because the BEAM benchmark ships as parquet. `numpy` is required because
benchmark modules import it directly.

## Development Tools

`pytest-timeout` is part of the dev extra so a nonterminating test fails instead of hanging CI.

`ruff` is capped below 0.16 because new stabilized lint rules changed the result of `ruff check .`
on unchanged code. Adopting those rules should be a repository-wide sweep, not a side effect of an
unrelated dependency resolution.
