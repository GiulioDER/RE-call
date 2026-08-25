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

PostgreSQL 16, 17, and 18 are supported with pgvector. The local Docker default tracks PostgreSQL
18 through `pgvector/pgvector:pg18`, while CI keeps migration coverage for 16 and 17 as older
supported majors.

The Compose file declares a named `recall_pgdata` volume for new PostgreSQL 18 containers. Existing
local data created by a PostgreSQL 16 Compose container is not upgraded in place by changing the
image tag; dump it from the old container and restore it into the PostgreSQL 18 container.

`pgvector>=0.4` is required because `from pgvector import Vector` is a top-level export starting in
0.4. On 0.3.x, importing `recall.store` fails.

`psycopg-pool` is optional because CLI use can run on a single connection. Server processes should
install the `pool` extra or an extra that includes it.

## MCP Extra

`mcp` is **pinned exactly at 2.1.0**, not ranged. RE-call uses the MCP 2 server import path,
context injection, and snake_case tool annotation fields. The boundaries below are why the floor
sits where it does; they are kept as a record of what each version added, and as the map for a
future bump:

| Version boundary | Why it matters |
|---|---|
| `>=1.10.0` | Adds the resource-server auth split needed by `recall_mcp.auth`. |
| `>=1.27.2` | Adds `AccessToken.subject` and `.claims`, used to carry tenant identity. |
| `>=2.0.0` | Provides `MCPServer`, typed request `Context` injection, and snake_case `ToolAnnotations` fields used by `recall_mcp.server`. |
| `<3` | Next major is reserved as a port until its server, auth, and context APIs are tested here. |
| `==2.1.0` | The pin itself. See below. |

### Why this one is pinned rather than ranged

Changed 2026-08-25, from `>=2,<3`. A minor release is enough to change behaviour this project
asserts on: **2.1.0 redacts a tool exception's message** unless the exception derives from the
SDK's `ToolError`, so the refusal raised by `recall_mcp.limits.RateLimited` reached clients as a
bare `Error executing tool recall_search` instead of its retry guidance. Three tests in
`tests/test_rate_limit_http.py` caught it, but only after it had broken master three times
(#497, #498, #499), because nothing in the range said "2.1.0 is new here".

Two things made the drift invisible, and both still hold:

- **CI installs with `pip install -e`, which ignores `uv.lock`.** The lock said 2.0.0 the whole
  time. It constrains `uv sync` and nothing else, so it is not a control on what CI resolves.
- **A range admits a version no one has run the suite against.** The pin converts that into a
  deliberate bump: a version change is now a diff, reviewed, with a full CI run behind it.

`RateLimited` keeps its `ToolError` base regardless — the pin controls *when* a new SDK arrives,
not whether the code survives it.

**To bump:** change the three sites in `pyproject.toml` (`mcp`, `dev`, `desktop`) together, run
`uv lock`, and run the suite. Watch `tests/test_rate_limit_http.py` in particular: it is the file
that fails when the SDK changes how tool errors surface.

`PyJWT[crypto]` is declared directly rather than inherited transitively from `mcp`, because
`recall_mcp.oidc` imports RSA support at module import time.

Keep the `mcp`, `dev` and `desktop` extras in step so CI exercises the same API surface users install.

## Cloud Embedding Extras

`voyage` installs the Voyage SDK for `VoyageEmbedder`. `openai` installs the OpenAI SDK used by
`OpenAICompatEmbedder`, including OpenRouter-backed embedding models such as
`gemini-embedding-2`. Both are opt-in because they send corpus text to third-party embedding APIs.

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

`python-dotenv` is part of the dev extra because mypy checks the benchmark harnesses and two MTRAG
modules import `dotenv_values` for optional `--dsn-env-file` support. It is not a runtime dependency
for the library or MCP server.

`ruff` is capped below 0.17. Ruff 0.16 is supported, but the project pins the pre-0.16 default rule
selection explicitly in `pyproject.toml` so a linter upgrade is not also a repository-wide style
rewrite. Adopting additional 0.16 rules should be a targeted cleanup.
