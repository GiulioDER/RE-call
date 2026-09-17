<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call: memory that knows when not to guess" width="900">
</p>

<!-- mcp-name: io.github.GiulioDER/re-call -->

<p align="center">
  <b>Memory that abstains instead of guessing.</b><br>
  RE-call is agent memory on your own PostgreSQL with pgvector. Each result carries a verdict,
  confidence, and provenance, while unsupported questions are refused instead of answered by the
  nearest match.
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/recall-rag/"><img src="https://img.shields.io/pypi/v/recall-rag" alt="PyPI version"></a>
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue.svg" alt="License: PolyForm Noncommercial"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="docs/postgresql-badge.svg" alt="PostgreSQL + pgvector">
</p>

<p align="center">
  <a href="#what-is-re-call">What it is</a>
  &nbsp;·&nbsp;
  <a href="docs/FOR_AGENTS.md">For agents</a>
  &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a>
  &nbsp;·&nbsp;
  <a href="#quickstart">Try the demo</a>
  &nbsp;·&nbsp;
  <a href="#install-and-integrate">Install and integrate</a>
</p>

## What is RE-call

RE-call is agent memory on your own PostgreSQL database. It indexes source documents with pgvector
and keeps validity, lineage, confidence, and provenance attached to every result.

Vector search returns nearby text. RE-call also checks whether that text is current, supported, and
trustworthy enough for the query. A superseded claim is marked `superseded`; a result that does not
clear the calibrated trust gate becomes `ABSTAIN` with a reason. Declared supersession makes the current memory win
over stale but similar memory.

The default path uses local embeddings plus hybrid dense and Postgres full text retrieval. It needs
no memory-layer LLM call. Additional retrieval, reasoning, and structured-fact modules are
optional.

For the agent reading this, [RE-call for agents](docs/FOR_AGENTS.md) explains the experience loop
and how to use persistent memory as a trustworthy past rather than an unverified scratchpad.

## How it works

RE-call builds an immutable, calibrated generation once and serves queries against that pinned
generation. The solid path is the default. Dashed arrows mark the optional reasoning graph path or
generation bindings. Nothing bypasses the generation or trust boundary.

```mermaid
flowchart LR

subgraph group_clients["Entry points"]
  node_cli["CLI<br/>command interface<br/>[cli.py]"]
  node_package_api["Python package API<br/>library API<br/>[__init__.py]"]
  node_agent_sdk["Agent memory SDK<br/>in-process SDK<br/>[memory.py]"]
  node_mcp_server["MCP server<br/>tool server<br/>[server.py]"]
end

subgraph group_build["Generation build"]
  node_setup_wizard["Setup wizard<br/>provisioning workflow<br/>[setup.py]"]
  node_generation_builder["Generation builder<br/>immutable indexing pipeline"]
  node_document_ingest["Document ingestion<br/>document processing<br/>[document.py]"]
  node_indexer["Index writer<br/>indexing service<br/>[index.py]"]
end

subgraph group_serving["Retrieval and trust"]
  node_retriever["Hybrid retriever<br/>query service<br/>[retriever.py]"]
  node_retrieval_legs["Dense, FTS, sparse legs<br/>retrieval components<br/>[embeddings.py]"]
  node_reranker["Reranker<br/>candidate ranking<br/>[rerank.py]"]
  node_trust_gate{{"Trust gate<br/>policy enforcement<br/>[trust.py]"}}
  node_calibration["Calibration<br/>generation readiness<br/>[calibration.py]"]
end

subgraph group_storage["Storage and operations"]
  node_postgres[("PostgreSQL + pgvector<br/>authoritative data store")]
  node_redis[("Redis<br/>rate limiting<br/>[redis.tf]")]
  node_aws_deployment["AWS deployment<br/>infrastructure<br/>[ecs.tf]"]
end

subgraph group_reasoning["Reasoning and provenance"]
  node_reasoning_planner["Reasoning planner<br/>bounded expansion planner"]
  node_reasoning_expansion["Graph expansion<br/>reasoning service"]
  node_semantic_graph["Semantic graph<br/>graph storage and serving<br/>[semantic_graph.py]"]
  node_provenance["Provenance controller<br/>evidence review"]
  node_fact_ledger["Append-only fact ledger<br/>structured fact record<br/>[fact_ledger.py]"]
end

node_cli -->|"provisions"| node_setup_wizard
node_cli -->|"indexes"| node_generation_builder
node_package_api -->|"queries"| node_retriever
node_agent_sdk -->|"queries"| node_retriever
node_mcp_server -->|"tool calls"| node_retriever
node_setup_wizard -->|"configures"| node_postgres
node_document_ingest -->|"parsed documents"| node_generation_builder
node_generation_builder -->|"validated chunks"| node_indexer
node_indexer -->|"commits generation"| node_postgres
node_retriever -->|"dispatches query"| node_retrieval_legs
node_retrieval_legs -->|"vector and full-text search"| node_postgres
node_retrieval_legs -->|"candidates"| node_reranker
node_reranker -->|"ranked evidence"| node_trust_gate
node_calibration -->|"readiness and confidence"| node_trust_gate
node_postgres -->|"generation state"| node_calibration
node_retriever -.->|"direct candidates"| node_reasoning_planner
node_reasoning_planner -->|"bounded plan"| node_reasoning_expansion
node_semantic_graph -->|"same-generation neighbors"| node_reasoning_expansion
node_semantic_graph -->|"graph records"| node_postgres
node_reasoning_expansion -->|"expanded evidence"| node_trust_gate
node_trust_gate -->|"trusted evidence"| node_provenance
node_provenance -->|"reviewed facts"| node_fact_ledger
node_fact_ledger -->|"protected append"| node_postgres
node_aws_deployment -->|"RDS"| node_postgres
node_aws_deployment -->|"operates"| node_redis

click node_cli "https://github.com/giulioder/re-call/blob/master/recall/cli.py"
click node_package_api "https://github.com/giulioder/re-call/blob/master/recall/__init__.py"
click node_agent_sdk "https://github.com/giulioder/re-call/blob/master/recall_agent/memory.py"
click node_mcp_server "https://github.com/giulioder/re-call/blob/master/recall_mcp/server.py"
click node_setup_wizard "https://github.com/giulioder/re-call/blob/master/recall/setup.py"
click node_generation_builder "https://github.com/giulioder/re-call/blob/master/recall/generation_build.py"
click node_document_ingest "https://github.com/giulioder/re-call/blob/master/recall/document.py"
click node_indexer "https://github.com/giulioder/re-call/blob/master/recall/index.py"
click node_retriever "https://github.com/giulioder/re-call/blob/master/recall/retriever.py"
click node_retrieval_legs "https://github.com/giulioder/re-call/blob/master/recall/embeddings.py"
click node_reranker "https://github.com/giulioder/re-call/blob/master/recall/rerank.py"
click node_trust_gate "https://github.com/giulioder/re-call/blob/master/recall/trust.py"
click node_calibration "https://github.com/giulioder/re-call/blob/master/recall/calibration.py"
click node_reasoning_planner "https://github.com/giulioder/re-call/blob/master/recall/reasoning_planner.py"
click node_reasoning_expansion "https://github.com/giulioder/re-call/blob/master/recall/reasoning_expansion_service.py"
click node_semantic_graph "https://github.com/giulioder/re-call/blob/master/recall/semantic_graph.py"
click node_provenance "https://github.com/giulioder/re-call/blob/master/recall/provenance_controller.py"
click node_fact_ledger "https://github.com/giulioder/re-call/blob/master/recall/fact_ledger.py"
click node_redis "https://github.com/giulioder/re-call/blob/master/infra/aws/redis.tf"
click node_aws_deployment "https://github.com/giulioder/re-call/blob/master/infra/aws/ecs.tf"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_cli,node_package_api,node_agent_sdk,node_mcp_server toneBlue
class node_setup_wizard,node_generation_builder,node_document_ingest,node_indexer toneAmber
class node_retriever,node_retrieval_legs,node_reranker,node_trust_gate,node_calibration toneMint
class node_reasoning_planner,node_reasoning_expansion,node_semantic_graph,node_provenance,node_fact_ledger toneRose
class node_postgres,node_redis,node_aws_deployment toneIndigo
```

Ordinary `recall search` follows the direct path. Explicit reasoning accepts `graph_expansion`:
`auto` is the default and resolves to bounded one hop expansion for every nonempty query; `off`
keeps direct retrieval only; `one-hop` forces the graph path. The CLI uses
`--graph-expansion auto|off|one-hop`; the MCP tool uses `graph_expansion="auto"|"off"|"one_hop"`.
Graph neighbors are generation bound, direct candidates remain first, and expanded candidates must
clear the same trust boundary before they can support a cited answer.

The opt in choices attach to different points in the system:

| Optional capability | Where it fits | What it adds |
|---|---|---|
| Hosted embedder | Build and query | Remote model calls for embeddings. Query and corpus text may leave the environment. |
| Learned sparse retrieval, SPLADE | Hybrid retrieval | A learned term weighted retrieval leg in addition to dense vectors and Postgres full text. |
| Reranker | After candidate fusion | Reorders the fused candidates with a cross encoder. |
| Entailment judge | After the trust decision | Demotes high similarity near misses that do not answer the question. |
| Evidence Graph version one | Explicit reasoning retrieval | Adds bounded, generation-bound structural neighbors to reasoning retrieval. See `graph_expansion` above for controls; graph candidates pass through trust before a cited answer can use them. |
| Structured fact application | Evidence cards | Lets a reviewed fact pass through the provenance controller into the append only ledger. |

For details, see the [architecture writeup](docs/WRITEUP.md), [provenance controller](docs/PROVENANCE_CONTROLLER.md), and [API reference](docs/API.md).

## Quickstart

Prerequisites: Python 3.11 or newer, Docker, and a Docker installation able to run PostgreSQL with
pgvector.

```bash
pip install "recall-rag[fastembed]"
recall quickstart
```

The demo starts a throwaway database, indexes a small corpus included in the package, and runs
three searches. It includes a normal answer, a stale claim that is returned as superseded, and a
question that is refused. The demo uses development trust and changes no personal files.

Remove the demo database when finished:

```bash
recall quickstart --remove
```

Already have PostgreSQL with pgvector? Use `recall quickstart --existing-dsn <dsn>` instead. The
demo is intentionally separate from a real install and is not calibrated for your data.

## Install and integrate

For your own corpus, provide PostgreSQL with pgvector and run the guided setup wizard after
installing `recall-rag[fastembed]`:

```bash
recall setup
```

It applies the schema, asks for the embedder and retrieval options, indexes the corpus, offers
calibration, and registers the selected agent integration. When the wizard asks whether to calibrate,
use a labeled query file that refers to the corpus you are installing. Calibration fitted
to the bundled demo is only an example, not a certification for your data. The schema uses an
ordered SQL migration path and pre-tenancy tables are migrated in place.

For Docker, an existing database, headless provisioning, manual calibration, and troubleshooting,
see [docs/INSTALLATION.md](docs/INSTALLATION.md) and [docs/WIZARD.md](docs/WIZARD.md).

### Choose an integration

| Use case | Install | Next step |
|---|---|---|
| CLI and Python | `pip install "recall-rag[fastembed]"` | Run `recall setup`, then use `recall search` or the [Python API](docs/API.md). |
| MCP, Claude Code, Claude Desktop, or Codex | `pip install "recall-rag[fastembed,mcp]"` | Run setup and follow [the MCP guide](docs/USING_WITH_CLAUDE.md). Host specific steps are below. |
| Claude Agent SDK | `pip install "recall-rag[agent,fastembed]"` | Use the in process integration in [USING_WITH_AGENT_SDK.md](docs/USING_WITH_AGENT_SDK.md). |
| LangChain or LlamaIndex | Install the matching extra | Use the adapters described in [API.md](docs/API.md). |
| Windows desktop UI | `pip install "recall-rag[desktop]"` | Run `recall-install`; the current release does not ship a standalone Windows binary. See [the wizard guide](docs/WIZARD.md). |

#### Claude Code

Inside Claude Code, install the plugin after installing the Python package:

```text
/plugin marketplace add GiulioDER/RE-call
/plugin install recall@re-call
```

The plugin supplies the MCP server, memory search skill, and lifecycle hooks. `recall setup` still
needs to run against the project and database that Claude should use. The plugin keeps credentials
out of the repository. Details and manual wiring are in [plugin/README.md](plugin/README.md).

#### Codex

Run `recall setup` from the project. When Codex is detected, setup installs the Codex MCP server,
plugin bundle, memory skills, and hooks into the user configuration. Restart Codex afterward. The
Codex and Claude Code integrations share the same memo format and trust layer. See
[docs/CODEX_RECALL_INTEGRATION.md](docs/CODEX_RECALL_INTEGRATION.md).

#### Claude Agent SDK

The SDK integration runs the same tools in process and does not start an MCP server:

```python
from recall_agent import RecallAgentMemory

with RecallAgentMemory.from_env() as memory:
    options = memory.options()
```

See [docs/USING_WITH_AGENT_SDK.md](docs/USING_WITH_AGENT_SDK.md) for the complete example and
write-tool boundaries.

#### LangChain and LlamaIndex

Both adapters use the same trusted retrieval path. If trust abstains, they return no document by
default, and returned documents retain verdict, confidence, cosine, and supersession metadata.
See [docs/API.md](docs/API.md) for the supported classes and methods.

### If an install is not working

```bash
recall doctor
```

The doctor checks the interpreter, console scripts, embedder, Docker, database, pgvector, schema,
configured table and tenant, calibration, and agent registration. It changes nothing and prints the
repair command for each problem.

## Product surface

| Area | What ships |
|---|---|
| Retrieval and memory | Dense vectors plus Postgres full text with hybrid RRF, validity, calibrated confidence, provenance, trust verdicts, immutable generations, incremental indexing, pruning, and source erasure. |
| Structured facts | Citable evidence cards, provenance controller, append only fact ledger, current fact projection, and optional materialization outbox. |
| Quality | Real pgvector integration tests, type checking, linting, dependency audit, and a claim gate that checks published evidence in CI. |

RE-call is not a hosted memory service, dashboard, or automatic truth extractor. It does not rewrite
corpus metadata from an agent's inference. Reasoning is opt in, citation constrained, and review
aware. See [docs/PRODUCTION.md](docs/PRODUCTION.md) for deployment boundaries.

## Read next

| Need | Document |
|---|---|
| Why an agent needs persistent, trusted memory | [docs/FOR_AGENTS.md](docs/FOR_AGENTS.md) |
| Full documentation map | [docs/README.md](docs/README.md) |
| Install and provision | [docs/INSTALLATION.md](docs/INSTALLATION.md), [docs/WIZARD.md](docs/WIZARD.md) |
| Python, CLI, and MCP reference | [docs/API.md](docs/API.md) |
| Trust, architecture, and provenance | [docs/WRITEUP.md](docs/WRITEUP.md), [docs/PROVENANCE_CONTROLLER.md](docs/PROVENANCE_CONTROLLER.md) |
| Security and operations | [docs/AUTH.md](docs/AUTH.md), [docs/PRODUCTION.md](docs/PRODUCTION.md), [docs/OPERATING_MODES.md](docs/OPERATING_MODES.md) |
| Measurements and limits | [docs/EVIDENCE.md](docs/EVIDENCE.md), [results/FINDINGS.md](results/FINDINGS.md) |

Published numbers are tied to committed artifacts, and the claim gate checks them in CI. Benchmark
interpretation and limits belong in [docs/EVIDENCE.md](docs/EVIDENCE.md), not in this overview.

## Citation

If you describe RE-call in a paper, post, talk, or README of your own, cite the project and credit
Giulio D'Erme. Use [CITATION.cff](CITATION.cff) as the canonical citation source.

## License

RE-call is source available under the [PolyForm Noncommercial License 1.0.0](LICENSE). Personal,
educational, and noncommercial research use is permitted. Commercial use requires a separate
written license from the copyright holder. See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for
the boundary between permitted use and commercial licensing, and preserve [NOTICE](NOTICE) when
redistributing the software.
