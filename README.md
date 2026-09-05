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
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="docs/postgresql-badge.svg" alt="PostgreSQL + pgvector">
</p>

<p align="center">
  <a href="#what-is-re-call">What it is</a>
  &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a>
  &nbsp;·&nbsp;
  <a href="#quickstart">Try the demo</a>
  &nbsp;·&nbsp;
  <a href="#install-and-integrate">Install and integrate</a>
</p>

## What is RE-call

RE-call is a retrieval and memory layer for agents that need to know when a result is safe to use.
It stores source documents in your PostgreSQL database, indexes them with pgvector, and keeps
validity and lineage attached to every hit.

Plain vector search returns nearby text. RE-call also asks whether that text is current, supported,
and trustworthy enough for the query. A superseded claim comes back marked `superseded`; a result
that does not clear the calibrated trust gate becomes `ABSTAIN` with a reason. Declared supersession makes the current memory win over stale but similar memory.

The core path does not require a memory-layer LLM call. Local embeddings are available by default,
while hosted embeddings, reranking, sparse retrieval, reasoning, and structured fact application
are opt in.

## How it works

The system has one build path and one trusted read path. Reasoning and structured fact writes are
optional consumers of trusted evidence.

```mermaid
flowchart TB
    subgraph BUILD["1. Build a generation"]
        direction LR
        SOURCE["Memo files<br/>frontmatter"] --> INDEX["Manifest, chunk, embed"]
        INDEX --> GEN[("Immutable generation<br/>PostgreSQL + pgvector")]
        GEN --> CAL["Published calibration"]
    end

    subgraph READ["2. Trusted read path"]
        direction LR
        QUESTION["Question"] --> PIN["Pin active generation"]
        PIN --> RETRIEVE["Hybrid retrieval<br/>dense + full text"]
        RETRIEVE --> GATE{"Calibrated<br/>trust gate"}
        CAL --> GATE
        GATE -->|"admit"| TRUSTED["Trusted evidence<br/>verdict + provenance"]
        GATE -->|"refuse"| ABSTAIN["ABSTAIN<br/>reason returned"]
    end

    subgraph OUTPUTS["3. Optional consumers"]
        direction TB
        TRUSTED --> ANSWER["Reasoning + citation validation<br/>answer, review, or ABSTAIN"]
        TRUSTED --> EVIDENCE["Citable evidence<br/>recall_evidence"]
        EVIDENCE --> CARDS["Immutable evidence cards"]
        CARDS --> CONTROLLER["Provenance controller<br/>recall_apply_fact<br/>recheck source and lineage"]
        CONTROLLER --> LEDGER[("Fact ledger<br/>assertions and refusals")]
        LEDGER --> CURRENT["Current facts<br/>recall_current_facts"]
        LEDGER -. "authorized events" .-> OUTBOX["Materialization outbox<br/>bounded recovery"]
    end

    CONTROLLER -. "at most one fresh search" .-> RETRIEVE
    GEN -. "active generation" .-> PIN

    classDef defaultPath fill:#e8f3ff,stroke:#2b6cb0,color:#102a43,stroke-width:1px;
    classDef optionalPath fill:#fff8e1,stroke:#b7791f,color:#5f370e,stroke-width:1px;
    classDef trustPath fill:#e8f5e9,stroke:#2f855a,color:#163b27,stroke-width:1px;
    class SOURCE,INDEX,GEN,CAL,QUESTION,PIN,RETRIEVE defaultPath;
    class ANSWER,EVIDENCE,CARDS,CONTROLLER,CURRENT,OUTBOX optionalPath;
    class ABSTAIN,GATE,TRUSTED,LEDGER trustPath;
```

In practical terms:

1. A manifest turns a corpus into an immutable, tenant scoped generation.
2. A query pins that generation, runs retrieval, and passes through calibration and trust policy.
3. Trusted evidence can feed an answer, citations, or a reviewed fact application. Unauthorized fact
   writes become recorded refusals rather than corpus rewrites.

The detailed architecture is in [docs/WRITEUP.md](docs/WRITEUP.md). The provenance boundary is
documented in [docs/PROVENANCE_CONTROLLER.md](docs/PROVENANCE_CONTROLLER.md), and the complete
API is in [docs/API.md](docs/API.md).

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

For your own corpus, install the package, provide PostgreSQL with pgvector, and run the guided setup wizard:

```bash
pip install "recall-rag[fastembed]"
recall setup
```

The wizard applies the schema, asks for the embedder and retrieval options, indexes the corpus,
offers calibration, and registers the selected agent integration. When the wizard asks whether to calibrate, use a labeled query file that refers to the corpus you are installing. Calibration fitted
to the bundled demo is only an example, not a certification for your data. The schema uses an
ordered SQL migration path and pre-tenancy tables are migrated in place.

For Docker, an existing database, headless provisioning, manual calibration, and troubleshooting,
see [docs/INSTALLATION.md](docs/INSTALLATION.md) and [docs/WIZARD.md](docs/WIZARD.md).

### Choose an integration

| Use case | Install | Next step |
|---|---|---|
| CLI and Python | `pip install "recall-rag[fastembed]"` | Run `recall setup`, then use `recall search` or the [Python API](docs/API.md). |
| MCP server | `pip install "recall-rag[fastembed,mcp]"` | Run `recall setup` or follow [the MCP guide](docs/USING_WITH_CLAUDE.md). |
| Claude Code | The MCP install plus the plugin | Install the package, then run the plugin commands below. `recall setup` configures the project corpus and hooks. |
| Claude Desktop | The MCP install | Run setup, add the server block from [the Claude guide](docs/USING_WITH_CLAUDE.md), then restart Claude Desktop. |
| Codex | `pip install "recall-rag[fastembed]"` | Run `recall setup` from the project. It detects Codex and installs the MCP server, plugin bundle, skills, and lifecycle hooks. |
| Claude Agent SDK | `pip install "recall-rag[agent,fastembed]"` | Use the in-process integration in [USING_WITH_AGENT_SDK.md](docs/USING_WITH_AGENT_SDK.md). |
| LangChain | `pip install "recall-rag[langchain,fastembed]"` | Use `recall.integrations.langchain.RecallRetriever`. |
| LlamaIndex | `pip install "recall-rag[llamaindex,fastembed]"` plus `llama-index-core` | Use `recall.integrations.llamaindex.RecallRetriever`. |
| Windows desktop installer | `pip install "recall-rag[desktop]"` | Run `recall-install`. See [the wizard guide](docs/WIZARD.md). |

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
| Retrieval | Dense, sparse, hybrid RRF, optional reranking, validity, calibrated confidence, provenance, and trust verdicts. |
| Storage | PostgreSQL with pgvector, immutable generations, migrations, incremental indexing, pruning, and source erasure. |
| Agent access | CLI, MCP, Claude Code, Claude Desktop, Codex, Claude Agent SDK, LangChain, LlamaIndex, and Python APIs. |
| Structured facts | Citable evidence cards, provenance controller, append only fact ledger, current fact projection, and optional materialization outbox. |
| Quality | Real pgvector integration tests, type checking, linting, dependency audit, and a claim gate that checks published evidence in CI. |

RE-call is not a hosted memory service, a dashboard, or an automatic truth extractor. It does not
rewrite corpus metadata from an agent's inference. Reasoning is opt in, citation constrained, and
review aware. See [docs/PRODUCTION.md](docs/PRODUCTION.md) for deployment boundaries.

## Read next

| Need | Document |
|---|---|
| Full documentation map | [docs/README.md](docs/README.md) |
| Installation and provisioning | [docs/INSTALLATION.md](docs/INSTALLATION.md) |
| Python, CLI, and MCP reference | [docs/API.md](docs/API.md) |
| Trust, architecture, and provenance | [docs/WRITEUP.md](docs/WRITEUP.md), [docs/PROVENANCE_CONTROLLER.md](docs/PROVENANCE_CONTROLLER.md) |
| Calibration and generations | [docs/FIRST_CALIBRATION.md](docs/FIRST_CALIBRATION.md), [docs/CALIBRATION.md](docs/CALIBRATION.md), [docs/GENERATIONS.md](docs/GENERATIONS.md) |
| Security and operations | [docs/AUTH.md](docs/AUTH.md), [docs/MIGRATIONS.md](docs/MIGRATIONS.md), [docs/OPERATING_MODES.md](docs/OPERATING_MODES.md) |
| Measurements and limits | [docs/EVIDENCE.md](docs/EVIDENCE.md), [results/FINDINGS.md](results/FINDINGS.md) |
| Upgrade notes | [CHANGELOG.md](CHANGELOG.md) |

Published numbers are tied to committed artifacts, and the claim gate checks them in CI. Benchmark
interpretation and limits belong in [docs/EVIDENCE.md](docs/EVIDENCE.md), not in this overview.

## Citation

If you describe RE-call in a paper, post, talk, or README of your own, cite the project and credit
Giulio D'Erme. Use [CITATION.cff](CITATION.cff) as the canonical citation source.

## License

Apache 2.0 license. See [LICENSE](LICENSE), and keep [NOTICE](NOTICE) with redistributed derivative
works.
