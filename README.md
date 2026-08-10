<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call: Retrieval-Augmented Self-Recall" width="900">
</p>

<!-- mcp-name: io.github.GiulioDER/re-call -->

<p align="center">
  <b>Trustworthy memory for AI agents.</b><br>
  RE-call gives retrieval results confidence, provenance, validity, tenant isolation, and an explicit abstention path when the memory does not support an answer.
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PostgreSQL-16%2F17%20%C2%B7%20pgvector-336791" alt="PostgreSQL + pgvector">
  <img src="https://img.shields.io/badge/CI-real%20pgvector%20·%20types%20·%20audit-brightgreen" alt="CI: real pgvector, types, audit">
  <a href="https://glama.ai/mcp/servers/GiulioDER/RE-call"><img src="https://glama.ai/mcp/servers/GiulioDER/RE-call/badges/score.svg" alt="RE-call MCP server"></a>
</p>

<p align="center">
  <a href="https://glama.ai/mcp/servers/GiulioDER/RE-call"><img src="https://glama.ai/mcp/servers/GiulioDER/RE-call/badges/card.svg" alt="RE-call MCP server"></a>
</p>

<p align="center">
  <a href="#why-re-call">Why RE-call</a>
  &nbsp;·&nbsp;
  <a href="#five-minute-proof">Five-minute proof</a>
  &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a>
  &nbsp;·&nbsp;
  <a href="#product-surface">Product surface</a>
  &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a>
  &nbsp;·&nbsp;
  <a href="#evidence">Evidence</a>
</p>

## Why RE-call

Most memory systems optimize for the nearest match. Agent memory needs a stricter contract: the
retriever must say whether a memory is current, where it came from, how confident it is, and when
the corpus does not contain an answer.

RE-call is built around that contract.

It is for teams putting agent memory behind real applications: support copilots, internal research
agents, compliance assistants, and long-running workflow agents where a stale or unsupported memory
is worse than no memory. The buyer story is simple: keep the memory layer local by default, attach
policy to every hit, calibrate the refusal threshold on your corpus, and let the application decide
what to do with a result that is not trustworthy enough to answer from.

| Capability | What it means in practice |
|---|---|
| Validity-aware retrieval | Superseded, expired, not-yet-valid, low-confidence, and not-entailed hits are surfaced as verdicts rather than flattened into ordinary search results. |
| Explicit abstention | When no valid result clears the calibrated threshold, callers receive an abstention with a reason instead of a nearest-neighbor guess. |
| Local operation | Ingest and retrieval run on PostgreSQL plus pgvector. Local embeddings are supported, so memory can be built and queried without a memory-layer LLM call. |
| Policy-driven configuration | Embedder, reranker, calibration, trust policy, and retrieval profile are selected to match legal, hardware, latency, quality, and cost requirements. The default is local and offline; higher-quality or hosted options are opt-in. |
| Production boundaries | Tenant IDs, row-level security, token-scoped MCP HTTP transports, erasure, quotas, timeouts, migrations, and observability are part of the shipped surface. |
| Reproducible evidence | Published numbers are tied to committed artifacts, and the claim gate checks them in CI. |

Measured strengths:

| Strength | Evidence boundary |
|---|---|
| Lower memory-layer cost | The LOCOMO head-to-head records no RE-call memory-layer LLM calls, while the comparator pays for extraction calls. See [benchmarks/REVIEW.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/REVIEW.md). |
| External abstention check | On MTRAG, IBM's multi-turn RAG benchmark, RE-call is second on correct refusals among the recomputed systems and stays near the top answer-quality rows. See [docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md). |
| Validity beats nearest-match retrieval | The stale rate-limit memory is more similar to the query in the demo, but declared supersession makes the current memory win. The larger trust study is in [results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md). |
| Stronger than a plain vector store | Returned hits carry verdicts, confidence, provenance, tenant scope, and validity metadata. Plain top-k retrieval returns neighbors and leaves trust to the caller. |
| Clear limits | The evidence states where RE-call works, where it does not, and when a corpus-specific measurement is required. |

The README is the product overview. For evidence behind these claims, start with
[docs/EVIDENCE.md](https://github.com/GiulioDER/RE-call/blob/master/docs/EVIDENCE.md), then use
[results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) for
the full interpretation and limits.

## Five-minute proof

Run the bundled demo to see the product behavior before reading the evidence docs:

```bash
docker compose up -d --wait
pip install "recall-rag[fastembed]"
python -m recall.cli --table recall_quickstart \
  --migration-dsn postgresql://recall:recall@localhost:5432/recall \
  schema --dim 384 apply
RECALL_TRUST_MODE=development python -m recall.cli --table recall_quickstart demo
```

Expected shape:

![RE-call demo output](docs/demo-output.svg)

```text
[DEGRADED:INDEX_NOT_READY] query='how many requests per second can a client make?'
  ok          conf=1.00  cos=0.784  rate_limits_v2.md
  superseded  conf=1.00  cos=0.806  rate_limits_v1.md -> use rate_limits_v2.md

[ABSTAIN GAP DEGRADED:INDEX_NOT_READY] query='how do we handle penguins on mars?'
  reason: no hit above the calibrated confidence threshold
```

The stale memory is more similar to the query, but it is declared superseded and loses to the
current memory. The unrelated query returns an abstention. The degraded marker is intentional here:
this is a sample-corpus demonstration, not a certified production calibration.

Runnable examples: [examples/README.md](https://github.com/GiulioDER/RE-call/blob/master/examples/README.md).

## Quickstart

After the demo, run the guided setup wizard for your own corpus. The wizard records the selected
embedder, retrieval options, and an optional calibration that is fitted to your labeled queries and
your corpus.

```bash
docker compose up -d --wait
pip install "recall-rag[fastembed]"
python -m recall.cli --table recall_quickstart \
  --migration-dsn postgresql://recall:recall@localhost:5432/recall \
  schema --dim 384 apply
python -m recall.cli setup
```

PowerShell:

```powershell
docker compose up -d --wait
pip install "recall-rag[fastembed]"
python -m recall.cli --table recall_quickstart `
  --migration-dsn postgresql://recall:recall@localhost:5432/recall `
  schema --dim 384 apply
python -m recall.cli setup
```

When the wizard asks whether to calibrate, provide a labeled query JSON and the corpus directory.
Use [recall/eval/queries.json](https://github.com/GiulioDER/RE-call/blob/master/recall/eval/queries.json)
as the input shape. Calibration is per embedder and per corpus, so a new model or substantially
changed corpus should be calibrated again.

The distribution is `recall-rag`; the import is `recall`. The name `recall` on PyPI belongs to an
unrelated package, so do not install both into the same environment.

Working from a clone:

```bash
pip install -e ".[fastembed]"
```

## How it works

```mermaid
flowchart TB
    M["Memo: markdown plus frontmatter"] --> CH["Chunk"]
    CH --> EW["Embed locally"]
    EW -. "optional" .-> SP["SPLADE encode"]
    EW --> DB
    SP -. "optional" .-> DB

    Q["Query"] --> EQ["Query encoder"]
    EQ --> DB[("PostgreSQL plus pgvector")]

    DB --> DN["Dense vector search"]
    DB --> SL["Postgres full-text search"]
    DB -. "optional" .-> LS["Learned sparse search"]

    DN --> F["Reciprocal Rank Fusion"]
    SL --> F
    LS -. "optional" .-> F

    F -. "optional" .-> RR["Cross-encoder rerank"]
    RR --> GP
    F --> GP{"Gap check: calibrated threshold"}
    GP --> TR{"Trust layer: supersession, validity, confidence"}
    CAL["Calibration: fitted per embedder and corpus"] --> TR
    TR -. "optional" .-> EJ{"Entailment judge"}
    EJ --> OUT
    TR --> OUT["Verdict, confidence, provenance, or ABSTAIN"]
```

## Product surface

| Area | Ships today |
|---|---|
| Retrieval | Dense, sparse, hybrid RRF, optional SPLADE, optional cross-encoder reranking, calibrated confidence, provenance, and trust verdicts. |
| Configuration | Guided setup, local and hosted embedder choices, retrieval cost profiles, optional reranking, strict or development trust policy, and per-corpus calibration. |
| Storage | PostgreSQL with pgvector, ordered SQL migration path, immutable generations, incremental indexing, pruning, and source-scoped erasure. |
| Agent integration | CLI, MCP server, LangChain retriever, LlamaIndex retriever, and injectable search seams for tests. |
| Security | Tenant isolation, row-level security checks, serving and migration DSNs, bearer-token HTTP transports, scopes, quotas, and unsafe-DSN refusal. |
| Operations | Timeouts, reconnect policy, structured logging, counters, latency percentiles, and MCP stats. |
| Quality gates | Real pgvector integration tests, type checking, linting, dependency audit, claim-artifact checks, and regression fixtures for known failure modes. |

Deliberately out of scope: an end-user dashboard, graph reasoning, entity synthesis, high
availability orchestration, and automatic truth inference from prose.

The ordered SQL migration path is versioned now, pre-tenancy tables are migrated in place, and runtime
`CREATE TABLE IF NOT EXISTS` remains bootstrap only.

## Use it

For an ad hoc local markdown folder, create a table for that index, index the corpus, and search it.
If you did not calibrate during setup, use development mode only for local evaluation and demos.
Replace `./notes` with your memo folder.

```bash
python -m recall.cli --table recall_notes \
  --migration-dsn postgresql://recall:recall@localhost:5432/recall \
  schema --dim 384 apply
RECALL_TRUST_MODE=development python -m recall.cli --table recall_notes index ./notes
RECALL_TRUST_MODE=development python -m recall.cli --table recall_notes search "what did we decide about caching?"
python -m recall.cli lint ./notes
python -m recall.cli check ./notes/new-memo.md --strict
```

PowerShell uses the same commands, but set development mode first when you are running an
uncalibrated local evaluation:

```powershell
$env:RECALL_TRUST_MODE = "development"
```

For production generation mode, build, validate, calibrate, and promote an immutable generation.
Then query the tenant's active generation:

```python
from recall.embeddings import FastEmbedEmbedder
from recall.generation_store import GenerationStore
from recall.trust import trusted_search

emb = FastEmbedEmbedder()
with GenerationStore(DSN, dim=emb.dim, tenant="acme", pool_size=8) as store:
    store.check_schema()
    result = trusted_search(store, emb, "what is the rate limit?")
    if result.abstained:
        ...  # say you do not know
    for hit in result.hits:
        hit.verdict
        hit.confidence
        hit.validity.superseded_by
```

Set `RECALL_SERVING_DSN` for application traffic and `RECALL_MIGRATION_DSN` only in the migration
job. `RECALL_DSN` remains a deprecated development fallback for the serving DSN. See
[docs/MIGRATIONS.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MIGRATIONS.md).
Configuration modes are summarized in
[docs/OPERATING_MODES.md](https://github.com/GiulioDER/RE-call/blob/master/docs/OPERATING_MODES.md).

Operational safety notes:

| Topic | Rule |
|---|---|
| Test database | The test suite drops tables. It uses `RECALL_TEST_DSN`, never `RECALL_DSN`. |
| Default credentials | The MCP server refuses a non-local built-in `recall:recall` DSN unless `RECALL_ALLOW_INSECURE_DSN=1` is set deliberately. |
| Tenancy | Set `RECALL_TENANT` or `PgVectorStore(tenant=...)`. Use an unprivileged database role, because PostgreSQL superusers bypass RLS. |

## MCP

The MCP server uses the default `chunks` table. Apply that schema for the embedder the server will
run, then point the client at `recall_mcp.server`.

```bash
python -m recall.cli --migration-dsn postgresql://recall:recall@localhost:5432/recall \
  schema --dim 384 apply
```

If an existing `chunks` table was created with another vector dimension, use a fresh database or an
embedder with the matching dimension. The MCP stdio server does not take a `--table` flag.

```json
{
  "mcpServers": {
    "recall": {
      "command": "python",
      "args": ["-m", "recall_mcp.server"],
      "env": {
        "RECALL_SERVING_DSN": "postgresql://...",
        "RECALL_TENANT": "acme",
        "RECALL_TRUST_MODE": "development"
      }
    }
  }
}
```

Omit `RECALL_TRUST_MODE` in production after you have built, calibrated, and promoted a generation.
Local uncalibrated MCP work needs the explicit development setting for the same reason the CLI demo
does.

Tools: `recall_search`, `recall_evidence`, `recall_index`, `recall_forget`, and `recall_stats`.

Full guide: [docs/USING_WITH_CLAUDE.md](https://github.com/GiulioDER/RE-call/blob/master/docs/USING_WITH_CLAUDE.md).
Authentication and tenancy: [docs/AUTH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/AUTH.md).

## LangChain and LlamaIndex

```bash
pip install "recall-rag[langchain]"
pip install "recall-rag[llamaindex]"
```

```python
from recall.integrations.langchain import RecallRetriever

retriever = RecallRetriever.from_store(store, emb, k=5)
docs = retriever.invoke("what is the rate limit?")
```

When the trust layer abstains, the adapters return no document by default. Returned documents carry
trust metadata, including verdict, confidence, cosine, and supersession details.

## Documentation

Start with [docs/README.md](https://github.com/GiulioDER/RE-call/blob/master/docs/README.md).

Core documents:

| Document | Purpose |
|---|---|
| [docs/WRITEUP.md](https://github.com/GiulioDER/RE-call/blob/master/docs/WRITEUP.md) | Architecture and design rationale. |
| [docs/API.md](https://github.com/GiulioDER/RE-call/blob/master/docs/API.md) | Supported Python, CLI, and MCP surface. |
| [docs/REPOSITORY_MAP.md](https://github.com/GiulioDER/RE-call/blob/master/docs/REPOSITORY_MAP.md) | What is product, evidence, benchmark support, and archive. |
| [docs/AUTH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/AUTH.md) | Authentication, scopes, and tenant isolation. |
| [docs/MIGRATIONS.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MIGRATIONS.md) | Migration roles, serving DSNs, and schema operations. |
| [docs/OPERATING_MODES.md](https://github.com/GiulioDER/RE-call/blob/master/docs/OPERATING_MODES.md) | Local, production, quality, hosted, and evaluation deployment modes. |
| [docs/CALIBRATION.md](https://github.com/GiulioDER/RE-call/blob/master/docs/CALIBRATION.md) | Calibration workflow and generation-aware serving. |
| [docs/CASE_STUDY.md](https://github.com/GiulioDER/RE-call/blob/master/docs/CASE_STUDY.md) | Where the system came from and what is public versus private. |
| [docs/RESEARCH_PROTOCOL.md](https://github.com/GiulioDER/RE-call/blob/master/docs/RESEARCH_PROTOCOL.md) | How benchmark runs are controlled and audited. |

Release notes and upgrade warnings live in [CHANGELOG.md](https://github.com/GiulioDER/RE-call/blob/master/CHANGELOG.md).

## Evidence

Start with [benchmarks/README.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/README.md).
The results directory has its own map at
[results/README.md](https://github.com/GiulioDER/RE-call/blob/master/results/README.md).

The short version:

| Question | Current evidence |
|---|---|
| Does declared supersession beat plain similarity search? | Yes, on the authored-edge cases measured in the trust and scale studies. |
| Can abstention be trusted everywhere? | No. It works on far gaps and fails on near-misses unless a stronger answerability layer is added. |
| Is retrieval quality universal? | No. Corpus shape dominates, and the measured recommendation is to benchmark your corpus before choosing an embedder. |
| Is the Mem0 comparison apples-to-apples? | The published head-to-head uses the same LOCOMO questions, generator, judge, and paired tests, with reader-tier limits stated in the benchmark review. |
| What does MTRAG add? | A third-party multi-turn benchmark with an official judge that gives full credit for correct refusal. RE-call does not top the benchmark, and that boundary is stated in [docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md). |

Important benchmark documents:

| Document | Purpose |
|---|---|
| [results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) | Interpretation, limits, and negative results. |
| [results/RESULTS.md](https://github.com/GiulioDER/RE-call/blob/master/results/RESULTS.md) | Complete result tables. |
| [results/ARTIFACTS.md](https://github.com/GiulioDER/RE-call/blob/master/results/ARTIFACTS.md) | Checksum and artifact map for readers auditing a claim. |
| [docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md) | MTRAG setup, results, and scope boundaries. |
| [benchmarks/REVIEW.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/REVIEW.md) | Adversarial review of the LOCOMO comparison. |
| [benchmarks/PREREGISTRATION.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/PREREGISTRATION.md) | Pre-registered rules for the main memory benchmark. |
| [benchmarks/archive/preregistrations/README.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/archive/preregistrations/README.md) | Archived preregistrations for follow-up benchmark arms. |

## When not to use RE-call

Use something else if you need managed hosting, per-chunk ACLs, graph reasoning, automatic truth
extraction from prose, or a memory system that rewrites facts for you. RE-call is a retrieval
library over your PostgreSQL database, not a hosted memory platform.

## What this does not do

RE-call is a retrieval library, not a general reasoning system. It does not infer every missing
supersession edge, prove that an on-topic memory answers a near-miss question, or replace database
operations with a managed service. It returns the trust signals the caller needs, and it refuses to
pretend that a nearest match is always usable evidence.

## Reproduce

```bash
make eval
python -m recall.eval.scale --embedder hashing --filler 50000
```

Cloud rows require the relevant API keys. Local rows run key-free.

## Citation

If you describe RE-call in a paper, post, talk, or README of your own, cite the project and credit
Giulio D'Erme. Use [CITATION.cff](https://github.com/GiulioDER/RE-call/blob/master/CITATION.cff)
as the canonical citation source.

## License

Apache 2.0 license. See [LICENSE](https://github.com/GiulioDER/RE-call/blob/master/LICENSE), and
keep [NOTICE](https://github.com/GiulioDER/RE-call/blob/master/NOTICE) with redistributed
derivative works.
