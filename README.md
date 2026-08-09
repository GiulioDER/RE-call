<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call: Retrieval-Augmented Self-Recall" width="900">
</p>

<p align="center">
  <b>Trustworthy memory for AI agents.</b><br>
  RE-call gives retrieval results confidence, provenance, validity, tenant isolation, and an explicit abstention path when the memory does not support an answer.
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PostgreSQL-16%2F17%20%C2%B7%20pgvector-336791" alt="PostgreSQL + pgvector">
  <img src="https://img.shields.io/badge/tests-1300%2B%20·%20real%20pgvector-brightgreen" alt="1300+ tests">
</p>

<p align="center">
  <a href="#why-re-call">Why RE-call</a>
  &nbsp;·&nbsp;
  <a href="#showcase">Showcase</a>
  &nbsp;·&nbsp;
  <a href="#product-surface">Product surface</a>
  &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a>
  &nbsp;·&nbsp;
  <a href="#benchmarks">Benchmarks</a>
</p>

## Why RE-call

Most memory systems optimize for the nearest match. Agent memory needs a stricter contract: the
retriever must say whether a memory is current, where it came from, how confident it is, and when
the corpus does not contain an answer.

RE-call is built around that contract.

| Capability | What it means in practice |
|---|---|
| Validity-aware retrieval | Superseded, expired, not-yet-valid, low-confidence, and not-entailed hits are surfaced as verdicts rather than flattened into ordinary search results. |
| Explicit abstention | When no valid result clears the calibrated threshold, callers receive an abstention with a reason instead of a nearest-neighbor guess. |
| Local operation | Ingest and retrieval run on PostgreSQL plus pgvector. Local embeddings are supported, so memory can be built and queried without a memory-layer LLM call. |
| Production boundaries | Tenant IDs, row-level security, token-scoped MCP HTTP transports, erasure, quotas, timeouts, migrations, and observability are part of the shipped surface. |
| Reproducible evidence | Published numbers are tied to committed artifacts, and the claim gate checks them in CI. |

Measured strengths:

| Strength | Evidence boundary |
|---|---|
| Lower memory-layer cost | The LOCOMO head-to-head records no RE-call memory-layer LLM calls, while the comparator pays for extraction calls. See [benchmarks/REVIEW.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/REVIEW.md). |
| External abstention check | On MTRAG, IBM's multi-turn RAG benchmark, RE-call is second on correct refusals among the recomputed systems and stays near the top answer-quality rows. See [docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md). |
| Validity beats nearest-match retrieval | The stale rate-limit memory is more similar to the query in the demo, but declared supersession makes the current memory win. The larger trust study is in [results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md). |
| Stronger than a plain vector store | Returned hits carry verdicts, confidence, provenance, tenant scope, and validity metadata. Plain top-k retrieval returns neighbors and leaves trust to the caller. |
| Honest about limits | The published results include negative findings on near-miss abstention, corpus sensitivity, and benchmark scope. Those limits are first-class documentation, not footnotes. |

The README is the product overview. The evidence is deliberately separated:
[results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) explains
what the measurements support, [results/RESULTS.md](https://github.com/GiulioDER/RE-call/blob/master/results/RESULTS.md)
contains the tables, and [results/ARTIFACTS.md](https://github.com/GiulioDER/RE-call/blob/master/results/ARTIFACTS.md)
maps result files to configurations.

## Showcase

Run the built-in demo:

```bash
python -m recall.cli demo
```

It indexes the sample corpus, searches for the active rate-limit decision, and then asks a question
the corpus cannot answer.

```text
[ok] query='how many requests per second can a client make?'
  ok          conf=1.00  cos=0.784  rate_limits_v2.md
  superseded  conf=1.00  cos=0.806  rate_limits_v1.md  superseded_by=rate_limits_v2

[ABSTAIN · gap] query='how do we handle penguins on mars?'
  reason: no hit above the calibrated confidence threshold
```

The stale memory is more similar to the query, but it is declared superseded and loses to the
current memory. The unrelated query returns an abstention. That is the core behavior.

## How it works

```mermaid
flowchart TB
    M(["memo · markdown + frontmatter<br/>supersedes · valid_from · valid_until"]) --> CH[chunk]
    CH --> EW["embed · local, no API call"]
    EW -. optional .-> SP[SPLADE encode]
    EW --> DB
    SP -. optional .-> DB

    Q([query]) --> EQ["embed · query encoder"]
    EQ --> DB[("PostgreSQL + pgvector<br/>vectors and full-text in one DB")]

    DB --> DN["dense · pgvector cosine"]
    DB --> SL["sparse · Postgres full-text"]
    DB -. optional .-> LS["learned sparse · SPLADE"]

    DN --> F[Reciprocal Rank Fusion]
    SL --> F
    LS -. optional .-> F

    F -. optional .-> RR[cross-encoder rerank]
    RR --> GP
    F --> GP{{"gap check · calibrated threshold"}}
    GP --> TR{"trust layer<br/>supersession · validity · confidence"}
    CAL[/"calibration · fitted per embedder and corpus"/] --> TR
    TR -. optional .-> EJ{{entailment judge}}
    EJ --> OUT
    TR --> OUT(["verdict + confidence + provenance<br/>or ABSTAIN, with a reason"])

    classDef opt stroke:#d29922,color:#d29922,stroke-dasharray:5 4
    class SP,LS,RR,EJ opt
```

## Product surface

| Area | Ships today |
|---|---|
| Retrieval | Dense, sparse, hybrid RRF, optional SPLADE, optional cross-encoder reranking, calibrated confidence, provenance, and trust verdicts. |
| Storage | PostgreSQL with pgvector, ordered SQL migration path, immutable generations, incremental indexing, pruning, and source-scoped erasure. |
| Agent integration | CLI, MCP server, LangChain retriever, LlamaIndex retriever, and injectable search seams for tests. |
| Security | Tenant isolation, row-level security checks, serving and migration DSNs, bearer-token HTTP transports, scopes, quotas, and unsafe-DSN refusal. |
| Operations | Timeouts, reconnect policy, structured logging, counters, latency percentiles, and MCP stats. |
| Quality gates | **1,300+ tests**, real pgvector integration tests, type checking, linting, dependency audit, claim-artifact checks, and regression fixtures for known failure modes. |

Deliberately out of scope: an end-user dashboard, graph reasoning, entity synthesis, high
availability orchestration, and automatic truth inference from prose.

The ordered SQL migration path is versioned now, pre-tenancy tables are migrated in place, and runtime
`CREATE TABLE IF NOT EXISTS` remains bootstrap only.

## Quickstart

```bash
docker compose up -d --wait
pip install "recall-rag[fastembed]"
python -m recall.cli --migration-dsn postgresql://recall:recall@localhost:5432/recall \
  schema --dim 384 apply
python -m recall.cli demo
```

The distribution is `recall-rag`; the import is `recall`. The name `recall` on PyPI belongs to an
unrelated package, so do not install both into the same environment.

Working from a clone:

```bash
pip install -e ".[fastembed]"
```

For a guided local setup:

```bash
python -m recall.cli setup
```

## Use it

```bash
python -m recall.cli index ./notes
python -m recall.cli search "what did we decide about caching?"
python -m recall.cli lint ./notes
python -m recall.cli check ./notes/new-memo.md --strict
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

Operational safety notes:

| Topic | Rule |
|---|---|
| Test database | The test suite drops tables. It uses `RECALL_TEST_DSN`, never `RECALL_DSN`. |
| Default credentials | The MCP server refuses a non-local built-in `recall:recall` DSN unless `RECALL_ALLOW_INSECURE_DSN=1` is set deliberately. |
| Tenancy | Set `RECALL_TENANT` or `PgVectorStore(tenant=...)`. Use an unprivileged database role, because PostgreSQL superusers bypass RLS. |

## MCP

```json
{
  "mcpServers": {
    "recall": {
      "command": "python",
      "args": ["-m", "recall_mcp.server"],
      "env": {
        "RECALL_SERVING_DSN": "postgresql://...",
        "RECALL_TENANT": "acme"
      }
    }
  }
}
```

Tools: `recall_search`, `recall_index`, `recall_forget`, and `recall_stats`.

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
| [docs/AUTH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/AUTH.md) | Authentication, scopes, and tenant isolation. |
| [docs/MIGRATIONS.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MIGRATIONS.md) | Migration roles, serving DSNs, and schema operations. |
| [docs/CALIBRATION.md](https://github.com/GiulioDER/RE-call/blob/master/docs/CALIBRATION.md) | Calibration workflow and generation-aware serving. |
| [docs/CASE_STUDY.md](https://github.com/GiulioDER/RE-call/blob/master/docs/CASE_STUDY.md) | Where the system came from and what is public versus private. |
| [docs/RESEARCH_PROTOCOL.md](https://github.com/GiulioDER/RE-call/blob/master/docs/RESEARCH_PROTOCOL.md) | How benchmark runs are controlled and audited. |

Release notes and upgrade warnings live in [CHANGELOG.md](https://github.com/GiulioDER/RE-call/blob/master/CHANGELOG.md).

## Benchmarks

Start with [benchmarks/README.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/README.md).

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
| [docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md) | MTRAG setup, results, and scope boundaries. |
| [benchmarks/REVIEW.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/REVIEW.md) | Adversarial review of the LOCOMO comparison. |
| [benchmarks/PREREGISTRATION.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/PREREGISTRATION.md) | Pre-registered rules for the memory benchmark. |

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
