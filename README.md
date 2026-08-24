<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call: memory that knows when not to guess" width="900">
</p>

<!-- mcp-name: io.github.GiulioDER/re-call -->

<p align="center">
  <b>Memory that abstains instead of guessing.</b><br>
  RE-call is agent memory on your own PostgreSQL with pgvector: every hit carries a verdict,
  confidence and provenance, a retracted claim comes back marked <code>superseded</code>, and a
  question the corpus cannot answer is refused rather than answered from the nearest neighbour.
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/recall-rag/"><img src="https://img.shields.io/pypi/v/recall-rag" alt="PyPI version"></a>
  <a href="https://pypi.org/project/recall-rag/"><img src="https://img.shields.io/pypi/dm/recall-rag" alt="PyPI downloads"></a>
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="docs/postgresql-badge.svg" alt="PostgreSQL + pgvector">
  <img src="https://img.shields.io/badge/CI-real%20pgvector%20·%20types%20·%20audit-brightgreen" alt="CI: real pgvector, types, audit">
  <a href="https://glama.ai/mcp/servers/GiulioDER/RE-call"><img src="https://glama.ai/mcp/servers/GiulioDER/RE-call/badges/score.svg" alt="RE-call MCP server"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/launch/terminal-video/out/re-call-terminal-demo-preview.gif" alt="Terminal demo: a stale rate-limit memo outranks its correction on cosine, RE-call marks it superseded, and an unanswerable query is refused" width="720">
</p>

<p align="center">
  ATM-Bench Recall@10 <b>92.8924</b> <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10_percent -->
  against 79.09 <!--@ citation-pending: the board's published Memexa row, reproduced in docs/ATM_BENCH.md, not a RE-call measurement -->
  for the best published row (<a href="https://github.com/GiulioDER/RE-call/blob/master/docs/ATM_BENCH.md">limits</a>)
  &nbsp;·&nbsp;
  second of ten on MTRAG correct refusals (<a href="https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md">limits</a>)
  &nbsp;·&nbsp;
  zero memory-layer LLM calls to build memory, where Mem0 pays one per session (<a href="https://github.com/GiulioDER/RE-call/blob/master/benchmarks/REVIEW.md">limits</a>)
</p>

<p align="center">
  <a href="#why-re-call">Why RE-call</a>
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

<p align="center">
  <a href="https://giulioder.github.io/RE-call/">Setup guide: install, configure and run RE-call</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/GiulioDER/validity-frontmatter">Validity Frontmatter: the open spec RE-call implements</a>
</p>

## Why RE-call

Nearest-match retrieval cannot tell the difference between what is true and what merely reads like
it. When a corpus keeps its history, and real agent memory does, the retracted claim and its
correction are both retrievable, and the retracted one is often the nearer match. That is not a
tuning problem. A ranker with no notion of validity has no way to prefer the correction.

RE-call came out of a production, long-running trading-research agent: months of operation,
792 <!--@ citation-pending: measured in docs/CASE_STUDY.md, not backed by a committed results artifact -->
typed memos, 6,469 <!--@ citation-pending: measured in docs/CASE_STUDY.md, not backed by a committed results artifact -->
chunks, re-indexed daily by a session-end hook (counts from the private corpus behind the case
study, so no committed artifact backs them). Every guard in this repository
exists because that agent failed a specific way without it. See
[docs/CASE_STUDY.md](https://github.com/GiulioDER/RE-call/blob/master/docs/CASE_STUDY.md).

It is for teams putting agent memory behind real applications, where a stale or unsupported memory
is worse than no memory: keep the memory layer local by default, attach policy to every hit,
calibrate the refusal threshold on your corpus, and let the application decide what to do with a
result that is not trustworthy enough to answer from. Memory that knows what it no longer
believes, and says so.

How that compares to the usual choices (feature rows; the only measured column is Mem0, from the
paired head-to-head in
[benchmarks/REVIEW.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/REVIEW.md)):

| | RE-call | Mem0 | Zep / hosted memory | Plain pgvector / Chroma |
|---|---|---|---|---|
| LLM calls to build memory | none | one extraction call per session (measured: 272 <!--@ citation-pending: metered in benchmarks/REVIEW.md's cost table; no committed artifact retains the raw usage log --> calls for the LOCOMO corpus RE-call built at zero) | provider-dependent | none |
| Runs on your own database | yes, PostgreSQL + pgvector | self-host or SaaS | SaaS first | yes |
| Supersession and validity | declared in frontmatter, enforced per hit | no equivalent | no equivalent | none |
| Explicit abstention | calibrated threshold, refusal with a reason | no | no | no, top-k always answers |
| Trust metadata per hit | verdict, confidence, cosine, provenance, tenant | score | score | score |
| License | Apache 2.0 | Apache 2.0 | proprietary SaaS / OSS core | Apache 2.0 / MIT |

The rows for Zep and plain vector stores are feature comparisons, not measurements; nothing here
claims a benchmark against them.

The vocabulary that carries that validity, `supersedes`, `valid_from` and `valid_until` in a
document's frontmatter, is published separately as
[Validity Frontmatter](https://github.com/GiulioDER/validity-frontmatter): MIT licensed, with a
zero-dependency TypeScript implementation beside it. RE-call is its Python implementation, not its
owner. The specification is deliberately licensed more permissively than this repository, so
adopting the vocabulary carries no obligation to adopt the engine.

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
| Retrieval on a third-party personal-memory benchmark | On ATM-Bench, across 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count --> questions of personal memory QA, the benchmark's own evaluator scores this run at Recall@10 **92.8924** <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10_percent --> and QS **68.4264** <!--@ atm/atm_bench_full_20260821.json # official_score.qs_percent -->. The leaderboard submission was merged 2026-08-23, the answer model is not matched to the published baselines, and the limits are stated in [docs/ATM_BENCH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/ATM_BENCH.md). |
| Validity beats nearest-match retrieval | Declared supersession makes the current memory win over stale but similar memory. The larger trust study is in [results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md). |
| Stronger than a plain vector store | Returned hits carry verdicts, confidence, provenance, tenant scope, and validity metadata. Plain top-k retrieval returns neighbors and leaves trust to the caller. |
| Clear limits | The evidence states where RE-call works, where it does not, and when a corpus-specific measurement is required. |

The README is the product overview. For evidence behind these claims, start with
[docs/EVIDENCE.md](https://github.com/GiulioDER/RE-call/blob/master/docs/EVIDENCE.md), then use
[results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) for
the full interpretation and limits.

## Quickstart

Two commands, and the second one starts its own database:

```bash
pip install "recall-rag[fastembed]"
recall quickstart
```

The distribution is `recall-rag`; the import and the command are `recall`. The name `recall` on
PyPI belongs to an unrelated package, so `pip install recall` gets you something else entirely.
Do not install both into the same environment.

That provisions a throwaway PostgreSQL with pgvector in Docker, indexes a small corpus that ships
inside the package, and answers three questions: one it can answer, one whose nearest match is a
claim that was later retracted, and one it refuses. The middle one is the point.

Measured 2026-08-22 on one Windows machine with the pgvector image already pulled:
about **50 seconds** <!--@ citation-pending: one machine, one hand-timed run, not backed by a committed results artifact -->
cold, and about **22 seconds** <!--@ citation-pending: one machine, one hand-timed run, not backed by a committed results artifact -->
on a re-run that reuses the container. A machine without the image also pays for that pull, which
is the largest and most variable part and is not included here. Re-measure with
`time recall quickstart`.

Nothing is calibrated and nothing is registered with an agent. It prints the next command for each.

```bash
recall quickstart --remove          # stops the database and destroys its volume
```

Already running PostgreSQL with pgvector? `recall quickstart --existing-dsn <dsn>` skips Docker
entirely.

### The full install

The quickstart is a demonstration, not an install: it answers questions about a sample corpus with
an uncertified threshold, and it leaves your own notes untouched. What follows is the different and
longer thing, which points RE-call at your memory, fits a threshold to it, and registers the MCP
server with your agent.

RE-call keeps memory in your own PostgreSQL with pgvector, so a database comes first.

**Already running PostgreSQL with pgvector?** Skip ahead and point the DSN at it.

**Want a throwaway one?** Save this as `docker-compose.yml`, then start it:

```yaml
services:
  db:
    image: pgvector/pgvector:pg18
    environment:
      POSTGRES_USER: recall
      POSTGRES_PASSWORD: recall
      POSTGRES_DB: recall
    volumes:
      - recall_pgdata:/var/lib/postgresql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U recall"]
      interval: 2s
      timeout: 3s
      retries: 30

volumes:
  recall_pgdata:
```

```bash
docker compose up -d --wait
```

Then install, create the schema, and run the guided setup wizard. The wizard records the selected
embedder, retrieval options, and an optional calibration that is fitted to your labeled queries and
your corpus.

```bash
pip install "recall-rag[fastembed]"
python -m recall.cli --migration-dsn postgresql://recall:recall@localhost:5432/recall schema --dim 384 apply
python -m recall.cli setup
```

Those three run unchanged in PowerShell.

The schema command targets the default `chunks` table deliberately. Global migrations have to be
applied there before any other table, so starting with `--table something_else` on a fresh database
stops with `SchemaTooOld`. To add a separate index later, apply the default target first, then pass
`--table`.

When the wizard asks whether to calibrate, it wants a labeled query file and the corpus those
queries refer to. You do not have to build either to try it: both ship inside the installed
package, next to each other.

```bash
python -c "import recall.eval, pathlib; print(pathlib.Path(recall.eval.__file__).parent)"
```

That prints a directory holding `queries.json`, a labeled set covering both answerable and
unanswerable questions, and `corpus/`, the documents those questions are labeled against. Give the
wizard those two paths and calibration runs end to end. Sources:
[recall/eval/queries.json](https://github.com/GiulioDER/RE-call/blob/master/recall/eval/queries.json)
and [recall/eval/corpus/](https://github.com/GiulioDER/RE-call/tree/master/recall/eval/corpus).

A calibration fitted that way belongs to that sample, not to your data: it shows the mechanism
working and gives you a labeled file to copy the shape of. What makes a calibration valid, when a
changed corpus needs a new one (`recall calibration drift` measures that), and what a labeled
file must contain are covered in
[docs/FIRST_CALIBRATION.md](docs/FIRST_CALIBRATION.md) and
[docs/CALIBRATION.md](docs/CALIBRATION.md).

Working from a clone:

```bash
pip install -e ".[fastembed]"
```

## How it works

The solid path is the default retrieval flow. The dashed branch is the optional Evidence Graph
path, enabled only with `graph_expansion=one_hop`; it can add evidence, but it cannot bypass the
normal trust checks.

```mermaid
flowchart TB
    subgraph BUILD["Build the memory index"]
        direction LR
        M["Memo<br/>markdown + frontmatter"] --> I["Chunk + embed"] --> DB[("PostgreSQL<br/>+ pgvector")]
    end

    subgraph QUERY["Every query · default path"]
        direction LR
        Q["Question"] --> H["Hybrid retrieval<br/>dense + full-text<br/>optional sparse / rerank"]
        H --> GP{"Calibrated<br/>gap check"}
        GP --> TR{"Trust layer<br/>validity + confidence"}
        TR --> E["Trusted evidence<br/>with provenance"]
    end

    subgraph REASONING["Reasoning and citations"]
        direction LR
        E --> RP["Reasoning policy<br/>+ budget"]
        RP --> RV{"Citation + trust<br/>validation"}
        RV --> ROUT["Cited answer, review,<br/>clarification, or ABSTAIN"]
        RG["Generation-bound reasoning graph<br/>authored edges + review proposals"] --> RP
    end

    subgraph GRAPH["OPTIONAL · Evidence Graph V1 · off by default"]
        direction LR
        SG["Deterministic semantic graph<br/>entities + mentions + relations"] --> EC["Bounded one-hop<br/>neighbor evidence"]
        EC --> GT["Run normal trust<br/>evaluation again"]
    end

    DB -. "generation-bound" .-> H
    DB -. "generation-bound" .-> RG
    CAL["Corpus calibration"] --> TR
    TR -. "opt in:<br/>graph_expansion=one_hop" .-> SG
    GT -. "accepted evidence" .-> RP

    classDef defaultPath fill:#e8f3ff,stroke:#2b6cb0,color:#102a43,stroke-width:1px;
    classDef optionalPath fill:#fff8e1,stroke:#b7791f,color:#5f370e,stroke-width:1px;
    classDef trustPath fill:#e8f5e9,stroke:#2f855a,color:#163b27,stroke-width:1px;
    class Q,H,GP,E,RP,RV,ROUT defaultPath;
    class SG,EC,GT,RG optionalPath;
    class TR,CAL trustPath;
```

## Product surface

| Area | Ships today |
|---|---|
| Retrieval | Dense, sparse, hybrid RRF, optional SPLADE, optional cross-encoder reranking, calibrated confidence, provenance, and trust verdicts. |
| Configuration | Guided setup, local and hosted embedder choices, retrieval cost profiles, optional reranking, strict or development trust policy, and per-corpus calibration. |
| Storage | PostgreSQL with pgvector, ordered SQL migration path, immutable generations, incremental indexing, pruning, and source-scoped erasure. |
| Agent integration | CLI, MCP server, LangChain retriever, LlamaIndex retriever, and injectable search seams for tests. |
| Reasoning | Explicit opt-in reasoning API, CLI, and MCP tools over trusted retrieval, generation-bound authored and semantic Evidence Graph V1 projections <!--@ citation-pending: Evidence Graph V1 implementation artifact -->, proposal inspection, budgets, and citation validation. |
| Security | Tenant isolation, row-level security checks, serving and migration DSNs, bearer-token HTTP transports, scopes, quotas, and unsafe-DSN refusal. |
| Operations | Timeouts, reconnect policy, structured logging, counters, latency percentiles, and MCP stats. |
| Quality gates | Real pgvector integration tests, type checking, linting, dependency audit, claim-artifact checks, and regression fixtures for known failure modes. |

Deliberately out of scope: an end-user dashboard, entity synthesis, high availability orchestration,
automatic truth extraction from prose, and corpus rewrites from inference proposals. Reasoning is
opt in, citation constrained, and review aware.

The ordered SQL migration path is versioned now, pre-tenancy tables are migrated in place, and runtime
`CREATE TABLE IF NOT EXISTS` remains bootstrap only.

## When not to use RE-call

Use something else if you need managed hosting, per-chunk ACLs, automatic truth extraction from
prose, or a memory system that rewrites facts for you. RE-call is a retrieval library over your
PostgreSQL database, not a hosted memory platform.

## What this does not do

RE-call is a retrieval library with an opt-in reasoning layer, not a general reasoning system. It
does not infer every missing supersession edge, prove that an on-topic memory answers a near-miss
question, promote proposals into corpus truth, or replace database operations with a managed
service. It returns the trust signals the caller needs, and it refuses to pretend that a nearest
match is always usable evidence.

## Use it

For an ad hoc local markdown folder, create a table for that index, index the corpus, and search it.
If you did not calibrate during setup, use development mode only for local evaluation.
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

**On Claude Code, the plugin does all of this for you**, including the hooks and a skill that
teaches Claude when to search:

```
/plugin marketplace add GiulioDER/RE-call
/plugin install recall@re-call
```

It asks for a DSN, a tenant and a trust mode, and keeps the DSN in your OS keychain rather than in
`settings.json`. You still need a database first, which is what `recall quickstart` above is for.
See [plugin/README.md](plugin/README.md).

For every other MCP client, the manual wiring (schema, server block, trust mode) is in
[docs/USING_WITH_CLAUDE.md](https://github.com/GiulioDER/RE-call/blob/master/docs/USING_WITH_CLAUDE.md).
Core tools include `recall_search`, `recall_evidence`, `recall_index`, `recall_forget` and
`recall_stats`; the authoritative list of all tools is
[docs/API.md](https://github.com/GiulioDER/RE-call/blob/master/docs/API.md).
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
| [docs/REASONING_GRAPH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/REASONING_GRAPH.md) | Authored reasoning projection and deterministic Evidence Graph V1 semantics <!--@ citation-pending: Evidence Graph V1 implementation artifact -->. |
| [docs/REASONING_OPERATIONS.md](https://github.com/GiulioDER/RE-call/blob/master/docs/REASONING_OPERATIONS.md) | Opt-in reasoning tools, graph expansion, traces, review policy, and operational behavior. |
| [docs/AUTH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/AUTH.md) | Authentication, scopes, and tenant isolation. |
| [docs/MIGRATIONS.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MIGRATIONS.md) | Migration roles, serving DSNs, and schema operations. |
| [docs/OPERATING_MODES.md](https://github.com/GiulioDER/RE-call/blob/master/docs/OPERATING_MODES.md) | Local, production, quality, hosted, and evaluation deployment modes. |
| [docs/FIRST_CALIBRATION.md](https://github.com/GiulioDER/RE-call/blob/master/docs/FIRST_CALIBRATION.md) | Walkthrough from an indexed folder to a trusted, certified corpus, with the traps named where you hit them. |
| [docs/CALIBRATION.md](https://github.com/GiulioDER/RE-call/blob/master/docs/CALIBRATION.md) | Calibration workflow and generation-aware serving. |
| [docs/CASE_STUDY.md](https://github.com/GiulioDER/RE-call/blob/master/docs/CASE_STUDY.md) | Where the system came from and what is public versus private. |
| [docs/RESEARCH_PROTOCOL.md](https://github.com/GiulioDER/RE-call/blob/master/docs/RESEARCH_PROTOCOL.md) | How benchmark runs are controlled and audited. |
| [benchmarks/PREREGISTRATION-evidence-graph-v1.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/PREREGISTRATION-evidence-graph-v1.md) | Preregistered Evidence Graph V1 quality evaluation and relation controls <!--@ citation-pending: preregistration artifact -->. |

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
| What does ATM-Bench add? | A third-party personal-memory QA benchmark over eleven thousand email, image and video items, scored by its own evaluator, where half the questions are graded deterministically rather than by a judge. RE-call's retrieval leads the published board by a wide margin; the answer score is not answer-model-matched and the submission has not been accepted yet. Both limits are stated in [docs/ATM_BENCH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/ATM_BENCH.md). |

Important benchmark documents:

| Document | Purpose |
|---|---|
| [results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) | Interpretation, limits, and negative results. |
| [results/RESULTS.md](https://github.com/GiulioDER/RE-call/blob/master/results/RESULTS.md) | Complete result tables. |
| [results/ARTIFACTS.md](https://github.com/GiulioDER/RE-call/blob/master/results/ARTIFACTS.md) | Checksum and artifact map for readers auditing a claim. |
| [docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md) | MTRAG setup, results, and scope boundaries. |
| [docs/ATM_BENCH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/ATM_BENCH.md) | ATM-Bench official results, comparability boundaries, and where the remaining loss is. |
| [benchmarks/REVIEW.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/REVIEW.md) | Adversarial review of the LOCOMO comparison. |
| [benchmarks/PREREGISTRATION.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/PREREGISTRATION.md) | Pre-registered rules for the main memory benchmark. |
| [benchmarks/archive/preregistrations/README.md](https://github.com/GiulioDER/RE-call/blob/master/benchmarks/archive/preregistrations/README.md) | Archived preregistrations for follow-up benchmark arms. |

## Reproduce

From a git clone (the eval harness is repo-only; it is not shipped in the `recall-rag` wheel):

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

<p align="center">
  <a href="https://glama.ai/mcp/servers/GiulioDER/RE-call"><img src="https://glama.ai/mcp/servers/GiulioDER/RE-call/badges/card.svg" alt="RE-call MCP server"></a>
</p>
