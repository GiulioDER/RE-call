<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call — Retrieval-Augmented Self-Recall" width="900">
</p>

<p align="center">
  <b>Trustworthy retrieval for an AI agent's own memory.</b><br>
  Every hit comes back with confidence, provenance, and validity — or the honest answer is <i>"I don't know."</i>
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/recall-rag/"><img src="https://img.shields.io/pypi/v/recall-rag.svg?color=blue" alt="PyPI"></a>
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PostgreSQL-16%2F17%20%C2%B7%20pgvector-336791" alt="PostgreSQL + pgvector">
  <img src="https://img.shields.io/badge/tests-1300%2B%20·%20real%20pgvector-brightgreen" alt="1300+ tests">
</p>

<p align="center">
  <a href="#why-re-call">Why RE-call</a>
  &nbsp;·&nbsp;
  <a href="#what-re-call-can-do-for-you">What it can do for you</a>
  &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a>
  &nbsp;·&nbsp;
  <a href="#quickstart--2-minutes-no-api-key">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/GiulioDER/RE-call/blob/master/docs/EVIDENCE.md">Evidence</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/GiulioDER/RE-call/blob/master/docs/PRODUCTION.md">Production &amp; limits</a>
</p>

<p align="center">
  <b>📐 <a href="https://github.com/GiulioDER/RE-call/blob/master/docs/pipeline.png">THE FULL PIPELINE, CORPUS TO ANSWER</a></b><br>
  <i>every phase in one diagram, with what each option costs</i>
</p>

<details>
<summary>show it inline</summary>
<br>
<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/pipeline.png" alt="The RE-call pipeline, top to bottom: lint, chunking, contextualisation, embedding, the optional SPLADE sidecar, the Postgres store, then query embedding, three retrieval legs, RRF fusion, optional rerank, gap warning, the trust layer, an optional entailment judge, abstention, evidence construction and the MCP surface. Colour encodes what an option costs: shipped default, best measured, free and local, cloud egress, opt-in, or rejected.">
</p>
</details>

---

## Why RE-call

Give your AI agent, app, or team a long-term memory that is **free to run**, **stays on your
machines**, and **tells the truth about what it knows**.

Measured against **[Mem0](https://github.com/mem0ai/mem0)** on the public **LOCOMO** benchmark,
with an *identical* generator, judge, and paired questions (full table, losses and caveats
included in [FINDINGS §9d](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md)):
RE-call comes out **more accurate**, **$0 to build at any scale**, and it keeps your data on
infrastructure you own. I publish the configuration where it loses, because a benchmark you
cannot lose isn't one.

<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/mem0_comparison.svg" alt="RE-call vs Mem0 on the LOCOMO benchmark. Judged answer accuracy: RE-call 0.420, Mem0 0.366, n=1,540, paired p=0.0002 to 0.0065, Holm-corrected. Cost to build the benchmark memory: RE-call $0.00, Mem0 $7.29." width="900">
</p>

On **[MTRAG](#judged-by-someone-else-mtrag)**, IBM's multi-turn RAG benchmark, RE-call **refuses
2.3× more unanswerable questions than `gpt-4o`** while placing 4th of ten on answer quality: the
only system in that top four that is also top three on abstention.

Its real strength, though, is that it is built to be **tuned, not fixed**. Every stage of the
pipeline (embedder, reranker, the SPLADE sidecar, the entailment judge) is opt-in and swappable,
so the same system runs comfortably at either end of the spectrum. On MTRAG's 777 judged queries
that span is **nDCG@5 0.2930 to 0.4342**, a 48% relative range across configurations of one engine,
and every rung is a named flag with a measured price:

- 🔒 **Fully local and air-gapped**, for maximum data protection: local embeddings, your own
  PostgreSQL, zero external calls, zero per-query egress.
- ☁️ **With a cloud embedder and reranker**, when a jargon-heavy corpus makes the extra accuracy
  worth the cost.
- ⚙️ **From a laptop to production hardware**, without re-architecting: the free default path is
  fast out of the box, and every heavier option is named and measured, never silently switched on.

Every setting keeps the same guarantee: superseded or expired memories are demoted, not served, so
the agent says "I don't know" instead of guessing. Whatever the constraint (data residency,
latency, GPU availability, or plain cost), RE-call's pipeline reconfigures around it instead of
forcing one shape on every deployment.
→ [see it in one screen](https://github.com/GiulioDER/RE-call/blob/master/docs/EVIDENCE.md#see-it-in-one-screen)

## What RE-call can do for you

- **Stop an agent from re-litigating settled decisions or contradicting its own memory.**
  Supersession and validity are enforced at retrieval, abstention is a first-class return value,
  and drop-in LangChain / LlamaIndex retrievers plus an MCP server for Claude fit the stack you
  already run.
- **Remove the per-write API bill.** Embed locally, store in the Postgres you already run, and
  scaling up never creates a new invoice.
- **Keep user data on infrastructure you control.** Multi-tenant with database-enforced row-level
  security, token auth, `recall forget` for right-to-erasure, MIT license.
- **Keep closed experiments closed.** Built inside a production trading-research agent so a stale
  conclusion never outranks its own correction.

**Try it in 2 minutes, no API key** → [Quickstart](#quickstart--2-minutes-no-api-key).

## Judged by someone else: MTRAG

**[MTRAG](https://github.com/IBM/mt-rag-benchmark)** (IBM, TACL 2025): 842 human-written
multi-turn tasks, nine published systems, and an official `gpt-4o-mini-2024-07-18` judge that pays
a **full 1.0 on every metric** for correctly saying *"I do not have that information"*. Almost no
other benchmark scores a refusal as anything but a miss.

### RE-call keeps the promise it makes

Correct refusals on the 55 unanswerable tasks, same judge, same tasks, every system:

| # | system | refuses what it cannot answer |
|---|---|---|
| 1 | llama-3.1-8b | 32.7% |
| **2** | **RE-call** | **29.1%** |
| 3 | llama-3.1-70b | 29.1% |
| 4 | gpt-4o-mini | 23.6% |
| 6 | **gpt-4o** | **12.7%** |
| 7 | llama-3.1-405b | 5.5% |
| 9 | qwen-2.5-72b | 1.8% |

**RE-call refuses 2.3× more often than `gpt-4o`, and 16× more often than `qwen-2.5-72b`.**

### From the top of the table, not the bottom

Abstention usually costs answer quality. Here is what it cost:

| # | system | answer quality |
|---|---|---|
| 1 | llama-3.1-405b | 0.5691 |
| 2 | qwen-2.5-72b | 0.5625 |
| 3 | gpt-4o | 0.5591 |
| **4** | **RE-call** | **0.5527** |
| 5 | c4ai-command-r-plus | 0.5502 |

**RE-call is the only system in that top four that is also top three on abstention.** The three
above it refuse 5.5%, 1.8% and 12.7%. The gap to `gpt-4o` is **0.0064**.

And RE-call's retrieval beat the benchmark's own: **0.5527 against 0.5516** on identical
generator, prompt and judge, with only the contexts swapped.

### One engine, whatever you can afford

| configuration | nDCG@5 | cost |
|---|---|---|
| **SPLADE learned sparse** *(the free default)* | **0.3573** | local, $0 |
| **+ Voyage rerank** *(one flag)* | **0.4342** | paid API |

A 48% relative span between two flags, measured on 777 judged queries. The reranker is
**+0.0769 nDCG@5** and **worse on 162 of the 777**, which is why it is off by default.

→ Every number, the six runs behind them, and the scoring bug I reported upstream:
**[docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md)**.

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

**The solid path is what runs if you change nothing.** Writing a memory is a local embedding, no
LLM call, so it stays free at any scale. On the read path, dense and sparse search feed **RRF
fusion**, the **gap check** refuses to dress up nearest-noise as an answer, and the **trust layer**
enforces supersession, validity, and a confidence threshold fitted per embedder and corpus, never a
shipped constant, before anything reaches the agent.

**Everything dashed and amber is opt-in and off by default:** the SPLADE leg, the cross-encoder
reranker, the entailment judge. Each costs something measurable, so each is switched on by name,
never inferred for you. Full derivation of the threshold and every measured trade-off:
[FINDINGS](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md).

→ Every phase in full, every embedder measured so far, and what each option costs:
**[docs/pipeline.png](https://github.com/GiulioDER/RE-call/blob/master/docs/pipeline.png)**.

## Quickstart · 2 minutes, no API key

```bash
docker compose up -d --wait          # PostgreSQL + pgvector
pip install "recall-rag[fastembed]"  # local embeddings, no API key
python -m recall.cli --migration-dsn postgresql://recall:recall@localhost:5432/recall \
  schema --dim 384 apply             # explicit, versioned DDL step
python -m recall.cli demo            # index corpus/ and run the sample queries
```

> **The distribution is `recall-rag`; the import is `recall`.** `pip install recall` gets you an
> unrelated RPC framework last released in 2014 — that name was taken and is not reclaimable, and
> `re-call` is rejected by PyPI as too similar to it. Both `recall` and this package provide a
> top-level `recall` module, so do not install `recall` and `recall-rag` into one environment.
>
> Working from a clone instead? `pip install -e ".[fastembed]"`.

## Use it

```bash
python -m recall.cli index ./notes                       # index a folder of markdown
python -m recall.cli search "what did we decide about caching?"
python -m recall.cli lint ./notes                        # supersession-graph health (no DB)
python -m recall.cli lint ./notes --fix                   # propose missing edges (dry run)
python -m recall.cli check ./notes/new-memo.md --strict    # write-time gate, for a pre-commit hook
```

For the generation path, after building, validating, calibrating, and promoting it as described
below, query the tenant's active immutable generation:

```python
from recall.embeddings import FastEmbedEmbedder
from recall.generation_store import GenerationStore
from recall.trust import trusted_search

emb = FastEmbedEmbedder()
with GenerationStore(DSN, dim=emb.dim, tenant="acme", pool_size=8) as store:
    store.check_schema()  # SELECT-only compatibility check; provisioning applied migrations
    result = trusted_search(store, emb, "what is the rate limit?")
    if result.abstained:
        ...  # say you don't know — do not answer from these hits
    for hit in result.hits:
        hit.verdict      # ok | superseded | expired | not_yet_valid | low_confidence | …
        hit.confidence   # calibrated; 0.5 sits exactly on the abstention boundary
        hit.validity.superseded_by
```

Set `RECALL_SERVING_DSN` for application traffic and `RECALL_MIGRATION_DSN` only in the migration
job. See [database migrations and roles](docs/MIGRATIONS.md). `RECALL_DSN` remains a deprecated
development fallback for the serving DSN.

### Multi-query fusion (`search_fused`)

Fuses the current turn with prior turns before reranking once, for a small, reranker-gated accuracy
gain on multi-turn conversations. It requires a reranker and refuses without one, and it is opt-in
by data: no `history`, no fusion, every existing `search()` call is unaffected.

```python
result = retriever.search_fused("and what about the deadline?", history=["what is the policy?"])
```

<details>
<summary>Measured gains, costs, and refusal conditions</summary>

Measured on MTRAG-human dev at `candidate_k=100`, with a reranker: **+0.0084 nDCG@5**<!--@ citation-pending: measured in `/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/`, an archive outside this repo --> (Holm-significant, cross-encoder/`ms-marco-MiniLM-L-6-v2`) and
**+0.0842 R@100**<!--@ citation-pending: measured in `/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/`, an archive outside this repo --> over single-query `search`. Gains proved significant and directional under BAAI/`bge-reranker-v2-m3` (+0.0117 nDCG@5)<!--@ citation-pending: measured in `/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/`, an archive outside this repo -->. The effect is small in absolute terms, and it was measured on one dev split, so treat it as
directional evidence rather than a guarantee on your corpus.

⚠️ **Requires a reranker, and refuses without one.** Raw, this arm scores **0.0447 nDCG@5 below**<!--@ citation-pending: measured in `/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/`, an archive outside this repo -->
`search()`; the cross-encoder is what repairs the ranking damage a concatenated query does. RE-call
ships with the reranker off by default (see above), which is exactly why `search_fused` refuses
rather than merely warns when none is configured.

It also costs roughly **twice the retrieval** of `search()`, plus mandatory reranking (~1,050<!--@ citation-pending: measured in `/var/lib/recall-benchmarks/2026-08-07-mtrag-rerank-conversion/`, an archive outside this repo -->
ms/query on CPU). Whether that trade is worth it is an operator decision.

Histories whose concatenation exceeds 4,096<!--@ citation-pending: source constant, not a measurement: `FUSED_HISTORY_MAX_CHARS` in recall/retriever.py --> characters are **refused, not truncated**: a truncated
history is a configuration that was never measured. A result can also carry fewer than `k` hits,
when a chunk is deleted between retrieval and the final rescore.

`search_fused` is library only for now; it is not exposed as an MCP tool. Adding `history` to a
public tool surface needs its own auth, limits, and query length contract.

</details>

### Immutable index generations

Builds a replacement index generation without mutating the active one, then promotes or rolls back
by atomically switching tenant pointers, so a re-index never risks the serving index. Full manifest
commands, lifecycle rules, and retention: [docs/GENERATIONS.md](docs/GENERATIONS.md).

<details>
<summary>Fingerprinting, calibration status, and operational notes</summary>

The generation index path fingerprints the embedder revision, chunker configuration, FTS configuration,
and immutable object manifest. Equal vector dimensions do not permit reuse across models or
revisions. Legacy rows are registered as `legacy_unverified` and never become an active strict
generation index.

Production object access requires a deployment-owned bucket and prefix allowlist; requests cannot
supply credentials or an endpoint. Every object version, length, and cryptographic digest is
verified before embedding.

Tenant and generation bound calibration now ships. Strict production enforcement is the next
implementation session, so generation promotion remains blocked in production and requires an
explicit unsafe flag in development. A published artifact does not yet make this release's
production path fail closed when calibration is absent.

> **Two operational notes.** The test suite **DROPs tables**, so it reads a separate
> `RECALL_TEST_DSN` and never the serving DSN, exporting your real DSN and running `pytest` cannot
> touch it. And the MCP server **refuses to start** if `RECALL_SERVING_DSN` (or its deprecated
> `RECALL_DSN` fallback) carries the built-in
> `recall:recall` credentials against a non-local host; set a real password, or
> `RECALL_ALLOW_INSECURE_DSN=1` to accept the risk deliberately.

> **Multi-tenancy.** Set `RECALL_TENANT` or `PgVectorStore(tenant=...)`. RLS enforces the same
> boundary in the database, so a forgotten `WHERE` returns nothing rather than another tenant's
> memories. ⚠️ **RLS is bypassed by a superuser or a `BYPASSRLS` role**, including the one in this
> repo's `docker-compose.yml`. Connect as an unprivileged role, or that second layer is decoration;
> `store.check_rls_effective()` tells you which you have, and the server warns at startup.

</details>

## Use it with Claude (MCP)

```json
{ "mcpServers": { "recall": {
    "command": "python", "args": ["-m", "recall_mcp.server"],
    "env": { "RECALL_SERVING_DSN": "postgresql://...", "RECALL_TENANT": "acme" } } } }
```

Five tools: `recall_search` (verdict + confidence + provenance, or an explicit abstention),
`recall_evidence` (the same retrieval as a citable evidence bundle plus the prompt to answer it
with — see below), `recall_index`, `recall_forget` (permanently delete a source's chunks —
irreversible, tenant-scoped), `recall_stats` (size, freshness, and the process metrics). Full
guide →
[docs/USING_WITH_CLAUDE.md](https://github.com/GiulioDER/RE-call/blob/master/docs/USING_WITH_CLAUDE.md).

## Use it with LangChain or LlamaIndex

```bash
pip install "recall-rag[langchain]"     # or: "recall-rag[llamaindex]"
```

```python
from recall.integrations.langchain import RecallRetriever   # or .llamaindex

retriever = RecallRetriever.from_store(store, emb, k=5)
docs = retriever.invoke("what is the rate limit?")          # LlamaIndex: .retrieve(...)
```

Both adapters are **strict by default**, like everything else, and raise `TrustRefusal` when the
gate cannot certify an answer. Pass `policy=TrustPolicy.development()` to `from_store` for local
work against an uncalibrated corpus.

Both also expose the **evidence boundary**, for when you are about to *answer* from memory rather
than just retrieve from it. `retriever.evidence(query)` returns only the passages the trust layer
cleared, in retrieval order, and `retriever.evidence_prompt(query)` renders them into a fixed
library-authored system instruction plus a delimited, JSON-escaped data message — so corpus text
reaches your model as data and never as an instruction. `recall.validate_answer` then checks the
returned envelope structurally: at least one citation, every citation resolving to a supplied
chunk id. It does **not** check that a cited passage supports the answer, and says so.

Every document and node carries the trust and lineage identity in `metadata` —
`recall_trust_state`, `recall_failure_code`, `recall_calibrated`, plus the tenant, generation,
pipeline, corpus and query-set identifiers — because a framework will happily hand a single
document to a chain that never sees the result object:

```python
doc = docs[0]
doc.metadata["recall_trust_state"]   # "trusted" | "degraded"
doc.metadata["recall_calibrated"]    # False unless a certified artifact backed this
```

The signal is also **in band**, in `page_content` itself, not only in metadata. LangChain's stock
`stuff_documents_chain` and LlamaIndex's default node handling render the text alone into the
prompt, so a warning that lives only in metadata never reaches the model. A degraded hit's text is
prefixed with a warning saying the trust layer did not run — which is a different sentence from
the one a *judged* hit gets, because "was not checked" and "was checked and failed" are different
facts and the prompt is the only place the model will ever see either.

Drop-in retrievers for both frameworks, so RE-call can be the `retriever=` behind any chain, agent
or query engine. They differ from an ordinary vector retriever in exactly one way, and it is the
whole point:

> **When the trust layer abstains, they return nothing** — an empty `list[Document]` /
> `list[NodeWithScore]`, not a best-effort neighbour.

A plain similarity retriever always hands back its top-k, so a chain cites the closest vector even
when that memory is stale or superseded — and the stale hit is often the *highest*-cosine one. Here
the chain gets nothing instead of a confident wrong memory. Every returned document carries the
trust signal in `metadata` (`recall_verdict`, `recall_confidence`, `recall_cosine`,
`superseded_by`), so a downstream prompt or reranker can use it; pass
`return_abstention_reason=True` if you would rather receive one empty document carrying
`recall_reason` than an empty list.

`from_store()` takes the same knobs as `trusted_search` — `k`, `source`, `calibration`, `reranker`,
`entailment`. Both adapters accept an injectable search function, so they are unit-tested without a
database, and both ship in `dev` as well as their own extra — the `test` and `typecheck` jobs
install `.[dev]` only, so otherwise they would be shipped but never CI-tested or type-checked.

## Read next

| | |
|---|---|
| [docs/EVIDENCE.md](https://github.com/GiulioDER/RE-call/blob/master/docs/EVIDENCE.md) | The problem precisely, the demo screen, every verified claim with its limit, the withdrawn ones, and retrieval quality per corpus |
| [docs/PRODUCTION.md](https://github.com/GiulioDER/RE-call/blob/master/docs/PRODUCTION.md) | Production posture row by row, what this deliberately does not do, and the upgrade notes that can break a deployment |
| [docs/PRIOR_ART.md](https://github.com/GiulioDER/RE-call/blob/master/docs/PRIOR_ART.md) | Graphiti, Mem0, Letta, LangMem, and where this genuinely differs |
| [docs/ENGINEERING.md](https://github.com/GiulioDER/RE-call/blob/master/docs/ENGINEERING.md) | The test suite, the type gate, and the defects a real corpus found |
| [docs/MTRAG_BENCHMARK.md](https://github.com/GiulioDER/RE-call/blob/master/docs/MTRAG_BENCHMARK.md) | Multi-turn RAG scored by MTRAG's own judge: the tuning curve, where it places against the baselines, and the scoring bug I reported upstream |
| [results/RESULTS.md](https://github.com/GiulioDER/RE-call/blob/master/results/RESULTS.md) · [results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) | Every number with its command, and what each one means |

## License

MIT — see [LICENSE](https://github.com/GiulioDER/RE-call/blob/master/LICENSE).
