<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/banner.png" alt="RE-call — Retrieval-Augmented Self-Recall" width="900">
</p>

<p align="center">
  <b>Trustworthy retrieval for an AI agent's own memory.</b><br>
  Every hit comes back with confidence, provenance, and validity — or the honest answer is <i>"I don't know."</i>
</p>

<p align="center">
  <a href="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml"><img src="https://github.com/GiulioDER/RE-call/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/GiulioDER/RE-call/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PostgreSQL-16%2F17%20%C2%B7%20pgvector-336791" alt="PostgreSQL + pgvector">
  <img src="https://img.shields.io/badge/tests-584%20·%20real%20pgvector-brightgreen" alt="584 tests">
</p>

<p align="center">
  <a href="#see-it-in-one-screen">See it</a>
  &nbsp;·&nbsp;
  <a href="#what-is-actually-verified">What's verified</a>
  &nbsp;·&nbsp;
  <a href="#production-posture">Production posture</a>
  &nbsp;·&nbsp;
  <a href="#quickstart--2-minutes-no-api-key">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="#what-this-does-not-do">Limits</a>
</p>

---

**Most RAG hands back the closest vector match. That's the wrong answer more often than you'd think.**

A long-running agent piles up memory — decisions, closed experiments, incident notes — and then it
**re-litigates settled decisions**, **hallucinates over gaps** the memory can't fill, and **builds on
facts that are no longer true**. The catch: when you've reversed a decision, the *stale* memory of it
is often the **highest-cosine hit in the whole result**. Similarity search serves it, confidently.

RE-call is a retrieval engine for that memory that is *honest about what it doesn't know*. It returns
**verdict + confidence + provenance** with every hit — not just similarity — demotes memories that
were superseded or expired, and prefers an explicit **abstention** over confident noise.

## See it in one screen

<p align="center">
  <img src="https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/superseded-catch.png" width="740" alt="recall demo: the stale rate-limit memory has the highest cosine (0.806) but is flagged superseded and demoted below the current memory; an unanswerable query returns an explicit ABSTAIN.">
</p>

<details>
<summary>same run, as text</summary>

```text
$ python -m recall.cli demo

[ok] query='how many requests per second can a client make?'
  ok           conf=1.00  cos=0.784  rate_limits_v2.md                       '# API rate limits (revised)'
  superseded   conf=1.00  cos=0.806  rate_limits_v1.md → use rate_limits_v2  '# API rate limits … limited to 100'

[ABSTAIN · gap] query='how do we handle penguins on mars?'
  reason: no hit above the calibrated confidence threshold (probable corpus gap)
```
</details>

Look at the cosines. The **stale** memory scores **higher (0.806)** than the current one — plain vector
search returns it, and the agent builds on a limit that no longer exists. RE-call flags it
`superseded`, points at its successor, and puts the *current* memory on top. When the memory genuinely
has no answer, it says so. **That ordering decision is the whole thesis.**

## What is actually verified

Every headline number below was measured, and every one carries its limit. Where a claim could not be
supported, it was withdrawn rather than softened — the withdrawals are listed too, because a claims
table without them is marketing.

| Claim | Measurement | Limit |
|---|---|---|
| **Supersession beats similarity** | Superseded-trust rate **0.00**, 95% Wilson **[0.00, 0.02]**, n=250, against a baseline of **1.00** — plain search returns the stale memory *every time* on adversarially-worded queries | Generated corpus; the successor/abstain columns on it are **not** meaningful (below) |
| **Abstention is calibrated, not guessed** | On the real corpus: threshold **0.728 ± 0.042** over 4 index rebuilds, false-abstain **0.015**, gap false-confidence **0.000** | Needs ≥ ~20 labelled samples; below that the rule loses its outlier robustness |
| **Timestamps cannot replace declared supersession** | "Trust the newest relevant hit", steelmanned, still trusts the stale memory **83–100%** of the time | — |
| **Reranking rescues a weak embedder** | Hybrid + cross-encoder lifts MRR **0.63 → 1.00** offline | Situational: a strong embedder already saturates this corpus |
| **Fine-tuning pays only for a vocabulary gap** | **+0.00** on a rich corpus; **0.31 → 0.55** held-out MRR on opaque jargon → [study](https://github.com/GiulioDER/RE-call/blob/master/docs/RAG_TRAINING_STUDY.md) | Measure your gap first |
| **Near-misses need a judge, not a threshold** | QNLI stage cuts near-miss false-confidence **1.00 → 0.60**, **0.80 → 0.50**, same judge across embedders → [study](https://github.com/GiulioDER/RE-call/blob/master/docs/ENTAILMENT_SUPERSESSION_STUDY.md) | Judge-alone *degrades* far-gap detection — the two stack, neither replaces the other |

Full methodology, per-embedder tables and the negative results → **[results/FINDINGS.md](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md)**.
Design rationale and the reasoning behind each guard → **[docs/WRITEUP.md](https://github.com/GiulioDER/RE-call/blob/master/docs/WRITEUP.md)**.

### Claims that were withdrawn

A previous version of this file published each of these. They did not survive re-measurement:

- **"FCR @calibrated 0.00"** — the threshold was fitted and scored on the same samples. On separable
  data that is 0.00 by arithmetic. Now cross-validated, and the fitting rule was
  [replaced outright](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) after it proved to let **20.5%** of unanswerable queries through.
- **Coverage and abstention accuracy on generated corpora** — the "unanswerable" queries were an
  answerable query plus a nonsense suffix, so nothing could separate them. Rebuilt as genuinely
  off-topic questions; the *document*-level degeneracy remains and is stated as unmeasured.
- **"6× faster incremental re-index"** — understated. Measured on a Linux server it is **33×**.
- **Real-corpus recall@5 of 0.945** — that used document *headings* as queries, which is known-item
  retrieval. Against 110 hand-labelled questions phrased the way a person actually asks, hit@5 is
  **0.33** on that corpus. → [FINDINGS §7](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md)
- **"Retrieval is the weakest part of this system"** — the sentence this file carried after that
  measurement. A replication on a public corpus scored **0.705** with the same local embedder, so
  0.33 was a property of *that corpus*, not of this software. Corrected rather than quietly
  deleted, because the claim was published. → [FINDINGS §8](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md)
- **"ANN recall is tuned on the filtered path"** — the heading this file gave the HNSW fix, which
  reads as a recall improvement. Two measurements were taken and only the flattering one reached
  the docs: the 0.38 → ~0.90 lift comes from a fixture corpus the test *retries until it
  reproduces the pathology*, while an independent A/B on a normally-built corpus moved recall the
  other way (0.523 → 0.483). What was actually fixed is **truncation** — filtered search returning
  fewer results than requested — and the PR that shipped it said so at the time. Reworded above,
  and corrected in FINDINGS §5b and the changelog, rather than deleted.
  → [#57](https://github.com/GiulioDER/RE-call/pull/57)

## Production posture

"Enterprise-grade" is not a single property, so here is the itemised version — verified on a real
host (PostgreSQL 17, pgvector 0.8.2, Python 3.12, connecting as an **unprivileged** role), not only
on a laptop.

| Property | Status | Evidence |
|---|---|---|
| **Multi-tenancy** | ✅ `tenant_id` on every row and every query, plus a row-level-security policy (`ENABLE` + `FORCE`) | Verified as a `NOSUPERUSER NOBYPASSRLS` role — a superuser bypasses RLS, so testing it as one would have passed vacuously |
| **Concurrency** | ✅ async MCP tools + `psycopg_pool`; the server previously served exactly **one** request at a time | FastMCP awaits async tools and calls sync ones *inline* — there is no thread offload |
| **Timeouts / resilience** | ✅ `statement_timeout`, `connect_timeout`, narrow reconnect-and-retry | The retry refuses to re-run a `QueryCanceled`, which would escape the very timeout that fired |
| **Security posture** | ✅ fail-closed on published default credentials; index-root confinement that survives symlinks on 3.11/3.12 | `pathlib` only gained `recurse_symlinks` in 3.13 |
| **Observability** | ✅ `logging` (text/JSON), counters and latency percentiles for abstention, verdicts, reconnects; surfaced through the MCP `recall_stats` tool | The library never attaches handlers — that is the host's job |
| **Incremental indexing** | ✅ content-hash skip, bounded-memory batched writes, prunes files deleted from disk | 5,100 chunks / 1,120 files: full **7.4 s**, unchanged re-index **0.22 s** |
| **Scale characteristics** | ✅ measured at **50,600 chunks**: recall@5 1.00 filtered and unfiltered, search p50/p95/p99 | Templated text; absolute retrieval quality is optimistic |
| **Real-corpus operation** | ✅ 794 hand-written memos → 6,491 chunks, p50 **78 ms** | Works at this size; see the retrieval row for how well |
| **Retrieval quality, real questions** | ✅ **hit@5 0.705** [0.56, 0.82] on a public 746-doc corpus with the free local embedder · ⚠️ **0.348** on an idiosyncratic private one — see [the tables below](#retrieval-on-a-real-corpus-what-actually-moved-it) | Measured on 110 hand-labelled questions per corpus, not on headings. Corpus vocabulary dominates: a cloud embedder is worth +0.28 on the hard corpus and +0.02 on the ordinary one |
| **Data erasure** | ✅ `recall forget` / `recall_forget` permanently delete a source's chunks; previews by default, `--yes` to act | The right-to-erasure path — irreversible, so it refuses to act unattended without the flag |
| **Abuse bounds** | ✅ `recall_index` refuses before embedding anything if a request exceeds `RECALL_INDEX_MAX_FILES` / `RECALL_INDEX_MAX_BYTES` | A client-callable indexer with no cap is an unbounded spend on a cloud embedder |
| **Authentication** | ✅ bearer tokens on the HTTP transports, three scopes, one tenant per principal — see [docs/AUTH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/AUTH.md) | Starting an HTTP transport without tokens **refuses to boot** rather than warning. stdio stays unauthenticated by design: it is a private pipe, not a listener |
| **Schema migrations** | ❌ runtime `CREATE TABLE IF NOT EXISTS`, no versioned upgrade path | Pre-tenancy tables *are* migrated in place, with a test |
| **HA / replication** | ❌ out of scope — this is a library over your Postgres | — |

## Retrieval on a real corpus: what actually moved it

110 hand-labelled questions against 794 real memos (6,491 chunks), phrased the way a person asks
rather than as document headings. Four hypotheses, tested one at a time on the **same** 46 held-out
questions — three eliminated, one confirmed:

| change | hit@5 | Δ | cost |
|---|---|---|---|
| baseline — bge-small, hybrid dense+sparse | 0.348 [0.23, 0.49] | — | 45 ms |
| + cross-encoder rerank | 0.391 [0.26, 0.54] | +0.043 *(within noise)* | **57× latency** |
| candidate pool 20 → 100 | 0.348 | **+0.000** | — |
| chunk size 400 / 800 / 1600 | 0.326 / 0.348 / 0.348 | **+0.000** | a re-index each |
| **embedder → voyage-3** | **0.630 [0.49, 0.76]** | **+0.282** | 246 ms, API dependency, data egress |

`hit@50` plateaus at ~0.50 in every local configuration: for half the questions the right document
was nowhere in the top *fifty*, which is why reordering and bigger pools could not help — a
reranker only reorders what was retrieved. **The ceiling was the representation, not the pipeline.**

The three eliminations are what make the fourth result a diagnosis rather than a lucky guess. The
abstention layer was never the bottleneck: 89% of unanswerable questions correctly refused, 4–7%
of answerable ones wrongly refused, on every arm.

### Replicated on a public corpus — and it narrows the conclusion

The above is one corpus, and a private one. Repeated on the **public Python PEP corpus** — 746
files matched by `**/*.rst`, the figure the runner reports and the one used in the table — with 110
hand-labelled questions that **ship in this repo** — a corpus anyone can fetch and check:

| hit@5 | bge-small (local) | voyage-3 (cloud) | Δ |
|---|---|---|---|
| private memory corpus, 794 docs | 0.348 [0.23, 0.49] | **0.630** [0.49, 0.76] | **+0.282** |
| **PEPs, 746 docs (public)** | **0.705** [0.56, 0.82] | 0.727 [0.58, 0.84] | **+0.022** |

**The pipeline was never the cap** — on ordinary technical prose the *free local* embedder reaches
0.705. And **the cloud embedder's win is corpus-specific**: +0.28 where the vocabulary is
idiosyncratic, +0.02 on the PEPs, the latter inside the noise for ~5× latency, an API dependency
and data egress. So the rule is conditional, not "buy the better embedder".

Abstention accuracy is **1.00** on the PEPs for both embedders — the trust layer was never the
bottleneck on either corpus.

#### Against a baseline — because 0.705 means nothing on its own

A hit@5 is only a result next to what a boring baseline scores on the *same* corpus, chunks and
questions. So the runner now reports four arms, not one. On the PEPs, bge-small, 44 held-out
answerable questions:

| arm | hit@5 | MRR | p50 | reading |
|---|---|---|---|---|
| **BM25** (Okapi, untuned) | 0.455 [0.32, 0.60] | 0.313 | 150 ms | the thirty-year-old anchor |
| sparse only (Postgres FTS) | 0.023 [0.00, 0.12] | 0.023 | 24 ms | near-useless alone on this corpus |
| dense only (pgvector) | 0.682 [0.53, 0.80] | 0.483 | 31 ms | carries almost all of the result |
| **hybrid** (dense + sparse + RRF) | **0.705** [0.56, 0.82] | 0.494 | 26 ms | the published number |

Two things this makes honest that the single number could not. **The pipeline beats BM25 by
+0.25** (0.705 vs 0.455) — a real margin, not a rounding artifact, so the embedding stack is
earning its keep. And **dense is doing the work**: hybrid's +0.023 over dense-alone is inside the
interval, and the sparse leg scores 0.023 by itself, so on ordinary prose like the PEPs the
fusion is barely moving the top-5 — its value is on the rare identifiers and error codes that a
memory corpus has and this one does not. Stated as a margin over a baseline, "hybrid reaches
0.705" becomes a measurement instead of an assertion. (The BM25 tokeniser has no stemming while
the FTS leg does, so BM25 is mildly handicapped on morphology — noted in `recall/eval/bm25.py`;
it does not move the +0.25 conclusion.)

```bash
git clone --depth 1 https://github.com/python/peps
python -m recall.eval.labelled --corpus peps/peps     --questions recall/eval/peps_questions.json --glob '**/*.rst'
```

→ [FINDINGS §7–§8](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md) for the misses, the labelling errors found by inspecting
them, and the one experiment still untested.

## How it works

```mermaid
flowchart LR
    Q([query]) --> E[embed]
    E --> D[dense · pgvector cosine]
    Q --> S[sparse · Postgres full-text]
    D --> F[Reciprocal Rank Fusion]
    S --> F
    F --> R[cross-encoder rerank]
    R --> G{trust layer}
    G --> O([verdict + confidence + provenance per hit, or ABSTAIN])
```

Dense semantic search and sparse keyword search each retrieve candidates; **Reciprocal Rank Fusion**
merges them, a cross-encoder reranks, and the **trust layer** judges every hit — supersession,
validity window, calibrated confidence — before it reaches the agent. Validity is plain frontmatter
in the memory itself (`supersedes: old_doc.md`, `valid_until: 2026-06-30`) — *authored, not inferred*,
because a claim honoured as written is safe and a claim guessed at is not.

## Prior art — and where this genuinely differs

Agent memory is a crowded field. Everything below is Apache-2.0 and further along than this
project; a claim to novelty has to survive them, so here is the comparison rather than an
implication that the corner is empty.

| | what it is | how it handles a fact that stopped being true | what it needs |
|---|---|---|---|
| **[Graphiti](https://github.com/getzep/graphiti)** (powers [Zep](https://github.com/getzep/zep)) | temporal knowledge-graph engine | bi-temporal validity windows; contradicted facts are **invalidated, not deleted** — **inferred by an LLM at ingestion** | a graph DB (Neo4j / FalkorDB / Neptune) + an LLM call per episode |
| **[Mem0](https://github.com/mem0ai/mem0)** | memory layer (lib · self-host · cloud) | as of its 2026 redesign, **ADD-only** — no update or delete; memories accumulate and temporal reasoning happens at *retrieval* | an LLM for extraction; hybrid semantic + BM25 + entity linking |
| **[Letta](https://github.com/letta-ai/letta)** (ex-MemGPT) | stateful-agent **runtime** | memory blocks + context management, at the agent layer | an agent runtime — a different layer entirely, not a retrieval library |
| **[LangMem](https://langchain-ai.github.io/langmem/)** | memory-management toolkit | not addressed in its docs | pairs with LangGraph, though not required |
| **RE-call** | retrieval library over Postgres | validity **declared by the author** in frontmatter (`supersedes:`, `valid_until:`), enforced as a post-processing layer | PostgreSQL + pgvector. No LLM in the retrieval path, no graph DB |

**The one real difference is who decides that a memory is stale.** Graphiti infers it; RE-call
requires the author to have written it down. That is not obviously the better choice, and this
repo has the measurement that shows the cost: on the reference corpus, **2 of 792** memos declared
`supersedes:` while **60** closed a decision only in prose. Authored edges are trustworthy and
have terrible coverage.

It also has the measurement that argues for it. `recall lint --fix` was built to close that gap by
inference and, after review, could safely declare **zero** of those 60
([#29](https://github.com/GiulioDER/RE-call/issues/29)) — narrating vs declaring, part vs whole,
augmenting vs replacing are invisible to a pattern and obvious to the author. An LLM will do
better than a regex there. It will not do *reliably* better, and this library's whole thesis is
that a confidently wrong supersession is worse than a missing one. So the honest statement is a
trade, not a win: **RE-call buys precision on the edges it has, and pays for it in coverage.**

Two further differences, and one deficit:

- **Abstention is a returned value, not an error path.** `trusted_search` answers "should you
  trust any of this at all" with a calibrated threshold and a reason. The neighbours return
  memories; the caller decides.
- **No LLM and no graph database anywhere in the path.** Retrieval is pgvector plus Postgres
  full-text over a table you already know how to back up. That is cheaper and auditable; it is
  also why there is no entity reasoning here at all.
<<<<<<< Updated upstream
- **No standard-benchmark number.** Mem0 publishes LoCoMo and LongMemEval scores. This repo has
  never run either — every number here is on its own corpora, plus the public PEP replication.
  Until that changes, nothing in this README is comparable to a published memory-benchmark
  result, and it should not be read as if it were.
=======
- **A standard-benchmark number — with a hard boundary on what it compares to.** LOCOMO now runs
  against this library ([FINDINGS §9](https://github.com/GiulioDER/RE-call/blob/master/results/FINDINGS.md)), but **not** the metric Mem0 and Zep
  report: their **J** score (LLM-as-a-Judge ≈66) grades a *generator* this library does not ship,
  so no number here belongs beside it. What is measured is the retrieval substrate underneath such
  a system — evidence-turn **hit@5 0.615** [0.59, 0.64] with the free local embedder — and the one
  axis no published LOCOMO result scores at all: the **446 adversarial questions** (22.5% of the
  set) that test whether a system knows what it doesn't know. There, out of the box, RE-call
  abstains on **zero** — the on-topic-wrong-attribution case is the §4 stale-hit geometry under
  load — and its shipped levers (calibration, an entailment judge) raise that to 0.37–0.77 only by
  refusing a quarter to half of *legitimate* questions. The residual is the entity reasoning the
  bullet above says this library deliberately omits. A measured boundary, not a leaderboard win.
>>>>>>> Stashed changes

## Where this comes from

RE-call is extracted from the memory system behind a production trading-research agent whose memory
outgrew its context window. That corpus is the one the numbers above were measured against:
**794 hand-written markdown memos → 6,491 chunks**, re-indexed daily.

Every guard here is a scar from a real failure — re-litigating a falsified experiment, trusting a
weak hit on an unanswerable question, building on a fact that had been reversed. Running the library
back against that corpus is also what exposed the defects listed under [Engineering](#engineering):
real files carry stray bytes, real authors write `[[wikilinks]]` where the parser expected filenames,
and real closure notes hedge.

**→ [Redacted case study](https://github.com/GiulioDER/RE-call/blob/master/docs/CASE_STUDY.md)** — the real structure, the guards in action, and
exactly what is public versus private.

## Quickstart · 2 minutes, no API key

```bash
docker compose up -d --wait          # PostgreSQL + pgvector
pip install "recall-rag[fastembed]"  # local embeddings, no API key
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

```python
from recall.store import PgVectorStore
from recall.embeddings import FastEmbedEmbedder
from recall.trust import trusted_search

emb = FastEmbedEmbedder()
with PgVectorStore(DSN, dim=emb.dim, tenant="acme", pool_size=8) as store:
    store.ensure_schema()
    result = trusted_search(store, emb, "what is the rate limit?")
    if result.abstained:
        ...  # say you don't know — do not answer from these hits
    for hit in result.hits:
        hit.verdict      # ok | superseded | expired | not_yet_valid | low_confidence | …
        hit.confidence   # calibrated; 0.5 sits exactly on the abstention boundary
        hit.validity.superseded_by
```

Point `RECALL_DSN` at any Postgres.

> **Two operational notes.** The test suite **DROPs tables**, so it reads a separate
> `RECALL_TEST_DSN` and never `RECALL_DSN` — exporting your real DSN and running `pytest` cannot
> touch it. And the MCP server **refuses to start** if `RECALL_DSN` carries the built-in
> `recall:recall` credentials against a non-local host; set a real password, or
> `RECALL_ALLOW_INSECURE_DSN=1` to accept the risk deliberately.

> **Multi-tenancy.** Set `RECALL_TENANT` or `PgVectorStore(tenant=...)`. RLS enforces the same
> boundary in the database, so a forgotten `WHERE` returns nothing rather than another tenant's
> memories. ⚠️ **RLS is bypassed by a superuser or a `BYPASSRLS` role** — including the one in this
> repo's `docker-compose.yml`. Connect as an unprivileged role, or that second layer is decoration;
> `store.check_rls_effective()` tells you which you have, and the server warns at startup.

## Use it with Claude (MCP)

```json
{ "mcpServers": { "recall": {
    "command": "python", "args": ["-m", "recall_mcp.server"],
    "env": { "RECALL_DSN": "postgresql://...", "RECALL_TENANT": "acme" } } } }
```

Four tools: `recall_search` (verdict + confidence + provenance, or an explicit abstention),
`recall_index`, `recall_forget` (permanently delete a source's chunks — irreversible,
tenant-scoped), `recall_stats` (size, freshness, and the process metrics). Full guide →
[docs/USING_WITH_CLAUDE.md](https://github.com/GiulioDER/RE-call/blob/master/docs/USING_WITH_CLAUDE.md).

## What this does not do

Stated plainly, because the failure mode this library exists to prevent is confident overreach.

- **No token revocation without a restart.** Authentication shipped — bearer tokens, scopes and
  one tenant per principal ([docs/AUTH.md](https://github.com/GiulioDER/RE-call/blob/master/docs/AUTH.md)) — but the token file is read at startup,
  so removing access takes effect on reload, not on save. (Per-tenant rate limits and an indexing
  byte quota *do* ship — see [SECURITY.md](https://github.com/GiulioDER/RE-call/blob/master/SECURITY.md) — but their buckets are per process, so N
  workers admit roughly N times the rate.) For revocation, rotation or per-request identity, front
  this with a real identity provider and supply the MCP SDK's `auth_server_provider`.
- **Validity is authored, not inferred.** On the reference corpus — 792 memos at the time this was
  measured — **2** declared `supersedes:` while **60** described a closure only in prose. `recall
  lint --fix` was built to close that gap and, after review, could safely declare **zero** of them:
  narrating vs declaring, part vs whole, augmenting vs replacing are invisible to a pattern and
  obvious to the author. It ships as a **reviewing aid**, with `recall check` moving the question
  to write time. → investigated and settled in
  [#29](https://github.com/GiulioDER/RE-call/issues/29), now closed; the limitation stands
- **Successor and abstention accuracy are unmeasured on generated corpora.** Every synthetic
  document is the same sentence with a different opaque token, so those columns measure token
  discrimination, not the trust layer. STR, latency and scale figures are unaffected.
- **Gap detection is bounded by the embedder.** With a weak one, no threshold separates answerable
  from unanswerable — measured, not assumed.
- **Filtered ANN search stopped truncating — which is not the same as better recall.** An HNSW
  walk is filter-blind, so a `source`-filtered query exhausted its candidate list before finding
  `k` matches: at pgvector's defaults, **40/40** queries silently returned fewer results than
  asked for. `hnsw.ef_search=200` + `hnsw.iterative_scan=relaxed_order` on the filtered path fix
  that, unambiguously, in both measurements taken (**0/40** and **0/30**). The two **disagree on
  recall**, so both are published: on the test fixture's corpus — which the test deliberately
  rebuilds until it reproduces a strong pathology — recall@10 moves 0.38 → ~0.90, while on a
  corpus built the way a real multi-file index run builds one it moves **0.523 → 0.483**.
  `relaxed_order` fills to `k` with approximate matches, so this trades truncation for
  approximation rather than buying recall. The unfiltered path still runs at the defaults, where
  it measured 1.000 — but every query now also carries a `tenant_id` predicate, and that
  combination has not been measured on a multi-tenant table. HNSW build nondeterminism also
  measurably moves calibration. → measured in
  [#57](https://github.com/GiulioDER/RE-call/pull/57); issue
  [#11](https://github.com/GiulioDER/RE-call/issues/11), now closed

## Engineering

**584 tests, 7 skipped.** The database-touching ones run against a real pgvector container — no mock
DB. CI runs `ruff`, `mypy`, the suite against PostgreSQL under coverage, the suite *again* at the
declared dependency floor, and `pip-audit` over a checked-in `uv.lock` — each as a gate rather than
a report.

Type checking arrived late and is worth being specific about, because "we added mypy" is usually a
non-event. 81% of functions here already carried a return annotation and **nothing verified any of
them**. Running the checker over that found two things a green test suite had not:
`RECALL_TRANSPORT` was an unvalidated environment string flowing into a `Literal`-typed SDK
parameter — a typo reached `mcp.run()` as an arbitrary value after startup had already opened a
store and read the token file — and `ensure_schema` indexed a `None` row when pointed at an
existing table that was not a recall table. Both now fail early and by name. The gate is
`disallow_untyped_defs`, not a permissive baseline: a partially-checked package stops checking
wherever an annotation is missing, so a lenient gate passes while its coverage shrinks.

Tests are written to fail for the right reason. A representative sample:

- the RLS tests connect as a role that **cannot bypass RLS**, because as a superuser they would pass
  while testing nothing;
- the cross-tenant test asserts the other tenant's row **exists** before checking it is invisible,
  so a silently failed write cannot make it green;
- the supersession-cache test counts real table scans, so a "fix" that quietly became *rescan every
  search* would be caught;
- the metrics test asserts the counters move on the **real retrieval path** — instrumentation that
  is never wired up reports zero forever and reads as "nothing is going wrong".

Several defects were found only by running the library against a real corpus and a real server, and
each has a regression test quoting the input that caused it: a single NUL byte in one file aborting a
792-file index; every declared supersession edge failing on reference *formatting*; five tests that
encoded the developer's own environment and failed on a correctly-configured host.

## Upgrading to 0.5.0

**The chunks table gains a `tenant_id` column and its primary key becomes `(tenant_id, id)`.**
`ensure_schema()` performs that migration in place on an existing table and assigns existing rows to
the `default` tenant, which is also the default `tenant=` — so a single-tenant deployment upgrades
without noticing. There is a test that builds an old-shape table, inserts a row, opens it with this
version, and asserts the row survives and is still retrievable.

The key had to change: chunk ids derive from the file path, so two tenants indexing the same layout
produced the *same id*, and under the old single-column key one tenant's re-index silently
overwrote the other's row.

Two behavioural changes worth knowing before you upgrade:

- **The abstention threshold is fitted differently** (mid-gap rather than on the lowest answerable
  sample). It abstains *more*, and on measured data far more accurately — on the held-out sweep,
  false-confidence on unanswerable queries drops from **0.205 to 0.045**, for an additional
  **0.7%** of answerable queries wrongly abstained on (false-abstain 0.003 → 0.010). A separate
  end-to-end run of the shipped rule on the same host measured gap FCR **0.000** at false-abstain
  0.015; that number is not comparable to the 0.205, which comes from the sweep. Re-run
  `recall calibrate` and re-check any threshold you have pinned.
- **`supersedes:` matching is more tolerant.** `name`, `name.md`, `[name]` and `[[name]]` now all
  resolve to the same document, so edges that were silently dangling may start applying. That is the
  intent — on the reference corpus it took working edges from 0 to 2 — but it does mean memories
  that were served as `ok` can now correctly come back `superseded`.

## Upgrading to the next release (unreleased)

Five changes on `main` that are not in 0.5.0 yet, listed here because each can make something
that currently succeeds start failing. Full detail in [CHANGELOG.md](https://github.com/GiulioDER/RE-call/blob/master/CHANGELOG.md).

- **`RECALL_ALLOW_INSECURE_DSN` is now an explicit allowlist** — only `1|true|yes|on` disable the
  guard, and **every other value, including `0` and `false`, keeps it ON**. A deployment relying
  on `=0` to switch the check off will now refuse to start. The most likely of these to bite,
  because `0` previously did the opposite of what it reads as.
- **The `mcp` extra requires `mcp>=1.27.2`** (was `>=1.10`). Versions 1.10–1.27.1 installed
  cleanly and then failed on every authenticated call, so this now fails at install time instead.
  Upgrade with `pip install -U "recall-rag[mcp]"`.

- **`recall index` refuses a mass prune.** A re-index that would delete 50% or more of the
  sources under a root (`RECALL_MAX_PRUNE_FRACTION`, default `0.5`, above a floor of 5 indexed
  sources) raises `PruneGuardTripped` and deletes nothing — that is how a *missing* corpus stops
  being indistinguishable from a *deleted* one. It is a behaviour change for any scripted
  re-index: confirm the files really are gone, then re-run with `--allow-prune`.
- **The MCP HTTP transports require authentication and meter per tenant.** Starting
  `streamable-http` or `sse` without `RECALL_AUTH_TOKENS_FILE` refuses to boot. Per-tenant rate
  limits and an hourly indexing byte budget are on **by default** there
  (`RECALL_RATE_*_PER_MIN`, `RECALL_INDEX_BYTES_PER_HOUR`; `off` disables one). `stdio` is
  unchanged — unauthenticated by design, and not metered.
- **Schema DDL gives up after 5s of lock contention** (`RECALL_SCHEMA_LOCK_TIMEOUT_MS`; `0`
  restores the old unbounded wait). The DDL is idempotent and retried on the next store open.

## Reproduce

```bash
make eval                                        # ablations + trust + near-miss → results/
python -m recall.eval.scale --embedder hashing --filler 50000    # scale + latency
```

## License

MIT — see [LICENSE](https://github.com/GiulioDER/RE-call/blob/master/LICENSE).
