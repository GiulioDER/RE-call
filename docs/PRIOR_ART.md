# Prior art, and where this comes from

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
- **A standard-benchmark number — with a hard boundary on what it compares to.** LOCOMO now runs
  against this library ([FINDINGS §9](../results/FINDINGS.md)), but **not** the metric Mem0 and Zep
  report: their **J** score (LLM-as-a-Judge ≈66) grades a *generator* this library does not ship,
  so no number here belongs beside it. What is measured is the retrieval substrate underneath such
  a system — evidence-turn **hit@5 0.671** [0.65, 0.69] with the free local embedder, rising to
  **hit@20 0.855** [0.84, 0.87] across the measured depth curve ([FINDINGS §9a](../results/FINDINGS.md)).
  Both depths are quoted deliberately: `hit@k` is a *ceiling* on any downstream J, and a ceiling
  published at one depth reads as a ceiling at every depth — it is not. Depth is not free either,
  since k=20 spends four times the generator's context to buy it — and the one
  axis no published LOCOMO result scores at all: the **446 adversarial questions** (22.5% of the
  set) that test whether a system knows what it doesn't know. There, out of the box, RE-call
  abstains on **zero** — the on-topic-wrong-attribution case is the §4 stale-hit geometry under
  load — and its shipped levers (calibration, an entailment judge) raise that to 0.37–0.77 only by
  refusing a quarter to half of *legitimate* questions. The residual is the entity reasoning the
  bullet above says this library deliberately omits. A measured boundary, not a leaderboard win.

## Where this comes from

RE-call is extracted from the memory system behind a production trading-research agent whose memory
outgrew its context window. That corpus is the one the numbers above were measured against:
**794 hand-written markdown memos → 6,491 chunks**, re-indexed daily.

Every guard here is a scar from a real failure — re-litigating a falsified experiment, trusting a
weak hit on an unanswerable question, building on a fact that had been reversed. Running the library
back against that corpus is also what exposed the defects listed under
[Engineering](ENGINEERING.md):
real files carry stray bytes, real authors write `[[wikilinks]]` where the parser expected filenames,
and real closure notes hedge.

**→ [Redacted case study](CASE_STUDY.md)** — the real structure, the guards in action, and
exactly what is public versus private.
