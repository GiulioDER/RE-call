# Agentic RAG — state of the art scan, read against RE-call's measured bottlenecks

**Date:** 2026-07-28 · **Scope:** literature + tooling published roughly 2025-09 → 2026-07
**Method:** web/arXiv sweep, then every finding filtered through RE-call's own FINDINGS.md
**Evidence tier of this document:** *survey*. No number here was reproduced locally. Numbers are
attributed to their source; where a fetch returned a vague summary that is stated explicitly.

> ## ⚠️ Status — one recommendation in here is already falsified
>
> This survey was written **before** the Phase 0 diagnostic it motivated. Read it with that
> ordering in mind, because it recommends a lane that has since been closed.
>
> - ❌ **§4.3 (gap-triggered PRF) and shortlist entry #2 are FALSIFIED.** The trigger they propose
>   was measured on LOCOMO (n=1,536) on 2026-07-28 and selects for retrieval **successes**, not
>   failures: hit@5 **0.7081** when it fires against **0.6164** when it does not — delta
>   **+0.0917**, the opposite of the predicted sign.
>   → [`results/legdiag/FINDINGS_phase0.md`](../results/legdiag/FINDINGS_phase0.md), registered as
>   [`results/FINDINGS.md`](../results/FINDINGS.md) §12.
> - ✅ **§4.2 (weighted fusion) survives untouched** — it never depended on the trigger, and Phase 0
>   measured its target at **349 of 1,536 questions (22.7%)** whose gold chunk sits in the fused
>   candidate pool but below k=5.
> - ❌ **§1's framing of BEAM as an untapped opportunity is WRONG.** BEAM was already run — on
>   `bench/beam-1m`, the same week, results in `results/FINDINGS.md` §9d–§9n: **RE-call 0.594 vs
>   Mem0 0.650**. RE-call loses the aggregate, though after a calibration fix none of the three
>   paired families is significant against it. The survey asserted "nobody has run it" because its
>   author searched the *literature* and not this repo's own memory. §1's description of what BEAM
>   measures is accurate; its claim about our position on it was not.
> - ⏳ Genuinely unexamined: §2 the LOCOMO judge audit, §4.1 BGE-M3, §5.1 conformal abstention. The
>   §6 ranking predates the only measurement run against it, so treat it as a reading list, not a
>   plan.
>
> Kept rather than deleted, and kept with the falsification attached: a survey that quietly drops
> its own dead recommendation teaches nothing about why it was wrong.

---

## 0. The filter this survey was run through

RE-call is not a generic RAG stack, so "best practice" is mostly irrelevant to it. Three hard
constraints and three measured facts decide what is worth reading at all.

**Constraints (from README — these are the product):**
- **C1. No LLM in the ingest or retrieval path.** The `$0.00 vs Mem0's $7.29` claim dies otherwise.
- **C2. Stays on your Postgres.** No graph DB, no vendor cloud, offline-capable.
- **C3. Trust layer is the thesis** — verdict + confidence + provenance, supersession demoted,
  abstention as a first-class return value.

**Measured facts (from `results/FINDINGS.md` — these decide what's worth *doing*):**
- **M1. The bottleneck is candidate recall, not ranking.** §7: the cross-encoder converted 3 of 31
  misses; for ~28 the right document was never in the window. *"A reranker can only reorder what
  fusion already retrieved."*
- **M2. RRF's symmetry is a defect, and it's already been caught.** §9a: a 5× deeper pool
  *lowered* hit@5 (0.671 → 0.596) because `_rrf` scores `dense[r]` and `sparse[r]` identically, so
  a deeper pool interleaves five times as many low-rank candidates into every prefix.
- **M3. Representation is the residual.** §7: swapping bge-small → voyage-3 was worth **+0.282**
  where rerank, chunk size and pool size were all null. §2: the calibrated gap threshold **does not
  transfer across embedders** — a fitted constant, not a law.

Everything below is scored against those six lines. A large fraction of the 2026 "agentic RAG"
literature fails C1 outright, and a further chunk is answering M1 with more ranking.

---

## 1. ~~The headline finding:~~ the benchmark that scores RE-call's thesis — **and we had already run it**

> **Correction.** This section was written as a discovery: *"here is an untapped benchmark that
> scores your differentiators."* The description of BEAM below is accurate. The framing is not.
> **BEAM had already been run against RE-call on branch `bench/beam-1m`**, the same week, with
> results in `results/FINDINGS.md` §9d–§9n: **RE-call 0.594 vs Mem0 0.650** on BEAM 1M, 300
> questions, same judge, both at shipped defaults. RE-call loses the aggregate; after a calibration
> fix none of the three paired families is significant against it.
>
> The survey missed this because its author searched the **literature** and never searched **this
> project's own memory** — the failure `CLAUDE.md` has a standing rule against (*docs_search before
> manual recall*) and which another session had recorded as a lesson three hours earlier. Read
> §9d–§9n before acting on anything in this section.
>
> One thing from that run worth carrying: **BEAM is adversarial by construction** — its unanswerable
> questions ask for details never stated about topics discussed at length, so overlap measures score
> *highest* exactly where there is no answer. Its numbers are an upper bound on difficulty, not
> deployed behaviour.

`FINDINGS.md` §9 calls trust/supersession *"the one axis nobody else scores."* As of ICLR 2026 that
is no longer true.

**BEAM — "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs"**
([arXiv 2510.27246](https://arxiv.org/pdf/2510.27246) · [code](https://github.com/mohammadtavakoli78/BEAM) ·
[data](https://huggingface.co/datasets/Mohammadta/BEAM) · [project page](https://mohammadtavakoli78.github.io/beam-light/))

100 conversations across 19 domains at **128K / 500K / 1M / 10M tokens**, 2,000 human-validated
questions, scored over **ten** memory abilities:

| ability | RE-call surface |
|---|---|
| **Contradiction Resolution** *(new in BEAM)* | the supersession demotion — the whole `superseded-catch` demo |
| **Knowledge Update** | `supersedes:` frontmatter + trust verdicts |
| **Abstention** | the calibrated ABSTAIN return value |
| **Temporal Reasoning** | `valid_until:` / `expired` verdict |
| **Event Ordering** *(new in BEAM)* | unclaimed |
| Information Extraction, Multi-hop, Summarization, Preference Following, Instruction Following | generic |

Four of the ten are things RE-call is *specifically built to do* and LOCOMO does not measure at all.
Scoring is **nugget-based** (reference answers decomposed into atomic units) rather than a single
LLM-judge verdict — materially more resistant to the judge failure documented in §2 below.

The paper's own result is the interesting part: **all models, including their improved LIGHT
method, still struggle with contradiction resolution** — the authors call maintaining globally
consistent state an unsolved problem. Their reported averages across all ten abilities are low in
absolute terms (128K: vanilla 0.239–0.280, RAG 0.269–0.323, LIGHT 0.294–0.358; at 10M everything
collapses to 0.10–0.27). LIGHT's own gain is +3.50%–12.69% over the strongest baseline.

**Why this matters more than any retrieval tweak below:** RE-call's differentiator is currently
argued *by demo* (the `rate_limits_v1 → v2` screenshot) and by a private-corpus study (§4). BEAM
converts it into a public, peer-reviewed, per-ability number on a benchmark where the field is
visibly failing. It is the difference between "we believe validity beats similarity" and "here is
the contradiction-resolution column."

⚠️ **Caveats before treating this as a free win.** BEAM is an end-to-end QA benchmark, so RE-call
enters as the memory layer behind a reader — same shape as the Mem0 head-to-head, same
reader-tier-conditional caveat that already bit the LOCOMO article. Also: the 10M-token arm is a
real ingest cost even at $0/embedding, and I have not verified the licence on the HF datasets.

---

## 2. The defensive finding: the LOCOMO headline is standing on softer ground than it looks

RE-call's public claim rests on a paired LOCOMO margin of **+0.046 → +0.057**, p ≤ 0.0014.

**Penfield Labs audited LOCOMO** ([writeup](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)):

- **99 score-corrupting errors in 1,540 questions (6.4%)** — hallucinated facts in the answer key
  (a car model that appears only in annotator search fields, never in the conversation), temporal
  arithmetic errors, and **24 speaker-attribution errors**. The cross-benchmark baseline for
  ML answer-key error rates is ~3.3%; LOCOMO is roughly double it.
- **The judge experiment is the alarming one.** They generated intentionally *wrong but
  topically adjacent* answers for all 1,540 questions and scored them with `gpt-4o-mini` using the
  published eval prompts: **62.81% were accepted.** Specific factual errors were caught ~89% of the
  time; vague answers that hit the right topic but missed the detail passed ~67% of the time.
- No corrected subset was released — audit scripts only.

Independently, [Locomo-Plus](https://arxiv.org/html/2602.10715v1) documents item-level flaws
(ambiguous phrasing, label-granularity mismatch, duplicates) and notes per-category n often below
100. And a widely-cited observation this cycle: **plain filesystem operations score ~74% on
LOCOMO**, matching or beating purpose-built memory systems.

**What this does and does not do to RE-call's claim.** It does *not* invalidate it — the
head-to-head is **paired**, identical generator, identical judge, identical questions, so a shared
6.4% label error and a shared loose judge are largely differenced out; that design is exactly the
defence. What it *does* is bound the interpretation: a permissive judge compresses the achievable
margin toward zero, so a **+0.046 margin measured through a judge that accepts 63% of wrong answers
is a floor on the true separation, not an estimate of it** — and it makes the reader-tier
sensitivity already found (+0.054 → +0.040 → **−0.043** on Sonnet) more plausible as judge
artefact than as a real capability inversion.

**Concrete, cheap, and in the house style:** re-score the retained per-question dumps on the
audit's flagged item list (their scripts are public), and report the margin on the clean subset
next to the full one. If the margin survives, that is a *stronger* claim than the current one, for
the price of a re-score with no new API spend. If it moves, the article needs a restatement — which
is the same move already made three times on this project.

Also worth a paragraph in the README's honest-limits section:
**[ConvoMem](https://arxiv.org/pdf/2511.10523)** argues that for roughly the first ~150
conversations, a long context window handles recall without a memory layer at all. RE-call's answer
is already better than most vendors' — the pitch is $0 cost, locality and *honesty*, not raw recall
— but the paper is going to be cited at anyone claiming a memory layer, and pre-empting it reads
better than answering it.

---

## 3. The negative result that saves work: don't build the agentic orchestration

**"Dissecting Agentic RAG: A Component Ablation for Multi-Hop QA with a Local 7B Model"**
([arXiv 2606.21553](https://arxiv.org/abs/2606.21553)) — eight ablation conditions on HotpotQA.

| component | verdict |
|---|---|
| **Rule-based adaptive routing** | **−1.8 EM / −1.9 F1 vs fixed hybrid RRF.** It over-routes to BM25 because it fires on named entities, which appear in nearly every multi-hop sub-question. |
| **Retrieval iterations** | **Two iterations capture 95% of the gains of five.** No meaningful benefit from deeper loops. |
| Query decomposition | significant (p<0.01), smaller than the loop itself |
| Cross-encoder reranking | significant (p<0.001) but "limited practical impact" |

Full pipeline 53.2 EM / 61.6 F1 vs 43.1 / 54.0 baseline — and **simpler fixed strategies beat
adaptive variants under resource constraints.**

Three things follow directly for RE-call:

1. **The fixed hybrid RRF that RE-call already ships is the correct baseline**, and query-adaptive
   routing (the obvious "make it agentic" move) is measured to be *worse* than it. Do not build it.
2. **The loop is the biggest agentic lever, and it saturates at two passes** — which is exactly the
   budget an LLM-free system can afford (see §4.3).
3. The cross-encoder result independently corroborates §7's own null: reranking is real but small
   once fusion is working. Consistent with RE-call's finding that rerank's LOCOMO win (+0.106) came
   from a benchmark where fusion *was* finding the documents, and its private-corpus null (+0.065
   n.s.) came from one where it wasn't.

Corroborating: **["Rethinking Reasoning-Intensive Retrieval"](https://arxiv.org/pdf/2605.04018)**
finds NDCG/MRR correlate poorly with agentic end-task performance, and that retrievers should be
selected for coverage and diversity rather than isolated relevance — i.e. **optimise recall of the
candidate pool, not the ranking of it.** That is M1, restated by someone else.

---

## 4. Retrieval ideas that survive C1 (no LLM), ranked by expected value

### 4.1 BGE-M3 — one local model that upgrades all three legs at once ★ top pick

[BAAI/bge-m3](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) emits **dense,
learned-sparse, and ColBERT-style multi-vector representations from a single forward pass**, is
Apache-licensed, runs locally, and is described as the 2026 production default (usually paired with
`bge-reranker-v2`). It maps onto RE-call's existing architecture with unusual precision:

| RE-call today | BGE-M3 replacement | which measured fact it attacks |
|---|---|---|
| `bge-small` dense leg | M3 dense | **M3** — the +0.282 gap voyage-3 exposed, without leaving your infrastructure |
| Postgres `ts_rank` disjunctive FTS | M3 learned-sparse (term weights + **expansion**) | **M1** — a lexical leg that *expands* terms bridges paraphrase→jargon in the lexical space, where the dense leg can't |
| `CrossEncoderReranker` (~1,050 ms) | M3 multi-vector MaxSim | latency: late interaction needs **~180× fewer FLOPs than BERT rerankers at k=10** |

The learned-sparse leg is the interesting one, because it is a *direct* answer to §7's residual.
The diagnosis there was that `bge-small` cannot connect a paraphrased question to documents whose
identifying vocabulary — codenames, venue names, internal shorthand — was never in pretraining. A
learned sparse model attacks the same gap from the opposite side: it keeps the exact-term matching
that already works for jargon *and* adds learned expansion. The generalisation evidence supports
this — SPLADE-family sparse models were specifically found to **generalise better out-of-domain on
BEIR** than dense equivalents, which is precisely the regime §7 and §8 are in.

**Honest caveats:** M3 is ~568M params vs bge-small's 33M — CPU-only indexing gets materially
slower (relevant: the fine-tune attempt was already abandoned on CPU grounds at 629% across 63
threads). Storing sparse term-weight vectors and per-token multi-vectors in Postgres is real schema
work, not a config flag. And this is a **four-way confound if shipped as one change** — dense,
sparse, rerank and vocabulary all move together, which would violate the project's own
"assert the invariant / one lever at a time" standard. Sequence it: dense swap → measure; sparse
leg → measure; multi-vector rerank → measure.

**Prediction to write down before measuring:** the *sparse* leg carries most of the private-corpus
gain (jargon), the *dense* swap carries most of the LOCOMO gain (ordinary prose, where §8 already
established the local embedder is not the bottleneck).

### 4.2 Fix the fusion before anything else ★ cheapest, already predicted

M2 is not a hypothesis — §9a *measured* the dilution and diagnosed the mechanism. The external
literature agrees on both the defect and the fix:

- RRF's known weakness is that **it discards score information entirely**, using only ranks.
- **Weighted RRF (WRRF)** multiplies the reciprocal-rank term by a per-query normalized confidence,
  so a document that is both highly ranked *and* confidently retrieved outranks one that is merely
  highly ranked ([WRRF paper](https://uregina.ca/~nss373/papers/Rag-CCNC2026.pdf)).
- The practitioner consensus is blunt and useful: **normalization matters more than the fusion
  algorithm**, with percentile-rank normalization the robust default. Naive weighted-average of
  BM25 and cosine fails because BM25 is unbounded and the scales are incompatible — which is why
  RRF exists, and why the fix is *normalize then weight*, not *abandon ranks*.

For RE-call this is a ~30-line change in `_rrf` plus a per-leg weight parameter, and it is the
precondition for §4.3 and for any pool-depth work: **until the legs are weighted, "raise
`candidate_k`" is measuring the fusion, not the retrieval** — as §9a already proved the hard way.
This is the `FUSION FIRST` note in MEMORY.md, now with an external mechanism and a named method.

### 4.3 Pseudo-relevance feedback — the LLM-free agentic loop ❌ FALSIFIED 2026-07-28

> **This section's proposal is dead.** Everything below was written before it was measured, and is
> retained as the record of what was argued. The gap-trigger idea it builds to — "fire the second
> pass only when the honesty signal says the query is failing" — was operationalised as leg
> disagreement and tested on LOCOMO: it fires on the queries retrieval already gets **right**
> (hit@5 0.7081 firing vs 0.6164 not, delta **+0.0917**, intervals disjoint, n=1,536). The
> mechanism is now understood: a decisive lexical leg means the question carried a sharp lexical
> signal — a name, a date, a number — and those questions are easy. See
> [`results/legdiag/FINDINGS_phase0.md`](../results/legdiag/FINDINGS_phase0.md).
>
> Note what this does *not* falsify: PRF itself was never reached. What died is **this trigger**.
> Any replacement is a new hypothesis needing its own preregistration, and Phase 0 bounds what it
> could ever buy at `b_unretrieved` = **162/1,536 (10.5%)** — the share of questions whose gold
> chunk is in neither leg's pool, since anything already pooled is fusion's job, not a second
> retrieval's.

§3 says the retrieval **loop** is the biggest agentic lever and saturates at two passes. C1 says
you cannot spend an LLM call on the second pass. Classical IR has solved exactly this since 1971.

**Rocchio / PRF:** run the first retrieval, treat the top-*n* as pseudo-relevant, move the query
vector toward their centroid, retrieve again, fuse. Cost: one extra vector op and one extra
Postgres round trip — **no model call, no API, no LLM.** It is the second iteration the ablation
says is worth 95% of the gains, at roughly the cost of a cache miss.

Evidence: PRF is described as *foundational* for vocabulary mismatch, semantic gap and **recall**
limitations in both sparse and dense regimes — the three words that describe §7's failure.
`ANCE-PRF` learns a query encoder for it; `TPRF` does it with a small transformer; and notably
**ColBERT-PRF** clusters token-level embeddings from feedback docs, picks IDF-discriminative
centroids, and augments the query in embedding space — MAP up to **+26% on TREC 2019**. Rocchio
remains the recommended baseline for recall@1000 specifically.

⚠️ **The known failure mode is query drift**, and IR research is explicit that PRF helps some
queries and hurts others — so it must ship behind a selective trigger, not unconditionally. RE-call
already has the perfect trigger and nobody else does: **fire the second pass only when
`gap_warning` is set or the top cosine sits in the uncertain band.** That is a genuinely novel
composition — the honesty signal that already exists becomes the loop's control input, so the extra
latency is spent only on the queries that are failing. Confident queries stay at 45 ms.

**Prediction:** PRF moves recall on the private corpus (paraphrase→jargon, high gap-warning rate)
and does approximately nothing on LOCOMO (already 0.671@5 on ordinary prose).

### 4.4 Late-interaction rerank inside Postgres — VectorChord

If §4.1's multi-vector leg is worth having, it needs somewhere to live that isn't a second database
(C2). **[VectorChord](https://github.com/tensorchord/VectorChord)** (successor to pgvecto.rs) ships
a decomposed **MaxSim** implementation in Postgres and is
[pgvector-compatible in types and syntax](https://blog.vectorchord.ai/vectorchord-03-bringing-efficient-multi-vector-contextual-late-interaction-in-postgresql).
Their recommended pattern is exactly RE-call's shape: **sentence-level ANN search first, token-level
late-interaction rerank second.**

This is the credible path to killing the 1,050 ms cross-encoder while *keeping* the rerank win
(PR #103: hit@5 0.671 → 0.777) — late interaction is less effective than a full cross-encoder but
dramatically cheaper, and the 2026 work is largely about closing that gap
([token pruning](https://arxiv.org/pdf/2603.09933), [prune-then-merge](https://arxiv.org/pdf/2602.19549),
[MICE](https://arxiv.org/pdf/2602.16299) — a masked cross-encoder claiming *below-ColBERT* cost at
cross-encoder quality). ECIR 2026 has a [first workshop dedicated to it](https://arxiv.org/abs/2511.00444).

⚠️ **This is a new Postgres extension**, i.e. a deployment dependency on every consumer's DB. That
cuts against "the PostgreSQL you already run." Ship it as an *optional* tier like `[rerank]` is
today, never as a requirement.

### 4.5 Late chunking — small, free, no retraining

**[Late Chunking](https://arxiv.org/abs/2409.04701)** (Günther et al.): embed the *whole* long
document first, chunk after the transformer and just before mean pooling. Each chunk embedding then
carries the surrounding context instead of being encoded in isolation. **Works without additional
training** and applies to any long-context embedding model.

Relevance: §7 measured chunk *size* (400/800/1600) as a flat null — but that experiment varied the
boundary, not the *encoding*. Late chunking is an untested axis, not a re-run of a closed one, and
it addresses the specific failure §7 observed qualitatively: misses "landing in the right topic
family but the wrong document," which is what context-free chunk embeddings produce. Requires a
long-context embedder (M3 qualifies), so it composes with §4.1.

### 4.6 Corpus-specific associations — the LLM-free route to multi-hop

cat3 is RE-call's floor: 0.228@k=1 → 0.620@k=20, never closing on cat1's 0.837. Multi-hop is where
graph memory systems win, and **HippoRAG 2** is the reference result — dual-node KG plus
**Personalized PageRank**, lifting MuSiQue recall@5 69.7% → 74.7% and 2Wiki 76.5% → **90.4%**.

But HippoRAG builds its graph with **LLM-based OpenIE at index time** — a straight C1 violation.
Two LLM-free routes exist:

- **[Association Is Not Similarity](https://arxiv.org/pdf/2604.20850)** learns corpus-specific
  associations **unsupervised, from corpus structure only** — no LLM annotation at index time. This
  is the most on-thesis multi-hop paper found: its entire premise is that *association ≠ similarity*,
  which is a sibling of RE-call's own *validity ≠ similarity*. **The fetch did not yield usable
  numbers — this needs a real read before it is costed.**
- **mem0's entity layer is spaCy-based, i.e. LLM-free** — already established in
  `docs/mem0-teardown-2026-07-27.md`. Entities extracted without a model call can seed a graph
  whose query-time traversal (PageRank) is also LLM-free.

Deprioritized relative to §4.1–4.3: it is the largest build in this document, and M1 says the win
is in candidate recall, which the cheaper items address first.

---

## 5. The trust layer: two ideas that are more novel than the retrieval ones

### 5.1 Conformal prediction for abstention ★ fixes a documented negative result *by construction*

§2 of FINDINGS is an honest negative: **a fixed gap threshold does not transfer across embedders**,
and §2b records that the fitting rule had to be replaced. §10b is worse — on LongMemEval the
abstention layer failed and *"no available signal fixes it."* Both are the same shape: a
hand-calibrated scalar threshold that has no distribution-free guarantee and silently re-fits when
anything upstream moves.

**Conformal prediction is the standard machinery for exactly this**, and the 2025–2026 RAG
literature has now instantiated it at every stage of the pipeline:

- **[CONFLARE](https://arxiv.org/abs/2404.04287)** calibrates the *retrieval similarity threshold*
  so the retrieved set contains the true answer with a **user-specified confidence** — this is
  RE-call's `gap_threshold`, with a coverage guarantee instead of a fitted constant.
- **[Conformal Abstention](https://arxiv.org/html/2604.27914)** gives two finite-sample guarantees:
  a *participation* bound (probability of answering at all) and a *conditional correctness* bound
  (probability the given answer is right).
- **[Principled Context Engineering for RAG](https://www.arxiv.org/pdf/2511.17908)** applies
  conformal filtering post-retrieval to guarantee coverage of relevant evidence while **cutting
  retained context 2–3×** — it consistently meets its target coverage.
- **[Is Conformal Factuality for RAG Robust?](https://arxiv.org/pdf/2603.16817)** is the
  counterweight — read it before believing the guarantees transfer under distribution shift.

**Why this is the best idea in this document after BEAM.** RE-call's product claim is *"every hit
carries a confidence"*. Today that confidence is a calibrated number with a known transfer failure.
Conformal turns it into **"this result set contains the answer with probability ≥ 1−α, and here is
the finite-sample argument"** — with the recalibration procedure defined rather than improvised
when the embedder changes. It converts §2 from a limitation into a feature, it is
**LLM-free and pure Python over a held-out calibration split**, it needs no new infrastructure, and
it maps onto BEAM's *Abstention* column. It is also, as far as this sweep found, **not shipped by
any competing memory layer** — Mem0, Zep, Letta and Eywa all compete on recall and graph structure,
none on calibrated refusal.

### 5.2 Automatic supersession detection — the moat, and the thing to check for prior art

RE-call's supersession is currently **declared** (`supersedes:` / `status: closed` / `valid_until:`
frontmatter). That is honest and cheap, and it is also the ceiling: it only protects memories
someone remembered to annotate.

The field is converging on doing it automatically, and the reference implementation is
**Zep/Graphiti's bi-temporal model**: four timestamps per fact — `t_created` / `t_expired`
(system time) and `t_valid` / `t_invalid` (world time) — where superseded facts are **invalidated,
not deleted**, and a new edge that temporally overlaps a contradicting one sets the old edge's
`t_invalid` to the new edge's `t_valid`. Zep reports 94.7% on LOCOMO and 90.2% on LongMemEval
(vendor-reported; note §2 above before weighting those).

Two things to take, one to reject:

- ✅ **Take the bi-temporal schema.** RE-call currently has *one* time axis (`indexed_at`, used for
  staleness). Splitting system-time from validity-time is a schema change, not an LLM dependency,
  and it is what makes "was this true *then*?" answerable — BEAM's **Temporal Reasoning** and
  **Event Ordering** columns.
- ✅ **Note the 2026 counter-current:** rather than auto-invalidating, keep contradictions **live,
  each with its provenance**, and let the agent or human adjudicate. This is arguably *more*
  RE-call-shaped than Zep's approach — it is the same instinct as returning `superseded` with a
  successor pointer instead of hiding the row.
- ❌ **Reject Graphiti's detection mechanism**: it uses an **LLM to compare each new edge against
  semantically related existing edges.** Straight C1 violation. The LLM-free substitute is the
  entailment/NLI stage RE-call *already has* (`recall/entailment.py`, `DEFAULT_QNLI_REVISION`) —
  a cross-encoder NLI model scoring `contradiction` between a new memory and its nearest neighbours
  at write time is the same operation at ~0 marginal cost. **This is the single most defensible
  new capability available**: an existing component, pointed at a new job, that no LLM-free
  competitor has.

⚠️ **Prior-art check required before building.** Two papers land close enough to matter and neither
fetch produced hard numbers:
- **[Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors](https://arxiv.org/pdf/2606.26511)**
  — the title is RE-call's thesis sentence. The fetch suggested automatic supersession without
  LLM-based detection as the primary mechanism, but returned **no numbers and no method detail**.
  **Read this properly before writing a line of code.**
- **[Eywa: Provenance-Grounded Long-Term Memory](https://arxiv.org/pdf/2605.30771)** — competes on
  *provenance*, RE-call's third pillar; claims contradiction resolution via derivation chains.
  Fetch was likewise thin on numbers. Treat as a named competitor until read.

### 5.3 Consolidation — a rule worth stealing, and a trap worth avoiding

**[Retain or Consolidate?](https://arxiv.org/html/2607.17545v2)** frames memory management as four
actions — retain raw, or Merge / Abstract / Rewrite — and finds the choice is **budget-dependent**:
on LongMemEval, consolidation was worth **+48 points** at a 32-token budget, but **retention won by
8–11 points** at 256 tokens. The mechanism: consolidation buys *coverage* when evidence doesn't
fit, and costs *replacement* when it does — generation can drop a timestamp, blur an ordering, or
**discard a correction**.

That last failure mode is RE-call's thesis in reverse, which makes the paper's rule the right one
to adopt and its operators the wrong ones to adopt: **all three consolidation operators are
generative (C1 violation), and their principal risk is destroying exactly the corrections RE-call
exists to preserve.** The finding to keep is the *negative* one — **at any sane context budget,
retention beats consolidation** — which retroactively justifies RE-call's append-only,
never-summarize design. That is worth one line in the README, and zero lines of code.

---

## 6. Ranked shortlist

Ordered by (expected effect on a *measured* bottleneck) ÷ (cost), with C1/C2 compliance as a gate.

| # | idea | attacks | LLM-free | cost | confidence |
|---|---|---|---|---|---|
| 1 | **Weighted / normalized fusion** (§4.2) | M2 — *measured* dilution | ✅ | ~30 lines | **high** — mechanism already proven locally |
| ~~2~~ | ~~**Gap-triggered PRF second pass** (§4.3)~~ | ~~M1 — candidate recall~~ | — | — | ❌ **FALSIFIED 2026-07-28** — the trigger fires on the queries retrieval already gets right (+0.0917, wrong sign) |
| 3 | **Conformal abstention** (§5.1) | §2 + §10b negatives | ✅ | medium | **high** — turns a limitation into a guarantee |
| 4 | **BEAM evaluation** (§1) | *evidence*, not retrieval | ✅ (eval) | medium | **high** — the only public scoreboard for the thesis |
| 5 | **LOCOMO clean-subset re-score** (§2) | protects the headline | ✅ | small | **high** — public scripts, retained dumps, $0 |
| 6 | **BGE-M3 three-leg upgrade** (§4.1) | M3 + M1 | ✅ | large | medium — big win, big confound, ship in 3 steps |
| 7 | **NLI-based auto-supersession** (§5.2) | the moat | ✅ (reuses existing NLI) | medium | medium — **gated on the prior-art read** |
| 8 | **Late chunking** (§4.5) | representation | ✅ | small | medium — untested axis, needs long-ctx embedder |
| 9 | **VectorChord MaxSim rerank** (§4.4) | rerank latency | ✅ | medium | medium — new DB extension, must stay optional |
| 10 | **Corpus-specific associations** (§4.6) | cat3 multi-hop | ✅ | large | low — paper unread, largest build |
| — | ~~Adaptive query routing~~ | — | — | — | **measured harmful, −1.8 EM (§3). Do not build.** |

**If only three things happen:** ~~#1 and #2 together, measured as one causal chain (fusion is the
precondition for the loop being measurable at all), then #5 because it costs nothing and protects
the claim the whole project rests on, then #4 because it is the only place the thesis can be scored
in public.~~

**Superseded 2026-07-28.** #2 is falsified, so the "one causal chain" framing no longer applies —
and its premise, that fusion had to land first for the loop to be measurable, was itself wrong:
Phase 0 answered the loop's load-bearing question with **no fusion change at all**, for one CPU
afternoon. The revised reading, after the only measurement run against this list:

1. **#1 weighted fusion** — still standing, and now with a measured target rather than a mechanism
   argument: 349 of 1,536 questions (22.7%) are mis-ranked inside the pool.
2. **#5 the LOCOMO clean-subset re-score** — unchanged, still ~free, still protects the headline.
3. **#4 BEAM** — unchanged, and arguably promoted: it is the only public scoreboard for the axis
   this library is built on, and nothing in the repo references it yet.

The general lesson is cheaper than any of them: **the diagnostic that could kill an idea should
run before the work the idea implies**, not after. That ordering cost one afternoon here and would
have cost two build phases the other way round.

---

## 7. What this document does *not* establish

Per `docs/RESEARCH_PROTOCOL.md` — the artifact, not the process:

- **No number here was reproduced.** Every figure is as-reported by its source; several sources are
  vendor blogs with an interest in their own benchmark (Zep's 94.7%, Mem0's 92.5%).
- **Three fetches returned summaries without extractable numbers** and are flagged inline as
  needing a real read: `2606.26511` (temporal validity), `2605.30771` (Eywa), `2604.20850`
  (associations). Two of those three are potential prior art on RE-call's core claim, so this is
  the highest-priority follow-up in the document.
- **Every "prediction" in §4 is unscored** — written down here so they can be scored later, per the
  predict-before-measuring standard. None of them has been tested.
- **BEAM's licence and RE-call's eligibility to run it were not checked**, only that data and code
  are on GitHub/HF.
- **The LOCOMO audit is a blog post with public scripts, not a peer-reviewed paper.** Its judge
  experiment is the load-bearing claim and should be reproduced on the retained dumps before it is
  cited in the article.

---

## Sources

**Benchmarks & evaluation**
- [BEAM / LIGHT — Beyond a Million Tokens (ICLR 2026)](https://arxiv.org/pdf/2510.27246) · [code](https://github.com/mohammadtavakoli78/BEAM) · [project page](https://mohammadtavakoli78.github.io/beam-light/)
- [LoCoMo audit: 6.4% of the answer key is wrong](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)
- [Locomo-Plus: beyond-factual cognitive memory evaluation](https://arxiv.org/html/2602.10715v1)
- [ConvoMem: why your first 150 conversations don't need RAG](https://arxiv.org/pdf/2511.10523)
- [Is Mem0 really SOTA in agent memory? (Zep)](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)

**Agentic RAG**
- [Dissecting Agentic RAG: a component ablation for multi-hop QA](https://arxiv.org/abs/2606.21553)
- [Agentic RAG: a survey](https://arxiv.org/abs/2501.09136)
- [Rethinking reasoning-intensive retrieval in agentic search](https://arxiv.org/pdf/2605.04018)
- [Agent-orchestrated adaptive RAG](https://arxiv.org/abs/2606.05658v1)

**Retrieval methods**
- [SPLADE v2](https://arxiv.org/abs/2109.10086) · [SPLADE repo](https://github.com/naver/splade) · [SPRINT zero-shot sparse toolkit](https://arxiv.org/pdf/2307.10488)
- [Late Chunking](https://arxiv.org/abs/2409.04701) · [Jina writeup](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [ColBERT-PRF / dense PRF reproducibility](https://arxiv.org/abs/2112.06400) · [TPRF](https://arxiv.org/pdf/2401.13509) · [Rocchio PRF baseline](https://uwspace.uwaterloo.ca/bitstreams/22034c4d-79d3-4355-b711-17b8b78f0171/download)
- [Weighted RRF](https://uregina.ca/~nss373/papers/Rag-CCNC2026.pdf) · [RRF score-normalization discussion](https://avchauzov.github.io/blog/2025/hybrid-retrieval-rrf-rank-fusion/)
- [LIR: first workshop on late interaction @ ECIR 2026](https://arxiv.org/abs/2511.00444) · [MICE](https://arxiv.org/pdf/2602.16299) · [Voronoi token pruning](https://arxiv.org/pdf/2603.09933)
- [BRIGHT benchmark](https://github.com/xlang-ai/BRIGHT) · [survey of reasoning-intensive retrieval](https://arxiv.org/pdf/2605.00063)
- [Association Is Not Similarity](https://arxiv.org/pdf/2604.20850) · [HippoRAG](https://github.com/osu-nlp-group/hipporag)

**Memory systems & trust**
- [Zep: temporal knowledge graph architecture for agent memory](https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf)
- [Temporal Validity in Retrieval Memory](https://arxiv.org/pdf/2606.26511) · [Eywa: provenance-grounded memory](https://arxiv.org/pdf/2605.30771)
- [Retain or Consolidate?](https://arxiv.org/html/2607.17545v2) · [Rate–distortion view of memory compaction](https://arxiv.org/html/2607.08032v1)
- [Always-On Agents: survey of persistent memory](https://arxiv.org/pdf/2606.30306) · [Graph-based agent memory taxonomy](https://arxiv.org/pdf/2602.05665)
- [CONFLARE](https://arxiv.org/abs/2404.04287) · [Geometry-calibrated conformal abstention](https://arxiv.org/html/2604.27914) · [Principled context engineering via conformal prediction](https://www.arxiv.org/pdf/2511.17908) · [Is conformal factuality for RAG robust?](https://arxiv.org/pdf/2603.16817)

**Infrastructure**
- [VectorChord 0.3: multi-vector late interaction in Postgres](https://blog.vectorchord.ai/vectorchord-03-bringing-efficient-multi-vector-contextual-late-interaction-in-postgresql) · [ColBERT rerank in Postgres](https://docs.vectorchord.ai/vectorchord/use-case/colbert-rerank.html) · [repo](https://github.com/tensorchord/VectorChord)
- [Open-source embedding models guide (BGE-M3)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
