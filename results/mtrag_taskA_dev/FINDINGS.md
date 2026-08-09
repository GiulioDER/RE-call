# MTRAG-human dev, Task A — Voyage rerank vs the measured MiniLM baseline

**Prior work** (searched before writing, per CLAUDE.md):
`docs_search(source_type='memory', "reranker null falsified BGE-reranker-v2-m3 cross-encoder MTRAG
candidate_k pool size rerank OFF closed")`. It returned
[[project-recall-splade-learned-sparse-measured-2026-08-06]] and
[[closed-hypothesis-recall-rerank-pool-interaction-2026-08-05]], and it **overturned the framing
this file was first written with**. See "What I got wrong" below; that correction is the main
reason to read this.

777 judged queries, four domains, `candidate_k=100`, embedder `BAAI/bge-small-en-v1.5`, learned
sparse `prithivida/Splade_PP_en_v1` (apache-2.0). Every figure recomputed by a scorer written
separately from the harness; all six arms agree to **±0.000000**.

| arm | rerank | nDCG@5 | R@100 |
|---|---|---|---|
| `hybrid_lexical` | — | 0.2930 | 0.6865 |
| `dense_only` | — | 0.3024 | 0.6736 |
| `splade_only` | — | 0.3286 | 0.7074 |
| `hybrid_both` | — | 0.3348 | 0.7347 |
| `hybrid_splade` | — | 0.3573 | 0.7377 |
| **`hybrid_splade_voyage`** | voyage rerank-2.5 | **0.4342** | **0.7668** |

`hybrid_splade_voyage` vs `hybrid_splade`, paired over 777 queries: **+0.0769**, bootstrap 95% CI
**[+0.0571, +0.0964]**, permutation **p = 0.00010**. Better on 302, **worse on 162**, unchanged on
313. R@100 moves too because the arm reranks the whole fused pool before truncating, so top-100
membership changes and not merely its order.

## ⚠️ What I got wrong, and the number that actually matters

The first version of this file claimed the Voyage lift **contradicted a decision closed with
data**, on the strength of MEMORY.md's one-line summary ("leva RERANKER CHIUSA") read from context.
The memo says something much narrower, and reading it changes the result:

**Reranking is a large, already-measured, positive effect.**
[[project-recall-splade-learned-sparse-measured-2026-08-06]] records the reranker lever at
**+0.0864, CI [+0.0671, +0.1061]**, reproduced by two independent harnesses. Its per-arm table:

| arm | raw | + MiniLM rerank |
|---|---|---|
| `hybrid_lexical` | 0.2926 | 0.3811 |
| `hybrid_splade` | 0.3573 | **0.3769** |

What is **closed** is that a *bigger* cross-encoder does not rescue the SPLADE-only gold buried
below rank 10: MiniLM (22M) and BGE-reranker-v2-m3 (568M) bury the same 90 vs 91 of 123, which was
read as "not a capacity problem, those documents are not rankable from a `(query, passage)` pair".
The lever is closed **for the residual coverage gap**, not for ranking in general.

So most of the +0.0769 is the lever that was already known. Against the unreranked arm it is
largely a re-measurement. **The new quantity is Voyage against MiniLM on the same arm:**

```
hybrid_splade + MiniLM  (memo, 2026-08-06)  0.3769
hybrid_splade + Voyage  (this run)          0.4342
                                            ------
                                            ~+0.057
```

That is the figure worth defending, and it is the one that bears on the closed reading: Voyage is a
**third** model, outside the 22M/568M pair the "not a capacity problem" conclusion rests on. If it
holds, the conclusion needs re-stating as "not a capacity problem *within the cross-encoders
tested*", which is a weaker claim than the one on file.

⛔ **It is not yet a paired comparison.** The 0.3769 comes from a different run of a different
harness; the two numbers share a corpus, a split and a candidate_k, but not a process, and a
cross-run delta is not a paired CI. `hybrid_splade_rerank` (MiniLM, `candidate_k=100`, this
harness) was running when this was written, and its job is twofold: reproduce 0.3769, and make the
Voyage-vs-MiniLM delta paired and testable. Until then this is a lead, not a result.

🔑 162 queries got **worse** under Voyage. Any writeup quoting +0.0769 without that is quoting half
of it.

## Deployability is not performance

Voyage rerank-2.5 is a **paid hosted API billed per query**; `hybrid_splade` runs locally at no
marginal cost; MiniLM reranking is local and free and already recovers 0.3769. A comparison that
reports only the hosted number is the same error as reporting a non-commercially-licensed
checkpoint as if it shipped. Report the tier alongside the score.

## Provenance

The five non-Voyage arms carry **no recorded revision**. Every run wrote its manifest to the fixed
name `preregistered_manifest.json` and the Voyage run reused this directory, so the record of what
produced them was overwritten; they can be dated only by the ABSENCE of a `retried_queries` key,
which is forensics, not provenance.

Their numbers are still trustworthy on a check made afterwards rather than by the artifacts:
`recall/retriever.py` is **byte-identical** from before the first arm to `7886aa1`, and
`query_dense` / `query_sparse` / `query_learned_sparse` are unchanged. That is luck standing in for
a record, and it is why `manifest_filename()` and the per-arm `recall_revision` / `recall_dirty`
fields now exist.

The Voyage arm ran `2356663` (manifest copied aside before a later run could overwrite it) and took
**0 retries**, so it is unaffected by the retry-inside-the-timed-region bug `f47d927` fixed after it
started.

Structural checks on all six: 777 rows, 777 unique ids, 100 contexts each, no duplicate document
ids within a ranking, every file a distinct sha256. `splade_only` returns 6 empty rankings (a query
encoding to no terms contributes no leg) and `gap_warning` on all 777, both correct: the gap is
computed from DENSE cosines and that arm has no dense leg.
