# Pre-registration: re-measuring the dense floor with a corrected instrument

**Date:** 2026-08-18   **Status:** predicted, not yet measured

## Registration

```yaml
supersedes_artifact: results/enterprise_rag/dense_floor_strat100.retrieval.json
superseded_because: the probe and the leg it bounded ran at different HNSW ef_search
index: ber_voy_lex_12k_full, 519,245 rows, single tenant (verified 2026-08-18)
```

## The question

PR #334 measured how many EnterpriseRAG questions would lose all evidence to the uncalibrated
trust layer's 0.50 floor on the dense cosine, and published **"at least 5.6% of 500"**. Bug review
found the instrument does not support the word "least":

- `best_dense_score` probes with `query_dense(k=1)`. `recall/store.py:1329` widens `hnsw.ef_search`
  only when `k * multiplier > 40`, so at k=1 the probe runs at pgvector's default **ef_search=40**,
  while the retrieval leg it claims to bound runs at k=200 and therefore **ef_search=800**. The
  same file records ef_search=40 at 0.385 recall against ef_search=200 at 0.942.
- An approximate walk can only under-report the top score, and under-reporting moves questions
  **into** the below-floor count, which is the direction that breaks a lower bound.
- The two sides were not even taken on the same query vector: `best_dense_score` issued a second
  live `embed_query`, and this project has measured 42.5% of repeat Voyage query embeddings as
  different. See [[voyage-query-embeddings-are-not-deterministic]].

**One sentence: with the instrument corrected, does the measured demotion rate go up, and was the
k=1 probe actually lossy on this corpus?**

## What changes in the instrument

1. **The exact quantity replaces the bound.** The floor is applied per returned hit, so a question
   loses all evidence exactly when `max(dense over the returned top-k) < 0.50`. Those hits are
   already in hand, so this is free, needs no second query, and is an equality rather than a bound.
2. **The corpus probe is fixed and kept**, at `k=200` so `ef_search` widens, and driven through
   `QueryCachedEmbedder` so both sides use **one** query vector.
3. **The sample is drawn by a committed seeded sampler.** The published sample is the first 10 ids
   of each category block, which is the sorted-then-truncated head bias this project has been bitten
   by twice, and no sampler exists in the tree at all.
4. `summarize` refuses a sample that does not cover every stratum, because its numerator sums over
   strata present while its denominator is always 500.

## What I predict

`n = 100` sampled questions, 10 per `question_type`, over a 500-question population.

| # | quantity | point | interval |
|---|---|---|---|
| D1 | `sample_below_0_50` using the **exact** measure, max dense over the returned top-8 | **22** | 14 to 35 |
| D2 | the same count using the **corpus top-1** measure, probe fixed to k=200 | **16** | 10 to 26 |
| D3 | rows violating `max_returned <= corpus_top_1` | **0** | exactly 0 |
| D4 | 🔑 rows where the OLD k=1 probe differs from the fixed k=200 probe | **8** | 0 to 40 |
| D5 | population-weighted estimate below the floor, exact measure | **8.0%** | 4% to 16% |

**Ordering predictions:**

- **O1.** D1 > D2. The corpus top-1 can exceed every returned hit, so counting on it under-counts
  demotions. This is the effect the old artifact called a lower bound, and it should be visible.
- **O2.** D1 > 16, the previously published sample count, even though the sample has changed.

## What would falsify this

- 🔑 **D4 = 0**: the k=1 probe returns exactly the k=200 answer on all 100 questions, the audit's
  blocking finding is theoretical on this corpus, and I must report that as loudly as a
  confirmation. The lower-bound wording would still be wrong for the sampling reason, but the
  instrument would be vindicated.
- **D3 > 0**: the invariant fails even with one shared query vector and matched ef_search, which
  would mean the returned hits do not carry the dense cosine I believe they do, and nothing here
  can be interpreted until that is understood.
- **O1 fails** (D1 <= D2): the returned top-8 essentially always contains the corpus nearest
  neighbour, so the old bound was tight and the correction buys nothing but rigour.

## Confounds I can name now

1. **The sample changes at the same time as the instrument**, so D1 against the old 16 mixes two
   causes. D2 is measured on the SAME new sample precisely so the instrument effect (D1 vs D2) is
   isolated from the sampling effect.
2. **10 per stratum stays a sample.** Even corrected, the population figure is a stratified
   estimate with a real interval, not a bound. Only `high_level` is a census.
3. **Voyage query embeddings are non-deterministic**, so a re-run of this very measurement moves by
   a small amount. Measured elsewhere in this repo at about ±0.01 on an AUC; here it should mostly
   affect borderline rows near 0.50.
