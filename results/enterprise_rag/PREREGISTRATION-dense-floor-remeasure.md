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

---

## Result (2026-08-18)

**Status: measured.** New artifact `dense_floor_strat100.retrieval.json`, 100 questions, seed
20260818, 10 per stratum, corpus probe at k=200, both sides sharing one query vector through the
embedding cache. **No prediction above is edited.**

### Scorecard: three right, four wrong

| # | predicted | measured | verdict |
|---|---|---|---|
| D1 | 22, [14, 35] | **9** | ⛔ **FALSIFIED**, far below |
| D2 | 16, [10, 26] | **9** | ⛔ **FALSIFIED**, just below |
| D3 | exactly 0 | **0** | **CORRECT** |
| D4 | 8, [0, 40] | **25** | **CORRECT** (inside; the point estimate was 3x low) |
| D5 | 8.0%, [4%, 16%] | **5.7%** | **CORRECT** |
| O1 | D1 > D2 | 9 = 9 | ⛔ **FALSIFIED** |
| O2 | D1 > 16 | 9 < 16 | ⛔ **FALSIFIED** |

### 🔑 D4: the audit's blocking finding is real, and it is bigger than I predicted

**The k=1 probe disagrees with k=200 on 25 of 100 questions, and it always under-reports.** Mean
signed delta **-0.0598**, largest **-0.2234**. **Three of 100 rows have their floor verdict flipped
by the probe width alone.**

⚠️ **This corrects what I told the reviewer earlier in the session.** I checked that
`ber_voy_lex_12k_full` is single-tenant, concluded the tenant post-filter could not bite, and said
the pathology did not reach this measurement. The tenant filter is only one of the two mechanisms.
The other is that k=1 never trips the `ef_search` widening, and it bites on a single-tenant table
just as hard. Measured, not argued.

### ⛔ O1 and D1: I had the direction of the instrument change backwards

I predicted the exact measure would catch **more** demotions than the corpus top-1, because
`max(dense over returned k) <= corpus top-1` and the gap would be material. The gap is real but
tiny: **the returned top-8 contains the corpus nearest neighbour on 91 of 100 questions**, and the
mean gap is **+0.0027**. So the two measures return the **same count**, 9 and 9.

**The old artifact's "lower bound" framing was therefore nearly tight on the quantity it named,
and wrong for a different reason than I argued.** Its error was not the bound; it was the narrow
probe feeding the bound, which inflated the count.

### The headline moved, and the sampling explains more of it than the instrument

| | old artifact | this one |
|---|---:|---:|
| sample below the floor | 16 of 100 | **9 of 100** |
| population estimate | 5.6% | **5.7%** |

The population figures agree almost exactly while the sample counts do not, because the old sample
concentrated its demotions in small-population strata that carry little weight.

⚠️ **A narrative claim of PR #334 does not survive the new draw.** It reported `miscellaneous` at
**60% below the floor** and used that to argue the loss is concentrated. On a seeded sample it is
**1 of 10**. That claim was an artifact of taking the head of each category block.

✅ **And one result reproduces exactly, which is the pipeline's own check.** `high_level` is a
census, the same ten questions under either design, and it returns **5 of 10 below the floor** in
both. A measurement that changed there would have meant the instrument, not the sample, moved.

### 🔑 The real fragility, which neither the PR nor the audit named

**One sampled question carries 17.5 of the 28.5 estimated counts.** `basic` has 1 of 10 below the
floor against a population of 175, so a single row is worth **3.5 percentage points** of a 5.7%
headline. Drop it and the estimate is 2.2%; add one more and it is 9.2%.

**No wording fixes that, and this re-measurement does not either.** A figure whose dominant term is
one Bernoulli draw needs a bigger `basic` sample, not a better sentence. That is the honest next
step, and it is cheap: the strata are not equally informative, so sampling proportional to
population would spend the same 100 questions far better.

---

## Correction, 2026-08-18: D3 was a guard that could not fail, and I scored it as a success

Found by bug review, not by me. **D3 is withdrawn as a result.**

I registered "rows violating `max_returned <= corpus_top_1`: exactly 0" and reported it CORRECT.
It could not have been anything else, for two independent reasons:

1. **Both enforcement points abort before a file exists.** `retrieval_calibration` raises on the
   first violation and `write_dense_floor_artifact` refuses through `_score`, so a run with a
   violation leaves no artifact to score. The only artifacts that can be examined are the ones that
   passed.
2. **The inequality is arithmetically forced.** `best_dense_score` issues
   `query_dense(qvec, k=200)` and the retriever's dense leg issues `query_dense(qvec, k=candidate_k)`
   with `candidate_k=200`, on the same cached vector over the same deterministic index. The two
   candidate lists are therefore identical, and a sparse-only chunk outside that list cannot
   out-score its 200th member. Nothing was compared; an identity was restated.

⚠️ **This is my recorded recurring failure mode**, and registering it as a prediction did not
protect me from it: a prediction whose outcome is determined by construction is not a prediction.
The check is still worth keeping as a **guard**, because it would catch two legs running on
different query vectors, but it is not evidence and it is not scored.

**The related claim is also weaker than I wrote it.** "The returned top-8 contains the corpus
nearest neighbour on 91 of 100 questions" should read: the returned top-8 contains **the argmax of
the 200-candidate dense pool** on 91 of 100. That pool is itself an approximate walk, so the corpus
nearest neighbour is not established by it.

**To make D3 measure something**, the corpus probe would have to run strictly wider than
`candidate_k`, so that it is a different measurement rather than the same query twice.

## Correction, 2026-08-18: the narrow probe is unstable, not merely biased

The paired probe was re-run to produce a committed artifact
(`dense_floor_probe_width.json`), because the first run printed its numbers and wrote no file, and
those unbacked numbers had already reached a draft public comment.

The second run does not reproduce the first:

| | first run | committed artifact |
|---|---:|---:|
| rows where k=1 differs from k=200 | 25 | **31** |
| floor verdicts flipped | 3 | **4** |
| worst under-report | 0.2234 | 0.2234 |

Same query vectors, same index, same wide side. Part of the reported mean difference was my own
denominator change (the first averaged over differing rows only, the artifact averages over all
100), but the **counts** moved, which the denominator cannot explain.

🔑 **So `query_dense(k=1)` on this index is not a stable measurement**, not just a biased one. That
is a stronger argument for widening the probe than the bias alone, and it is the reason the
committed artifact is cited rather than either individual run.
