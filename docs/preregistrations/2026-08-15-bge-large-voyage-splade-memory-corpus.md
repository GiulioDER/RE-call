# Pre-registration: bge-large + Voyage rerank on our own memory corpus

> The filename still says `splade`. It is deliberately not renamed: the file path is this
> record's identity, and a pre-registration that moves is one nobody can cite. See the
> amendment below for why SPLADE left the configuration.

**Date:** 2026-08-15   **Status:** measured 2026-08-17, results appended below

> 🔁 **Status header corrected 2026-08-18.** It read "predicted, not yet measured"
> for a day after the results were appended, which is the failure this format exists to
> prevent: a record that has been scored and one that never was looked identical. **No
> prediction, confound or falsification criterion was touched**, only this line and the
> matching one in the Result section.

Written and committed **before** the new generation is indexed or calibrated. The gap between these
predictions and the measurement is the output; the pass rate is not.

> ## Amendment, 2026-08-15, before any measurement
>
> **SPLADE is out. The sparse leg is standard Postgres FTS** (`RECALL_SPARSE=fts`), the shipped
> default. Operator decision, taken after the predictions below were committed and **before the
> generation was indexed or anything was measured**.
>
> This is an amendment and not a rewrite. The SPLADE prediction stays exactly as written and is
> marked **NOT UNDER TEST** rather than deleted: it is a real, testable prediction and erasing it
> would make this record look as though SPLADE was never considered. If SPLADE is enabled later it
> is scored against the number below, unchanged.
>
> **Revised combined prediction, with SPLADE removed:** **+0.06 to +0.12** R@100 against the
> baseline, still dominated by the reranker.
>
> One risk this retires. SPLADE requires CUDA, and I had flagged that I did not know whether VPS2
> has a GPU. With FTS that question no longer gates the deploy, and the configuration is buildable
> on the target machine by construction rather than by hope.

## The question

Does the target configuration retrieve better than what we serve today, on **our own memory
corpus**, and what abstention threshold does it calibrate to?

- **Baseline:** `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dim), Postgres FTS sparse, no reranker,
  uncalibrated, served in `development` trust mode.
- **Target:** `fastembed:BAAI/bge-large-en-v1.5` (1024 dim), `RECALL_SPARSE=fts` (amended from splade), Voyage
  `rerank-2.5` with the local cross-encoder as fallback, reasoning layer not exposed, calibrated
  and published against a new immutable generation.

**One `memory` corpus holding both stores**, with each chunk stamped `project` and
`indexed_commit` so a hit says where it came from and can be filtered by it. Amended from the
earlier two-tenant plan: separating them by tenant made every cross-project question two
queries and a manual merge, and provenance-in-metadata gives the attribution without the split.

| Source | project stamp | Size today |
|---|---|---|
| recall's own store | `recall` | 61 files, 338 chunks |
| sentiment-agent store | `sentiment` | 933 files, not yet indexed |

**Code stays a separate corpus** under a code embedder, because a threshold is bound to its
corpus AND its embedder, and one abstention floor cannot serve two score distributions.

## What I predict

| Change | Predicted effect on R@100 | Confidence |
|---|---|---|
| bge-small → bge-large (384 → 1024) | **+0.01 to +0.03** | low |
| ~~FTS → SPLADE~~ **NOT UNDER TEST** (see amendment) | **+0.02 to +0.04** | medium |
| no rerank → Voyage rerank-2.5 | **+0.06 to +0.11** | medium |
| ~~All three together~~ superseded by the amendment | **+0.08 to +0.15**, sublinear in the sum | low |
| bge-large + Voyage rerank, FTS sparse (**the config actually under test**) | **+0.06 to +0.12** | low |

I predict the **reranker dominates** by a wide margin, and that the embedder upgrade is small
enough that it may not clear noise on a 61-file corpus at all. With SPLADE withdrawn there are
two levers under test, not three, and the reranker is expected to carry most of the difference.

**Calibrated abstention threshold:** I predict the published cosine floor lands **above** the
untuned 0.50 default currently in use, somewhere in **0.55 to 0.70**, and that it differs between
the two tenants by more than 0.05. If the two corpora calibrate to nearly the same threshold, that
is evidence the labels are not discriminating rather than evidence the corpora are alike.

## What would falsify this

- bge-large no better than bge-small, or worse, with a CI including zero. **Plausible**: these are
  short memo chunks, and a larger model helps least where the text is already lexically distinctive.
- The combined change under +0.04, which would mean the two remaining levers overlap far more
  than assumed, or that the reranker does not transfer from benchmark text to memo text.
- A calibrated threshold at or below 0.50, which would make the whole calibration a no-op against
  today's untuned default.
- Voyage rerank not beating the local cross-encoder by enough to justify a paid network call per
  query. **This is the one I most expect to be surprised by**, and it is the reason the local
  reranker stays as a fallback rather than being removed.

## How it will be measured

`recall calibration calibrate --generation <G> --queries <FILE> --publish`, per tenant, against a
new immutable generation. Metric: R@100 and the abstention threshold, on a labelled query set of
roughly 30 to 50 entries per tenant, authored for this run and **reviewed by the user before the
run**, including deliberately unanswerable queries, since abstention is the thing being calibrated.

**n is small and I am stating it up front.** 61 files is a small corpus and 30 to 50 queries is a
small evaluation. Treat a difference under 0.02 as unresolved rather than real.

## What I already know

From the sentiment-agent memory store, which holds recall's benchmark record while recall's own
store does not (itself a reason both are being indexed):

- **SPLADE measured +0.0303 R@100, p=0.0002**, merged as `d12ebf0` / #222. That is on a benchmark
  corpus, not on memos, so I am predicting slightly wider here rather than reusing the number.
- **Rerank measured +0.0864, CI [+0.0671, +0.1061].** ⚠️ Scope correction recorded 2026-08-08: the
  claim "the reranker lever is CLOSED" applied **only to the coverage gap**, not to rerank itself.
  Reusing the shorter phrasing is what produced a wrong conclusion once already.
- **Voyage embedder measured +0.0769, explicitly NOT paired**, so it does not transfer to a paired
  comparison and is not the basis of the embedder prediction above.
- Calibration belongs at install time and is per embedder: the untuned 0.50 floor is "0th percentile
  of five top-1 distributions, 16th of a sixth", so it is not comparable across models. Changing the
  embedder **requires** recalibration; that is a fact, not a finding this run can produce.

## Confounds I can name now

- **The embedder change and the dimension change are the same change.** 384 → 1024 alters the index,
  the ANN behaviour and the storage. I cannot separate "bge-large is better" from "1024 dims are
  better" in this design, and I am not claiming to.
- **I author the labels and I predict the outcome.** That is a real bias. Mitigation: the user
  reviews the labels before the run, and unanswerable queries are drawn from topics genuinely absent
  from the corpus rather than invented to be missed.
- **Voyage rerank is a network call.** Latency and failure rate are confounded with quality, and a
  fallback to the local reranker mid-run would silently mix two configurations. The run must record
  which reranker actually served each query, or it measures a blend.
- **The corpus is being written by the sessions that evaluate it.** Both stores gained memos today.
  The generation must be pinned before labelling, or the labels drift against a moving corpus.

## Result

**Status:** measured 2026-08-17, in the three sections below. Nothing above this line has
been edited, and nothing above it may be.

> ## Amendment, 2026-08-17, written BEFORE the calibration was run
>
> Three things recorded here because they were known before measuring and would otherwise read as
> post-hoc explanation. Nothing above this line is edited.
>
> ### 1. The corpus under test is now complete
>
> One `memory` tenant in `recall_repos` on VPS2: **8,716 chunks from 1,080 files**, every file on
> disk indexed, **0 chunks without a `project` stamp**.
>
> | project | chunks | files |
> |---|---:|---:|
> | sentiment-agent | 8,261 | 987 |
> | recall | 439 | 86 |
> | ai-boost-cad / cca-demos / ai-boost-av-safety | 13 | 6 |
> | steel | 3 | 1 |
>
> Larger than the "61 files, 338 chunks" this record predicted against, and the sentiment-agent
> store it listed as "not yet indexed" is now in. The prediction about noise on a 61-file corpus
> is therefore scored against a **1,080-file** corpus.
>
> ### 2. ⚠️ THE ARTEFACT'S EMBEDDER LABEL WILL BE WRONG, AND IS KNOWN TO BE WRONG
>
> The corpus is embedded with `fastembed:BAAI/bge-large-en-v1.5`. Every chunk records
> `embedding_profile = bge-small-symmetric-v1`, **naming the wrong model**.
>
> Not a mis-run. `recall/embeddings.py:794-801` hardcodes that literal as the fallback
> `profile_id` whenever no `identity` is supplied, independent of `model_name`, and
> `resolve_embedder` (`embeddings.py:1139`) constructs `FastEmbedEmbedder(model_name=...)` without
> one. Measured:
>
> ```
> fastembed:BAAI/bge-large-en-v1.5   dim=1024   profile_id=bge-small-symmetric-v1
> fastembed  (= bge-small)           dim= 384   profile_id=bge-small-symmetric-v1
> voyage:voyage-4                    dim=1024   profile_id=voyage:voyage-4          (correct)
> ```
>
> **The vectors are the evidence, not the label:** `vector_dims(embedding) = 1024` on all 8,716
> rows; bge-small is 384. So the corpus really is bge-large.
>
> Operator decision, taken deliberately: **calibrate now and record the defect** rather than fix
> the profile id first. The fix changes `profile_id`, a term in `index_fingerprint`, so it would
> re-embed all 8,716 chunks — a migration. The cost of recording instead is that this artefact
> cannot, alone, prove which model it calibrated. A reader must use the dimension.
>
> ### 3. Observed while validating the query set, BEFORE calibrating
>
> Stated here because it bears on a prediction above and was known first. **The prediction is not
> edited.**
>
> All **28** unanswerable queries have a top-1 hit at cosine **0.565 to 0.716**. Three of the
> highest were read in full and confirmed non-answers: "GraphQL schema migration" retrieves a
> Postgres advisory-lock memo, "Kafka migration benchmark" retrieves a Redis latency memo,
> "penetration test / payments" retrieves a memo about a guard that cannot fire. The labels are
> sound; the corpus has a **high similarity floor**, which is what a homogeneous body of technical
> memos under a strong dense embedder produces.
>
> Consequence for "the published cosine floor lands in **0.55 to 0.70**": a floor inside that band
> cannot reject queries the corpus demonstrably cannot answer, since those reach 0.716. I expect
> this prediction to be **falsified upward**. Recorded now so the falsification counts.
>
> ### The query set
>
> `docs/preregistrations/2026-08-17-memory-queries.json`, committed with this amendment and before
> the run. 50 entries: **22 answerable, 28 unanswerable** — the built-in eval set's 20/26 ratio,
> for the same reason: a threshold calibrated mostly on answerable queries has never been shown a
> case where it should abstain.
>
> Two labelling decisions that move the metric, stated rather than buried:
>
> - **`relevant_ids` were resolved from the built index, never typed.** They are `<file>:<ord>`
>   and the ordinals are a property of how the corpus chunked.
> - **Every chunk of an answering memo counts as relevant** (mean 6.6 per query), not the single
>   best-matching one. Choosing one by cosine would let the embedder under test pick its own
>   labels. This inflates precision for multi-chunk memos.
>
> Verified before use: no answerable label spans more than one project — which matters because
> `relevant_ids` key on the root-relative filename and **three filenames exist in several stores**
> (`MEMORY.md` in all six, `feedback_index.md` and `reference_index.md` in two).

## Measured, 2026-08-17

Artefact: `results/calibration-memory-2026-08-17.json`. Measured against the LIVE `memory` tenant
(8,716 chunks), not a rebuilt copy — `recall calibrate` builds a throwaway `cal_<uuid>` table and
re-indexes into it, which would have calibrated a different index from the one served. The
measurement functions are recall's own and unmodified (`measure_top_cosines`, `from_samples`,
`loo_threshold_rates`); only the store differs.

| | value |
|---|---|
| **Calibrated threshold** | **0.7100** (scale 0.0155) |
| Answerable (n=22) | min 0.668, p25 0.729, median 0.741, max 0.815 |
| Unanswerable (n=28) | min 0.565, median 0.651, p75 0.667, max 0.716 |
| **Separation** (min ans − max unans) | **−0.048** |
| Separability | 0.989, CI [0.957, 1.000] |
| Leave-one-out false-confident | 0.036 |
| Leave-one-out false-abstain | 0.045 |

### Scoring the predictions

**"The published cosine floor lands above the untuned 0.50, somewhere in 0.55 to 0.70."**
**FALSIFIED**, upward, at **0.7100**. Above 0.50 as predicted, but outside the stated band. The
direction was recorded in the amendment above before the number existed, because validating the
query set had already shown unanswerable queries reaching 0.716 — a floor inside the predicted
band could not have rejected them. The band was too low because it was reasoned from a 61-file
corpus of one store; a 1,080-file corpus of technical memos has a markedly higher similarity
floor.

**"...and that it differs between the two tenants by more than 0.05."** **NOT TESTABLE as
designed.** The amendment of 2026-08-15 collapsed the two tenants into one `memory` corpus, so
there is no second threshold to compare. Marked ineligible rather than scored, for the same reason
the SPLADE prediction was.

**The R@100 predictions (bge-small → bge-large +0.01 to +0.03; rerank +0.06 to +0.11; combined
+0.06 to +0.12) are NOT MEASURED.** This run produced a calibrated threshold, not a retrieval
comparison. Scoring them needs the baseline arm — bge-small, no reranker — measured on this same
corpus and query set. Nothing here should be read as evidence for or against them.

### What the numbers say beyond the predictions

**Separability 0.989 with separation −0.048 is not a contradiction, and the pair matters more
than either alone.** The distributions are almost perfectly ordered (a randomly drawn answerable
query out-scores a randomly drawn unanswerable one ~99% of the time), yet they still OVERLAP: the
worst answerable query (0.668) scores below the best unanswerable one (0.716). So 0.7100 is a
least-bad compromise, not a clean boundary, and no single cosine threshold can separate these two
sets. Reading only the separability would hide that.

The leave-one-out rates put numbers on the residual cost: **3.6% false-confident, 4.5%
false-abstain**. Roughly one query in 22 is answered when it should have abstained, and one in 22
abstains when it should have answered.

⚠️ **The artefact's corpus label is wrong and this is known.** Rows carry
`embedding_profile = bge-small-symmetric-v1` while holding 1024-dim bge-large vectors; see the
amendment above. The JSON records `embedding_profile_label_is_known_wrong: true` so the defect
travels with the artefact rather than living only here.

⚠️ **Confound not retired:** I authored the labels and predicted the outcome. The pre-registration
named this and proposed operator review of the labels as the mitigation; that review has not
happened yet. The labels are committed and readable at
`docs/preregistrations/2026-08-17-memory-queries.json`.

## Measured, 2026-08-17: the rerank arm

Artefact: `results/rerank-arm-memory-2026-08-17.json`. Dense pool of 300 retrieved ONCE per query
and scored twice — fusion order versus Voyage `rerank-2.5` order — so the arms are paired by
construction. Retrieving separately per arm would have let Voyage's non-deterministic query
embeddings put the two arms on different pools.

**Voyage served 22 of 22 queries; 0 fell back to the local cross-encoder.** The pre-registration
names a mid-run fallback as a confound that would silently measure a blend; it did not occur.

| metric | baseline | + Voyage rerank | delta |
|---|---:|---:|---:|
| R@5 | 0.4337 | 0.4649 | **+0.0311** |
| R@10 | 0.4827 | 0.5518 | +0.0691 |
| **R@100** | 0.6979 | 0.7689 | **+0.0710** |
| nDCG@10 | 0.5180 | 0.5478 | +0.0299 |
| MRR | 0.8016 | 0.7845 | **−0.0170** |

### Scoring the prediction

**"no rerank → Voyage rerank-2.5: +0.06 to +0.11 R@100."** **CONFIRMED** at **+0.0710**, inside
the band and near its middle. This is the one prediction in the record that has now been measured
as written, on the metric it was written in.

**"I predict the reranker dominates by a wide margin."** **NOT YET SCORED.** Dominance is a
comparison against the embedder arm, which is unmeasured — scoring it needs a bge-small index of
this same corpus.

### Two things the registered number hides

**MRR fell, −0.0170.** The reranker improves BREADTH and degrades the TOP. It pulls additional
relevant chunks into the retrieved set while demoting some first-relevant hits that dense
retrieval had already ranked first. A record that quoted only R@100 would report an unambiguous
win where the measurement shows a trade.

**The served shape gains less than half of the registered metric.** R@5 is +0.0311 against
R@100's +0.0710. The shipped profiles are `candidate_k=20, returned_k=5` (`profiles.py:105-114`,
both `fast` and `quality`), so **no served configuration returns 100 hits**. R@100 scores the
prediction as written; R@5 is what a user experiences. Both are reported rather than one being
substituted for the other, because the registered metric is the one the prediction was made in
and the served metric is the one that matters operationally.

⚠️ Also note the label design caps small-k recall: with a mean of **6.6 relevant chunks per
query**, R@5 cannot exceed 5/6.6 = 0.76 for any system. The choice to mark every chunk of an
answering memo relevant was recorded above as inflating precision; it deflates recall at small k,
and that applies equally to both arms so the DELTA is unaffected.

## Measured, 2026-08-17: the embedder arm and the combined configuration

Artefact: `results/embedder-arms-memory-2026-08-17.json`. The same 1,080 files indexed a second
time under bge-small into `chunks_bge_small` — a **separate table**, because `chunks.embedding` is
`vector(1024)` and bge-small is 384-dim, so a tenant alone cannot hold both. Same tenant name,
same query set, same pool of 300: the arms differ in the embedder and nothing else. Both arms
produced **8,716 chunks from 1,080 files**, confirming the chunking is embedder-independent and
the comparison is properly paired.

| metric | bge-small | bge-large | bge-large + rerank |
|---|---:|---:|---:|
| R@5 | 0.3433 | 0.4337 | 0.4649 |
| R@10 | 0.3825 | 0.4827 | 0.5518 |
| **R@100** | 0.6609 | 0.6979 | 0.7689 |
| nDCG@10 | 0.4145 | 0.5180 | 0.5478 |
| MRR | 0.6297 | **0.8016** | 0.7845 |

Deltas on R@100 with a **paired bootstrap 95% CI**, 10,000 resamples, seed 20260817:

| comparison | delta | 95% CI | predicted | verdict |
|---|---:|---|---|---|
| bge-small → bge-large | +0.0370 | **[−0.0325, +0.1154]** | +0.01 to +0.03 | **INDISTINGUISHABLE FROM ZERO** |
| bge-small → bge-large + rerank | +0.1080 | [+0.0421, +0.1799] | +0.06 to +0.12 | **CONFIRMED** |

### Scoring the predictions

**"bge-small → bge-large: +0.01 to +0.03 R@100, confidence low."** **INDISTINGUISHABLE FROM
ZERO.** The point estimate is +0.0370 — above the band — but the interval spans zero, so this run
cannot say the effect is real, let alone that it falls outside the prediction. Reporting
"falsified above" from the point estimate alone would have been wrong, and that is exactly the
error the CI exists to prevent. The record itself anticipated this: it rated the prediction low
confidence and said it "may not clear noise", which at **n=22 answerable queries** is what
happened.

**"All three together (now: bge-large + Voyage rerank, FTS sparse): +0.06 to +0.12."**
**CONFIRMED** at **+0.1080**, inside the band with a CI excluding zero.

**"I predict the reranker dominates by a wide margin."** **CONFIRMED**, and this is now scorable.
The reranker alone is +0.0710 with the CI of the combined arm excluding zero, while the embedder
alone cannot be distinguished from no effect. Nearly all of the combined +0.1080 is the reranker.

### What R@100 could not see

**The embedder's real effect is on ranking, not recall.** bge-small → bge-large moves **MRR from
0.6297 to 0.8016 (+0.17)** and **nDCG@10 from 0.4145 to 0.5180 (+0.10)**, while its R@100 delta is
noise. A bigger embedder is placing the right chunk near the top far more often; it is not finding
many chunks the smaller one missed within 100. The pre-registration chose R@100, so that is what
is scored — but a reader concluding "bge-large did nothing" from the scored line would be wrong.
These two figures are descriptive: no CI was computed for them, and they were not registered.

**Rerank trades top-1 for breadth, in both arms.** MRR falls from 0.8016 to 0.7845 when the
reranker is added, the same direction as the standalone rerank arm.

## Appended 2026-08-18: the number that SERVES is 0.7120, not the 0.7100 recorded above

Nothing above is retracted. The 0.7100 fit stands exactly as measured. But it is not the value the
server applies, and both numbers were in circulation for a day without either being wrong.

| | fitted | store it was fitted against | date |
|---|---:|---|---|
| `results/calibration_memory.json` | **0.7100** | legacy `chunks` table | 2026-08-17 |
| published row in `recall_calibrations` | **0.7120** | generation `recall_chunks_v1` | 2026-08-18 |

Every other field is identical: scale 0.0155, separability 0.9886363636363636 to all sixteen
digits, n=22 answerable and n=28 unanswerable, same 50 query file. Only the threshold moved, by
0.002.

**What is verified:** the two rows exist with those values, the fits are one day apart, and
`bin/install_calibrations.py` applies no transform to a threshold, so nothing rewrote 0.7100 into
0.7120 in transit. Confirm with:

```bash
psql "$RECALL_DSN" -x -c "SELECT threshold, scale, separability, n_answerable, n_unanswerable, created_at FROM recall_calibrations WHERE tenant_id='memory';"
```

**What is inferred and NOT measured:** that the difference comes from the generation build
re-chunking the corpus, so a few top cosines shift while the ranking (and therefore separability)
does not. That is consistent with an identical rank statistic beside a moved threshold, and it is
consistent with the legacy `chunks` table now holding **0 rows** for this tenant, but I did not
re-fit against both stores to demonstrate it. Do not cite it as measured.

🔑 **The operational rule this yields: a threshold in a results file is not necessarily the
threshold in service.** They are bound to different stores, and a generation re-embeds and
re-chunks. Read the published row, not the artefact, when you want to know what the server will
do.
