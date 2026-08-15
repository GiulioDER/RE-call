# Pre-registration: bge-large + Voyage rerank + SPLADE on our own memory corpus

**Date:** 2026-08-15   **Status:** predicted, not yet measured

Written and committed **before** the new generation is indexed or calibrated. The gap between these
predictions and the measurement is the output; the pass rate is not.

## The question

Does the target configuration retrieve better than what we serve today, on **our own memory
corpus**, and what abstention threshold does it calibrate to?

- **Baseline:** `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dim), Postgres FTS sparse, no reranker,
  uncalibrated, served in `development` trust mode.
- **Target:** `fastembed:BAAI/bge-large-en-v1.5` (1024 dim), `RECALL_SPARSE=splade`, Voyage
  `rerank-2.5` with the local cross-encoder as fallback, reasoning layer not exposed, calibrated
  and published against a new immutable generation.

Two corpora, separate tenants, calibrated separately because **a threshold is bound to its corpus**:

| Tenant | Source | Size today |
|---|---|---|
| `memory` | recall's own store | 61 files, 338 chunks |
| `sa-memory` | sentiment-agent store | 933 files, not yet indexed here |

## What I predict

| Change | Predicted effect on R@100 | Confidence |
|---|---|---|
| bge-small → bge-large (384 → 1024) | **+0.01 to +0.03** | low |
| FTS → SPLADE | **+0.02 to +0.04** | medium |
| no rerank → Voyage rerank-2.5 | **+0.06 to +0.11** | medium |
| All three together | **+0.08 to +0.15**, sublinear in the sum | low |

I predict the **reranker dominates**, that SPLADE is second, and that the embedder upgrade is the
smallest of the three and may not clear noise on a 61-file corpus at all.

**Calibrated abstention threshold:** I predict the published cosine floor lands **above** the
untuned 0.50 default currently in use, somewhere in **0.55 to 0.70**, and that it differs between
the two tenants by more than 0.05. If the two corpora calibrate to nearly the same threshold, that
is evidence the labels are not discriminating rather than evidence the corpora are alike.

## What would falsify this

- bge-large no better than bge-small, or worse, with a CI including zero. **Plausible**: these are
  short memo chunks, and a larger model helps least where the text is already lexically distinctive.
- The combined change under +0.05, which would mean the three levers overlap far more than assumed.
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

**Status:** not yet measured. Append below; do not edit anything above.
