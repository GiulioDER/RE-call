# ATM-Bench RE-call pipeline and threshold calibration

Date: 2026-08-19

## Objective

Select the RE-call retrieval configuration for the ATM-Bench run, then calibrate
the gap threshold against the ATM corpus without using the held-out questions.
The selection target is retrieval and short answer evidence retention, not answer
length or an LLM judge score.

## Data and split

Use the official full ATM-Bench question file and the locally available official
image, video, and email descriptions. Split by SHA-256 of the question ID:

* development: integer SHA-256 modulo 10 in 0 through 6
* holdout: integer SHA-256 modulo 10 in 7 through 9

The 31-question official hard file is an external check and is not used for
selection or threshold fitting. The corpus index contains all official memory
items and is built separately for each embedding profile.

ATM-Bench contains no unanswerable label. Consequently, this experiment must
not claim a calibrated answerable versus unanswerable abstention boundary. The
threshold experiment is limited to an answer-preservation floor: fit the dense
top-1 cosine threshold at the fifth percentile of development questions, then
report false abstention on holdout answerable questions.

## Frozen arms

All arms use candidate_k=200 and score depths 1, 5, 10, 25, 50, and 100.

1. `fastembed:sentence-transformers/all-MiniLM-L6-v2`, dense and lexical hybrid.
2. `fastembed:BAAI/bge-small-en-v1.5`, dense and lexical hybrid.
3. `fastembed:BAAI/bge-large-en-v1.5`, dense and lexical hybrid.
4. The best local hybrid arm above plus the pinned local
   `cross-encoder/ms-marco-MiniLM-L-6-v2` over the full candidate pool.
5. A SPLADE hybrid arm using `prithivida/Splade_PP_en_v1`, to be run on a
   GPU instance only if the local wiring and data transfer checks pass.

The Voyage embedding and Voyage reranker are recorded as paid comparison arms,
but are not called in this no-credit run. A cloud comparison requires explicit
credit authorization.

## Primary metrics

Retrieval metrics are reported separately:

* official per-evidence item recall at 10
* question hit recall at 10
* complete evidence recall at 10

Short-answer retention is reported separately on `list_recall` questions:

* official list Jaccard at 5
* gold answer containment at 5
* answer hit at 5

No metric rewards emitting more non-gold IDs. Mean and p95 latency are
secondary. A configuration is preferred only when it improves the primary
retrieval or short-answer metric on development and does not reverse on
holdout. If metrics disagree, the result remains a tradeoff and no automatic
promotion is allowed.

## Predictions recorded before measurement

* BGE-small hybrid will beat the existing MiniLM hybrid on complete evidence
  recall at 10 by at least 0.02, while BGE-large will not beat BGE-small by
  more than 0.02 after latency is considered.
* Local cross-encoder reranking over candidate_k=200 will improve development
  complete evidence recall at 10 by at least 0.02. If the gain is absent or
  reverses on holdout, reranking remains opt-in.
* The development fifth-percentile answer-preservation threshold will keep
  holdout false abstention at or below 0.10 for the selected embedder. This is
  a prediction about answerable-question retention, not abstention accuracy.
* SPLADE will improve candidate-pool coverage, especially at depth 100, but is
  not predicted to improve short-answer Jaccard at depth 5 without a separate
  ranking or answer-selection mechanism.

## Decision rule

Fit the threshold and choose the pipeline using development only. Freeze the
choice before reading holdout outcomes. Then evaluate the frozen choice on the
holdout and on the hard file. No threshold, embedder, reranker, candidate pool,
or answer depth may be tuned after seeing holdout results.

## Cost and external systems

The local arms use no paid API. Voyage calls and any Vast.ai rental are excluded
from the no-credit result and require explicit authorization before execution.

