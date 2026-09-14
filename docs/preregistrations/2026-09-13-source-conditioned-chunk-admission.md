# Pre-registration: source conditioned chunk admission on memory gold

**Date:** 2026-09-13  
**Status:** predicted, not yet measured  
**Population:** the committed 50 query source gold set, comprising 22 answerable queries,
25 essential facts, and 28 unanswerable controls  
**Generation:** `gen_56dce932a4444411b488bc860fea4fb7`  
**Calibration:** `cal_b458bf825c6d47469c5511b3a5038dff`  
**Embedder:** `voyage:voyage-4`

## Question

Can evidence accumulated across chunks from one source distinguish a genuinely relevant memory
from a coincidental high cosine, while every admitted chunk still has to contribute its own query
cosine?

The preceding source gold audit found the gold source in the first 20 dense candidates for all 22
answerable queries, but the served five item evidence contained it for 17. The essential fact audit
then found every fact for 14 queries and 16 of 25 facts overall. Four of the 28 unanswerable controls
received trusted evidence. These are disclosed prior results, not predictions. Their reproduction is
recorded in `docs/results/2026-09-13-live-retrieval-leg-source-gold.md` and
`docs/results/2026-09-13-live-document-expansion-essential-facts.md`.

The current Context 4 generation is outside this experiment. The collector must pin the generation
and calibration above. A different generation, calibration, pipeline fingerprint, or corpus
fingerprint aborts the run.

## Candidate collection

For each query, retain the exact production fused pool formed from the first 20 dense candidates and
the first 20 lexical candidates with reciprocal rank fusion constant 60. Retain pool position,
source, ordinal, text, raw query cosine, calibrated confidence, and the normal trust verdict for
every distinct candidate. The private collector is available only when both the generation pin and
the source admission audit flag are set.

The ordinary served five item evidence is collected in the same request. Its score must reproduce
the prior baseline exactly: 14 complete answerable queries, 16 covered facts, 17 source hits,
context precision 0.4677, and 4 unanswerable answers. Any mismatch aborts the treatment analysis as
substrate drift.

## Control arm

`trust_backfill` uses the full fused pool without source features. It keeps only candidates whose
normal trust verdict is `ok`, preserves fused pool order, and returns the first five. This isolates
the value of retrieving deeply and backfilling after trust evaluation from the value of source
conditioning.

## Source model

Candidate chunks are collapsed to one row per query and source. The row has seven features:

1. maximum raw query cosine among that source's candidates;
2. reciprocal rank of the source's best dense candidate, using `1 / (60 + rank)`, or zero;
3. reciprocal rank of the source's best lexical candidate, using the same formula, or zero;
4. total chunk level reciprocal rank fusion mass accumulated by that source;
5. fraction of that source's distinct candidates present in both legs;
6. `log1p` of its distinct candidate count;
7. its fusion mass minus the strongest competing source's fusion mass.

The label is one only when the query is answerable and the source is in its committed source gold
set. Every source for an unanswerable query is labelled zero.

Source support is estimated out of fold by leave one query out logistic regression. Every held out
query is scored by a model fitted on the other 49 queries. Features are standardized from the
training fold only. The fit uses deterministic iteratively reweighted least squares, an unregularized
intercept, L2 coefficient 1.0 on feature weights, a maximum of 100 iterations, and convergence at a
maximum absolute Newton step below `1e-8`. Sample weights first make each query contribute total
weight one, then balance the two source label classes to equal total weight inside the training
fold. The reported support value is the sigmoid of the held out logit. No source or query identity
is a feature.

## Chunk admission arms

For each candidate, define:

`adjusted_score = raw_cosine + alpha * (source_support - 0.5)`

The fixed `alpha` grid is `0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20`.

A candidate can be selected only when all of these conditions hold:

1. its normal trust verdict is `ok` or `low_confidence`;
2. its raw cosine is at least 0.30;
3. its adjusted score is at least the certified raw cosine threshold read from the pinned
   calibration.

Candidates with any validity, supersession, dependency, metadata, or entailment rejection remain
rejected. They cannot be rescued by their source. Selected candidates are ordered by adjusted score,
then by original fused position, and truncated to five. This is a per chunk decision. It never copies
a source verdict to its chunks.

`alpha=0.00` is reported as a deep pool admission control. It can differ from `trust_backfill` only
at the raw cosine floor, which should be inactive for normal `ok` hits. A difference is an apparatus
warning.

## Metrics

The primary quality metric is the count of answerable queries whose selected evidence covers every
committed essential fact. Secondary metrics are total essential facts covered, source hit queries,
and context precision, defined as gold source items divided by all selected items across answerable
queries. The binding safety metric is the number of unanswerable queries receiving any selected
evidence.

The report also records per query source support, selected candidates, admissions and demotions,
plus each arm's changes relative to both served baseline and `trust_backfill`.

## Predictions

1. At least one source conditioned arm will reach 16 complete queries and 19 covered facts without
   exceeding the baseline 4 unanswerable answers.
2. At least one arm will reduce unanswerable answers to 3 or fewer while retaining at least the
   baseline 14 complete queries and 16 covered facts.
3. Any arm satisfying either prediction will keep context precision at or above 0.44.
4. `trust_backfill` alone will not satisfy prediction 1. If it does, the simpler backfill is the
   preferred mechanism unless source conditioning improves the primary metric again or improves
   safety without reducing it.
5. Collection and offline analysis will cost 0.00 USD beyond the already authorized hosted query
   embeddings and will complete in under 15 minutes on the 50 query set.

## Decision rule

`BUILD SOURCE CONDITIONING` when the smallest alpha satisfying either quality condition below also
has context precision at least 0.44 and does not lose to `trust_backfill` on complete queries:

1. at least 16 complete queries, at least 19 covered facts, and at most 4 unanswerable answers; or
2. at least 14 complete queries, at least 16 covered facts, and at most 2 unanswerable answers.

`BUILD BACKFILL` when `trust_backfill` satisfies the first quality condition and no source
conditioned arm improves complete queries or reduces false answers without reducing complete
queries.

`GATE` when the best source conditioned arm gains exactly one complete query or one to two facts
with no safety regression, or reduces false answers by exactly one with no quality regression.
This outcome requires a larger held out gold set before serving work.

`CLOSE` when no arm reaches a build or gate condition. A result that increases unanswerable answers
above 4 cannot license implementation regardless of fact coverage.

## Reproduction command

After the collector and runner are committed to a pinned remote checkout:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioned-admission-<commit>'
$env:RECALL_SOURCE_COMMIT='<commit>'
python -u scripts/run_live_source_conditioned_admission.py `
  --generation-id gen_56dce932a4444411b488bc860fea4fb7 `
  --output docs/results/2026-09-13-live-source-conditioned-admission.json
```

The result report must include the final committed remote path and commit, because placeholders in
this frozen prediction section are not edited after measurement.

<!-- frozen_above -->

## Result

Not measured yet.

### Pre-measure apparatus clarification, 2026-09-13

The sentence above saying `alpha=0.00` can differ from `trust_backfill` only at the raw cosine
floor is wrong. The registered ordering rule sorts every source admission arm by adjusted score,
so `alpha=0.00` is a raw cosine rerank plus deep trust backfill, while `trust_backfill` preserves
fused order. I found the contradiction while implementing the runner, before collecting or
inspecting any source admission result. The runner therefore names `alpha=0.00` as
`cosine_backfill`, reports it separately, and attributes a gain over `trust_backfill` to cosine
ordering rather than source conditioning. All registered numbers and decision thresholds remain
unchanged.

### Measured result, 2026-09-13

The registered decision is `BUILD SOURCE CONDITIONING` at the smallest qualifying arm,
`alpha=0.08`. It preserved 14 complete queries and 16 facts, reduced unanswerable answers from 4
to 2, and raised context precision from 0.4677 to 0.4941. The larger `alpha=0.15` arm reached 15
complete queries, 17 facts, 2 false answers, and context precision 0.5227, but remains an
independent validation candidate because the frozen rule selects the smallest qualifying alpha.

Full result, case analysis, reproduction command, and artifact digest:
`docs/results/2026-09-13-live-source-conditioned-admission.md`.
