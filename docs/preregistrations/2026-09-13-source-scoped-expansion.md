# Pre-registration: source scoped expansion on Context 4 memory gold

**Date:** 2026-09-13  
**Status:** predicted, not yet measured  
**Population:** the committed 50 query memory gold set, comprising 22 answerable queries,
25 essential facts, and 28 unanswerable controls  
**Query set SHA256:** `06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f`  
**Fact labels SHA256:** `45c38731e138f6aed425635b78ee79b692e142f5ef040e0efffeb042ad47186b`  
**Generation:** `gen_18d5edd2e5e847c0af1ee37e40d27893`  
**Calibration:** `cal_23d8708ac550444fa4274ac617df0870`  
**Embedder:** `voyage-context:voyage-context-4`, profile `voyage-context-4-v1`

## Question

Can the source support model improve essential fact recall when it controls both which sources are
searched and which chunks are admitted, rather than only rescoring chunks already present in the
global dense 20 plus lexical 20 union?

The preceding Voyage 4 experiment found the gold source among the top three out of fold source
support ranks for five of the eight incomplete answerable queries. Its registered `alpha=0.08` arm
preserved 14 complete queries and 16 facts while reducing false evidence answers from four to two.
Its exploratory `alpha=0.15` arm reached 15 complete queries and 17 facts with two false answers.
These are disclosed prior results, not predictions. Full evidence is in
`docs/results/2026-09-13-live-source-conditioned-admission.md`.

Context 4 is a new retrieval substrate. No exact baseline count is assumed. All decisions below are
paired within the pinned Context 4 generation and calibration. A different generation, calibration,
pipeline fingerprint, corpus fingerprint, query set digest, or fact label digest aborts the run.

## Pass one: global source support

For every query, collect the served five item baseline and the exact production fused candidate
union from dense rank 20 and lexical rank 20. Retain the same seven source features and the same
leave one query out logistic regression defined in
`docs/preregistrations/2026-09-13-source-conditioned-chunk-admission.md`:

1. maximum raw query cosine;
2. reciprocal rank of the best dense candidate with constant 60;
3. reciprocal rank of the best lexical candidate with constant 60;
4. total source reciprocal rank fusion mass;
5. fraction of source candidates present in both legs;
6. `log1p` of distinct source candidate count;
7. source fusion mass minus the strongest competing source mass.

Each held out query is scored only by a model fitted on the other 49 queries. Features are
standardized from that training fold. The fit keeps the registered query weighting, class balance,
unregularized intercept, L2 coefficient 1.0, 100 iteration limit, and `1e-8` convergence tolerance.
No query identity, source identity, fact term, or held out label is a feature.

Sources are ordered by descending held out support, then by the best original fused pool rank, then
lexicographically by source. Only sources present in the global fused union are eligible.

## Pass two: source scoped retrieval

Keep the same freshly launched MCP process alive after pass one. For each query, search each of its
top three sources separately through the pinned generation. Each source request uses the ordinary
source filter and exposes its private benchmark trust pool. The hosted query embedder may be called
again because the current serving surface does not expose vectors across requests. Provider cost is
zero under the current account, but request count and wall time are reported rather than claimed as
production latency.

Within each selected source, order distinct candidates by raw query cosine descending, then their
source scoped fused pool rank, and retain the first eight. Pool candidates must carry their normal
trust verdict. Merge the retained candidates across sources by chunk identity.

For every retained candidate define:

`adjusted_score = raw_cosine + alpha * (source_support - 0.5)`

A candidate is eligible only when its normal verdict is `ok` or `low_confidence`, its raw cosine is
at least 0.30, and its adjusted score is at least the certified threshold read from the pinned
calibration. All other trust rejections remain binding. Eligible chunks are ordered by adjusted
score descending, then source support descending, source rank ascending, source scoped pool rank
ascending, and chunk identity. Return at most five chunks.

## Registered arms and controls

The primary treatment is `scoped_sources_3_alpha_0.08`.

The complete offline matrix uses source budgets one, two, and three with alpha values 0.00, 0.08,
and 0.15. Source budget prefixes the already collected top three list, so the matrix adds no hosted
requests. Alpha 0.00 isolates source selection plus within source cosine ordering. Alpha 0.08 is the
registered treatment from the preceding experiment. Alpha 0.15 is a separately named validation
candidate and cannot replace 0.08 merely because it wins this reused gold set.

Controls are:

1. `baseline`, the ordinary served five item Context 4 evidence;
2. `global_alpha_0.08`, the registered source admission rule on the global fused union;
3. `global_alpha_0.15`, the prior exploratory rule on the same global union.

Every arm has the same five item output budget. The 28 unanswerable queries remain binding and go
through both passes exactly like answerable queries.

## Metrics

The primary quality metric is complete answerable queries whose evidence covers every committed
essential fact. Secondary quality metrics are total essential facts covered, gold source hit
queries, and context precision, defined as gold source items divided by all selected items for
answerable queries. The binding safety metric is unanswerable queries receiving any evidence.

The artifact records all source ranks, source scoped candidate pools, selected chunks, fact matches,
per arm deltas, client observed wall time, and the exact serving lineage. It also reports retrieval
request count. It does not infer provider billing or production p95 latency from this serial audit.

## Predictions

1. The primary top three alpha 0.08 treatment will gain at least one complete query or at least two
   essential facts over `global_alpha_0.08`, with no loss on the other metric.
2. It will not exceed `global_alpha_0.08` on unanswerable answers and will keep that count at two or
   fewer.
3. Its context precision will be at least 0.48.
4. A top three source budget will cover more facts than the corresponding top one source budget for
   at least one of alpha 0.08 or 0.15.
5. The full 200 request audit, 50 global requests plus 150 source scoped requests, will complete in
   under 30 minutes. Provider request accounting, retries, and billing are not available from this
   apparatus, so no dollar cost prediction is registered.

## Decision rule

`BUILD SOURCE SCOPED EXPANSION` when the smallest source budget at alpha 0.08 satisfies all of:

1. at least one more complete query or at least two more facts than `global_alpha_0.08`;
2. no loss in complete queries or covered facts versus `global_alpha_0.08`;
3. no more unanswerable answers than `global_alpha_0.08` and no more than two overall;
4. context precision at least 0.48.

`VALIDATE ALPHA 0.15` when no alpha 0.08 arm builds but the smallest alpha 0.15 arm meets the same
conditions against `global_alpha_0.15`. This licenses a fresh held out validation set, not serving.

`GATE` when the best scoped arm gains exactly one fact without losing complete queries, or gains one
complete query while losing exactly one other fact, with no safety regression. It then needs more
gold before implementation.

`CLOSE` when no scoped arm reaches a build, validation, or gate condition. Any arm with more than
two unanswerable answers or worse safety than its corresponding global control cannot license work.

The primary decision is evaluated before inspecting individual query cases. Case analysis may
explain the result but cannot alter these thresholds.

## Reproduction command

After the runner is committed to a pinned remote checkout:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-scoped-expansion-<commit>'
$env:RECALL_SOURCE_COMMIT='<commit>'
python -u scripts/run_live_source_scoped_expansion.py `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-scoped-expansion.json
```

The result report must provide the committed checkout path, source commit, artifact SHA256, exact
command, and any pre-measure apparatus correction. Frozen numbers above are never edited after the
first preregistration commit.

<!-- frozen_above -->

## Result

Not measured yet.
