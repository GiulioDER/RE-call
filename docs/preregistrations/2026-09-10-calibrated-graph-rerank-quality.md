# Pre registration: calibrated semantic graph reranking quality

Date: 2026-09-10

Status: locked before the live measurement

## Question

Does the combined semantic graph path with calibrated candidate reranking improve trusted
retrieval quality over graph off retrieval on the frozen VPS2 memory query set, while preserving
the strongest original retrieval items and avoiding trust or refusal regressions?

The candidate implementation is commit `ab688757`, `feat: calibrate semantic graph candidate
reranking`. The hard relative cosine admission rule is removed from the candidate path. The
reranker combines calibrated cosine relevance, relation confidence, inverse path length, and
bounded corroboration, then keeps two strongest original trusted retrieval items as anchors.

## Prediction

Before measuring, I predict that graph one hop with the candidate implementation will increase
answerable query hit at five by `0.02` to `0.08` absolute versus graph off. I predict paired MRR
will be noninferior within `-0.01` absolute, false refusal will not increase by more than `0.01`,
and no recorded trust state will regress.

## Frozen inputs

1. Query file: `docs/preregistrations/2026-08-17-memory-queries.json`.
2. Query count: 50.
3. Query file SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
4. Tenant: `memory`.
5. Embedder: `voyage:voyage-4`.
6. Retrieval profile: `fast`.
7. Result size: `k=5`.
8. Reasoning budget: `max_steps=12`, `max_graph_nodes=32`, `max_evidence_tokens=2048`.
9. The active VPS2 generation is selected once immediately before the run and pinned for both
   arms with `RECALL_BENCHMARK_PIN=1` and `RECALL_PINNED_GENERATION_ID`. A generation change
   invalidates the run.
10. Query order, process settings, host, database endpoint, and environment remain fixed between
    arms.

## Arms

1. `off`: `graph_expansion=off`.
2. `one_hop`: `graph_expansion=one_hop`, `RECALL_GRAPH_PRECISION_VARIANT=combined`, relation
   control `none`.

The legacy `RECALL_GRAPH_COSINE_MARGIN` value is supplied only for runner compatibility. It is
inert in the candidate policy and is not an admission threshold.

## Repetition and pairing

1. Start one fresh server process per arm with the pinned generation.
2. Run one unscored warmup pass over all 50 queries.
3. Run five recorded passes over all 50 queries, preserving order.
4. Pair treatment and control by query identity and recorded pass.
5. Collapse each arm to one value per query by averaging its five recorded pass values before
   computing the primary paired delta. Repeated passes are not treated as 250 independent
   questions.
6. Keep refused, empty, slow, and failed requests in the artifact and classify them separately.

## Metrics and decision rule

The primary metric is answerable query hit at five, where a hit is present when at least one
trusted evidence `chunk_id` is in that query's `relevant_ids`. The primary population excludes
queries marked `answerable: false` and queries without labelled `relevant_ids`.

The secondary retrieval metric is MRR, using the reciprocal rank of the first trusted evidence
chunk in `relevant_ids`, or zero when no relevant chunk is present. I will also report paired
false refusal rate, empty trusted evidence rate, trust state counts, refusal reason counts, and
evidence set changes.

The candidate passes the quality gate only if all conditions hold:

1. Mean answerable hit at five improves by at least `0.02` absolute.
2. The paired bootstrap 95 percent interval for the hit at five delta has a lower bound above
   zero. Bootstrap uses 10,000 resamples of the 50 query level pairs with seed `20260910`.
3. MRR is noninferior within `-0.01` absolute.
4. False refusal delta is at most `+0.01` and no trust state regresses.
5. No query population with at least five labelled answerable queries regresses by more than
   `0.03` hit at five.

The pre registered prediction is supported when the observed hit at five delta is within
`[+0.02, +0.08]` and the other predicted noninferiority conditions hold. Failure to meet the
prediction is reported separately from the quality gate. No rollout recommendation is made from
a quality only result when the availability or latency gates are not satisfied.

## Statistical analysis

All quality calculations use the query as the unit. I will report nearest rank p50 and p95 for
latency fields only, paired bootstrap 95 percent intervals for quality deltas, and a paired
sign flip permutation p value with seed `20260911`. The raw artifact is immutable and the quality
summary is generated only after both arms finish.

The evidence assembly mode does not run an answer generator, so unsupported claim rate and answer
correctness are not measured by this run and will not be inferred from retrieval metrics.

## Reproduction

The runner is:

```powershell
.venv/Scripts/python.exe scripts/run_live_graph_performance_attribution.py `
  --query-set docs/preregistrations/2026-08-17-memory-queries.json `
  --generation-id <selected-immediately-before-run> `
  --output <immutable-quality-artifact>.json `
  --passes 5 --warmup-passes 1 --variant combined --control none --timeout 240
```

The preregistration is not edited after the first live request. Results belong in a separate
artifact and result document.
