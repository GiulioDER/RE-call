# Pre registration: graph quality benefit

Date: 2026-09-10

Status: locked before the live measurement

## Question

Does the deployed one hop graph path improve retrieval evidence quality, and is the benefit
concentrated in queries whose target is backed by an authored graph relation?

## Frozen inputs

1. Query file: `docs/preregistrations/2026-09-10-graph-quality-queries.json`.
2. Query count: 50, consisting of 30 `graph_relation` cases and 20 `ordinary_control` cases.
3. Query file SHA256: `59dc6a2f258e7bb331124fd30239d34f2c0776eb30a08c41d34d9e1b1aa8f80a`.
4. Graph relation cases were selected deterministically from the first 30 authored `references`
   relations between `recall/` entities in the active VPS2 `memory` generation. Their expected
   relevant evidence is the relation evidence chunk recorded in the graph database.
5. Ordinary controls are the first 20 answerable cases from the previously frozen memory query
   set. Their expected evidence remains the existing `source:ordinal` labels.
6. Tenant: `memory`.
7. Embedder: `voyage:voyage-4`.
8. Retrieval profile: `fast`.
9. Result size: `k=5`.
10. Reasoning budget: `max_steps=12`, `max_graph_nodes=32`, `max_evidence_tokens=2048`.
11. Generation: `gen_b02a44a99917424ba3bd8011280e3712`.
12. Corpus fingerprint: `958147ad6f3a7269940077954449772e2d1cf98191c2594d507e5d66d8f8f3b2`.
13. Pipeline fingerprint: `77c918cd93f9200b36e505ae874d49a1949f04a8a85ce5b8b72a8c135b472db7`.
14. Query order, host, database endpoint, process settings, and serving checkout remain fixed.

## Arms

1. `off`: `graph_expansion=off`.
2. `one_hop`: `graph_expansion=one_hop` with the deployed combined precision policy.

Both arms use the same pinned generation and query order. The one hop arm is the production
serving path under test, including its selective gate. A refusal is retained as an observed
outcome and is not discarded.

## Repetition protocol

1. Start one fresh MCP server process for each arm.
2. Run one unscored warmup pass over all 50 queries.
3. Run five recorded passes over all 50 queries, in the frozen order.
4. Preserve every response, refusal, trust state, graph diagnostic, and latency field.
5. Do not edit the query file or this preregistration after the first measured request.

## Primary quality metric and gate

For each query and pass, an evidence hit means that at least one of the top five trusted evidence
items matches an expected relevant id. A graph relation expected id matches `chunk_id`. An
ordinary control expected id matches the item `source` plus `ordinal` in `source:ordinal` form.
The query is the unit of analysis. Report paired per query means across the five recorded passes.

The primary quality gate passes only if both conditions hold:

1. On the 30 graph relation cases, one hop improves mean hit@5 by at least 0.05 absolute versus
   graph off.
2. On the 20 ordinary controls, one hop is noninferior within -0.02 absolute hit@5 versus graph
   off.

The quality gate is `PENDING` if the graph relation cases do not produce enough admitted graph
expansions to exercise the mechanism. The report must state the admitted case count and show
results for the full graph relation subset regardless.

## Secondary quality measures

Report these paired measures separately for graph relation cases, ordinary controls, and all 50:

1. Mean hit@5 and nearest rank p50 and p95 hit@5 by query.
2. Mean reciprocal rank of the first matching evidence item.
3. Trusted evidence set size and overlap between arms.
4. New trusted evidence ids introduced by graph expansion.
5. Graph candidate discovery, admission, refusal, and gate reason.
6. Trust state, outcome, and refusal reason.
7. Per query delta, identifying which cases gain, lose, or stay unchanged.
8. Server and client latency, database calls, transferred bytes, and cache behavior as secondary
   performance attribution.

Because the answer provider is disabled for this run, answer correctness, citation correctness,
and unsupported claim rate are not estimable. The experiment measures retrieval evidence quality,
trust behavior, and mechanism attribution only.

## Interpretation limits

The graph relation subset is a targeted mechanism coverage set, not a production traffic estimate.
It can show whether graph expansion helps when an authored relation is relevant. It cannot estimate
the fraction of all user queries that will benefit. The ordinary controls test collateral effects,
not production prevalence.

## Analysis rules

1. Keep slow, refused, and failed requests in the denominator and classify them separately.
2. Use paired query identity as the unit of comparison.
3. Report the raw per query table before aggregate conclusions.
4. Report bootstrap intervals and a paired permutation test for hit@5 and MRR deltas.
5. Do not claim a quality benefit from latency alone or from graph diagnostics without a relevant
   evidence hit.

## Reproduction commands

Freeze verification:

```powershell
Get-FileHash -Algorithm SHA256 docs/preregistrations/2026-09-10-graph-quality-queries.json
```

Measurement command:

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-quality-queries.json --output <immutable-quality-artifact>.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined --control none
```

Results belong in a separate report. This preregistration is not edited after the first live
request.
