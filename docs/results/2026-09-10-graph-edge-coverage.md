# Graph edge coverage baseline

Measured 2026-09-10 UTC before any threshold change. The production baseline is the frozen
`graph-quality` run against generation `gen_b02a44a99917424ba3bd8011280e3712`.

## Metric definition

1. A category is a query population in the frozen query set.
2. Entities and authored relations are unique endpoint labels and relation identifiers declared
   by that category.
3. Seed coverage counts unique query cases with at least one recorded relation seed activation.
4. Candidate counts are sums over the five recorded one hop passes.
5. Gold recovery counts expected evidence newly added by graph expansion relative to the paired
   graph off row. Baseline retrieval hits are not graph recovery.

## Production baseline

`graph_relation`: 47 entities, 30 authored relations, 0 of 30 queries with a seed relation,
0 candidates discovered, 0 candidates admitted, and 0 gold evidence recovered.

`ordinary_control`: 0 entities, 0 authored relations, 1 of 20 queries with a seed relation,
40 candidates discovered, 10 candidates admitted, and 0 gold evidence recovered.

The graph relation category therefore has zero query edge coverage. Threshold tuning cannot affect
those cases until seed coverage is nonzero.

## Corpus graph inventory

The live projection endpoint reported 6,499 semantic entities and 4,364 authored semantic
relations. All 4,364 relations are `references`; `supports`, `contradicts`, `depends_on`, `caused`,
and `same_entity` each have zero authored relations. The graph is populated globally, but its
edges are not reached by the production graph relation queries.

## Relaxed activation diagnostic

The separate directional diagnostic had 5 of 5 queries with a seed relation, 679 candidates
discovered, 679 admitted, and one query with graph added gold evidence. This is evidence that the
underlying graph can activate under relaxed policy. It is not evidence for changing production
thresholds.

## Frozen inputs

1. Query set: `docs/preregistrations/2026-09-10-graph-quality-queries.json`
2. Query set SHA256: `59dc6a2f258e7bb331124fd30239d34f2c0776eb30a08c41d34d9e1b1aa8f80a`
3. Production artifact SHA256: `c86007255345b8405316da8edeb5ccb585808f081e2e7e8dff6cdc08d908bb39`
4. Projection endpoint: `recall_reasoning_projection(include_text=false)`

## Reproduction

Run the frozen production collection:

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-quality-queries.json --output $env:TEMP/recall-live-graph-quality-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined --control none
```

Then read each recorded `one_hop` payload's `diagnostics` fields:
`graph_relation_seed_activations`, `graph_candidates_discovered`,
`graph_relation_candidates_accepted`, and the paired `evidence_ids`.
