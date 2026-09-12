# Preregistration: graph candidate metadata first fetch

Date: 2026-09-12

Status: locked before the live performance remeasurement

## Objective

Reduce graph one hop database payload by ranking bounded candidate IDs before fetching passage text.
The graph first path should fetch text only for candidates that can fill the final graph context;
metadata and stored vectors remain available for temporal, cosine, and structural checks. Requests
with an explicit security policy retain the existing full text path until an authorization specific
metadata contract is measured separately.

## Frozen design

1. Query set: the frozen 50 query set used by the live graph performance artifact.
2. Arms: graph off and graph one hop, paired by query identity.
3. The graph policy, generation, embedder, budgets, and database remain unchanged.
4. The control is the pre change live attribution artifact measured on 2026-09-12.
5. The treatment records candidate payload bytes, fetched candidate count, database calls, total
   latency, and the existing quality fields.

## Decision rule

The change is accepted for a quality replay only if:

1. One hop candidate payload bytes p95 fall by at least 50 percent.
2. One hop candidate fetched count p95 falls by at least 50 percent.
3. Total one hop latency p95 does not regress.
4. No graph readiness, refusal, trust, or schema compatibility regression occurs.

A separate answer quality replay must then confirm that valid envelopes, citation validity, gold
citation recall, and correct abstention do not regress by more than two percentage points.

## Reproduction

```powershell
python scripts/run_live_graph_performance_attribution.py `
  --query-set docs/preregistrations/2026-08-17-memory-queries.json `
  --output docs/results/2026-09-12-live-graph-performance-attribution-candidate-fetch.json
```
