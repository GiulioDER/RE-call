# Retrieval serving SLO

## Quality profile decision

The initial measured policy is for the shipped `quality` profile with `k=5`, `candidate_k=20`,
the pinned local `bge-small` embedder, and the pinned local `ms-marco-MiniLM-L-6-v2` reranker.
The workload used the immutable performance fixture, 10 warmup requests, and 50 measured warm
requests at each offered concurrency.

| Policy | Value |
| --- | ---: |
| Served warm request p95 | ≤ 2,000 ms |
| Served warm request p99 | ≤ 2,200 ms |
| Recommended offered concurrency | ≤ 4 |
| Running concurrency | 2 |
| Queue capacity | 8 |
| Maximum process RSS alert | 1.25 GiB |
| Error rate alert | > 1% over 5 minutes |

The committed benchmark contains 50 warm requests at offered concurrency 1, 2, 4, and 8.
The table below is derived from its raw request rows using the preregistered nearest rank
percentile rule. It does not claim a measurement at offered concurrency 12.

| Offered concurrency | p95 | p99 | Peak RSS | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1,582.4 ms | 1,838.8 ms | 712.1 MiB | within SLO |
| 2 | 1,226.9 ms | 1,359.7 ms | 856.0 MiB | within SLO |
| 4 | 1,854.9 ms | 1,928.0 ms | 881.2 MiB | within SLO |
| 8 | 2,100.0 ms | 2,194.5 ms | 934.2 MiB | p95 breach |

The benchmark artifact is
`benchmarks/results/performance_baseline_20260906T111559Z.json`. It is supplementary evidence
from a local production-shaped workload, not a promotion certification for an idle reference host.

## Alerts

The service already emits the following metric families through `recall_stats`; deployments may
forward the same snapshot to their alerting backend:

* `recall_retrieval_total_ms{profile="quality"}`
* `recall_retrieval_rejected_total{profile="quality",reason=...}`
* `recall_retrieval_failed_total{profile="quality"}`
* `recall_retrieval_budget_exceeded_total{profile="quality"}`

Page on any of these conditions over a five minute window:

1. served p95 above 2,000 ms
2. served p99 above 2,200 ms
3. rejected or failed requests above 1% of attempts
4. process RSS above 1.25 GiB
5. any sustained `queue_full` or `budget_exhausted` refusal while offered concurrency is at or
   below 4

The first response is to reduce offered concurrency or route new work to the fast profile. Do not
increase queue depth as the first mitigation, because the load test shows queueing increases tail
latency before it improves completed throughput.

These values are deployment policy and must be remeasured when the corpus scale, embedder,
reranker artifact, hardware, or database topology changes.
