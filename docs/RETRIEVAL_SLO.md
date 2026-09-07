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

The measured p95 and p99 at offered concurrency 1, 2, 4, 8, and 12 were:

| Offered concurrency | p95 | p99 | Peak RSS | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1,108 ms | 1,132 ms | 713 MiB | within SLO |
| 2 | 1,424 ms | 1,665 ms | 857 MiB | within SLO |
| 4 | 1,308 ms | 1,353 ms | 878 MiB | within SLO |
| 8 | 2,310 ms | 2,361 ms | 939 MiB | p95 breach |
| 12 | 2,722 ms | 2,806 ms | 988 MiB | p95 and p99 breach |

The benchmark artifact is
`benchmarks/results/performance_baseline_20260907T202202Z.json`. It is supplementary evidence
from a local production-shaped workload, not a promotion certification for an idle reference host.

## Alerts

The service already emits the following metric families through `recall_stats` and its exporter:

* `recall_retrieval_total_ms{profile=\"quality\"}`
* `recall_retrieval_rejected_total{profile=\"quality\",reason=...}`
* `recall_retrieval_failed_total{profile=\"quality\"}`
* `recall_retrieval_budget_exceeded_total{profile=\"quality\"}`

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
