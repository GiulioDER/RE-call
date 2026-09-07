# Supplementary quality-profile load test

This protocol supplements the frozen retrieval performance baseline without changing its matrix.
It measures the shipped `quality` profile at the point where its bounded admission queue saturates.

## Fixed workload

* fixture: `benchmarks/fixtures/performance_baseline.json`
* profile: `quality`, `k=5`, `candidate_k=20`
* embedding: the pinned local `bge-small` profile used by the baseline
* reranker: the pinned local `cross-encoder/ms-marco-MiniLM-L-6-v2` tree
* offered concurrency: `1,2,4,8,12`
* sampling: one cold request, 10 warmup requests, then 50 measured warm requests per level
* database: a disposable migrated table and `benchmark` tenant; never a shared or production table

The run records every request, including admission refusals, and reports wall latency for all
requests and successful requests separately. It also records process RSS, admission wait, stage
timings, error reasons, and the profile's running and queued capacity.

## SLO and capacity decision rule

The SLO is for served warm requests, not shed requests. Set the p95 and p99 ceilings after the run
from the highest offered concurrency that remains within the measured latency knee, with a small
round-number headroom. The first offered level that violates either ceiling is the load boundary,
not a reason to enlarge the queue. A capacity alert must cover both latency and explicit shedding.

The result artifact is supplementary evidence. It does not certify the promotion gate, which still
requires a separately certified idle-reference-host measurement.
