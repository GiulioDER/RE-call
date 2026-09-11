# Pre registration: live graph performance attribution

Date: 2026-09-10

Status: locked before the live measurement

## Question

Does the deployed graph serving path reduce warm graph cost through projection caching and
bounded reuse, while preserving the quality and trust behavior of graph off retrieval?

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
10. Query order, process settings, host, and database endpoint remain fixed between arms.

## Arms

1. `off`: `graph_expansion=off`.
2. `one_hop`: `graph_expansion=one_hop` with the deployed combined precision policy.
3. `directional_coverage`: an additional diagnostic arm using the declared directional control.
   It exists only to ensure that candidate fetch, cosine rescoring, trust reevaluation, cache,
   and adjacency spans have positive observations when the production combined policy selectively
   refuses expansion. It cannot pass or fail the production gate.

The primary paired comparison is `off` versus `one_hop`. The directional coverage arm is not a
replacement for it.

## Repetition and cache protocol

1. Start one fresh server process per arm with the pinned generation.
2. Run one unscored warmup pass over all 50 queries.
3. Run five recorded passes over all 50 queries, preserving order.
4. For the one hop arm, the first graph eligible request after process start is the projection
   cache miss population. Recorded passes after the first eligible pass are the warm population.
5. A warm p95 is valid only when at least 100 graph eligible warm requests have complete required
   spans. Otherwise the performance gate is `PENDING`, not green.
6. Run a separate eight request same key concurrency probe to measure single flight. Its latency
   is reported separately and is not mixed into the sequential p95.

## Locked primary gate

The proposed 50 percent reduction is defined precisely as:

`warm one hop graph stage p95 <= 0.50 * cold one hop graph stage p95`.

This denominator is the cold one hop graph stage, not graph off. Graph off does not execute a
graph stage, so using it as the denominator would turn absence of work into a false graph
optimization claim. Graph off remains the required baseline for total request latency and quality.

The primary gate passes only when all of the following hold:

1. The warm graph stage p95 is at most 50 percent of the cold graph stage p95.
2. Overall quality is noninferior to graph off within 0.01 absolute.
3. No corrected query class regresses by more than 0.03 absolute.
4. False refusal does not increase by more than 0.01 absolute.
5. Unsupported claim rate does not increase at all.
6. Every required instrumentation field is present and finite for every scored request.

The existing serving availability guard remains secondary: one hop total p95 must not exceed
twice the paired graph off total p95 at the same configuration.

## Required attribution fields

Every scored request must record the following fields, with milliseconds unless stated otherwise:

1. Baseline retrieval latency.
2. Graph readiness check latency and readiness result.
3. Projection cache hit, miss, and single flight wait state.
4. Adjacency construction latency and cache state.
5. Query embedding latency, including whether the baseline vector was reused.
6. Candidate fetch latency, candidate count, and returned byte count.
7. Cosine rescoring latency and scored candidate count.
8. Trust reevaluation latency and reevaluated candidate count.
9. Planner execution latency, operation count, and budget result.
10. Total server latency and client observed latency.
11. Database statement count and transferred bytes.
12. Cache hits, cache misses, hit rate, single flight owners, and single flight waiters.

Database transferred bytes means driver observable result and parameter bytes. If the driver cannot
expose wire bytes, the artifact must report application payload bytes in a separate field and mark
wire bytes unavailable. It must not silently substitute one for the other.

## Quality comparison

Quality is paired by query identity. The report includes trusted evidence set, hit at five,
abstention state, trust state, refusal reason, graph additions, citation set, unsupported claim
rate, and answer correctness where an answer provider is enabled. The evidence assembly run is
the latency comparison because it avoids generator variance. The planner run records planner
execution as a separate attribution series and is not mixed into the evidence assembly p95.

## Analysis rules

1. Report nearest rank p50 and p95 for every latency field.
2. Report counts, rates, and missing field counts for every categorical field.
3. Use the query as the unit of quality analysis, not individual chunks.
4. Report paired bootstrap intervals and a paired permutation test for quality deltas.
5. Do not remove slow, refused, or failed requests from total latency. Classify them separately.
6. Do not claim performance improvement from a single sample or from a warmup request.

## Reproduction commands

The query file digest is rechecked with:

```powershell
Get-FileHash -Algorithm SHA256 docs/preregistrations/2026-08-17-memory-queries.json
```

The live runner must use the pinned generation and emit one immutable raw artifact per arm. The
artifact must include the checkout commit, imported `recall_mcp/service.py` SHA256, generation id,
policy fingerprint, query set digest, host load, and all required attribution fields before the
quality or latency summary is generated.

The preregistration is not edited after the first live request. Results belong in a separate
artifact.
