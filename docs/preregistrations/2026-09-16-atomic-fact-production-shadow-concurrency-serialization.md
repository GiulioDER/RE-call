# Atomic fact production shadow concurrency serialization

Status: preregistered after runtime isolation missed only the concurrent p99 gate and before code or
measurement.

## Purpose

Bound the remaining concurrent selector tail without changing scores, exclusions, ranking, artifact,
or serving behavior.

## Frozen change

Add one process-local lock around the production selection critical section. At most one request may
normalize the query, score the matrix, apply exclusions, and select the deterministic winner at a
time. Artifact loading retains its separate existing single-flight lock. Benchmark full-sort work
must remain outside the production selection lock.

The current one-thread BLAS selector measured 3.573 ms median and 41.829 ms p99 with eight workers.
Serialization should replace CPU overlap with a queue whose expected upper bound is approximately
eight ordinary selector durations. This prediction uses only the already measured timing shape and
does not change the frozen 40 ms gate.

## Frozen checks

A concurrent test must prove that no two production selector critical sections overlap. Removing
the lock must make that test fail. All exact selection, scope, fail-open, artifact, redaction,
configuration, and optional-import tests remain green.

## Frozen live gate

Repeat the runtime-isolation protocol from one committed source tree against the same frozen
generation and artifacts. Return `PASS_ATOMIC_PRODUCTION_SHADOW_CONCURRENCY_SERIALIZATION` only if
every runtime-isolation gate passes unchanged, including eight-worker selector p99 at most 40 ms.
The active generation and serving route must remain unchanged.

Otherwise return `STOP_ATOMIC_PRODUCTION_SHADOW_CONCURRENCY_SERIALIZATION`. A pass authorizes the
separate active-serving preregistration, not direct activation.
