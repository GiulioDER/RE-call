# Atomic fact production shadow runtime isolation

Status: preregistered after the remediation shadow stopped and before implementation or live
measurement.

## Purpose

Measure the confirmed atomic rescue production path without observer overhead or BLAS
oversubscription. This remains an off-by-default shadow and does not authorize active serving.

## Frozen diagnosis

The remediation established exact same-vector full-sort parity for 96 sequential and 256 concurrent
requests, exact internal public-result immutability, zero concurrent errors, and complete fail-open
and rollover safety. It stopped on historical receipts and latency.

The live process used 12 OpenBLAS threads for each matrix-vector operation. A non-pool isolated
diagnostic with one BLAS thread measured the original matrix multiplication at p99 2.472 ms
sequential and 10.088 ms with eight workers. The benchmark-only full sort also ran inside every
timed request and raised the total stage cost. Neither issue is part of the serving mechanism.

## Frozen implementation

1. Restore the original `matrix @ query` scoring kernel used by the release confirmation.
2. Launch atomic shadow processes with `OPENBLAS_NUM_THREADS=1`. The runner must verify that value
   before accepting measurements.
3. Run correctness and performance in separate fresh processes. The correctness process enables the
   private expectation receipt and same-vector full-sort reference. Its latency is not a production
   cost measurement.
4. The performance process disables both the historical receipt and full-sort reference. It retains
   only the production selector, bounded diagnostics, and benchmark-only internal public-result
   immutability boolean.
5. Keep the artifact, frozen generation, query order, parent exclusion, deterministic tie break,
   scope skips, and failure behavior unchanged.

Historical identity and score parity remain diagnostics only. They compare different live query
embeddings and numerically different BLAS reductions. The same-vector full-sort result is the exact
selector correctness invariant.

## Frozen checks

Before live measurement, tests must prove that the command sets one OpenBLAS thread only when atomic
shadow is enabled, the original matrix kernel is used, benchmark reference fields remain absent in
the performance process, and correctness reference fields remain exact in the correctness process.
The existing scope, cache, artifact, fail-open, redaction, and cost-surface tests remain green.

## Frozen live measurement

Use fresh MCP processes from one committed source tree pinned to generation
`gen_dff506e12f494965af9f109671a99e63`.

First run the correctness process across all 96 rows with the private receipt and same-vector
full-sort reference enabled. Then start a fresh performance process, warm the artifact once, replay
all 96 rows sequentially, and replay 256 requests in batches of eight. Run malformed, missing,
stale-generation, and active-generation rollover probes in fresh processes.

## Frozen gates

Return `PASS_ATOMIC_PRODUCTION_SHADOW_RUNTIME_ISOLATION` only if:

1. all 96 correctness selections have exact same-vector parent and score parity;
2. all 96 performance requests report internal public-result immutability and status ok;
3. performance selector p95 is at most 10 ms and p99 is at most 25 ms;
4. performance total shadow-stage p95 is at most 15 ms and p99 is at most 30 ms;
5. the 256-request performance replay has zero errors, exact internal public-result immutability,
   and selector p99 at most 40 ms;
6. cold artifact load, resident memory, and disk size retain their prior passing gates;
7. every malformed, stale, and rollover probe passes;
8. the measured process reports one OpenBLAS thread;
9. no request adds an embedding or database retrieval;
10. the active generation and production serving route remain unchanged.

Otherwise return `STOP_ATOMIC_PRODUCTION_SHADOW_RUNTIME_ISOLATION`. Do not alter the quality
mechanism or tune a retrieval threshold on these rows.

## Path after a pass

A pass authorizes the separate active-serving preregistration already required by the original
shadow protocol. Active work must build against the then-current certified generation, integrate
before fusion and trust evaluation, preserve scopes, and prove generation promotion and rollback.
