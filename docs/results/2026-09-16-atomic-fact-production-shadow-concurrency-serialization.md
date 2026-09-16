# Atomic fact production shadow concurrency serialization

Decision: `PASS_ATOMIC_PRODUCTION_SHADOW_CONCURRENCY_SERIALIZATION`.

All frozen gates passed. The exact selector matched the same-vector full-sort reference on all 96
correctness rows. The clean production path preserved every public result and measured selector p95
4.251 ms, p99 6.529 ms, total stage p95 6.151 ms, and total stage p99 8.175 ms.

With eight workers and 256 requests, there were zero errors and zero public immutability failures.
Selector p95 was 13.875 ms, p99 was 26.926 ms, and maximum was 39.724 ms. One BLAS thread was active
in every measured request, and benchmark reference fields were absent from every performance
request.

Artifact size, cold load, resident memory, missing artifact, wrong generation, wrong digest,
truncated artifact, process survival, and active-generation rollover checks all passed. Production
routing and the active generation did not change.

This pass authorizes a separately preregistered active-serving experiment. It does not itself
authorize activation.
