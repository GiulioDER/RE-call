# Atomic fact production shadow runtime isolation

Decision: `STOP_ATOMIC_PRODUCTION_SHADOW_RUNTIME_ISOLATION`.

Every correctness, safety, resource, and sequential performance gate passed. The clean production
path measured selector p95 7.985 ms and p99 12.238 ms. Total shadow-stage p95 was 11.969 ms and p99
was 15.881 ms. All 96 same-vector correctness checks passed exactly.

The only failed gate was the eight-worker selector p99: 41.829 ms against a frozen maximum of
40 ms. Median was 3.573 ms, p95 was 23.578 ms, and maximum was 46.383 ms. There were zero request
errors, zero public immutability failures, one BLAS thread in every request, and no reference work
in the performance process.

The result is a stop and production remains unchanged. A separate preregistration targets only the
remaining overlapping-selector tail with a bounded critical section.
