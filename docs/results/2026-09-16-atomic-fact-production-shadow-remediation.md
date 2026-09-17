# Atomic fact production shadow remediation

Decision: `STOP_ATOMIC_PRODUCTION_SHADOW_REMEDIATION`.

The remediation passed every same-request correctness and safety invariant. All 96 sequential and
all 256 concurrent fast selections matched the same-vector full-sort reference exactly. All public
results were internally unchanged, there were zero concurrent errors, and every artifact, failure,
and rollover probe passed.

The run stopped because one historical parent identity differed, the benchmark full sort raised
total stage latency, and concurrent selector p99 was 100.666 ms. Historical exact scores matched
only 18 of 96, confirming that scores from an earlier live query embedding are not stable receipts.

The process used the NumPy OpenBLAS default of 12 threads. A subsequent non-pool diagnostic on the
same matrix measured the original matrix kernel with `OPENBLAS_NUM_THREADS=1`: sequential p99 was
2.472 ms and eight-worker p99 was 10.088 ms. The one-thread and 12-thread score arrays were not byte
identical, so historical identities cannot replace a same-vector reference gate.

The second stop remains valid. A separate preregistration freezes a final shadow with one BLAS
thread and separates full-sort correctness observation from production-cost timing.
