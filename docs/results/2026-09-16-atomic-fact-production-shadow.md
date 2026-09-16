# Atomic fact production shadow

Decision: `STOP_ATOMIC_PRODUCTION_SHADOW`.

The first production shadow preserved all 96 sequential public responses and reproduced all 96
selected parent identities. Artifact integrity, size, load, resident memory, malformed artifact,
stale lineage, process survival, and generation rollover checks passed. The active production route
did not change.

The run stopped for three reasons:

1. Exact historical score parity was 88 of 96 even though parent identity parity was 96 of 96.
   The historical score was produced by an earlier live query embedding. The selected parent is the
   functional invariant because the rescue is appended at a fixed sixth rank.
2. The selector missed its latency gates. Sequential p95 was 34.922 ms and p99 was 90.048 ms.
3. The concurrent replay compared fresh live requests with an earlier sequential response. It saw
   29 public response differences and 21 historical identity differences despite zero request
   errors. That comparison confounded the feature with live query embedding and retrieval drift.

An isolated VPS2 diagnostic used the same frozen matrix with a non-pool vector. The existing matrix
multiplication measured p99 28.942 ms sequential and 116.176 ms with eight workers. The equivalent
single-pass NumPy contraction measured p99 5.102 ms sequential and 13.667 ms with eight workers.
This diagnostic selected the performance repair only. It did not measure retrieval quality.

The frozen result remains a valid stop. A separate corrective preregistration defines the next run
before the selector or runner changes.
