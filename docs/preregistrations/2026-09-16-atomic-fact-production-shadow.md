# Atomic fact generation-bound production shadow

Status: preregistered after the 96 row release confirmation passed and before implementing or
measuring the production shadow.

## Purpose

Prove that the confirmed dense-five plus one atomic rescue can run inside the real MCP retrieval
process with bounded load, memory, and request overhead while leaving every served result
unchanged. This protocol authorizes an off-by-default shadow only. It does not authorize active
candidate replacement.

## Frozen mechanism

Build one immutable atomic artifact for certified generation
`gen_dff506e12f494965af9f109671a99e63`, calibration
`cal_6171177aeb614d288baa4e28600caca1`, pipeline
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, corpus
`522e9d1c506142d8a6a5a1f5be9073ff9409132d356e29f61fbf8e7d6d2044a0`, embedding profile
`voyage-context-4-v1`, dimension 1,024, and 11,385 ordinary chunks.

The artifact contains normalized float32 atomic vectors and deterministic view metadata bound to
the generation's parent chunk identifiers, source identifiers, parent ordinals, and view ordinals.
Its metadata records schema version, generation, calibration, pipeline, corpus, profile, dimension,
view count, parent count, matrix SHA256, metadata SHA256, source commit, and construction timestamp.
The builder must verify manifest bytes and parent mapping exactly as the release-confirmation
runner did. Build only on VPS2 under the shared embed lock and frozen resource policy.

Add these typed, documented settings:

1. `RECALL_ATOMIC_RESCUE_MODE`, allowed values `off` and `shadow`, default `off`.
2. `RECALL_ATOMIC_RESCUE_ARTIFACT`, the immutable artifact metadata path.
3. `RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE`, a deterministic fraction from zero through one,
   default zero.

Any `active` value is refused under this protocol. With mode off, no artifact is opened, no matrix
is allocated, no trace is captured, and the existing retrieval cost surface remains unchanged.

For a sampled unscoped request, reuse the existing query vector and already fetched dense
candidate trace. Load and validate the artifact once per process, encode parent identities once,
mask the first five dense parents, and select the exact highest-scoring remaining atomic parent.
Do not embed, query the database again, rerank, trust-evaluate, or alter any serving candidate.

Source-scoped or security-scoped requests are skipped with a bounded reason code until an active
design has a measured scope-preserving mechanism. Artifact, lineage, load, shape, finite-value, or
selection failures are caught by the shadow boundary and must not fail or change baseline search.

Expose only bounded status fields and aggregate metrics: sampled, skipped, lineage error,
artifact error, computation error, selected parent already equal to dense rank six, selector
latency, artifact load latency, and process resident-memory delta. Do not expose vectors, query
text, parent text, source identifiers, chunk identifiers, or row-level candidate scores.

## Frozen implementation checks

Before live measurement, tests must prove:

1. mode off performs no file read, allocation, trace capture, metric increment, or result change;
2. deterministic sampling has exact zero and one boundaries;
3. artifact lineage, digest, shape, dimension, normalization, finite values, and metadata are
   validated before use;
4. one process loads one artifact once under concurrent first access;
5. exact masked maximum matches the full deterministic sort, including repeated parent views and
   score ties;
6. sampled shadow reuses the existing query vector and dense trace with no embedding or database
   call;
7. normal, source-scoped, security-scoped, malformed-artifact, stale-generation, and selection
   failure paths preserve the exact baseline response;
8. environment schema, generated environment documentation, MCP and in-process descriptions, and
   cost-surface tests remain consistent.

Every material test requires an assertion-level red proof against the baseline or a plausible
production mutation before green.

## Frozen live measurement

Run two fresh MCP processes from the same commit and certified generation. The control uses mode
off. The candidate uses shadow mode, the frozen artifact, and sample rate one. Replay the private
96 row release-confirmation pool in the same order through both processes, reusing no response
between arms. Raw queries and row-level results remain private.

Then run an eight-worker, 256-request concurrency replay against the shadow process. Run explicit
failure probes with a wrong generation, wrong matrix digest, truncated matrix, and missing artifact.
Run a rollover probe that changes the process's active generation binding while leaving the old
artifact configured. These probes must return the same public baseline results while recording a
bounded shadow error.

Measure artifact build time, artifact bytes, one-time load time, resident-memory delta, selector
p50, p95, p99 and maximum, total shadow-stage p50, p95, p99 and maximum, public response parity,
candidate parity against the release-confirmation private artifact, error counts, and process
survival.

## Frozen gates

Return `PASS_ATOMIC_PRODUCTION_SHADOW` only if all conditions hold:

1. all 96 sequential public responses have exact control-versus-shadow parity;
2. all 96 selected atomic parent identities and scores match the release-confirmation artifact;
3. selector p95 is at most 10 ms and p99 is at most 25 ms;
4. total shadow-stage p95 is at most 15 ms and p99 is at most 30 ms;
5. one-time artifact load takes at most 2,000 ms and increases resident memory by at most 128 MiB;
6. the artifact is at most 64 MiB on disk;
7. the 256-request concurrent replay has zero baseline or shadow errors, exact public parity, and
   selector p99 at most 40 ms;
8. every malformed, missing, stale-generation, and rollover probe preserves the baseline response,
   records the expected bounded error, and leaves the process healthy for a subsequent valid
   request;
9. no request embeds an atomic document, performs an extra query embedding, or executes an extra
   database retrieval;
10. the active production generation and serving route remain unchanged.

Otherwise return `STOP_ATOMIC_PRODUCTION_SHADOW`. Do not tune the selector or quality mechanism on
the 96 confirmation rows.

## Path after a pass

A passing shadow authorizes a separate preregistration for active dense-leg replacement. That
protocol must measure the actual fused and trust-evaluated served result, preserve source and
security scopes, bind artifacts to generation promotion and rollback, and ship behind a default-off
serving flag. Production activation before the next release requires both that gate and an explicit
rollback verification.

## Prediction

I predict the shadow will pass. The release confirmation measured selector p95 3.697 ms and p99
7.510 ms over 6,322 views, leaving margin for artifact checks and metrics. A float32 matrix of
6,322 by 1,024 values is about 24.7 MiB before metadata and remains well below the 64 MiB disk and
128 MiB resident-memory gates.
