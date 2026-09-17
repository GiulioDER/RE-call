# Atomic fact production shadow remediation

Status: preregistered after the first production shadow stopped and before implementing or measuring
the remediation.

## Purpose

Repair the measured production kernel and invalid concurrency comparator without changing the
confirmed dense-five plus one atomic rescue quality mechanism. This protocol still authorizes only
an off-by-default shadow. It does not authorize active serving.

## Frozen diagnosis

The first shadow preserved 96 of 96 public responses and reproduced 96 of 96 selected parent
identities. It stopped because exact historical scores drifted on eight rows, selector latency
missed its gates, and the concurrent runner compared fresh live requests with an earlier sequential
response.

The selector score is not used to choose a rank after the parent is selected. The rescue parent is
always appended at dense rank six. Therefore historical score equality is not a functional serving
invariant. The exact same-vector score from an independent full-sort reference remains an invariant.

An exploratory VPS2 kernel diagnostic on the frozen matrix and a non-pool vector measured the
existing matrix multiplication at p99 28.942 ms sequential and 116.176 ms with eight workers. A
single-pass `numpy.einsum` contraction with optimization disabled measured p99 5.102 ms sequential
and 13.667 ms with eight workers. No quality row was used in that diagnostic.

## Frozen remediation

1. Replace only the atomic matrix score kernel with
   `numpy.einsum("ij,j->i", matrix, query, optimize=False)`.
2. Add an independent benchmark-only full-sort reference over the same current query vector,
   exclusion set, matrix, and deterministic tie break. Report only identity and exact score parity
   booleans.
3. Record an internal public-result immutability boolean before and after shadow computation. Do not
   compare one live request with a different live request.
4. Warm and validate the immutable artifact before measured steady-state requests. Continue to
   record cold load latency and resident memory separately.
5. Keep the artifact, frozen generation, 96-row order, deterministic selection, security skips,
   failure behavior, and all public response semantics unchanged.

Historical score parity against the earlier release-confirmation receipt remains informational.
Historical parent identity parity remains a required sequential gate.

## Frozen implementation checks

Tests must prove that the new kernel has exact parent and score parity with a full deterministic sort
for ordinary, repeated-parent, exclusion, and tie cases. A plausible return to matrix multiplication
must fail the kernel-selection test. Benchmark-only reference and immutability booleans must remain
bounded and absent outside a benchmark-pinned process. Existing artifact, scope, fail-open, cache,
and cost-surface tests remain green.

## Frozen live measurement

Use the same commit for two fresh MCP processes and pin both to generation
`gen_dff506e12f494965af9f109671a99e63`. Warm the shadow artifact once before collecting steady-state
latency. Replay the same private 96 rows sequentially. Then replay 256 requests in batches of eight
against one shadow process.

For every measured request, compare the fast selector with the independent full-sort reference using
that request's current query vector. Preserve only aggregate counts outside the private boundary.
Repeat the missing, wrong-generation, wrong-digest, truncated, and active-generation rollover probes.

## Frozen gates

Return `PASS_ATOMIC_PRODUCTION_SHADOW_REMEDIATION` only if all conditions hold:

1. all 96 sequential public results are internally unchanged by shadow computation;
2. all 96 sequential selected parent identities match the release-confirmation receipt;
3. all 96 sequential fast selections have exact parent and score parity with the same-vector
   full-sort reference;
4. steady-state selector p95 is at most 10 ms and p99 is at most 25 ms;
5. steady-state total shadow-stage p95 is at most 15 ms and p99 is at most 30 ms;
6. cold artifact load is at most 2,000 ms, resident memory increase is at most 128 MiB, and artifact
   size is at most 64 MiB;
7. the 256-request replay has zero errors, exact same-vector reference parity, exact internal public
   immutability, and selector p99 at most 40 ms;
8. all malformed, stale, and rollover probes preserve the public result, record the expected bounded
   error, and leave the process healthy;
9. no request adds an embedding or database retrieval;
10. the active production generation and serving route remain unchanged.

Otherwise return `STOP_ATOMIC_PRODUCTION_SHADOW_REMEDIATION`. Do not alter the quality mechanism or
tune any threshold on the 96 rows.

## Path after a pass

A pass authorizes a separate active-serving preregistration. Active serving must still integrate the
candidate before fusion and trust evaluation, preserve scoped requests, bind artifact promotion and
rollback to generation lineage, and prove an explicit rollback before release.
