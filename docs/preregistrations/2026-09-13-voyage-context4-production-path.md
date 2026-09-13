# Pre registration: Voyage Context 4 production path comparison

Date: 2026-09-13

## Question

Does the production generation and query path preserve the retrieval gain previously observed for
Voyage Context 4, and does the existing Voyage reranker change the conclusion?

## Arms

The four arms use the same frozen LoCoMo question set, corpus bytes, chunker, candidate depth,
random seed, database schema, and evaluation code.

1. Voyage 4, production path, no reranker.
2. Voyage Context 4, production path, no reranker.
3. Voyage 4, production path, existing Voyage reranker.
4. Voyage Context 4, production path, existing Voyage reranker.

Voyage 4 uses the registered profile `voyage-4-v1` if that profile exists in the measured checkout.
Voyage Context 4 uses the registered profile `voyage-context-4-v1`. If the existing Voyage 4
profile has a different registered id, the report must record that exact id and explain the mapping.

## Primary outcome

The primary outcome is answerable hit at 5 over the 1,536 paired answerable questions. The
estimand is the paired treatment minus control difference for the two no reranker arms. The
reranker arms are a prespecified secondary comparison.

Secondary outcomes are hit at 1, 3, 10, and 20, category deltas, bootstrap 95 percent confidence
intervals over paired questions, rescues, regressions, net change, request count, provider error
count, retry count, wall time, and billed token metadata when returned by the provider.

## Prediction

The production path will retain a positive hit at 5 difference for Voyage Context 4 over Voyage 4,
with an expected difference near the prior benchmark result of plus 6.25 percentage points. The
95 percent paired bootstrap interval is expected to remain above zero. The gain may shrink after
the production path applies per source grouping and the same grouping contract to every arm.

The reranker is expected to improve both embedding arms, while preserving the direction of the
Context 4 comparison. A direction reversal or a confidence interval spanning zero is a failure to
confirm the prediction, not a reason to edit it.

## Frozen procedure

Use the benchmark question and corpus artifacts committed with the benchmark source revision. The
production path must be the registered profile resolver, manifest generation builder, query encoder,
and existing reranker implementation. No benchmark only indexer subclass or direct provider call is
allowed.

Use candidate depths 1, 3, 5, 10, and 20, with candidate_k 20. Record the source revision, manifest
digest, corpus fingerprint, pipeline fingerprint, profile id, profile fingerprint, generation id,
calibration id, query set digest, environment, provider SDK version, and every output artifact hash.

No result is reportable if any input chunk is dropped, reordered, truncated, assigned to a different
context group, or returned without an aligned vector. The run stops on any such violation.

## Resource and stop rules

Run no more than one embedding job on VPS2 at a time. Abort before measurement if the provider key,
SDK, database, or deployment route is not verified. Stop a run after three consecutive transient
provider failures, an unbounded retry, a request over the recorded provider limit, a vector width
mismatch, or a production generation write failure. Preserve the failed run artifacts and previous
generation identifier.

The benchmark is retrieval quality only. It does not authorize promotion, deployment, provider
configuration changes, deletion of an old generation, or a production cutover.

## Artifacts

The final report must include the preregistration path, commit, exact source revision, command,
frozen input digests, all four arm identities, generation and calibration identifiers, aggregate
metrics, confidence intervals, request and failure accounting, and the decision against the frozen
prediction.
