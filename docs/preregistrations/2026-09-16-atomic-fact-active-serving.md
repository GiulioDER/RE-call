# Atomic fact active serving

Status: preregistered after the production shadow passed and before active implementation,
current-generation artifact construction, or active measurement.

## Purpose

Prove that dense-five plus one atomic rescue improves the real fused and trust-evaluated serving
result on the current certified production generation, then deploy it behind an explicit default-off
flag with generation-bound rollback.

## Frozen production lineage

The active generation at registration is `gen_919991221e1045e69824baa1a9be4e30`, calibration
`cal_cf8477d2df95464a938a3bcceb097803`, pipeline
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus
`7e9e27ec7b5120476c3dd26a4f1a8d728cbc75ff4e117e5b1ff8b16a5b1ccd8c`.

The immediate rollback generation is `gen_dff506e12f494965af9f109671a99e63`. The artifact registry
must contain valid immutable artifacts for both generations before activation.

## Frozen active mechanism

1. Add `active` to `RECALL_ATOMIC_RESCUE_MODE`; the default remains `off`.
2. Add `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT`. Resolve only
   `<root>/<generation_id>/manifest.json`, validate containment, digests, calibration, pipeline,
   corpus, embedding profile, and dimension before use.
3. For an unscoped request with no security policy, select the exact best atomic parent outside the
   first five dense parents using the already computed query vector. Fetch that one parent by its
   generation-bound chunk identifier.
4. Insert the atomic parent at dense rank six and preserve the original rank six and later parents
   after it. Run ordinary fusion, reranking, gap evaluation, trust evaluation, entailment, and
   response truncation on the modified candidate pool.
5. Source, folder, facet, prefix, and security-scoped requests retain the baseline path until a
   separately measured scope-preserving artifact exists.
6. Missing, stale, malformed, unfetchable, or incompatible active artifacts fail closed. They must
   never silently serve the baseline while reporting active mode.
7. Record bounded counters and latency only. Never expose the selected candidate identity, score,
   text, source, query, or vector in diagnostics.
8. Atomic mode off opens no artifact, fetches no chunk, and preserves the existing cost surface.

Clarification recorded before implementation or measurement: if the selected atomic parent already
appears at dense rank six or later, move that parent to rank six and remove its later duplicate.
Every other later parent retains relative order. This is required to preserve the existing distinct
parent invariant; literal duplication would create an arm the confirmation experiment never tested.

## Frozen implementation checks

Tests must prove rank-six insertion, preservation of later dense parents, one bounded chunk fetch,
full fusion and trust consumption, off-path zero cost, all scoped bypasses, lineage refusal, missing
parent refusal, generation-root containment, one-thread selector serialization, and rollback to mode
off. Every material test requires an assertion-level red proof.

## Frozen current-generation measurements

Build the immutable artifact on VPS2 under the shared embedding lock and frozen resource policy.
Use the already frozen 96-query release pool only as a consumed release-regression set. Do not tune
the mechanism, exclusions, or thresholds on it.

First compare current-generation `dense6` with `dense5_atomic1` before fusion. The active lane may
continue only if exact-parent net gain is at least eight with at most one loss, and gold-source net
gain is at least five with at most one loss.

Then run fresh control and active MCP processes from the same commit and current generation. Replay
all 96 queries through the real fused and trust-evaluated retrieval-only surface. Measure exact gold
parent and gold source at served cutoffs one, three, and five, trust decisions, abstentions, public
lineage, latency, errors, and scoped parity. Raw rows remain private.

Active serving passes only if:

1. gold source at five has positive net gain with at most one loss;
2. exact gold parent at five has nonnegative net change with at most one loss;
3. no cutoff has more than one gold-source or exact-parent loss;
4. trust-state, refusal, and abstention changes are explained entirely by candidate changes and no
   trusted response is replaced by an untrusted one;
5. unscoped p95 added server latency is at most 20 ms and p99 at most 45 ms;
6. source and security scoped responses have exact baseline parity and no artifact access;
7. malformed, stale, missing-parent, and rollover probes fail closed while mode active;
8. the active generation and production route remain unchanged during measurement.

Otherwise return `STOP_ATOMIC_ACTIVE_SERVING` and do not deploy.

## Frozen deployment and rollback gate

After a measurement pass, deploy the exact measured commit and both generation artifacts. Keep mode
off for startup, verify import path, health, database readiness, and mode-off response parity, then
enable active mode and restart long-lived MCP services.

Verify new processes use the current artifact and one BLAS thread. Run production smoke queries,
scoped parity, metrics, and service health. Rollback is the environment change
`RECALL_ATOMIC_RESCUE_MODE=off` plus service restart; verify it restores exact baseline behavior.
Also verify a pinned process can serve the rollback generation from its own artifact. Do not change
the database generation merely to test application rollback.

Return `READY_FOR_RELEASE` only after active smoke and rollback rehearsal both pass. Otherwise leave
mode off and report the stop.
