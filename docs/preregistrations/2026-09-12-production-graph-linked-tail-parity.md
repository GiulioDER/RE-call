# Preregistration: production graph linked tail parity

Date: 2026-09-12

Status: locked before implementation measurement

## Objective

Test whether the production graph first path becomes retrieval equivalent to the successful
LoCoMo selective policy when it uses graph topology to identify candidates already present in the
ordinary top 20, preserves direct ranks 1 through 8, and lets at most two linked candidates compete
for the final two positions in a fixed ten item context.

The experiment separates selection parity from real corpus graph coverage. Exact LoCoMo parity
tests whether production code implements the measured mechanism. The frozen production query set
tests whether the current memory graph contains enough useful linked tail candidates for that
mechanism to affect gold evidence retrieval.

## Change under test

Add an explicit graph first candidate mode with three values:

1. `outside_pool`: preserve the current behavior, where graph candidates already present in the
   raw retrieval pool are excluded.
2. `linked_tail`: consider only graph connected candidates present below the protected direct
   prefix in the raw retrieval pool. Reuse their existing query cosine and chunk payload.
3. `hybrid`: admit eligible linked tail candidates first, then allow candidates outside the raw
   pool to compete for any remaining graph allocation.

The default remains `outside_pool` until a result passes the gates below. Every mode keeps the
ordinary retrieval pool at 20, protects the first 8 direct results, returns at most 10 context
items, and admits at most 2 graph selected items. Selective admission uses calibrated query cosine
with margin 0.05 over the weakest item in the original direct ranks 9 and 10. Gold labels, answers,
and citation outcomes are not available to the selector.

## Frozen evaluation inputs

### LoCoMo parity population

1. Dataset: `locomo10.json`.
2. Population: all 1,536 answerable questions across all 10 conversations.
3. Embedder: `voyage:voyage-4`.
4. Retrieval candidate count: 20.
5. Protected direct count: 8.
6. Final context count: 10.
7. Maximum graph selected count: 2.
8. Admission margin: 0.05.
9. Reference artifact: `docs/results/2026-09-12-selective-graph-admission.json`.

### Production memory population

1. Query set: `docs/preregistrations/2026-08-17-memory-queries.json`.
2. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
3. Population: all 50 frozen queries.
4. Tenant: `memory`.
5. Embedder: `voyage:voyage-4`.
6. Generation: pin the certified generation reported immediately before the run and record its
   generation, pipeline, corpus, and calibration identities in the artifact.
7. One warmup pass and five recorded passes.

## Arms and controls

The LoCoMo comparison contains direct top 10, the existing selective 8 plus 2 reference policy,
and the production linked tail helper. The production helper must also run with shuffled relation
endpoints and removed relations at the same budgets.

The live comparison contains graph off, `outside_pool`, `linked_tail`, and `hybrid`. Each graph mode
runs with true relations. The linked tail mode also runs with shuffled endpoints and removed
relations. Arm order alternates by query index so cache or second mover effects are not assigned to
one arm.

## Predictions and gates

### Primary implementation parity gate

For every one of the 1,536 LoCoMo questions, the production linked tail helper must return the exact
same ordered context identifiers as the existing selective margin 0.05 policy when both receive
the same top 20 retrieval result and graph adjacency. Any mismatch fails implementation parity and
blocks a live quality interpretation.

The first eight identifiers must always match direct retrieval. Every context must contain exactly
ten unique identifiers when at least ten direct candidates exist. No more than two identifiers may
be graph selected.

### Topology gate

True relation endpoints must retain the existing positive complete gold evidence delta over direct
top 10. Shuffled endpoints and removed relations must not reproduce the true relation result. If a
shuffled control matches or exceeds true topology, the result does not demonstrate graph value.

### Live activation and quality gates

The live linked tail arm must expose the number of eligible linked candidates, admitted linked
candidates, outside pool candidates, and final replacements. At least 5 of the 50 queries must have
an eligible linked tail candidate for the live set to be considered a coverage test. Fewer than 5
is reported as insufficient graph coverage, not as evidence that the selector failed.

Among activated queries, linked tail must produce at least one gold evidence rescue and no gold
evidence regressions relative to graph off. Across all 50 queries, any gold evidence rate, complete
gold evidence rate, mean gold recall, MRR, and evidence precision must not decline. These live gates
are guardrails on a small frozen set, not a production promotion claim.

### Cost guardrails

`linked_tail` must not perform candidate text fetches or cosine rescoring for candidates already in
the raw retrieval pool. Its database statement count and database result bytes must not exceed graph
off by more than the cached graph readiness and adjacency work required to identify topology.
`hybrid` may perform outside pool work, which must be reported separately.

Any changed context budget, changed retrieval pool, generation mismatch, query set mismatch,
untrusted evidence bypass, source authorization bypass, or missing topology control invalidates the
run.

## Required red proof

Before implementation, add a behavior test whose invariant is that a graph linked candidate below
direct rank 10 can compete for a final context slot without displacing the protected first eight.
Run it against the current `outside_pool` behavior and require the ordered identifier assertion to
fail because the linked candidate is excluded. Collection failures, missing symbols, fixture
failures, and exceptions are not valid red proof.

A second behavior test must prove that linked candidates reuse the raw retrieval chunk and score.
Mutating the implementation to fetch or rescore a linked candidate must fail an operation count or
loader assertion.

## Measurement commands

The exact LoCoMo parity and live VPS2 commands will be implemented with the feature. Their command
lines, source commit, artifact SHA256, environment values, and pinned generation identity must be
recorded in new immutable result files. No quality or latency result may be written into this
preregistration after the frozen section. Corrections and results belong in separate documents.

## Decision rule

Passing exact LoCoMo parity promotes `linked_tail` to the live experiment, not to the serving
default. Passing the live activation and quality guardrails makes it a production candidate.
`hybrid` is promoted only if it improves gold evidence beyond `linked_tail` without adding a
regression or violating the cost report. A failed linked tail activation gate redirects work to
typed relation coverage rather than threshold tuning.
