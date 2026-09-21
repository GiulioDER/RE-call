# C8 scope bound atomic qualification

## Supersedes

This record supersedes the atomic artifact portion of
`2026-09-21-aml-c8-five-condition-qualification.md`. The earlier record assumed that one artifact
directory addressed by generation ID could serve five distinct corpus scopes. That assumption is
false when the scopes have distinct corpus fingerprints.

## Question

Can C8 bind optional atomic rescue to the exact opaque corpus scope, generation identity, and
artifact lineage, while keeping ordinary retrieval available when that scope has no valid artifact?

This is a private operational qualification. It is not an official AML Smoke, Full run,
leaderboard submission, or a measure of agent task quality.

## Frozen apparatus

The artifact registry layout is
`<artifact-root>/<opaque-scope>/<generation-id>/manifest.json`. The scope is the service's opaque
tenant namespace, never a supplied raw user identifier. A resolver accepts no path separators,
dot segments, or nested components in either scope or generation identity.

Each independently frozen corpus scope receives an artifact constructed from that same scope's
Code4 windows. The serving binding contains the opaque scope, generation identity, corpus
fingerprint, calibration identity, pipeline fingerprint, Code4 embedding profile, and dimension.

## Predictions and pass criteria

1. Matching artifacts in two distinct scopes with the same generation identity will both load and
   activate in one process, with no cache collision and with each rescue selecting a candidate from
   its own scope.
2. A manifest located under a different scope, even if its remaining lineage fields match, will not
   be considered. Ordinary retrieval remains available and reports atomic fallback.
3. A manifest with a mismatched corpus fingerprint, profile, dimension, calibration identity, or
   pipeline fingerprint will fail closed into ordinary retrieval.
4. Traversal shaped scope or generation input will be rejected before an artifact is read.
5. C8 graph application will remain tenant scoped. Every valid graph relation observed in the
   later five condition replay will resolve to a raw Code4 window in its own scope. A scope with no
   eligible relation is a graph no-op, rather than a qualification failure.

## Interpretation

Passing establishes scope safe C8 atomic activation and graph isolation. It does not establish
atomic rescue availability for a corpus without eligible candidates, graph benefit, AMB agent
success, or authorization for an official Smoke.
