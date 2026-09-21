# C8 five condition corpus qualification

## Question

Can the C8 routed specialist, grounded graph, and generation-bound atomic rescue composition
operate through the public Add and Search API over an isolated five condition AML-shaped corpus
without breaking the C7 retrieval boundary or its graph and artifact safety contracts?

This is a private operational qualification. It is not an official AML Smoke, Full run, leaderboard
submission, or a measure of agent task quality.

## Frozen apparatus

The qualification uses an isolated C8 service at the committed source revision, one private tenant
per condition, and the frozen five-condition AMB corpus material. Corpus ingestion may call only
`/v1/add`; retrieval replay may call only `/v1/search`; `/version`, `/v1/corpus/status`, and
`/v1/delete` are setup, audit, and cleanup controls. The participant evaluator is not invoked.

The service configuration is `C8_routed_specialists_grounded_graph`, Voyage Code4 primary retrieval,
Context4 specialist storage, MM2-capable multimodal wiring, and OpenRouter
`openai/gpt-4o-mini` anchored compilation. The replay query set is the frozen AMB task prompt set
for each condition, with no query selection after observing a result.

## Predictions and pass criteria

1. Every accepted Add will complete without a compiler fallback, and each condition will have at
   least one persisted eligible grounded relation.
2. Every replay Search will report graph attempted and no graph fallback. Graph promotion will
   preserve top-100 membership and preserve the protected top-8 ordering.
3. Every persisted eligible graph relation will resolve to a raw Code4 window from the same source
   session and will have a server-validated compiler evidence span. No dangling, cross-session, or
   invalid relation is accepted.
4. After each condition corpus is frozen, its generated atomic artifact will match the exact served
   generation identity, corpus fingerprint, Code4 profile and dimension, calibration identifier,
   and pipeline fingerprint. Searches deliberately selected from the artifact's non-empty eligible
   candidate set will report atomic rescue attempted, active, candidate available, and no fallback.
5. A deliberately mismatched atomic artifact will fail closed and leave ordinary retrieval active.

## Interpretation

Passing establishes that C8 is operationally eligible for an official Smoke. It does not establish
that graph or atomic rescue improves AMB agent success, and it does not authorize an official
Smoke. A failure blocks Smoke authorization until the cause is corrected and this qualification is
re-run under a new preregistration.
