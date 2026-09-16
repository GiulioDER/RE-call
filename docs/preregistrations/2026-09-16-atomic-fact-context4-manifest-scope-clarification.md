# Atomic fact Context 4 manifest scope clarification

Status: registered before document embedding and before any retrieval outcome.

After the byte and parent alignment repairs, the full preflight reached seven objects in the
pinned generation manifest whose paths are under the production uploads area rather than any of
the seven roots frozen by `2026-09-16-atomic-fact-exhaustive-22-pilot.md`.

The auxiliary iterator now ignores every manifest object that is outside all seven exact frozen
roots. It reports the aggregate excluded count. A path inside more than one configured root remains
an error. Every object inside exactly one root remains subject to the frozen index filename rule,
manifest byte verification, extraction, and pinned parent containment checks.

This makes the manifest iterator implement the already frozen seven root source universe. It does
not change the roots, private queries, gold, lineage, embeddings, ranking arms, metrics, thresholds,
or decision rule.
