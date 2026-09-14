# Direct query anchor candidate diagnostic and development plan

Registered 2026-09-14 before measuring candidate-pool gold reachability or direct anchor ranking on
the consumed query-anchor holdout.

## Scope

The completed `query_anchor_empty_base_v1` holdout is development data only. Its registered
diagnostic found 30 empty-base rows, split 15 answerable and 15 controls. The inherited guarded
source conditioner proposed no candidate for any answerable row. This experiment asks whether the
gold evidence was nevertheless present in the existing audited candidate pool and whether a direct
query-anchor ranker could select it without activating matched controls.

This is a diagnostic and development screen. It cannot authorize serving. Any positive result must
be frozen and tested on a new source-disjoint paired holdout.

## Frozen direct rule

Re-run only the 30 consumed empty-base rows against the same immutable generation and lineage used
by the completed holdout. Enable both source-admission and retrieval-leg benchmark audits.

For each query, calculate the already frozen rare-anchor features for every audited candidate pool
chunk. Retain a chunk only when exactly three anchors were selected, no selected anchor has source
document frequency zero, and the chunk contains at least two of the three anchors. Rank retained
chunks by descending chunk anchor coverage, then descending complete-source anchor coverage, then
ascending original audited pool position. Select at most one chunk.

Do not use answerability, gold source, exact answer span, query identifier, or the known nonce
generator in selection. The private artifact may retain numeric features and gold booleans but must
remain outside the repository. Public output must contain aggregate counts only.

## Measurements and decision

For the 15 answerable rows, report gold-source and exact-span reachability in the complete audited
pool, dense leg, and sparse leg. Report the direct rule's selected count, exact-span gains, gold
source selections, source-pool exact-span selections, and exact addition precision. For the 15
controls, report selected count.

Proceed to freeze a new source-disjoint holdout only if the direct rule produces at least two exact
span gains, zero control selections, and exact addition precision of at least 50.0%. If answerable
gold evidence is usually absent from the audited pool, stop this lane and move earlier to lexical
candidate generation or query construction. Do not retune the thresholds on these consumed rows.
