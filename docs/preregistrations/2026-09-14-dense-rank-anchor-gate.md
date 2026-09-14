# Dense rank with query anchor safety gate development plan

Registered 2026-09-14 before inspecting dense exact-span ranks or measuring either candidate rule
below on the consumed 30-row empty-base cohort.

## Question

The prior screen found exact evidence in the dense candidate leg for 10 of 15 answerable rows, but
anchor-first ranking selected only two exact chunks among 14 additions. Can dense rank provide the
relevance order while the corpus-absent query anchor test provides control safety?

This is development data only. It cannot authorize serving. Do not change thresholds after seeing
the results.

## Frozen measurements and rules

Re-run only the 30 consumed empty-base rows against the immutable holdout lineage. Report minimum
dense rank for the gold source and exact span at cutoffs 1, 3, 5, 10, and 20.

Calculate the frozen three rare query anchors against the complete audited candidate pool. Reject
the entire query when fewer than three anchors are available or any selected anchor has source
document frequency zero. This query-level gate must be identical for both rules.

Compare exactly two deterministic rules:

1. `dense_first_anchor_safe`: select the dense rank-one chunk after the query-level safety gate.
2. `dense_first_anchor_compatible`: after the same query-level gate, select the first dense-ranked
   chunk whose chunk anchor coverage is at least two of three.

Append at most one item. Do not use answerability, gold source, exact span, query identifier, or the
known nonce generator in selection. Row-level output remains private. Public output is aggregate
only.

## Gate

Choose the higher exact-span precision rule, breaking a tie by more exact gains and then fewer
additions. Proceed to a new source-disjoint paired holdout only if the chosen rule has at least two
exact gains, zero control activations, and exact addition precision of at least 50.0%. Otherwise
close dense-rank rescue on this empty-base cohort and move to a new query representation experiment.
