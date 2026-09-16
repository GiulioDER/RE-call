# Anchor only auxiliary retrieval view development plan

Status: predicted, not yet measured.

Registered 2026-09-15 before implementing the runner, inspecting row level outcomes, or issuing
any live query for this experiment.

## Question

The consumed empty base cohort contains exact evidence in the original dense top 20 for 10 of 15
answerable rows. Repeating rare anchors inside the complete primary query failed to improve rank
one and reduced top 5 exact coverage. Can the same safe anchors retrieve complementary exact
evidence when they are embedded as a separately scored auxiliary view, while the original dense
ordering remains unchanged?

This is a candidate-generation screen on already consumed data. It cannot select a result,
authorize serving, or authorize a fresh holdout.

## Frozen population and lineage

Use only the 30 rows with an empty served base in the completed query anchor holdout, split into
15 answerable rows and 15 matched unanswerable controls. Use the existing query pool and holdout
artifacts without inspecting gold fields during query construction.

The query pool SHA256 is
`6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`. The completed holdout
SHA256 is `58b883af8d1527ef137913762f9a3a456198c8f818d8286bfc87a248c4d3fa5b`.

Pin generation `gen_2ccf2130f6c64d99a11a6bcb6f929dd8`, calibration
`cal_e50dac493112488ea5e7cf79d86c0099`, pipeline fingerprint
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus fingerprint
`f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312`.

## Frozen auxiliary representation

First retrieve the complete original question with the source admission and retrieval leg audit
surfaces enabled. Compute source document frequency for every eligible question token over the
complete audited candidate pool, using the existing query anchor token rules. Select exactly three
anchors by ascending source document frequency and then ascending SHA256 of the token, identical
to the existing anchor safety policy. Emit those selected anchors in their first occurrence order
in the original question.

The row is eligible only when exactly three anchors exist and none has source document frequency
zero. Build `anchor_only_aux_v1` as the three selected anchors joined by one ASCII space. Do not
include the remaining question, prompt words, punctuation, repeated anchors, synonyms, generated
text, corpus text, gold labels, answerability, source names, query identifiers, or the known
control nonce rule.

Retrieve `anchor_only_aux_v1` once through the same pinned server. The original and auxiliary
dense lists each contribute their first 20 items to a deduplicated union keyed by chunk ID. Union
membership is measured only. Preserve the original dense order, do not fuse scores or ranks, do
not select an addition, and do not change the served result.

## Apparatus gate

Before interpreting the auxiliary view, reproduce the frozen original dense aggregates exactly:

* Gold source reachability at ranks 1, 3, 5, 10, and 20 must be 6, 6, 7, 8, and 10.
* Exact span reachability at ranks 1, 3, 5, 10, and 20 must be 3, 3, 5, 8, and 10.
* The original query level anchor gate must admit 14 answerable rows and zero controls.

Every response must match all four pinned lineage fields. Any mismatch, missing audit, population
change, or aggregate mismatch is an apparatus failure and has no retrieval quality interpretation.

## Measurements

Report original and auxiliary gold source and exact span reachability at ranks 1, 3, 5, 10, and
20. Report gold source and exact span reachability in the deduplicated top 20 union. Report the
number of answerable rows where the auxiliary top 20 makes an exact span newly reachable beyond
the original top 20, the number where it makes the gold source newly reachable, and the number of
unique auxiliary chunks added to the union. Report eligible answerable and control counts.

An incremental exact row has no exact span in the original top 20 and at least one exact span in
the auxiliary top 20. An incremental gold source row has no gold source in the original top 20 and
at least one gold source in the auxiliary top 20. These definitions do not depend on union order.

Private output may contain row identifiers, anchors, ranks, chunk IDs, and gold booleans but must
remain outside the repository. Public output contains aggregate counts only.

## Prediction and decision

I predict the auxiliary view will make exact evidence newly reachable for two of the five
answerable rows missed by the original dense top 20, add no eligible controls, and add many
non-gold candidates. The likely value is complementary recall rather than direct precision, so no
selection precision claim is made in this screen.

Proceed to a separately preregistered one-slot selector study on this same consumed cohort only if
the auxiliary top 20 makes exact evidence newly reachable for at least two answerable rows and the
eligibility gate admits zero controls. Otherwise close lexical anchor query construction and move
to a genuinely independent learned late-interaction or chunk-level relevance view. Do not tune
anchor count, anchor order, query wording, cutoff, or the two-row gate after measurement, and do
not spend a fresh source-disjoint holdout on this candidate-generation result alone.
