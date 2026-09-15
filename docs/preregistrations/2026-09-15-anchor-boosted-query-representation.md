# Anchor boosted query representation development plan

Status: predicted, not yet measured.

Registered 2026-09-15 before implementing the runner, inspecting row level outcomes, or issuing
any live query for this experiment.

## Question

The consumed empty base cohort contains exact evidence in the original dense top 20 for 10 of 15
answerable rows, but the original dense rank one candidate is exact for only three. Can repeating
three rare, corpus supported terms from the original question alter the Context 4 query embedding
enough to promote exact evidence without replacing or semantically expanding the question?

This is a development screen on already consumed data. It cannot authorize serving. A positive
result must be frozen and tested on a new source disjoint paired holdout.

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

## Frozen representation

First retrieve the original question with the source admission and retrieval leg audit surfaces
enabled. Compute source document frequency for every eligible question token over the complete
audited candidate pool, using the existing query anchor token rules. Select exactly three anchors
by ascending source document frequency and then ascending SHA256 of the token, identical to the
existing anchor safety policy.

The row is eligible only when exactly three anchors exist and none has source document frequency
zero. Build `anchor_boost_once_v1` by normalizing whitespace in the complete original question,
then appending one space followed by each selected anchor exactly once in their original question
order. Do not add prompt words, synonyms, generated text, corpus text, gold labels, answerability,
source names, query identifiers, or the known control nonce rule.

Retrieve the resulting query once through the same pinned server. Do not replace the original
query result. The scored candidate is transformed dense rank one only when it differs from original
dense rank one. An unchanged rank one produces no proposal. Apply the original question's complete
query level anchor safety decision before any proposal, and append at most one item.

## Apparatus gate

Before scoring the transformed run, reproduce the frozen original dense aggregates exactly:

* Gold source reachability at ranks 1, 3, 5, 10, and 20 must be 6, 6, 7, 8, and 10.
* Exact span reachability at ranks 1, 3, 5, 10, and 20 must be 3, 3, 5, 8, and 10.
* The original query level anchor gate must admit 14 answerable rows and zero controls.

Every response must match all four pinned lineage fields. Any mismatch, missing audit, population
change, or aggregate mismatch is an apparatus failure and has no retrieval quality interpretation.

## Measurements

Report original and transformed gold source and exact span reachability at ranks 1, 3, 5, 10, and
20. Report how many answerable and control rows change dense rank one, how many changed candidates
come from the gold source, how many contain the exact span, exact addition precision, and the
number of answerable rows whose minimum exact rank improves, ties, or worsens.

Private output may contain row identifiers and gold booleans but must remain outside the
repository. Public output contains aggregate counts only.

## Prediction and decision

I predict the representation will change dense rank one for 6 to 10 of the 14 eligible answerable
rows, select zero controls, and raise exact rank one from three rows to between four and six. I
expect the result to remain below the 50.0 percent precision gate more often than not, because the
same anchors previously showed poor chunk level discrimination. The value of this screen is a
cheap causal test of whether Context 4 responds to deterministic anchor reweighting before any new
holdout is spent.

Proceed to a new source disjoint paired holdout only if the changed rank one proposals contain at
least two exact spans, select zero controls, achieve at least 50.0 percent exact addition precision,
and do not reduce exact span reachability at rank five. Otherwise close single view anchor boosting
and move to an independently scored extra retrieval view rather than tuning repetition count,
anchor count, wording, or thresholds on these rows.
