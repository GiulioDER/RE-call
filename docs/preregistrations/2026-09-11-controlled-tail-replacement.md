Date: 2026-09-11

Status: locked before implementation and measurement

## Question

Does allowing one graph candidate to replace only the weakest unprotected direct item improve
evidence recall while protecting strong direct evidence?

## Population and fixed apparatus

Use the checked out `locomo10.json`, categories 1 through 4, with every question that has
nonempty labelled evidence. Preserve the input order. Index each conversation once in an isolated
benchmark tenant using the production `Indexer`, `PgVectorStore`, and `HybridRetriever` path.
For each question, compute one top-20 hybrid retrieval result and reuse it for every arm.

The embedder is the repository's `fastembed` profile. The tail comparison uses the fixed
development calibration mapping `Calibration(embedder="fastembed", threshold=0.50, scale=0.05)`
and the already implemented margin `0.05`. This mapping is recorded for reproducibility and is
not evidence that the production generation has a certified calibration.

## Arms

All contexts are capped at their named budget and all arms use the same query, corpus, retrieval
pool, and question order.

1. `baseline_k10`: top ten direct retrieval items.
2. `tail_replacement_9_plus_1`: protect the top nine direct items; rank one-hop graph candidates
   by their existing retrieval score; replace direct rank ten only when the candidate's calibrated
   relevance is strictly greater than the direct tail's relevance plus `0.05`. If no candidate
   clears the margin, retain the direct tail.
3. `baseline_k5`: top five direct retrieval items.
4. `tail_replacement_4_plus_1`: protect the top four direct items; apply the same one-candidate
   calibrated tail replacement rule to direct rank five.
5. `established_8_plus_2`: the established retrieval-ranked comparison, with eight direct items
   followed by at most two unique one-hop graph candidates and no tail replacement rule.

Graph candidates must be source-backed, unique, reachable by an authored one-hop relation from
the protected direct prefix, and absent from the direct retrieval context before selection. No
gold evidence, answer text, category gating, or second semantic query may influence selection.

## Primary outcomes and pairing

The primary outcome is complete labelled evidence coverage, reported at question level. Secondary
outcomes are any-hit coverage, MRR, evidence precision, mean added items, and paired rescues and
regressions against the matching direct baseline: ten-item arms against `baseline_k10`, and the
public five-item arm against `baseline_k5`. Report category breakdowns and immutable per-question
contexts. The established 8 plus 2 arm is compared with `baseline_k10`.

## Prediction and decision rule

I predict that `tail_replacement_9_plus_1` will be positive but smaller than the established
8 plus 2 arm, because it permits only one graph rescue. I predict that `tail_replacement_4_plus_1`
will be noisier on the smaller public context and will not clearly exceed `baseline_k5`. I predict
that each controlled arm will have fewer regressions than its direct baseline has graph swaps, and
that the protected direct prefix will remain identical in every selected context.

This is a measurement, not a promotion gate. I will treat the controlled policy as promising only
if complete evidence coverage improves over its matching baseline, the paired bootstrap interval
for the delta excludes zero, and no more than five percent of protected-prefix positions change.
Failure to meet that rule leaves the production default unchanged. Answer quality and citation
support remain a separate replay stage.

## Integrity controls

Record the input hash, source revision, calibration mapping, margin, retrieval settings, graph
edge counts, all row-level contexts, and the exact arm configuration. Do not edit raw artifacts
after measurement. The existing behavioral red proof for the tail replacement implementation must
remain green after restoring the deliberate mutation before the measurement starts.
