# Atomic fact auxiliary retrieval view audit result

Measured 2026-09-16 on the consumed 250 row extractive positive pool. The frozen protocol is
[2026-09-16-atomic-fact-auxiliary-view-audit.md](../preregistrations/2026-09-16-atomic-fact-auxiliary-view-audit.md).

## Verdict

`STOP_ATOMIC_FACT_AUXILIARY_SHADOW`.

The audit failed four frozen gates: unique gold view coverage, gold parent agreement, gold family
coverage, and maximum rendered view length. No embedding or shadow generation is authorized from
this result.

## What worked

Source integrity was exact for all 250 rows. The audit reconstructed 1,569 ordinary chunks and 999
atomic fact views. Every source produced at least one view.

The proposed view was substantially smaller than the frozen safety bounds. Auxiliary row growth was
0.637 times the ordinary chunk count, against a maximum of 2.0. Rendered character growth was 0.341
times the ordinary chunk characters, against a maximum of 1.5. The p95 parent carried two views and
the maximum carried four, against limits of four and eight. There were no within-source fact
duplicates, no within-source rendered collisions, and no rendered collisions across sources.

The construction preserved a unique complete fact and the frozen parent ordinal for 248 of 250 gold
rows. Fallback coverage was 132 of 132 and heading coverage was 60 of 60. Field coverage was 56 of
58.

These measurements reject the concern that fact-sized views necessarily produce an unbounded flat
index or crowd the candidate pool. They do not establish a retrieval gain.

## Why it stopped

The frozen eligibility rule measured the complete field paragraph, including its label, against the
same 320 character ceiling used for fact content. Two otherwise eligible field values therefore did
not receive a view. This interpretation is fixed for this consumed audit and cannot be changed after
the result.

Eleven of 999 rendered views exceeded 800 characters. Fact content itself never exceeded 320
characters and its p95 was 306. The excess therefore came from structural context. The frozen
construction did not apply the production context policy's bounded section field or degradation
ladder.

The two failures are construction contract failures, not evidence of excessive corpus growth,
duplicate facts, or parent crowding. The verdict remains a stop because the gates were registered
before measurement.

## Relationship to existing heading contextualization

This experiment does not repeat deterministic section contextualization. The existing feature adds
title, heading hierarchy, and source path to the embedding text of the complete ordinary chunk. The
candidate measured here isolates one complete fact in a second embedding record and maps it back to
the unchanged ordinary chunk. It creates a different retrieval surface while preserving the same
evidence and citation boundary.

## Reentry condition

Do not tune this construction on the consumed 250 sources. Reenter only with a fresh source-disjoint
pool and a production-aligned rendering contract frozen before extraction:

1. apply the 40 to 320 character eligibility bound to fact content after separating an explicit
   field label
2. reuse the production title and section caps and a deterministic degradation ladder that always
   preserves complete fact content
3. keep the parent-mapped auxiliary record design and the same row growth, character growth,
   duplicate, collision, and parent crowding gates
4. require complete gold coverage before spending on Context 4 embeddings

If the fresh audit passes, build one isolated Context 4 shadow generation. Compare parent-deduplicated
dense retrieval with and without the auxiliary vectors on a second untouched source-disjoint exact
span set. The primary quality endpoints remain exact span and gold source rank one, with reach at
ranks 5, 10, and 20 as diagnostics.

## Integrity

The aggregate result is
`docs/results/2026-09-16-atomic-fact-auxiliary-view-audit.json`. It contains no source text, query
text, answer spans, or row-level outcomes. The audit used no model, embedding, retrieval, or external
inference.

The behavioral red proof changed every auxiliary parent ordinal by one. The parent identity test
failed at the intended assertion, then passed after restoration. The focused test suite, Ruff, and
targeted mypy checks passed before measurement.
