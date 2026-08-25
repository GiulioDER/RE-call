# Pre-registration: 1024-dimensional heading contextualization

Date: 2026-08-25

This is the VPS2-compatible follow-up to [the BGE-small heading contextualization
pre-registration](2026-08-24-heading-contextualization.md). VPS2's installed generation table is
`vector(1024)`, so the small profile cannot be built in that deployment without a separate schema.
This candidate keeps the same deterministic section rendering rule and changes only the registered
encoder profile to the 1024-dimensional BGE-large model.

## Frozen comparison

* Baseline: the currently active VPS2 generation and retrieval settings.
* Candidate: `bge-large-context-section-v1`, with the same immutable corpus, query set, candidate
  budget, fusion settings, reranker, and trust policy.
* Production remains on the baseline until all parity, calibration, safety, and quality gates pass.

The candidate must be built as a new generation and calibrated independently. The active
generation's calibration cannot be reused because both the pipeline fingerprint and passage
representation change.

## Primary and secondary measures

The primary measure is paired answerable Recall@100. Secondary measures are operational Recall@20,
paired reciprocal rank, false abstention, false acceptance, trust verdict agreement, evidence and
citation parity, indexing and passage embedding duration, and query latency at equal concurrency.

## Heading-dependent slice

Include a query when its gold chunk has a nonempty `heading_hierarchy` and normalized heading terms
do not already occur in the raw chunk text. Normalize by lowercasing, folding whitespace, and
removing punctuation. Define the slice before inspecting candidate rankings.

## Promotion rule

Promote only if:

1. source set, chunk count, raw text, content hashes, source offsets, heading hierarchy, and
   provenance are exactly identical;
2. overall answerable Recall@100 is noninferior within a one percentage point absolute margin;
3. false abstention and false acceptance do not regress;
4. the heading-dependent slice has a positive paired effect with a confidence interval whose lower
   bound is above zero; and
5. no trust, citation, entailment, or latency safety gate regresses.

If the heading-dependent slice is too small for a stable interval, the result is inconclusive and
the candidate is not promoted.

No quality measurement may be reported as confirmatory until the candidate generation and its
profile-specific calibration exist under this pre-registration.
