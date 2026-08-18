# Pre registration: gold conditioned real document selector test

Date: 2026-08-18

## Purpose

The EnterpriseRAG run showed that the hashing retriever often missed the expected document before
answer slot selection. This diagnostic holds retrieval coverage constant by adding every chunk from
the expected document to the candidate pool, while retaining the original retrieved distractors.
It tests selector behavior on real document text. It is not an end to end retrieval score and it
must not be used as launch evidence.

## Fixed data and labels

The test uses the same SHA256 identified EnterpriseRAG files recorded in
`2026-08-18-enterprise-real-corpus.md`, the local hashing index, and these 22 answerable questions:

`qst_0303`, `qst_0304`, `qst_0309`, `qst_0310`, `qst_0311`, `qst_0312`, `qst_0313`, `qst_0316`,
`qst_0318`, `qst_0320`, `qst_0322`, `qst_0323`, `qst_0324`, `qst_0325`, `qst_0326`, `qst_0327`,
`qst_0328`, `qst_0330`, `qst_0331`, `qst_0337`, `qst_0338`, `qst_0340`.

The runner must audit that every labeled slot is coverable by at least one chunk from its expected
document before measuring. If the audit fails, the run stops.

## Arms

1. `current_retrieval`: original retrieved hits and retrieval ordered evidence.
2. `document_grouping`: original retrieved hits and document ordered evidence.
3. `gold_document_grouping`: original hits plus expected document chunks and document ordered evidence.
4. `gold_answer_slots`: the same gold candidate pool with answer slot selection.
5. `gold_bundle_beam`: the same gold candidate pool with bounded bundle beam selection.

## Predictions

1. The two original retrieval arms will have lower complete slot recall than the three gold pool
   arms because the preceding 500 question run showed a retrieval coverage ceiling.
2. `gold_answer_slots` will select complete slot bundles on at least 18 of 22 questions.
3. `gold_bundle_beam` will match or exceed `gold_answer_slots` on complete slot recall, with higher
   selection latency.
4. Gold pool arms will select fewer non gold distractor chunks than the original retrieval arms.
5. No result from this diagnostic will authorize production promotion. Trust remains degraded and
   the gold candidate injection is intentionally oracle conditioned.

## Measurement result

To be appended after the diagnostic run. Predictions above must not be edited.
