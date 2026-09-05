# Heading contextualization evaluation

Prior work: `project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06` and the existing context mode tests. The prior work identifies retrieval coverage as the bottleneck. The existing tests establish raw text and metadata parity, but do not establish retrieval quality.

## Hypothesis

The section profile `bge-small-context-section-v1` improves retrieval for Markdown memories whose useful context is expressed in a heading, while preserving raw evidence, trust behavior, and abstention safety.

## Frozen comparison

* Baseline: the currently active profile and retrieval settings.
* Candidate: `bge-small-context-section-v1` with the same corpus, query set, candidate budget, fusion settings, reranker, and trust policy.
* Evaluation: paired queries. Each query is run against both generations.
* The active generation remains the serving baseline until every gate passes.

## Primary and secondary measures

* Primary retrieval measure: paired Recall@100.
* Operational retrieval measure: paired Recall@20.
* Ranking measure: paired reciprocal rank of the first relevant source.
* Safety measures: false abstention on answerable queries, false acceptance on unanswerable queries, and trust verdict agreement.
* Operational measure: indexing duration, passage embedding duration, and query latency at the same concurrency.

## Heading dependent slice

Include a query in the heading dependent slice when its gold chunk has a nonempty `heading_hierarchy` and no normalized heading term occurs in the raw chunk text. Normalization lowercases, folds whitespace, and removes punctuation. The slice is computed before candidate results are inspected.

## Promotion rule

Promote only when all of the following hold:

* Source set, chunk count, raw text, content hashes, source offsets, heading hierarchy, and provenance are identical between generations.
* Overall answerable Recall@100 is noninferior with a one percentage point absolute margin.
* False abstention and false acceptance do not increase by more than one percentage point.
* The heading dependent Recall@100 has a positive paired effect with a confidence interval whose lower bound is above zero.
* No trust verdict, evidence citation, or entailment regression is observed.

The paired analysis and confidence interval method are fixed before results are inspected. If the heading dependent slice is too small for a stable interval, the result is reported as inconclusive and the candidate is not promoted.

## Safety checks

The candidate must use a new generation bound calibration. The baseline calibration is not reused because the pipeline fingerprint and passage representation differ. The old generation remains available for rollback after cutover.
