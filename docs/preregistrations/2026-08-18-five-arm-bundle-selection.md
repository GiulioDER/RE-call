# Pre registration: five arm bundle selection benchmark

Date: 2026-08-18

## Question

Does answer aware evidence selection improve complete, citation safe bundles for questions that
require distant sections, while preserving abstention on partial or unanswerable queries?

## Fixed fixture

The fixture will contain at least these cases:

1. Two distant sections in one document are both required.
2. A nearby section contains a misleading partial answer.
3. An exception appears near the document conclusion.
4. Two documents contain similar entities with different outcomes.
5. A comparison requires one section per entity.
6. An unanswerable query has only one answer slot present.

Each case will define required chunk identifiers, required answer slots, and forbidden chunk
identifiers where a misleading or incomplete answer must not be selected.

## Arms

1. `current_retrieval`: initial retrieval and retrieval ordered evidence.
2. `document_grouping`: initial retrieval and document ordered evidence.
3. `structural_expansion`: source scoped expansion followed by document ordered evidence.
4. `answer_slots`: source scoped expansion, document grouping, and required slot coverage.
5. `bundle_beam`: source scoped expansion, slot coverage, and bounded bundle beam selection.

The third arm is explicitly named structural expansion for this first fixture, but the measured
implementation must report whether it is query expansion or true section adjacency. It must not be
described as heading aware structure unless the fixture and implementation contain that signal.

## Primary predictions

1. `document_grouping` will preserve more cross document coverage than `current_retrieval` but
   will not recover all distant evidence that was absent from initial retrieval.
2. `structural_expansion` will improve distant section recall over both baseline arms without
   increasing unanswerable false positives by more than one fixture case.
3. `answer_slots` will reduce misleading partial answer selection and will abstain when a required
   slot is absent.
4. `bundle_beam` will match or exceed `answer_slots` on complete slot coverage and forbidden chunk
   avoidance, at higher bundle selection CPU cost.
5. No arm will be promoted to the default from this fixture alone.

## Metrics

For every arm report:

1. Complete required chunk set recall.
2. Complete required slot recall.
3. Forbidden chunk selection count.
4. Partial answer false positive count on the one slot present cases.
5. Unanswerable false positive count.
6. Trust state and demoted chunk leakage count.
7. Mean and p95 selection CPU time.

The measurement is offline and deterministic. It does not claim database latency or general corpus
quality. A database backed latency measurement is required before any serving default changes.

## First measurement

Measured with `benchmarks/five_arm_bundle_probe.py` on 2026-08-18 after the preregistration commit.
There were 6 cases. The arm summaries were:

| arm | complete ids | complete slots | forbidden selected | false positives | mean selection ms | p95 selection ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current retrieval | 1 | 0 | 1 | 1 | 0.0810 | 0.1421 |
| document grouping | 1 | 0 | 1 | 1 | 0.0845 | 0.1282 |
| structural expansion | 6 | 5 | 1 | 1 | 0.1776 | 0.2582 |
| answer slots | 6 | 5 | 0 | 0 | 0.4449 | 0.8750 |
| bundle beam | 6 | 5 | 0 | 0 | 1.0838 | 2.0983 |

The measurement supports the registered direction on this fixture. Structural expansion recovers
distant evidence but can retain a misleading passage. Slot selection removes that passage and
abstains when a required slot is absent. Bundle beam matches slot selection here at higher CPU
cost. The fixture is too small to justify a serving default or a general quality claim.
