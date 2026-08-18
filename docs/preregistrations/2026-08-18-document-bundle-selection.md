# Pre registration: document expansion and evidence bundle selection

**Date:** 2026-08-18   **Status:** predicted, not yet measured

## The question

Does adaptive same document expansion, followed by document aware evidence selection, improve
coverage of answers that require distant sections without weakening trust or citation safety?

## Baseline and treatment

**Baseline:** the current calibrated `trusted_search` result and retrieval ordered evidence bundle.

**Treatment:** when the query is structurally relational and the initial trusted result identifies a
source, rerun calibrated retrieval inside the top source documents, merge unique trusted chunks, and
select a coherent bundle by document, section order, and score. Ordinary retrieval remains unchanged
when the feature is disabled or the query does not trigger expansion.

## Prediction

On a fixed fixture containing direct questions, distant section questions, cross document questions,
and unanswerable controls, I predict:

| Metric | Prediction |
| --- | --- |
| Required evidence set recall on distant section questions | **+0.20 to +0.40 absolute** |
| Overall evidence set recall | **+0.05 to +0.20 absolute** |
| Unsupported or demoted citation rate | **0.00**, unchanged from baseline |
| Correct abstention on unanswerable controls | **no decrease greater than 0.05** |
| Treatment latency relative to baseline | **no more than 3x** |

The main prediction is that the gain comes from finding a second section in the same document, not
from changing the score of the initial hit. I expect little or no gain on direct one section
questions.

## Invariants

- [ ] Expanded chunks receive their own cosine and trust verdict.
- [ ] No chunk with a verdict other than `ok` enters the evidence bundle.
- [ ] The baseline path is byte stable when expansion is disabled.
- [ ] Source expansion never crosses the tenant, generation, or corpus binding.
- [ ] Duplicate chunk ids are emitted at most once.
- [ ] A bundle never cites a chunk that was not present in the trusted result after expansion.

## What would falsify this

The treatment is falsified if distant section evidence recall improves by less than 0.05, if
unanswerable false confidence increases by more than 0.05, if any demoted chunk becomes citable, or
if the median latency exceeds 3x baseline on the fixture. A gain only on direct questions would
falsify the proposed mechanism even if the aggregate score improved.

## Decision rule, fixed in advance

| Outcome | Action |
| --- | --- |
| Distant evidence recall improves by at least 0.10, safety invariants hold, and latency stays within 3x | Keep the feature available and prepare a separate default promotion decision |
| Safety invariant fails | Reject the implementation regardless of retrieval gain |
| Gain is below 0.10 or latency exceeds 3x | Keep the feature opt in and revise the selector before promotion |
| Fixture apparatus fails an invariant | Do not interpret the quality result |

## Result

Measured 2026-08-18 with:

```powershell
.\.venv\Scripts\python.exe benchmarks/document_bundle_probe.py
```

**Status: partially confirmed, with the intended mechanism confirmed on the fixture.**

| Metric | Prediction | Measured |
| --- | --- | --- |
| Required evidence set recall on distant section questions | +0.20 to +0.40 | **0 of 4 to 4 of 4**, +1.00 |
| Overall evidence set recall | +0.05 to +0.20 | **7 of 12 to 12 of 12**, +0.417 |
| Unsupported or demoted citation rate | 0.00 | **0**, all treatment bundles remained trusted |
| Correct abstention on unanswerable controls | no decrease greater than 0.05 | **unchanged**, false positives 1 to 1 |
| Treatment latency relative to baseline | no more than 3x | **2.05x mean CPU bundle assembly**, database query latency not measured |

The direct question controls did not gain evidence, as predicted. The gain came from adding a
second trusted section from the same source, and the comparison case also gained sections from both
source documents. The result is strong enough to keep the feature available and continue toward a
database backed latency measurement, but it is not evidence for making the feature the default yet.
