# Controlled tail answer and citation replay

The full retrieval artifact was replayed on 2026-09-11 through the configured OpenRouter answer
provider, `deepseek/deepseek-v4-flash`, with the existing evidence prompt and envelope plus
citation validator. The replay covered 1,536 questions in each of five arms, using 4,945
deduplicated provider requests and 2,735 prompt-level cache reuses.

Retrieval input: [controlled-tail replacement result](2026-09-11-controlled-tail-replacement.md).
Raw replay artifact: [controlled-tail answer validation JSON](2026-09-11-controlled-tail-answer-validation-full.json).
Replay runner: [run_controlled_tail_answer_validation.py](../../scripts/run_controlled_tail_answer_validation.py).

## Results

`answer and citation valid` means the generated JSON envelope passed structural validation,
including citation identity validation. `cited gold recall` measures how much of the labelled
evidence set was cited by the answer. It is citation support, not factual answer correctness.

| arm | answer and citation valid | validation errors | cited gold recall | all gold cited | mean citations |
|---|---:|---:|---:|---:|---:|
| baseline, top 10 direct | 99.02% | 15 | 54.05% | 49.70% | 1.07 |
| controlled, 9 direct + 1 graph | 99.61% | 6 | 54.29% | 50.03% | 1.07 |
| baseline, top 5 direct | 99.41% | 9 | 47.90% | 43.71% | 0.89 |
| controlled, 4 direct + 1 graph | 99.48% | 8 | 48.02% | 44.03% | 0.89 |
| established, 8 direct + 2 graph | 99.41% | 9 | **55.81%** | **51.86%** | 1.09 |

## Paired answer and citation deltas

| comparison | valid delta | cited gold recall delta | all-gold-cited delta |
|---|---:|---:|---:|
| 9 direct + 1 graph vs top 10 | +0.59 pp | +0.52 pp | +0.33 pp |
| 4 direct + 1 graph vs top 5 | +0.07 pp | +0.12 pp | +0.33 pp |
| 8 direct + 2 graph vs top 10 | +0.39 pp | **+2.11 pp** | **+2.47 pp** |

All rows remained in the denominator. The replay recorded 47 validation or provider-error rows
across all arms, including timeouts, malformed JSON, empty responses, invalid envelope fields, and
unknown citation IDs. No unsupported-claim judge was run, and the provider output was not treated
as a factual correctness score.

## Decision

The answer and citation replay does not overturn the retrieval decision. The controlled `4+1`
arm shows a small positive citation-support movement over its `k=5` baseline, while `9+1` remains
inconclusive. The established `8+2` arm has the clearest citation-support gain in this replay.
No production change is authorized: the tail arms used a development calibration, factual answer
correctness and unsupported claims remain unjudged, and the production default remains unchanged.
