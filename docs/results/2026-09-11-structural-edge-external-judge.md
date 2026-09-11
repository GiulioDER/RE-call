# Structural edge external judge evaluation

Measured 2026-09-11 from the frozen answer artifact with 1,536 paired LoCoMo questions per
arm and 6,144 answer rows. Answers were generated previously by the fixed production answer
provider. This run only performed factual scoring.

Command:

```text
python -u scripts/run_structural_edge_external_judge.py docs/results/2026-09-11-structural-edge-answer-validation-full.json docs/results/2026-09-11-structural-edge-external-judge-locomo.json --workers 8
```

## Judge configuration

The same external judge was used for every arm:

| field | value |
|---|---|
| provider | OpenRouter |
| model | `openai/gpt-4o-mini` |
| temperature | 0.0 |
| maximum output | 8 tokens |
| prompt | `benchmarks.pipeline.JUDGE_SYSTEM_PROMPT` |
| unique judge prompts | 2,094 |
| provider failures | 0 |

OpenRouter reported 393,573 prompt tokens, 3,167 completion tokens, 396,740 total tokens, and
an estimated cost of $0.06093615 for the judge calls. Exact duplicate prompts were reused across
arms in the row materialization.

The gold answer was joined from `locomo10.json`. Category 5 adversarial rows were excluded by
the source answer artifact and are not treated as answerable gold rows. Invalid answer envelopes,
empty answers, and insufficient evidence remain false in the all row denominator.

## Results

`answer_score_all_rows` is the primary paired answer score. `judge_accuracy_successful_only` is
conditional on an eligible answer receiving a successful YES or NO response.

| arm | correct | all row score | successful judge accuracy | valid envelope | eligible rows |
|---|---:|---:|---:|---:|---:|
| baseline | 601/1536 | 39.13% | 56.43% | 99.48% | 1,065 |
| retrieval ranked, 8 direct + 2 structural | 636/1536 | 41.41% | 59.00% | 99.41% | 1,078 |
| retrieval ranked, 9 direct + 1 structural | 629/1536 | 40.95% | 58.95% | 99.41% | 1,067 |
| category selective, 8 direct + 2 structural | 639/1536 | 41.60% | 59.22% | 99.54% | 1,079 |

## Paired effects

| treatment | score delta, 95% normal CI | rescued baseline misses | regressed baseline correct |
|---|---:|---:|---:|
| 8 direct + 2 structural | +2.2786, [0.5368, 4.0204] | 111 | 76 |
| 9 direct + 1 structural | +1.8229, [0.2622, 3.3836] | 89 | 61 |
| category selective 8 direct + 2 structural | +2.4740, [1.0465, 3.9014] | 82 | 44 |

The baseline had 935 all row misses. The rescue rates were 11.87% for 8 plus 2, 9.52% for
9 plus 1, and 8.77% for category selective.

## Decision against the locked rule

The 8 direct plus 2 structural arm passes the external answer score gate, with a 2.2786 point
paired improvement and an 11.87% rescue rate. Category selective also passes, with a 2.4740
point improvement and an 8.77% rescue rate. The 9 direct plus 1 structural arm improves the
score, but does not reach the preregistered two point answer gate.

Retrieval precision did not fall for any treatment. It changed from 9.15% for baseline to
9.35% for 8 plus 2, 9.24% for 9 plus 1, and 9.29% for category selective. Answer validation
failure rates stayed below one percent in the preceding fixed answer stage. The preceding
answer stage also measured treatment provider p95 latency below twice baseline.

This validates 8 direct plus 2 structural as the leading global candidate and category selective
as a viable lower addition alternative for the measured LoCoMo population. It rejects 9 direct
plus 1 as the primary choice under the locked two point answer gate. The result is judge specific
to this fixed OpenRouter configuration and does not by itself justify changing the production
default without an explicit rollout decision.

Raw artifact: `2026-09-11-structural-edge-external-judge-locomo.json`.
