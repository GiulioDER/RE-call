# Selective graph blind answer judge

Measured on 2026-09-12 at `2026-09-12T19:47:47.775285Z` using the preregistered protocol
`2026-09-12-selective-graph-blind-judge`. The judge evaluated 276 paired questions from the
completed selective graph answer replay.

Raw artifact: [2026-09-12-selective-graph-blind-judge.json](2026-09-12-selective-graph-blind-judge.json)

Artifact SHA256: `7C1112485A860CDE20C0E9B6376B7B0A0D1AB773C7EAD98EDC878154FC66C045`.

Answer input SHA256: `94B5228D0025F3D3AC9A18C2C36067CD5334EB7ADC0CDD0DB7739491E35B277D`.

## Protocol

The sample included all 85 complete citation rescues, all 53 complete citation regressions, and a
deterministic sample of 138 unchanged pairs. The judge received the question, the union of both
arms' evidence contexts, and two answers labeled A and B. Arm labels were hidden and A or B order
was shuffled deterministically per question.

Judge model: `anthropic/claude-haiku-4.5` through OpenRouter. Temperature: 0. Reasoning effort:
none. Maximum completion tokens: 256. All 276 judge responses produced valid rubric JSON.

For each answer, the judge scored correctness, evidence support, and completeness from 0 through 2.
The total score therefore ranges from 0 through 6. The judge also counted unsupported factual claims.

## Primary result

| Metric | Direct 10 | Selective 8 plus graph | Paired delta |
| --- | ---: | ---: | ---: |
| Mean total judge score | 3.7790 | 4.1449 | +0.3659 |
| Mean unsupported claims | 0.0362 | 0.0217 | -0.0145 |

The preregistered judge gate did not pass. The point estimate exceeded the required +0.25 score
threshold, and the valid judgment rate was 100%, but the paired bootstrap 95% interval was -0.1739
to +0.8986 and included zero.

Selective won 95 pairs, baseline won 71, and 110 pairs tied on the six point rubric.

## Bucket results

| Bucket | Pairs | Direct 10 | Selective 8 plus graph | Delta |
| --- | ---: | ---: | ---: | ---: |
| Citation rescue | 85 | 2.2941 | 5.5059 | +3.2118 |
| Citation regression | 53 | 5.2830 | 1.7170 | -3.5660 |
| Unchanged control | 138 | 4.1159 | 4.2391 | +0.1232 |

The large rescue and regression differences are expected because the sample was selected from
those citation outcomes. The unchanged control is the less selected estimate of general answer
impact and shows only a small positive shift.

## Category results

| Category | Pairs | Mean score delta | Selective wins | Baseline wins |
| --- | ---: | ---: | ---: | ---: |
| cat1 | 46 | +0.565 | 39.1% | 19.6% |
| cat2-temporal | 69 | +0.101 | 34.8% | 29.0% |
| cat3 | 17 | +0.647 | 35.3% | 29.4% |
| cat4 | 144 | +0.396 | 32.6% | 25.7% |

The direction is positive in every sampled category, with the largest mean effects in cat3 and
cat1. The cat3 sample is small and should not be treated as stable evidence.

## Interpretation

The blind judge supports the mechanism suggested by the retrieval and citation results: graph
contexts can produce materially better answers when they rescue missing evidence. However, the
overall quality interval still crosses zero because the regressions offset much of the rescue gain.
The unchanged control shows only a small effect.

This experiment does not justify a general production quality claim. The strongest next experiment
is a larger confirmation focused on category 4 and on a production style selective gate, with a
predeclared guardrail against regressions and the same blind judge rubric. A useful design should
also test whether reducing graph admission further preserves the rescue wins while eliminating the
53 regression cases.

## Reproduction

The protocol was committed before judging in
`docs/preregistrations/2026-09-12-selective-graph-blind-judge.md`.

```powershell
python scripts/run_selective_graph_blind_judge.py docs/results/2026-09-12-selective-graph-answer-quality.json docs/results/2026-09-12-selective-graph-admission.json docs/results/2026-09-12-selective-graph-blind-judge.json --workers 8 --source-commit 13a6fd3b32bc338adbfa5adb0b94e05a69d4bc64
```
