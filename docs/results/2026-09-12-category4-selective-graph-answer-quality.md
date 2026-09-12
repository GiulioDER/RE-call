# Category 4 selective graph answer quality

Measured on 2026-09-12 at `2026-09-12T20:28:35.131190+00:00` on VPS2 using DeepSeek V4 Flash
through OpenRouter. The replay covered the fixed 841 category 4 questions and 1,682 answer
requests.

Raw artifact: [2026-09-12-category4-selective-graph-answer-quality.json](2026-09-12-category4-selective-graph-answer-quality.json)

Artifact SHA256: `9E2CE579515D40545D22339A75A840CE95F37B735CC22C0C70AE0BBEE26B0F78`.

Retrieval artifact SHA256:
`4E5DB6F5CED6BC368D61F174A19DEB592C587840056CD2882886A02012BAB166`.

Answer runner source revision: `83dc89fd7fae20feac276cdbe3df8018f4f964c0`.

Preregistration: [2026-09-12-category4-selective-graph-answer-quality.md](../preregistrations/2026-09-12-category4-selective-graph-answer-quality.md)

## Result

| Metric | Baseline | Selective margin 0.05 | Delta |
| --- | ---: | ---: | ---: |
| Questions | 841 | 841 | reference |
| Valid answer rate | 99.52% | 99.17% | -0.36 points |
| Nonempty answer rate | 84.66% | 84.90% | +0.24 points |
| Any gold citation | 73.60% | 74.20% | +0.59 points |
| Complete gold citation coverage | 70.87% | 72.29% | +1.43 points |
| Mean gold citation precision | 77.72% | 78.09% | +0.37 points |
| Mean gold citation recall | 72.20% | 73.20% | +1.00 point |
| Provider failures | 4 | 7 | +3 |
| Prompt tokens | 1,335,217 | 1,325,543 | -9,674 |
| Completion tokens | 48,864 | 50,236 | +1,372 |
| Provider latency p50 | 1,682 ms | 1,726 ms | +44 ms |
| Provider latency p95 | 10,535 ms | 7,952 ms | −2,583 ms |

The paired complete citation comparison produced 39 rescues and 27 regressions. The deterministic
10,000 resample bootstrap interval with seed `20260912` was `[-1.31, +4.28]` points.

## Gate decision

The answer quality gate failed. The point estimate of +1.43 points was below the preregistered
minimum of +2 points, and the confidence interval included zero. The valid answer rate guardrail
passed because it declined by only 0.36 points.

The retrieval gain from the 0.05 graph arm therefore did not transfer strongly enough to generated
answer citation quality on this category 4 population. I do not recommend production promotion.
The conditional blind judge was not run because the preregistered answer gate failed.

## Reproduction

```powershell
python scripts/run_voyage4_graph_tail_answer_quality.py docs/results/2026-09-12-category4-strict-graph-confirmation.json docs/results/2026-09-12-category4-selective-graph-answer-quality.json --treatment-arm selective_margin_005 --category 4 --model deepseek/deepseek-v4-flash --workers 8 --source-commit 83dc89fd7fae20feac276cdbe3df8018f4f964c0
```

The raw answer artifact was generated on VPS2 and copied without modification.
