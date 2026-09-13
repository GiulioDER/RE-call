# Selective graph admission answer quality

Measured on 2026-09-12 at `2026-09-12T19:38:47.947860Z` using the preregistered protocol
`2026-09-12-selective-graph-answer-quality`. The run covered all 10 LOCOMO conversations and
1,536 answerable questions in categories 1 through 4. Answers were generated with DeepSeek V4
Flash through OpenRouter.

Raw artifact: [2026-09-12-selective-graph-answer-quality.json](2026-09-12-selective-graph-answer-quality.json)

Artifact SHA256: `94B5228D0025F3D3AC9A18C2C36067CD5334EB7ADC0CDD0DB7739491E35B277D`.

Input retrieval artifact SHA256: `E3E729A9DD3D936F80245A477FAFEADB95F12A27E164DA1716626DF0DEE6E824`.

## Protocol

Both arms used the immutable contexts from the selective graph admission retrieval artifact. The
baseline provided direct ranks 1 through 10. The treatment provided direct ranks 1 through 8 plus
the selectively admitted graph items and direct fallback items. The question, context ordering,
answer prompt, model, temperature, reasoning effort, and maximum completion tokens were identical
across arms.

Model: `deepseek/deepseek-v4-flash`. Provider: OpenRouter chat completions. Temperature: 0.
Reasoning effort: none. Maximum completion tokens: 512.

## Primary result

| Metric | Direct 10 | Selective 8 plus graph | Paired delta |
| --- | ---: | ---: | ---: |
| Complete gold citation coverage | 52.80% | 54.88% | +2.08 points |
| Any gold citation rate | 63.54% | 64.84% | +1.30 points |
| Mean gold citation precision | 75.40% | 76.39% | +0.99 points |
| Mean gold citation recall | 57.77% | 59.35% | +1.59 points |
| Valid answer rate | 99.28% | 99.48% | +0.20 points |
| Nonempty answer rate | 73.70% | 74.54% | +0.85 points |

The preregistered answer quality gate did not pass. Complete gold citation coverage improved by
2.08 points, meeting the point threshold, but the paired bootstrap 95% interval was -0.07 to +4.30
points and included zero. Valid answer rate improved slightly and stayed within the guardrail.

## Provider and latency accounting

The baseline had 11 provider failures out of 1,536 requests, or 0.72%. The selective treatment had
8, or 0.52%. These failures remain in the fixed denominator.

Provider latency p50 was 1,827 ms for baseline and 1,762 ms for selective treatment. Provider
latency p95 was 13,570 ms and 9,176 ms. The baseline used 2,428,195 prompt tokens and 90,041
completion tokens. The selective treatment used 2,423,459 prompt tokens and 91,134 completion
tokens.

## Category results

| Category | Questions | Direct 10 | Selective 8 plus graph | Delta |
| --- | ---: | ---: | ---: | ---: |
| cat1 | 282 | 15.25% | 18.79% | +3.55 points |
| cat2-temporal | 321 | 54.21% | 54.52% | +0.31 points |
| cat3 | 92 | 10.87% | 11.96% | +1.09 points |
| cat4 | 841 | 69.44% | 71.82% | +2.38 points |

The treatment improved complete citation coverage in every category. The largest effects are in
cat1 and cat4, while cat2-temporal is nearly unchanged.

## Paired outcomes

The selective treatment produced 85 complete-citation rescues and 53 regressions, for a net gain of
32 questions. The any-citation delta was +1.30 points, with a paired bootstrap 95% interval of
-0.98 to +3.52 points.

## Interpretation

The selective gate improves retrieval with fewer graph additions and also produces a positive
answer-stage point estimate. The answer result is not statistically conclusive because the interval
barely crosses zero. This means the graph is promising for citation support, especially in cat1 and
cat4, but the current run does not establish a general answer quality improvement.

No independent factual correctness or citation entailment judge was run. I therefore do not claim
that the generated answers are more correct. The next highest value step is a blind judge over the
85 rescues, 53 regressions, and a matched sample of unchanged questions, followed by a larger
category 4 confirmation if the judged net quality remains positive.

## Reproduction

The protocol was committed before answer generation in
`docs/preregistrations/2026-09-12-selective-graph-answer-quality.md`.

```powershell
python scripts/run_voyage4_graph_tail_answer_quality.py docs/results/2026-09-12-selective-graph-admission.json docs/results/2026-09-12-selective-graph-answer-quality.json --treatment-arm selective_tail --model deepseek/deepseek-v4-flash --workers 8 --source-commit 52b264b061b756f2ef0a9d47ac49351c05553840
```
