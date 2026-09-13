# Voyage 4 bounded graph tail answer quality

Measured on 2026-09-12 at `2026-09-12T18:30:27.751794Z` using the preregistered protocol
`2026-09-12-voyage4-graph-tail-answer-quality`. The run covered all 10 LOCOMO conversations and
1,536 answerable questions in categories 1 through 4. Answers were generated with DeepSeek V4 Flash
through OpenRouter.

Raw artifact: [2026-09-12-voyage4-graph-tail-answer-quality.json](2026-09-12-voyage4-graph-tail-answer-quality.json)

Artifact SHA256: `92DB90BFAAD278806ADFCA367A95DD05E87A9E5B4E3B74161502210C68D85F76`.

Input retrieval artifact SHA256: `D51974F72E19AF2E5B978C3183090CA83C8D12BB134E91AF1A2117A2E46C4B07`.

## Protocol

Both arms used the immutable Voyage 4 retrieval contexts from the input artifact. The baseline
provided the top 10 direct contexts. The treatment provided the top 8 direct contexts plus at most
2 retrieval-ranked structural graph neighbors. The question, selected contexts, answer prompt,
model, temperature, reasoning effort, and maximum completion tokens were identical across arms.
Gold identifiers were used only after generation for scoring.

Model: `deepseek/deepseek-v4-flash`. Provider: OpenRouter chat completions. Temperature: 0.
Reasoning effort: none. Maximum completion tokens: 512.

## Primary result

| Metric | Direct 10 | Direct 8 plus graph 2 | Paired delta |
| --- | ---: | ---: | ---: |
| Complete gold citation coverage | 53.13% | 55.01% | +1.89 points |
| Any gold citation rate | 64.06% | 65.76% | +1.69 points |
| Mean gold citation precision | 75.68% | 76.71% | +1.02 points |
| Mean gold citation recall | 58.10% | 59.96% | +1.86 points |
| Valid answer rate | 99.41% | 99.15% | -0.26 points |
| Nonempty answer rate | 74.48% | 75.52% | +1.04 points |

The preregistered primary gate did not pass. Complete gold citation coverage improved by 1.89
points, below the required 2 point threshold, and the paired bootstrap 95% interval was -0.20 to
+3.91 points, which includes zero. The valid answer rate decline was within the allowed 2 point
limit.

## Provider and latency accounting

The baseline had 9 provider failures out of 1,536 requests, or 0.59%. The treatment had 13, or
0.85%. These failures remain in the fixed denominator. Provider latency p50 was 1,787 ms for the
baseline and 1,715 ms for the treatment. Provider latency p95 was 12,385 ms and 12,579 ms.

The baseline used 2,422,709 prompt tokens and 89,530 completion tokens. The treatment used
2,415,532 prompt tokens and 91,621 completion tokens.

## Category results

| Category | Questions | Direct 10 | Direct 8 plus graph 2 | Delta |
| --- | ---: | ---: | ---: | ---: |
| cat1 | 282 | 15.60% | 15.96% | +0.35 points |
| cat2-temporal | 321 | 55.76% | 55.14% | -0.62 points |
| cat3 | 92 | 11.96% | 11.96% | 0.00 points |
| cat4 | 841 | 69.20% | 72.77% | +3.57 points |

Cat4 contains the only material positive answer-stage effect. Cat2-temporal declines slightly and
cat3 is unchanged. The retrieval-stage improvement was broader than the generated-answer effect,
which indicates that added gold evidence did not consistently become cited in the final answer.

## Paired outcomes

The treatment produced 80 complete-citation rescues and 51 regressions, for a net gain of 29
questions. The any-citation paired delta was +1.69 points, with a bootstrap 95% interval of -0.59 to
+4.04 points.

This is a citation-support result, not an independent factual-correctness result. No judge was run,
so this experiment cannot establish that the answers were more factually correct. The modest valid
answer-rate decline and higher provider failure count should also be monitored in a larger or more
reliable replay.

## Interpretation

The preceding retrieval experiment passed its complete-gold-evidence gate at +2.34 points. This
answer replay does not confirm a corresponding answer-quality improvement under the preregistered
2 point threshold. The strongest follow-up is a targeted category 4 analysis with an independent
blind correctness judge, plus inspection of the 80 rescues and 51 regressions to determine whether
the graph adds usable evidence or merely increases context competition.

I do not recommend changing the production default based on this answer-stage result alone. The
retrieval result remains promising, especially for category 4, but answer quality needs a judge and
additional power before treating the graph tail as a general quality improvement.

## Reproduction

The protocol was committed before answer generation in
`docs/preregistrations/2026-09-12-voyage4-graph-tail-answer-quality.md`.

```powershell
python scripts/run_voyage4_graph_tail_answer_quality.py docs/results/2026-09-12-voyage4-graph-tail-quality.json docs/results/2026-09-12-voyage4-graph-tail-answer-quality.json --model deepseek/deepseek-v4-flash --workers 8 --source-commit 1df6feb95868d65698db6c5ac84d12afd6c50e75
```
