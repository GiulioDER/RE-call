# Graph answer context diagnostic correction

Measured 2026-09-12 after adding the explicit OpenRouter reasoning control. The run used the
one hop evidence from 22 answerable frozen queries at three context caps, with
`deepseek/deepseek-v4-flash`, temperature zero, `reasoning: {"effort":"none"}`, and a 512 token
maximum.

## Results

| cap | validation failures | valid answers | any gold citation | mean gold citation recall | prompt p95 | provider latency p95 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1 semantic validation error | 95.5% | 50.0% | 0.263 | 9,037 chars | 19,938 ms |
| cap5 | 0 | 100% | 50.0% | 0.263 | 5,858 chars | 17,051 ms |
| cap3 | 0 | 100% | 40.9% | 0.218 | 3,848 chars | 18,069 ms |

The original failures were caused by hidden reasoning consuming the entire 512 token completion
budget. The raw response showed `completion_tokens: 512`, `reasoning_tokens: 511`,
`finish_reason: length`, and null content. Sending the explicit no reasoning field removed those
empty and malformed JSON failures.

The five item cap is the best current candidate. It preserved full cap citation recall, reduced
the prompt p95 by 35%, and had no provider or validation failures in this 22 query diagnostic.
The three item cap reduced citation recall by 4.5 percentage points and is not preferred.

The one full cap failure was a semantic envelope error: the model returned an answer together with
`insufficient_evidence=true`. It was not a transport failure.

This is a local correction diagnostic against the patched checkout. It is not yet a production
VPS2 result until the provider change is merged and deployed.

Raw artifact:
`docs/results/2026-09-12-graph-answer-context-diagnostic-correction.json`.
