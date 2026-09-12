# Generation graph metadata cache answer quality result

Measured on 2026-09-12 against retrieval merge commit
`fe3a3bad7390197f35e91c6ed944fbb6a99c3574` using OpenRouter
`deepseek/deepseek-v4-flash`.

## Inputs

| Field | Value |
| --- | --- |
| Retrieval artifact | `2026-09-12-generation-graph-metadata-cache-performance.json` |
| Retrieval artifact SHA256 | `0333d36b3b4a93e2ae4dc87c7caae1a7fefacb278e1c09067ce8a68b761fa310` |
| Answer artifact SHA256 | `508ab4efb6e9ca5255a018060ea00a3c9cddc68cf4e7c9c2e437dcf0716e0097` |
| Generation | `gen_6559b6dbacfc4bc2847b65005f1ba1d4` |
| Query set | 50 queries, digest `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192` |
| Prompt digest | `067a63719933cdd9cfa2bbda9f68921fa14aa4dbe3259ffd754bfcfa14a27deb` |
| Sampling | Temperature zero, no reasoning, 512 token maximum |

## Results

| Metric | Graph off | Graph one hop | Delta |
| --- | ---: | ---: | ---: |
| Valid answer envelope | 98% | 98% | 0 pp |
| Nonempty answer | 36% | 36% | 0 pp |
| Correct abstention, unanswerable | 96.4% | 96.4% | 0 pp |
| Any gold citation | 22% | 18% | minus 4 pp |
| Complete gold citation coverage | 4% | 4% | 0 pp |
| Mean gold citation precision | 0.465 | 0.394 | minus 0.071 |
| Mean gold citation recall | 0.228 | 0.211 | minus 0.017 |
| Provider failure rate | 2% | 2% | 0 pp |
| Provider latency p50 | 3,565 ms | 3,100 ms | minus 465 ms |
| Provider latency p95 | 11,606 ms | 10,099 ms | minus 1,507 ms |
| Prompt tokens | 25,058 | 36,537 | plus 11,479 |

The paired result had one validity gain, one validity loss, and 48 unchanged queries. Gold citation
recall decreased by 1.7 percentage points. Any gold citation decreased by 4 percentage points.
The graph path also increased prompt tokens by 45.8 percent.

## Decision

The replay is usable at the two percent provider failure boundary. The preregistered quality guard
passed for valid envelopes, citation identity validity, gold citation recall, and correct
abstention because none dropped by more than two percentage points. The prediction of a positive
quality benefit was not supported. The latency gate had already failed, so this result does not
support enabling graph expansion as a performance or quality rollout.

This run does not measure factual correctness. The stored answers support a later fixed judge or
human review only.

## Reproduction

```powershell
$env:RECALL_REASONING_ANSWER_ENABLED='1'
$env:RECALL_REASONING_ANSWER_PROVIDER='openrouter'
$env:RECALL_REASONING_ANSWER_MODEL='deepseek/deepseek-v4-flash'
$env:RECALL_REASONING_ANSWER_MAX_TOKENS='512'
python scripts/run_live_graph_answer_quality.py docs/results/2026-09-12-generation-graph-metadata-cache-performance.json docs/results/2026-09-12-generation-graph-metadata-cache-answer-quality.json --source-commit fe3a3bad7390197f35e91c6ed944fbb6a99c3574 --workers 8
```
