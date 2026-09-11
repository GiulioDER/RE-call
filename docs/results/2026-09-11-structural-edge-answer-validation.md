# Structural edge answer-stage validation

Measured 2026-09-11 from the frozen full LoCoMo structural-edge retrieval artifact, with
1,536 questions per arm and 6,144 answer rows. The replay used the configured OpenRouter
answer provider `deepseek/deepseek-v4-flash`, temperature zero, an explicit reasoning effort
of `none`, a 512-token output ceiling, the production evidence prompt, and the production
answer envelope and citation validator.

Input artifact: `2026-09-11-structural-edge-performance-locomo-most-direct.json`.
Input source commit: `50818ede84db229abc397b7db4275cbf083c9f6b`.

## Provider preflight and repair

The first five-question-per-arm preflight exposed a configuration failure: without an explicit
OpenRouter reasoning setting, DeepSeek used the full 512-token ceiling for hidden reasoning and
returned empty or non-JSON content. The corrected preflight returned 20 valid envelopes with
zero errors. The provider now sends `reasoning: {"effort": "none"}` for this bounded JSON task.

## Results

| arm | valid envelope | any gold evidence cited | complete gold citation coverage | mean gold citation recall | provider calls | cache hits |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 99.48% | 58.66% | 49.54% | 0.5366 | 1,525 | 11 |
| retrieval-ranked, 8 direct + 2 structural | 99.41% | **60.22%** | **50.85%** | **0.5534** | 1,505 | 31 |
| retrieval-ranked, 9 direct + 1 structural | 99.41% | 59.11% | 50.13% | 0.5420 | 1,148 | 388 |
| category selective, 8 direct + 2 structural | **99.54%** | 60.35% | **51.17%** | 0.5522 | 0 | 1,536 |

The category-selective rows were all exact prompt duplicates of an already measured arm, so
their answers were safely reused rather than billed again. Across all arms, 4,178 unique
prompts produced 1,966 exact cache hits, reducing duplicate calls by 32.0%.

For the paired questions with usable citation metrics, 8 plus 2 rescued 74 complete gold
citation misses and regressed 55 complete cases relative to baseline. The 9 plus 1 arm
rescued 53 and regressed 45. The answer-stage result therefore follows the retrieval result:
8 plus 2 is stronger than 9 plus 1, while category-selective routing has the lowest marginal
answer-call cost and the highest valid-envelope rate.

Paid provider-call latency was 4,953 ms mean and 15,358 ms p95 for baseline, 5,096 ms mean
and 15,822 ms p95 for 8 plus 2, and 4,705 ms mean and 15,929 ms p95 for 9 plus 1. The
treatment p95 stayed below twice baseline. Mean total provider tokens were 1,609, 1,601,
and 1,603 respectively. Monetary cost was not recorded because the provider did not return a
cost field and no price was inferred.

## Scope and decision

This validates answer-envelope shape, citation identity, and whether generated answers cite
annotated evidence. It does not establish factual answer correctness. No external factual
judge was run, so these rows must not be reported as a LOC​​OMO LLM-as-a-judge score.

The answer-stage citation result supports keeping retrieval-ranked 8 direct plus 2 structural
as the leading global candidate, with category-selective 8 plus 2 as the lower-call alternative.
Production remains unchanged pending a separately authorized factual judge or human review.
