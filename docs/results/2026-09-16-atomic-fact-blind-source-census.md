# Atomic fact blind source census stopped one row below its gate

Measured 2026-09-16 under the original
[`source census`](../preregistrations/2026-09-16-atomic-fact-blind-source-census.md) and its committed
[`completion ceiling correction`](../preregistrations/2026-09-16-atomic-fact-blind-source-census-ceiling-correction.md).

## Verdict

`STOP_BLIND_SOURCE_CENSUS`.

The corrected run generated exactly one completion for each of the 45 frozen source candidates.
Thirty-one questions passed every frozen validation rule, one below the minimum of 32. The census
therefore remains stopped and cannot serve as confirmation or authorize production.

The 31 accepted rows are nevertheless a sealed exploratory set: their membership, exact answer
span, gold parent, and gold source were fixed before any retrieval call. A separate preregistration
may use them for a diagnostic paired comparison, provided it makes no promotion claim.

## Construction result

| Measure | Result | Frozen gate | Verdict |
| --- | ---: | ---: | --- |
| Frozen candidates | 45 | exactly 45 | pass |
| Provider completions | 45 | exactly 45 | pass |
| Accepted questions | 31 | at least 32 | fail |
| Answer five-token overlap rejections | 7 | diagnostic | measured |
| Forbidden structural word rejections | 4 | diagnostic | measured |
| Invalid JSON rejections | 3 | diagnostic | measured |
| Integrity or validation errors after acceptance | 0 | zero | pass |
| Distinct accepted sources | 31 | equal accepted rows | pass |
| Accepted roots | 4 | at least 3 | pass |
| Largest root share | 18 of 31 | at most 80 percent | pass |

The accepted root distribution is 18 `agent-memory-bench`, 11 `recall`, one
`ai-boost-av-safety`, and one `cca-demos`. The private pool passed the restricted NTFS access-list
check and remains outside the repository.

The provider reported 7,514 prompt tokens, 10,986 completion tokens, 18,500 total tokens, 210,694
milliseconds aggregate latency, and cost of 0.0036397606 dollars. The private pool SHA256 is
`97f77c71c1feb278b9d5297511e8b4fb913002208eaae2b3bbd2dc65d578fd18`.

No retrieval ran during construction. The machine-readable aggregate is
[`2026-09-16-atomic-fact-blind-source-census.json`](2026-09-16-atomic-fact-blind-source-census.json).
