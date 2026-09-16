# Atomic fact release confirmation passed

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-release-confirmation-retrieval.md`](../preregistrations/2026-09-16-atomic-fact-release-confirmation-retrieval.md).

## Verdict

`PASS_ATOMIC_RELEASE_CONFIRMATION`.

Dense-five plus one exact atomic rescue passed every prospective quality and integrity gate on 96
development-disjoint source questions. It improved exact parent reach by 16 rows and gold source
reach by 11 rows, with zero losses for either label.

| Rank-six arm | Exact parent | Gold source |
| --- | ---: | ---: |
| Dense six | 63 of 96 | 72 of 96 |
| Dense five plus one atomic | 79 of 96 | 83 of 96 |
| Absolute change | +16.67 points | +11.46 points |

The paired exact result was 16 gains, zero losses, and 80 ties, with one-sided exact probability
`0.0000152588`. The paired gold result was 11 gains, zero losses, and 85 ties, with one-sided exact
probability `0.000488281`. Both effects exceed the preregistered six-row minimum.

Gains were not confined to one root. Exact gains were 12 `sentiment-agent`, three `recall`, and one
`agent-memory-bench`. Gold gains were eight, two, and one respectively. No root recorded a loss.

## Selector and lineage checks

The exact masked maximum matched the full deterministic sort in identity and score on all 96
queries. Every score was finite, every arm contained six distinct parents, all rows were measured,
and the active generation remained unchanged.

| Selector latency | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| Full sort | 12.048 ms | 21.618 ms | 29.978 ms |
| Exact masked maximum | 1.684 ms | 3.697 ms | 7.510 ms |

The run used certified generation `gen_dff506e12f494965af9f109671a99e63`, calibration
`cal_6171177aeb614d288baa4e28600caca1`, pipeline
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, corpus
`522e9d1c506142d8a6a5a1f5be9073ff9409132d356e29f61fbf8e7d6d2044a0`, 11,385 ordinary
chunks, and 6,322 atomic views.

Embedding and measurement ran on VPS2 under the shared flock and the frozen 8 GiB memory, zero
swap, 250 percent CPU, nice 15, and four-thread policy. The scope was inactive, the lock was free,
and no experiment or index process remained after completion.

## Evidence boundary and production decision

This is a release-quality confirmation of candidate retrieval benefit, not yet a serving result.
The 96 sources were excluded from all atomic representation and rescue development populations,
questions were frozen before retrieval, and no parameter was tuned on their outcomes.

The result authorizes the preregistered off-by-default production shadow. That shadow must load a
generation-bound matrix once, reuse the request query vector and dense trace, record aggregate
latency and overlap, fail independently, and leave served candidates byte-for-byte unchanged.
Active serving remains unauthorized until the live shadow passes its load, memory, latency,
concurrency, rollover, and failure-isolation gates.

The private row artifact remains outside the repository with SHA256
`590548c5a7ed038bad4b245a6fc0f4bd8ae8c9f37e98f716e39148e6037b2442`. The machine-readable
aggregate is
[`2026-09-16-atomic-fact-release-confirmation-retrieval.json`](2026-09-16-atomic-fact-release-confirmation-retrieval.json).
