# Atomic fact current generation shadow stopped before live integration

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-current-generation-shadow.md`](../preregistrations/2026-09-16-atomic-fact-current-generation-shadow.md)
and its premeasurement
[`query scope clarification`](../preregistrations/2026-09-16-atomic-fact-current-generation-query-scope-clarification.md).

## Verdict

`STOP_BEFORE_LIVE_SHADOW`.

The current generation auxiliary index is small and deterministic, but the exact NumPy search
implementation missed both frozen tail latency gates. No live shadow was enabled and no served
answer changed.

This run also produced a useful warning about generalization. On the consumed 22 answerable
production queries, pure atomic retrieval did not improve rank one gold and was worse than dense
at every later cutoff. Those rows are diagnostic only and cannot establish a prospective quality
verdict, but they remove the basis for assuming that the first 22 row pilot uplift transfers to
ordinary query wording.

## Production corpus engineering result

| Measure | Result | Frozen gate | Verdict |
| --- | ---: | ---: | --- |
| Atomic views | 6,310 | at most 7,500 | pass |
| Artifact size | 25,107,048 bytes | at most 64 MiB | pass |
| Zero view sources | 151 of 1,563, 9.661% | at most 12% | pass |
| Ranking p50 | 88.854 ms | diagnostic | measured |
| Ranking p95 | 114.526 ms | at most 100 ms | fail |
| Ranking p99 | 176.258 ms | at most 150 ms | fail |
| Ranking maximum | 197.066 ms | diagnostic | measured |
| Determinism failures | 0 of 250 rankings | zero | pass |

The artifact covered 1,412 sources with views and mapped them to 11,365 ordinary chunks. The build
verified 1,563 manifest objects, excluded 15 aggregate or index sources by the frozen basename
rule, and excluded seven objects outside the seven frozen roots.

View construction took 8.376 seconds, document embedding took 47.339 seconds, query embedding took
11.759 seconds, and the 50 dense top 20 reads took 10.122 seconds. Total wall time was 112.786
seconds under the shared embedding lock, 8 GB memory cap, zero swap, 250 percent CPU quota, nice
level 15, and four embedding threads.

The reported additional maximum RSS was zero bytes. That value is not useful evidence of zero
memory cost: imports established a higher earlier process peak than loading the 25 MB artifact,
so `ru_maxrss` could not observe the incremental allocation. The artifact size remains valid, but
a future memory measurement must use current resident pages in a fresh worker instead of a delta
between historical maxima.

## Consumed query diagnostic

The 50 query file contains 22 answerable rows and 28 unanswerable controls. All 50 participated in
latency and movement measurements. Only the 22 labelled rows participated in gold source metrics.

| Arm | Gold @1 | Gold @3 | Gold @5 | Gold @10 | Gold @20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 17 | 20 | 21 | 21 | 22 |
| Atomic | 17 | 19 | 19 | 19 | 21 |

Atomic produced three rank one gains and three rank one losses. Rank one changed on 36 of all 50
queries. Mean dense and atomic overlap was 28.0 percent at rank one, 34.8 percent at rank five,
36.6 percent at rank ten, and 34.6 percent at rank twenty. One labelled row had no atomic view for
any gold source.

These numbers are not a confirmation set. They were exposed before this experiment and remain
excluded from promotion, tuning, and threshold selection. Their valid use is diagnostic: pure
atomic replacement is not safe enough to justify serving on the evidence available now.

## Integrity and production state

The run used detached source commit `c250c6517b2bbed306ece7805394847eb41ec91a` against production
generation `gen_83393e5c58524eecb9e390cb367e4d61`, certified calibration
`cal_86a67f38755c45f38456d6b2445767cc`, pipeline
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus
`5357c7fb736dfd8add6182e41f97932a8bd6c103daae0f9bf8d984f113f87c42`.

The active generation and its full binding were unchanged after measurement. Production routing
was not modified. The serving checkout itself advanced concurrently from `21d59ae6` to `5366770e`
for an unrelated commit named `preserve published ATM harness bytes`. Because the preregistration
also asked for an unchanged checkout, that literal condition cannot be certified even though the
database lineage and running route stayed fixed. The stop verdict already applies independently
because both latency gates failed.

Private vectors and per query candidates remain mode 0600 at
`/home/sentiment/.codex/evals/atomic-fact-current-generation-2026-09-16/`. Their hashes are recorded
in the public JSON. No query text, source text, or candidate identity is committed.

## Next decision

Do not enable a live atomic shadow from this implementation and do not promote pure atomic
retrieval.

The next highest return work is prospective quality, not another selector trained on these
consumed rows. Accumulate new sources absent from the pilot and freeze independently worded
questions plus gold before retrieval. In parallel, a small preregistered performance microbenchmark
can profile matrix scoring versus candidate selection and test an exact top candidate shortlist.
That optimization is worthwhile only as reusable infrastructure; it cannot repair the missing
quality confirmation.

The machine readable aggregate is
[`2026-09-16-atomic-fact-current-generation-shadow.json`](2026-09-16-atomic-fact-current-generation-shadow.json).
