# Exact atomic rescue shortlist passed

Status: `PROMISING_EXACT_RESCUE_SHORTLIST`.

## Outcome

The exact masked maximum removed the measured latency blocker without changing which atomic rescue
parent was selected. Across 31 real queries and 20 repetitions, both arms produced 620 timed
samples on the current production sized matrix of 6,317 atomic views.

| Measurement | Full sort reference | Exact masked maximum |
| --- | ---: | ---: |
| p50 | 8.634 ms | 1.373 ms |
| p95 | 49.169 ms | 3.445 ms |
| p99 | 54.101 ms | 40.495 ms |
| maximum | 226.907 ms | 43.442 ms |
| total over 620 samples | 9,442.372 ms | 1,393.933 ms |

The reference-to-candidate p95 ratio was 14.273. The candidate stayed below the frozen 25 ms p95
and 50 ms p99 gates. All 31 selected identities and scores matched exactly. Repeated candidate
selection produced zero determinism failures.

## What changed

The reference computes all cosine scores, sorts all 6,317 views, deduplicates parents, and scans
until it finds a parent outside dense top five. The candidate computes the same scores, applies a
vectorized mask for the five dense parent codes, takes the exact maximum, and resolves exact score
ties with the existing source, parent ordinal, and view ordinal order.

This is not approximate retrieval. It preserves the selected source, parent ordinal, and score.
The regression test also covers multiple atomic views for one parent and exact score ties. Its red
proof changed the production mask comparison from equality to inequality and failed at the intended
identity assertion before the correct implementation passed.

## Execution and lineage

The run used generation `gen_e5c95bffed8c41bb9c05680325fb8d60`, calibration
`cal_aa2d53e5d051418c8d7b28534b862afc`, pipeline
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus
`0587af4766daf313067eb07a72f7489765ff4b0192a96bf9c0d1142b24d5c0e7`. The corpus contained
11,376 ordinary chunks, 1,565 manifest objects, and 1,414 sources with atomic views.

Embedding and timing ran on VPS2 under the shared flock and the frozen 8 GiB memory, zero swap,
250 percent CPU, nice 15, four-thread policy. The active generation remained unchanged. The lock
was free and the experiment process was absent after completion.

The document embedding phase took 1,058.910 seconds because the hosted embedding calls were slow
during this run. That setup cost is outside the selector latency measurement. A production design
must build the atomic matrix during generation construction and reuse it rather than embedding the
corpus on each search.

## Evidence boundary and next step

This result preserves, but does not independently confirm, the earlier blind retrieval gain. The
31-query pool was reused only to supply realistic vectors and exclusions, so no new quality claim
is made here.

The next justified step is an off-by-default aggregate production shadow that loads a generation
bound atomic matrix once, runs the exact masked maximum after dense top five, records latency and
selection overlap without exposing row content, and never changes served results. That shadow must
measure end-to-end overhead, load time, memory, generation rollover behavior, and failure isolation
before any serving proposal. A larger prospective source-disjoint set is still required to confirm
the 28 of 31 versus 24 of 31 quality result.

The machine-readable aggregate is in
`docs/results/2026-09-16-atomic-fact-exact-rescue-shortlist.json`. Private query text, candidate
identities, and row-level outcomes remain outside the repository.
