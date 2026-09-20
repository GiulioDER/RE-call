# AML anchor compiler v3 admission result

Date: 2026-09-19

## Verdict

Anchor compiler v3 is source-grounded and operationally reliable, but it is not admitted into the
current AML retrieval path. The mechanical selector retained `V3_raw` because compiled records
reduced complete coverage at rank 100. The result explicitly does not authorize M2, M3, AML Smoke,
or AML Full.

## Frozen identities

| Item | Identity |
| --- | --- |
| RE-call commit | `97fdc2a8d160a559d78e496f49315d23d6116bb3` |
| Agent Memory Bench commit | `5b1cff3017e1180f56c43d3781569946ee5d9fc6` |
| Corpus manifest | `58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814` |
| Task set | `c886e9ffe47b9eaf5dfc8d97fb9dc54e589ec5037e345d2d1d2233b78fd539c6` |
| Selection | `2e4b55d8b8fe8e4227ea431b56ee81943d9a476faf14fdc1a5eb7547204d9357` |
| Identity artifact | `194ec04a313e4347232fd0c2dcd5d0043051d022554091e1bf4769458dcb7675` |

The immutable evidence is under
`results/aml-anchor-compiler-v3/97fdc2a8-5b1cff30-pilot-r2/`. Its `SHA256SUMS` verifies all
eight expected artifacts.

## Measured gates

| Gate | Result | Verdict |
| --- | ---: | --- |
| Accepted typed sessions | 194 of 196, `98.98%` | pass |
| Whole-session fallback | 2 of 196, `1.02%` | pass |
| Audited compiled records | 1,230 | pass |
| Unsupported claims | 0 | pass |
| Invalid spans | 0 | pass |
| Wrong profiles | 0 | pass |
| Sessions retaining raw records | 196 of 196 | pass |
| Complete coverage at rank 100 | raw `94.12%`; candidate `91.18%` | fail |

The one complete-coverage loss was `ts-golden-regen`. The candidate also reduced mean reciprocal
rank from `0.3217095592` to `0.2748844943`, an absolute change of `-0.0468250649`. Search p95
rose from `388.225 ms` to `532.347 ms`. Add p95 rose from `660.261 ms` to `20,973.655 ms`
because compiler v3 performs bounded `gpt-4o-mini` extraction during Add.

## What the experiment established

The grounding problem is largely solved. Deterministic anchors, field-level validation, exact
fallback, and unique record accounting produced audited records for almost the entire corpus.
The remaining failure is retrieval integration: 1,230 compiled records competed with raw evidence
inside a bounded candidate and Top-K budget. Raw retention at storage time did not guarantee raw
preservation at Search time.

This matters for the leader-inspired multi-view direction. Typed repository and engineering
experience views remain promising, but returning their records as peers of raw chunks is too
expensive in scarce rank slots. The next experiment should use typed retrieval as a sidecar signal:

1. Produce M0's raw candidate set unchanged.
2. Retrieve grounded typed records independently.
3. Map typed hits to their source sessions.
4. Apply a bounded score only to raw items already present in M0.
5. Return raw evidence only and require exact rank-100 membership and coverage equality with M0.

This design tests whether typed memory improves early ordering without letting generated records
displace exact code, commands, errors, or tests. It needs a new frozen preregistration and an
executable screen before promotion.

## Decision

Keep M0 raw dense plus exact lexical retrieval as the baseline. Close the direct compiled-record
admission lane. Do not run the draft M2/M3 experiment. Prepare a typed-sidecar session-reranking
ablation as the next high-ROI local experiment, then revisit query-shape aliases and task routing
only if that ablation preserves the raw tail and improves executable Task Solve.
