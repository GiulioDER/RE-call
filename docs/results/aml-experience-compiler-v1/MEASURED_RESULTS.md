# AML engineering experience compiler measured result

Date: 2026-09-18 UTC

Status: complete retrieval replay; compiler representations rejected by the frozen selector

## Outcome

The mechanical selector retained `E0_raw`. Neither compiler arm passed the preregistered
retrieval gate, so no compiler arm advances to the executable task screen.

`E1_compiled` reduced complete top 10 coverage from 11.76% to 0% and reduced mean reciprocal
rank from 0.2470 to 0.0321. `E2_compiled_raw` recovered all four complete top 10 cases lost by
`E1_compiled`, but its mean reciprocal rank was 0.2100, below the `E0_raw` value of 0.2470.
The frozen E2 rule requires recovery plus mean reciprocal rank no lower than raw, so E2 failed.

## Frozen identity

| Field | Value |
| --- | --- |
| RE-call commit | `952bc0482d23c1e2072fe4225a10fa9b8035c673` |
| agent-memory-bench commit | `4dce1bb931856351912f5754129915db4d34a41b` |
| RE-call `uv.lock` SHA256 | `c7d0396f68d2ad9de8e05bcbc5b3286e588fd9eda8b3250897c17eb20c4d672e` |
| Corpus manifest SHA256 | `58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814` |
| Task set SHA256 | `c886e9ffe47b9eaf5dfc8d97fb9dc54e589ec5037e345d2d1d2233b78fd539c6` |
| Population | 196 sessions, 2,281 messages, 34 tasks |
| HTTP timeout | 180 seconds in every arm |
| Product | `RE-call Hosted 1.0` |
| Generation | OpenRouter `openai/gpt-4o-mini` |
| Embedding | `voyage-4` |
| Retrieval | `hosted-quality` |
| Reranker identity | `voyage:rerank-2.5` |
| Compiler prompt digest | `b68f35c53538cc9cccd00d3ece1a0a04f6493a5ec7e0782e0aa1ec682836856e` |

All three result artifacts carry the exact application commit, product identity, population
digests, task rows, source counts, and 180 second timeout. The copied artifact hashes match VPS2.
No task identifier from the frozen task population appears in the privacy safe service logs.

## Retrieval measurements

| Metric | E0 raw | E1 compiled | E2 compiled plus raw |
| --- | ---: | ---: | ---: |
| Hit at 1 | 14.71% | 2.94% | 11.76% |
| Hit at 5 | 35.29% | 2.94% | 29.41% |
| Hit at 10 | 44.12% | 2.94% | 41.18% |
| Hit at 100 | 97.06% | 14.71% | 97.06% |
| Complete coverage at 5 | 11.76% | 0% | 8.82% |
| Complete coverage at 10 | 11.76% | 0% | 11.76% |
| Complete coverage at 100 | 76.47% | 2.94% | 76.47% |
| Mean reciprocal rank | 0.2470 | 0.0321 | 0.2100 |
| Mean source session recall | 1.0000 | 0.9118 | 1.0000 |
| Mean duplicate session concentration | 0.5174 | 0.0194 | 0.5291 |
| Mean returned characters | 123,678 | 113,566 | 123,807 |
| Add p50 | 529 ms | 10,303 ms | 12,985 ms |
| Add p95 | 3,482 ms | 31,258 ms | 29,465 ms |
| Search p50 | 354 ms | 311 ms | 377 ms |
| Search p95 | 460 ms | 449 ms | 582 ms |
| Compiler fallbacks | 0 | 168 | 171 |

Latency is descriptive only. The replay began while another authorized VPS2 workload was active,
and runtime CPU, memory, and swap caps were removed at 17:15 UTC after that workload ended. The
quality selector does not use latency, and every quality and identity invariant remained frozen.

## Compiler diagnostics

| Diagnostic | E1 compiled | E2 compiled plus raw |
| --- | ---: | ---: |
| Compiler provider calls | 207 | 204 |
| Prompt tokens | 458,142 | 449,075 |
| Completion tokens | 193,324 | 187,732 |
| Total tokens | 651,466 | 636,807 |
| Compile completion events | 125 | 121 |
| Proposed records | 495 | 467 |
| Accepted records | 58 | 50 |
| Rejected for evidence | 437 | 417 |
| Rejected for source session | 0 | 0 |
| Removed entities | 4 | 0 |
| Removed outcomes | 35 | 39 |
| Removed validations | 6 | 4 |
| Removed event times | 0 | 0 |
| Removed supersession references | 0 | 0 |
| Source NUL normalization | 1 event, 4 characters | 1 event, 4 characters |
| Compiler output NUL normalization | 0 events | 0 events |

The dominant failure is evidence grounding. E1 accepted 58 of 495 proposed records, while E2
accepted 50 of 467. The artifact level fallback counters show that 168 of 196 E1 sessions and
171 of 196 E2 sessions required deterministic fallback. Source session validation was not the
problem. Exact evidence support was.

## Artifact hashes

| Artifact | SHA256 |
| --- | --- |
| `952bc048-vps2-retry2/E0_raw.json` | `119af63e760a43990402838ec3d734c5a039eeeae12ec059e293709588f3f001` |
| `952bc048-vps2-retry2/E0_raw.service.log` | `992103992a4bb61d37816cdc4969e3b820462f5eaec09e228a090d8d7012b7d4` |
| `952bc048-vps2-retry2/E1_compiled.json` | `970bf53f7db346c792527975aeff6c739ccd73cb206847f89ee2a4eb8fa50722` |
| `952bc048-vps2-retry2/E1_compiled.service.log` | `cb6e36b330218b6b1469a1fc320b5f14d36935230cb9e6d6370db18e33889c1a` |
| `952bc048-vps2-retry2/E2_compiled_raw.json` | `dff43734cc4750957ff77bc8bf9312aa1e49a1c82d44413eace0a294604dd65e` |
| `952bc048-vps2-retry2/E2_compiled_raw.service.log` | `60832448674564cc461bc80cddc81b0ce0dafc126e8bc38ca7fb819cd9d64935` |
| `952bc048-vps2-retry2/selection.json` | `350c40a9706ddd4983466fb9ecfd9e59b3d63827207aad6c61ad00eac171d156` |

## Verification

The selector was rerun from the frozen benchmark commit. Its semantic output matched the recorded
selection exactly, and its LF normalized SHA256 matched the recorded selection hash. The focused
benchmark suite passed 16 tests. Ruff passed on the selector, replay, hosted selector, and focused
tests.

The result supports two conclusions. First, compiled only storage is not viable in its current
form because evidence rejection and fallback dominate. Second, adding compiled records beside raw
records preserves complete top 10 coverage but dilutes early rank. The next experiment should
improve deterministic evidence span binding and compiled record admission before retesting a
compiled plus raw arm. Threshold tuning against this result is not permitted.
