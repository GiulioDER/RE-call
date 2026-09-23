# Memory-tenant check of the C8-confirmed micro atomizer

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

The ungated `micro` atomizer passed confirmation on C8/CAMBench
(`2026-09-23-c8-atomizer-round3.md`). The program requires a memory-tenant check before any
atomizer change could reach production. On the memory tenant's production path, is `micro` at
least as good as production's own memo paragraph atomizer, and better than no rescue, on fresh
production-disjoint questions? And can it be served within production's frozen latency limits?

Context measured 2026-09-23, before this record: production atomic rescue is **off**
(`RECALL_ATOMIC_RESCUE_MODE=off` since 2026-09-17 08:26 UTC, reason unrecorded), and no artifact
exists for the active generation. This check cannot switch it on; it informs whether it should be,
and with which atomizer.

## Frozen apparatus

- VPS2, memory tenant, generation pinned to `gen_b6aefc110e0d42588f3a57c98aa29129`
  (`memory-20260923-voyage-r204`; corpus `f344b28e…`, pipeline `1c1be8a2…`), read only:
  11,735 chunks from 1,652 sources, Context4 (`voyage-context:voyage-context-4`). The user
  authorized this run on VPS2 on 2026-09-23; the official AML services there are not touched.
  Runs are niced and memory-bounded, holding `embed.lock` without waiting.
- Harness `scripts/memory_atomizer_check.py`, mirroring production's memory path: Context4 query,
  `query_dense` top 100, dense top five kept, one atomic winner at slot six
  (`insert_atomic_rescue_dense`).
- Arms: `off`; `prod`, the production memo paragraph atomizer built for the pinned generation by
  the unchanged `scripts.build_atomic_fact_production_artifact` into a private directory; `micro`,
  24-word windows with a 12-word stride inside each stored chunk, parent = that chunk, embedded
  with Context4 grouped by source.
- Questions: seed 20260924, one span of 12 to 30 words per sampled source, 240 sources drawn from
  those **no prior atomic evaluation record under `~/.codex/evals/atomic-fact-*` names as a
  source**. Gold is every chunk of that source containing the span. Writer
  `meta-llama/llama-3.3-70b-instruct` (strict schema, the reference record's rejection rules, a
  memory-note prompt). Split dev/confirm by source hash. Probe file hash appended before
  evaluation. Cap USD 0.50 in code.

## Metrics

Primary, on **dev**: paired exact-chunk gains and losses at rank 6 for `micro` against `off` and
for `micro` against `prod`. Secondary: rank 1 and 5, source-level hits, admitted rescues that are
exact gold, artifact size, and selector latency against production's frozen active limits
(p95 20 ms, p99 45 ms).

## Predictions

1. Writer yield: 170 to 225 kept questions.
2. `off` exact@6 on dev: 45% to 75%.
3. `prod` against `off`, exact@6: net +1 to +5, at most 2 losses.
4. `micro` against `off`, exact@6: net +1 to +6, at most 2 losses.
5. `micro` against `prod`, exact@6: net 0 to +3.
6. `micro` holds about 66,000 views (about 270 MB); its selector p95 exceeds 20 ms, while `prod`'s
   stays under 20 ms.
7. Exact-gold share of `micro`'s rescues: 5% to 15%.

## Decision rules

- `micro` **passes dev** if, at exact@6, it nets at least +2 against `off` with at most 2 losses
  **and** nets at least 0 against `prod`.
- If it passes dev, the confirm split runs **once** for all three arms. `micro` **passes the
  memory check** if confirm meets: net at least +1 against `off` with at most 2 losses, and net at
  least 0 against `prod`.
- The latency gate is reported separately. A quality pass with a failed latency gate is recorded
  as "better but not servable as configured", and the next step would be a smaller view set, not
  enablement.
- Nothing here changes production's mode, registry or generation.

## Result

Not yet run.

## Apparatus appendix, appended 2026-09-23 before any evaluation

Apparatus only; no arm had been evaluated and no view embedded when it was written.

- **The exclusion removed more than planned.** Applied as frozen (any source a prior
  `atomic-fact-*` record names as a source), it excludes **1,496 of 1,652 sources**, because the
  census and holdout records list nearly every source as a candidate, not only as gold. Only 153
  spans could be drawn instead of 240. The surviving sources are overwhelmingly memos written
  after the 2026-09-16 evaluations, which is a strict reading of "production-disjoint", at the
  cost of sample size. The rule and the thresholds are unchanged.
- The question step ran in two parts, because VPS2 holds no OpenRouter key outside the official
  AML run's own environment, which was not used: spans sampled on VPS2 (`spans`, file SHA-256
  `e49cdc0e1472f8d1…`), questions written on VPS3 (`write`). No credential moved between hosts.
  The sampling and writer are unchanged from the frozen apparatus.
- Question file SHA-256 `efc4c18332d00e52…`: **134 kept** (18 copied the span, 1 not answerable),
  67 dev and 67 confirm, 134 distinct sources (65 `sentiment-agent`, 64 `recall`, 4
  `agent-memory-bench`, 1 `steel`), no multi-chunk gold. Writer Llama 3.3 70B through five
  providers, USD 0.010342.
- Prediction 1 (170 to 225 kept) is already visible and **falsified low** (134), a direct result
  of the exclusion shortfall above. The dev split is about two thirds of the size the thresholds
  were written for; the thresholds stand.

## Result, measured 2026-09-23 on VPS2

Appended after the measurement; nothing above has been edited. Harness commit `e0bb253a`, pinned
generation unchanged throughout (`active_generation_unchanged: true` on both runs), `embed.lock`
held without waiting, both runs inside `MemoryMax=8G`, `CPUQuota=250%`, `nice 15`. VPS2 load
average stayed near 2.3. No AML service was touched.

Artifacts for the pinned generation: `prod` (the unchanged production builder) 6,515 views over
4,400 parents, 26.7 MB, builder decision `READY_FOR_ATOMIC_PRODUCTION_SHADOW`; `micro` 76,568 views
over 11,700 parents, 313.6 MB, embedded in 210.8 s.

| exact chunk | dev @1 | dev @5 | **dev @6** | dev @10 | conf @1 | conf @5 | **conf @6** | conf @10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `off` | 26 | 42 | 43 | 48 | 35 | 51 | 51 | 51 |
| `prod` | 26 | 42 | 44 | 49 | 35 | 51 | 52 | 52 |
| `micro` | 26 | 42 | **52** | 53 | 35 | 51 | **57** | 57 |

| paired at exact@6 | dev gains/losses (net) | confirm gains/losses (net) |
|---|---:|---:|
| `prod` vs `off` | 2 / 1 (+1) | 1 / 0 (+1) |
| **`micro` vs `off`** | **9 / 0 (+9)** | **6 / 0 (+6)** |
| **`micro` vs `prod`** | **8 / 0 (+8)** | **5 / 0 (+5)** |

Ranks 1 to 5 never move in any arm, as production's path guarantees. Source-level results track
the exact ones (dev +8, confirm +5 for `micro` against `off`). Rescued chunk exact gold: `micro`
10 of 67 (dev) and 6 of 67 (confirm); `prod` 2 and 1. No fallback in any run.

Selector p95: `micro` **27.24 ms** (dev) and **39.41 ms** (confirm); `prod` 5.01 and 5.72 ms. p99
was not computed by the harness although the Metrics section names the p99 limit; that is an
omission of this run, not a measured pass.

Predictions, scored:

1. Writer yield 170 to 225: **falsified low** (134), as stated in the appendix.
2. `off` exact@6 on dev 45% to 75%: **confirmed** (64.2%).
3. `prod` vs `off` net +1 to +5, at most 2 losses: **confirmed** (+1, 1 loss).
4. `micro` vs `off` net +1 to +6, at most 2 losses: **falsified high** (+9, 0 losses).
5. `micro` vs `prod` net 0 to +3: **falsified high** (+8).
6. About 66,000 views and 270 MB, `micro` p95 above 20 ms, `prod` below: size **falsified**
   (76,568 views, 313.6 MB, 16% more), latency **confirmed** (27.2 ms against 5.0 ms).
7. `micro` exact-gold rescue share 5% to 15%: **confirmed** (14.9%).

**Decision.** `micro` **passes dev** (net +9 against `off` with 0 losses, net +8 against `prod`)
and **passes the memory check on confirm** (net +6 against `off` with 0 losses, net +5 against
`prod`). It **fails production's latency limit as configured** (p95 27 to 39 ms against 20 ms).
Recorded outcome, per the frozen rule: **better but not servable as configured.** The next step is
a smaller `micro` view set that fits the latency budget, measured on fresh questions; nothing here
enables production, whose atomic rescue mode remains `off`.
