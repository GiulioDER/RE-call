# Pre-registration: can C9's Add be made faster without changing what it stores?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

Two independent questions, each answered by a number against a rule fixed below:

1. **Internal Add concurrency.** Does raising `RECALL_AML_ADD_CONCURRENCY` from 8 (as served) to
   12 or 16 raise C9's Add throughput by at least 20%, when Adds arrive as AML sends them?
2. **OpenRouter provider.** Does pinning the gpt-4o-mini provider (OpenAI or Azure) make C9's
   Add-time compile at least 20% faster than the other provider, without failing more often or
   keeping fewer records?

Neither changes the model, the prompt or the retrieval. Both are candidates for the AML Textual
and Coding Full runs, and neither is applied by this record: deploying either to the official C9
is a separate decision for the user after the result.

## Why it is being asked

The Textual smoke on the official C9 (2026-09-24, `385c6074`, platform concurrency 16/16, 134
Adds) showed where an Add's time goes:

| part | p50 | p90 | share |
| --- | --- | --- | --- |
| gpt-4o-mini anchored compile | 17.9 s | 32.0 s | 84% |
| embedding and storing after it | 2.4 s | 5.9 s | 16% |

The compile writes about 1,300 output tokens (p50 1,301, p90 1,692), so generation speed sets it.
The smoke's tail was one user's Adds arriving strictly one at a time, where per-Add latency, not
concurrency, sets the pace. C9 does not pin a provider, and OpenRouter serves
`openai/gpt-4o-mini` from OpenAI and Azure at the same price.

## What I already know

- **The 2026-09-23 ramp** (`docs/preregistrations/2026-09-23-aml-c9-max-concurrency.md`, branch
  `claude/aml-concurrency-probe`, C9 at `936b7bda`, platform Add 24) measured Add throughput by
  internal K: K=3 0.277/s, K=6 0.356/s (1.29x), **K=8 0.466/s (1.68x), K=12 0.444/s (1.60x)**.
  The gain stopped at 8. Search throughput did not move with K at all (about 2.4/s), which pointed
  at the process-wide flock that serialises every Voyage call (`recall_aml/embedding_lock.py`).
- **#751 (`385c6074`)** changed two things since: cache keys are hashed once per group, and a
  call's independent Voyage requests go out 4 at a time. It did not remove the flock. Its
  measured gain was on one 600,000 character Add (451.9 s to 136 s); on small Adds it should be
  small.
- **The same-user test** (`docs/preregistrations/2026-09-24-aml-c9-same-user-add-concurrency.md`)
  passed at K=8 on VPS3 with these LoCoMo sessions. It measured lock behaviour, not throughput.
- I over-predict effect sizes (memory: `i-over-predict-effect-magnitudes`), so the bands below
  sit at a quarter to a half of what the mechanism would allow at best.

## What I predict

**1. Internal Add concurrency.** Arms, in this order: K=8 (`k8a`), K=16, K=12, K=8 again (`k8b`).
The flock was the likely limit at K=8 before #751 and #751 left it in place, so:

| arm | Adds per minute, relative to the mean of `k8a` and `k8b` |
| --- | --- |
| K=12 | 0.95 to 1.15 |
| K=16 | 0.95 to 1.20 |

- Absolute throughput at K=8: 20 to 40 Adds per minute on these sessions (about 3,000
  characters each, far smaller than AML Textual Adds).
- `k8a` and `k8b` within 15% of each other.
- 48 of 48 Adds reach 200 in every arm, and at most 2 first attempts are not 200 in any arm.
- The part of each Add after its compile (the embedding under the flock, plus storing) grows
  with K. That is the mechanism check: if it does not grow, the flock is not what limits.
- Peak unit memory under 1.5 GiB in every arm.
- Probability that K=12 or K=16 passes the 1.20 rule below: about 20%.

**2. Provider.** 40 sessions, each compiled through `default` (unpinned, as served), `openai`
and `azure`, one call at a time, in rotating order:

- One provider is faster than the other by 10% to 30% in the paired median of compile time. I
  do not know which one, so I predict no direction.
- Probability that the faster one clears the 0.80 rule below: about 40%.
- `default` falls between the two, since OpenRouter splits traffic between them.
- Every pinned call is served by the provider asked for: 0 mismatches.
- 0 failed compiles on `openai`; 0 to 2 on `azure` (its content filter can refuse).
- Mean accepted records per compile within 0.5 of each other across the three routes.

## Decision rules, fixed now

**1.** Adopt the smallest K in {12, 16} for which all of these hold; otherwise keep 8:
- its Adds per minute are at least **1.20 times** the mean of `k8a` and `k8b`;
- 48 of 48 of its Adds reach 200, and at most 2 first attempts are not 200;
- its peak unit memory is at most 2 GiB (half the official C9's 4 GiB cap).

The run is **inconclusive**, and K stays 8, if `k8a` and `k8b` differ by more than 20% of their
mean: then drift, not K, would dominate the comparison.

**2.** Pin provider P, in production as `order: [P, other]` with fallbacks on, only if all of
these hold; otherwise do not pin:
- the paired median of P's compile time over the other provider's is **at most 0.80**, and P is
  faster on at least 65% of the sessions;
- 0 failed compiles on P;
- P's mean accepted records are no more than 1.0 below the other provider's.

The run is **invalid** if any pinned call reports a served provider other than the one pinned.

## What would falsify this

- For 1: any arm above its band, or a K=8 arm outside 20 to 40 Adds per minute. A flat result
  with the post-compile time also flat would falsify the flock mechanism, even if K stays 8.
- For 2: a gap under 10% or over 30% between the providers, any pinned mismatch, or a failure
  count outside the predicted range.

## How it will be measured

- **Where:** the VPS3 testbench only. Nothing runs on VPS2, and the official C9 is not touched.
- **Code:** the commit carrying this record, checked out in `/home/sentiment/locomo-route/code`
  (editable install, same venv as the same-user test).
- **Runner:** `bash scripts/aml_add_speed_vps3.sh <run-id>` as root on VPS3. It runs both
  measurements and writes everything to `/home/sentiment/c9-speed/<run-id>/`.
- **Server for 1:** one transient systemd unit per arm, `python -m recall_aml` on
  127.0.0.1:18121, variant `C9_routed_specialists_grounded_graph_atomic`, Search concurrency 3,
  `MemoryMax=4G` as on the official C9, database `locomo_route_20260923`, table
  `recall_aml_locomo_route_chunks`, and an embed lock and cache private to the run. The runner
  reads the arm's concurrency back from the server process's environment and stops if it differs
  (apparatus check).
- **Client for 1:** `scripts/aml_add_throughput.py`. Sixteen users at once, which is the
  platform concurrency used in the smokes; each user sends its 3 sessions one after another, as
  AML does. Retries on a 5xx or client timeout with the same `request_id`, backoff 5 s, up to 32
  attempts, client timeout 300 s. Each arm's users are deleted at the end.
- **Metric for 1:** Adds per minute, counted over successful Adds from the first send to the
  last reply. Also reported: characters per second, Add p50 and p90, first-attempt statuses,
  retries, the unit's `MemoryPeak`, and each Add's compile and post-compile time from the arm's
  journal.
- **Sessions:** `locomo10.json`, SHA256 `79fa87e9…` (the same-user test's file), 272 non-empty
  sessions. Every sixth in dataset order is reserved for measurement 2. The rest are sorted by
  size and dealt in snake order, 48 per arm, so the four arms carry 146,081 / 145,745 / 145,897 /
  146,057 characters and share no session (checked before commit).
- **Client for 2:** `scripts/aml_compiler_provider_latency.py`. It calls
  `OpenAICompiler.compile_anchored_v3` exactly as an Add does, with no prior records, through the
  client C9 builds (`build_openrouter_client`), adding only `provider: {order: [P],
  allow_fallbacks: false}` on the pinned routes. 40 reserved sessions (105,778 characters), each
  through all three routes, route order cycling through the six permutations.
- **Metric for 2:** each compile's wall time including its own retries, paired by session; also
  completion tokens per second per call, failed compiles, accepted records, and the served
  provider of every call.
- **Keys and spend:** the testbench Voyage and OpenRouter keys. Estimated about USD 1 in total
  (192 Adds, 120 compiles). The run stops for the user's decision if spend passes USD 3.

## Confounds I can name now

- **VPS3 is not VPS2.** VPS3 has 4 cores and 7 GB against VPS2's 12 and 47. C9 used about 0.6 s
  of CPU per Add on VPS2 today, so 16 in flight should not saturate VPS3, but a CPU limit would
  look exactly like a flock limit. The per-arm unit CPU time goes in the result to check this.
- **These Adds are small.** About 3,000 characters against AML Textual's typical 8,000 prompt
  tokens. Small Adds hold the flock briefly, so this may understate how much larger Adds contend.
- **Measurement 2 has no prior records**, so its prompts are shorter than an Add's. Generation
  length, which dominates, should be similar.
- **Provider load changes by the hour.** The rotation spreads that over all routes alike, but
  one evening is one sample of it.
- **One run of each.**

## Result (2026-09-24, run `c9speed1`)

**Status:** measured. Everything above this heading is as committed in `3fb00ac1`, before the run.

Run facts: VPS3, runner and harnesses at `3fb00ac1`, 19:35 to about 20:25 UTC. Artifacts in
`docs/results/2026-09-24-aml-c9-add-speed/`, SHA256 as recorded on the host:

| file | SHA256 |
| --- | --- |
| `throughput-k8a.json` | `7c51e0eee18cfc7d43f22b6cf3777d33501e4d7d8489ddf677d4a2d6147e22ab` |
| `throughput-k16.json` | `0a44c44ff5a98d7c61bacfd60f474940d1991a94392489b30482942e9f583be3` |
| `throughput-k12.json` | `ef255679f1578866f36841d5f5c1150faf1db2d79a149b74c0d23e7b534390cd` |
| `throughput-k8b.json` | `61248e120bbfd90300c13952b41100eb813331b2c19779520869db59944d210b` |
| `providers.json` | `e5d2f9bd2021af36455cd0d3c04927562675cd88ad8f75b778f7f4f4d7e36071` |

The four arm journals (about 330 KB each) stayed on VPS3 in `/home/sentiment/c9-speed/c9speed1/`.
Spend: OpenRouter 700,298 input and 202,425 output tokens, **USD 0.23**; Voyage not read from a
bill, a few cents at most for 192 Adds of about 3,000 characters.

### 1. Internal Add concurrency: keep 8

Apparatus check passed: each server process carried the concurrency its arm claimed (8, 16, 12, 8).

| arm | Adds per minute | vs K=8 mean (25.55) | 200 of 48 | first attempts not 200 | peak memory | compile p50 | after compile p50 / p90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `k8a` | 24.33 | | 48 | 0 | 155 MB | 15.7 s | 1.5 / 4.6 s |
| `k16` | 24.40 | **0.955** (predicted 0.95 to 1.20) | 48 | 0 | 157 MB | 25.3 s | 2.1 / 19.5 s |
| `k12` | 26.10 | **1.022** (predicted 0.95 to 1.15) | 48 | 0 | 155 MB | 20.3 s | 1.5 / 5.2 s |
| `k8b` | 26.76 | | 48 | 0 | 155 MB | 12.1 s | 1.7 / 6.3 s |

- Decision by the rule: neither K=12 nor K=16 reaches 1.20, so **K stays 8**. The run is valid:
  `k8a` and `k8b` differ by 9.5% of their mean, under the 20% limit and inside the predicted 15%.
- Absolute throughput at K=8 (24.3 and 26.8) is inside the predicted 20 to 40.
- Server CPU was 9.1 to 10.0 s per arm of about 110 s, under 0.1 of a core: not a CPU limit.

**Gap: the outcome held, the mechanism did not.** I predicted the embedding flock would bind, with
the time after the compile growing with K. It barely moved (p50 1.5 to 2.1 s). What grew was the
compile: p50 12.1 and 15.7 s at K=8, 20.3 s at K=12, 25.3 s at K=16. The K=8 figures include time
queued for a semaphore slot and are still the shortest, so each gpt-4o-mini call genuinely took
longer when more ran at once. The limit sits upstream, in OpenRouter or the provider for this key,
not in C9. This falsifies the flock explanation carried over from the 2026-09-23 ramp, at least
for Adds of this size after #751.

### 2. Provider: the rule selects OpenAI, narrowly, and the gain over today is small

Apparatus check passed: 0 pinned-provider mismatches; every pinned call was served by the provider
asked for (41 calls each, one compile retried once on each pinned route).

| route | compile p50 | p90 | output tokens/s p50 | failed compiles | accepted records, mean |
| --- | --- | --- | --- | --- | --- |
| `openai` | **5.72 s** | 8.86 s | 99.0 | 0 | 6.93 |
| `azure` | 10.16 s | 13.26 s | 59.4 | 0 | 7.90 |
| `default` (as served) | 6.47 s | 11.31 s | 90.6 | 0 | 7.63 |

`default` was served by OpenAI 23 times and Azure 17 times.

| paired over 40 sessions | median ratio | first faster |
| --- | --- | --- |
| openai / azure | **0.602** | 92.5% |
| openai / default | 0.946 | 67.5% |
| azure / default | 1.318 | 22.5% |

- Decision by the rule: OpenAI's paired median over Azure is 0.60 (at most 0.80 required), it is
  faster on 92.5% of sessions (65% required), it failed 0 compiles, and its mean accepted records
  are 0.975 below Azure's (at most 1.0 allowed). **All four hold, so the rule selects pinning
  OpenAI** as `order: [openai, azure]` with fallbacks on. The records condition held by 0.025.
- Against my predictions: the gap between providers was 40%, above the predicted 10 to 30%.
  `default` fell between them, 0 mismatches and 0 OpenAI failures held, and Azure's 0 failures
  sat inside 0 to 2. **Falsified:** accepted records within 0.5 across routes. OpenAI kept one
  record fewer than Azure per compile on average, and 0.7 fewer than `default`.

**What the rule does not show, and the user should weigh:**
1. **Against what C9 does today, the gain is small at the median.** `default` already sends most
   calls to OpenAI, so pinning cuts the paired median by about 5% (0.946) and the p90 from 11.3 s
   to 8.9 s (about 22%). The 40% gap is against Azure alone.
2. **Pinning changes what the compile stores.** OpenAI returns fewer records per compile than the
   mix C9 serves now (6.93 against 7.63). Those records are the graph's input, so this is not a
   pure latency change, and it is not the build the smokes tested.
3. Part of OpenAI's speed may come from writing less; its per-token rate is also higher (99
   against 59 tokens per second), so not all of it does.
4. These calls ran one at a time. Measurement 1 shows the compile slows under concurrency, and
   whether OpenAI alone slows more or less than the mix under load is unmeasured.
