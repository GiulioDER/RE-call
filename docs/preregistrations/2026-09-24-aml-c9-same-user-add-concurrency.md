# Pre-registration: do concurrent Adds for the same user all land on C9, when retried as AML retries?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

With C9 at master `13268614` (runtime-identical to the served `337f2537`) and
`RECALL_AML_ADD_CONCURRENCY=8`, as served, 16 Adds are sent at once and retried on a 5xx or a
client timeout with the same `request_id`, up to 32 attempts. Does **every** Add succeed, with
every receipt intact, in each of three scenarios?

| scenario | users | Adds per user |
| --- | --- | --- |
| `one_per_user` | 16 | 1 |
| `four_per_user` | 4 | 4 |
| `all_one_user` | 1 | 16 |

## Why it is being asked

The gate recorded in the official-config memo (2026-09-23) forbids the official AML Smoke on C9
until C9 is concurrency tested.
- **The named risk:** each Add holds its tenant's advisory lock for the whole request, and a
  waiter past the pool's 25 s `statement_timeout` is answered 503. C8 showed 11 of 60 at three
  workers.
- **What the existing ramp covers:** the 2026-09-23 ramp
  (`docs/preregistrations/2026-09-23-aml-c9-max-concurrency.md`, branch
  `claude/aml-concurrency-probe`) sent one user per worker with sequential Adds. So it never
  exercised that wait.
- **What changed since:** Add concurrency went from 3 to 8, which lets more same-user Adds queue
  on one lock at a time.

## What I predict

These come from the LoCoMo route run's mean Add time on the same data, 3,358 s / 272 = 12.3 s,
so a lock waiter at queue position p waits about 12.3 × (p - 1) s.

| scenario | first attempts not 200 (of 16) | eventual 200 | most attempts any Add used | wall clock |
| --- | --- | --- | --- | --- |
| `one_per_user` | 0 | 16 | 1 | 20 to 60 s |
| `four_per_user` | 3 to 8 | 16 | 2 to 5 | 60 to 180 s |
| `all_one_user` | 10 to 14 | 16 | 5 to 15 | 200 to 450 s |

Receipt replays: 48 of 48 return 200 with a nonzero raw count.

## Decision rule, fixed now

The gate **passes** when all three of these hold:
1. `one_per_user` has 0 first attempts other than 200;
2. every one of the 48 Adds reaches 200 within 32 attempts;
3. all 48 receipt replays are intact.

First-attempt 503s in the same-user scenarios are expected under AML's retry contract. They are
reported, not gating. If any Add needs more than 16 attempts, I report that half the retry
margin is used, which bears on whether Add concurrency 8 should go back to 3.

## What would falsify this

- Any Add that never reaches 200.
- Any lost receipt.
- Any non-200 in `one_per_user`.
- First-attempt failure counts outside the predicted bands. That would mean the lock-wait
  model is wrong.

## How it will be measured

- **Harness:** `scripts/aml_same_user_concurrency.py`, committed with this record.
- **Server:** VPS3 only, as the gate requires. One `python -m recall_aml` process on
  127.0.0.1:18120, at the commit carrying this record. It runs with
  `RECALL_AML_VARIANT=C9_routed_specialists_grounded_graph_atomic`,
  `RECALL_AML_ADD_CONCURRENCY=8`, `RECALL_AML_SEARCH_CONCURRENCY=3`, and the production pool
  (`statement_timeout` 25 s).
- **Storage and keys:** database `locomo_route_20260923`, table
  `recall_aml_locomo_route_chunks`, fresh user ids, a fresh embedding cache, and the testbench
  Voyage and OpenRouter keys.
- **Data:** 48 distinct LoCoMo sessions, 16 per scenario, so the embedding cache cannot shorten
  a later scenario's Adds. The dataset SHA256 is `79fa87e9…`, as in the route comparison.
- **Client:** the client timeout is 300 s. A retry waits 5 s after a 5xx or a timeout.
- **Order:** the scenarios run in the order above. Each ends by replaying every request, then
  deleting its users.
- **Recorded:** every attempt's status and duration. The server journal carries `latency_ms` and
  `error_class` per Add since #727.

## Confounds I can name now

- **VPS3 is not VPS2.** Provider latency and host load differ, so absolute times may not
  transfer. The lock mechanism and the 25 s timeout are the same code.
- **The platform's own behaviour is unknown.** This tests C9's worst case, 16 same-user Adds at
  once. Whether AML ever overlaps Adds for one user is not known. The official Multimodal run
  logged 0 statement timeouts over 55.5k Adds at platform concurrency 16, which suggests it
  rarely or never does, but that run's Adds are not these.
- **The backoff is mine.** AML's backoff between retries is not documented. A longer backoff
  lowers the attempt count and lengthens the wall clock.
- **One run.**

## Result (2026-09-24, run `c9su1`)

**Status:** measured. **The gate passes:**
1. `one_per_user` had 0 non-200 first attempts;
2. all 48 Adds reached 200;
3. all 48 receipt replays are intact.

The most attempts any Add needed was 5 of 32, so the retry margin was never near half used.
Add concurrency 8 stands.

Run facts:
- Harness and server both at `0f19e041`, on VPS3 127.0.0.1:18120, C9 with add-time atomic views
  and platform scope.
- Artifact `docs/results/2026-09-24-aml-c9-same-user-add-concurrency.json`, SHA256
  `9798213ca4d255bf2a9fc7ca1d0522834e864bf4802f28a63d78a90425e6716b`.

| scenario | first attempts not 200 | eventual 200 | most attempts | wall clock | median time to 200 |
| --- | --- | --- | --- | --- | --- |
| `one_per_user` (16x1) | **0** (predicted 0) | 16/16 | 1 (predicted 1) | 43.4 s (20 to 60) | 28.4 s |
| `four_per_user` (4x4) | **2** (predicted 3 to 8) | 16/16 | 2 (predicted 2 to 5) | 53.2 s (60 to 180) | 31.8 s |
| `all_one_user` (1x16) | **11** (predicted 10 to 14) | 16/16 | 5 (predicted 5 to 15) | 170.6 s (200 to 450) | 92.0 s |

Receipt replays: 48 of 48.

**Mechanism confirmed.** The server journal holds 26 `hosted_request_failed` records, exactly the
2 + 24 retries the client counted, and every one of them carries
`{"error_class":"QueryCanceled"}`: the 25 s `statement_timeout` firing on the advisory-lock wait.
No other error class appears. This is the first run that could read `error_class`, because #727
put it in the journal.

**Gap.** The same-user case landed inside its band. The mixed case (4x4) failed less often, and
both contended cases finished faster than predicted. My model used 12.3 s per Add, the
sequential mean from the route run. Under concurrency the Adds overlap their provider waits, so
the lock is held for less of the wall clock than that model assumed. The failure direction, the
mechanism and the recovery all held.

**What this does and does not show.** C9 recovers from its worst case (16 simultaneous Adds for
one user) within 5 attempts, using AML's documented retry-by-`request_id`. It does not show what
backoff AML uses, or whether AML ever overlaps Adds for one user. It is one run, on VPS3 rather
than VPS2.

🔁 **Correction to the gap paragraph above, appended the same day.** "Under concurrency the Adds
overlap their provider waits" cannot explain the `all_one_user` case. Adds for one user are
serialised by the lock and never overlap. The simpler account fits the numbers: 170.6 s over 16
serialised Adds is **10.7 s per Add**, about 13% below the 12.3 s I predicted from. These 48
sessions are the dataset's first 48, from its first conversations, not the full-corpus mix the
12.3 s came from. The overlap explanation applies only to the cross-user part of `four_per_user`.
