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
