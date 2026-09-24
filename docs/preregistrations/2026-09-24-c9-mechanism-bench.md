# Pre-registration: does live C9 pass a small Add and Search benchmark that exercises the atomizer and graph?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

This is a pre-run check before the official AML Textual and Coding runs. C9 serves on port 18015
behind `memory.pred-markets.com` at `385c6074`. `scripts/aml_c9_mechanism_bench.py` sends it one
throwaway user holding six multi-turn sessions (about 10,300 words, roughly 85 raw windows of 160
words). It uses only the official Add and Search envelopes, with `top_k: 100`, and asks 14 planted
questions plus one unanswerable one.

1. Does every hard contract check pass? The script lists them.
2. Do the two mechanisms that the existing 3-memory smoke cannot reach actually fire on the live
   service? Those are the Add-time atomic rescue (it needs 5 or more dense parents) and the
   grounded graph (it can promote only into ranks 9 and 10).
3. How many planted answers reach the top 10?

## Why it is being asked

The user wants C9 proven before the official run. `aml_text_route_smoke.py` passed 16/16, but it
stores three one-line memories. On that store, `select_view_rescue` refuses (fewer than five dense
candidates), and the graph has nothing below rank 8 to promote. So that smoke says nothing about
either mechanism.

## What I predict

These predictions are deliberately modest. `[[i-over-predict-effect-magnitudes]]` records eleven
of twelve predictions falsified, all of them too high.

| Quantity | Prediction |
| --- | --- |
| hard checks | all pass |
| Adds: status, echo, `compiler_fallback` | 6 of 6 are 200, echoes intact, no compiler fallback |
| compiled records, total over the 6 Adds | 15 to 45; at least 1 in each session that has needles |
| graph attempted and not fallen back | 14 of 14 searches |
| searches with graph relation hits above 0 | 5 to 14 of 14 |
| graph promotions, summed | 0 to 4 (point estimate 1) |
| atomic attempted and active | 14 of 14 |
| atomic `candidate-available` | 10 to 14 of 14 |
| atomic fallback | 0 to 3 of 14 |
| recall at 10 | 10 to 14 of 14 (point estimate 12) |
| recall at 1 | 5 to 11 of 14 (point estimate 8) |
| recall at 100 | 14 of 14. This one is close to trivial: the tenant holds about as many raw windows as `top_k`, so it tests shape, not ranking |
| slowest Add | under 60 s |
| Search seconds, client side | median under 3 s, max under 10 s |

## What would falsify "C9 is ready"

Any of these means C9 has a defect to find before the official run:

- A hard check fails.
- Relation hits are 0 on every search. That would mean the live service generates no grounded
  relations, or Search does not consume them.
- `candidate-available` is 0 on every search. That would mean the atomic views are not being
  read.
- An Add or Search needs a retry.

Recall at 10 below 10 of 14 does not falsify readiness. It is a quality observation on a
synthetic corpus, and it would be reported as such rather than gated.

## What this cannot show

The benchmark cannot show that the graph or the atomizer improves answers. The headers report
whether each mechanism acted, not the ranking without it, and nothing here runs an ablation. The
C9 baseline decision (`[[official-textual-coding-config-graph-on]]`) does not depend on it.

## Addendum before measurement: the concurrency stages (2026-09-24)

This was appended after the user asked for 32 concurrent Adds and 128 concurrent Searches, and
before any run. Nothing above it changed. The shape comes from `run_concurrency` in the script.

- **A:** 32 Adds sent at once, for 8 users with 4 sessions each. Each session is 1,200 to 2,000
  words, and a failed Add is retried with the same request id, up to 32 attempts.
- **B:** 128 Searches sent at once over those 8 users. Each user gets all 14 needles and the
  unanswerable question, and the 8 remaining searches repeat each user's first needle.
- **C:** 32 Adds for 8 new users, sent at once while 128 Searches run over the stage A users.

The basis is `[[2026-09-23-aml-c9-concurrency-ceiling]]`. Throughput is fixed on the server side:
about 0.28 Adds per second at internal concurrency 3, rising to about 1.68× that at the served
8, and about 2.4 Searches per second at any setting. The same-user test of 2026-09-24 saw 2 of 16
first attempts fail with four Adds per user.

| Quantity | Prediction |
| --- | --- |
| A: Adds stored | 32 of 32 |
| A: first attempts not 200 | 2 to 12 (point 5), all 503 lock waits |
| A: most attempts any Add needed | 2 to 6 |
| A: wall clock | 60 to 240 s (point 110) |
| A: compiler fallbacks | 0 to 2 |
| B: Searches answered 200 | 128 of 128, none retried |
| B: wall clock | 30 to 100 s (point 55) |
| B: Search seconds, median and max | median 15 to 50 (point 28); max 30 to 100 (point 55) |
| B: Searches over 60 s | 0 to 40 (point 3) |
| B: isolation, graph without fallback, atomic active | 128 of 128 each |
| B: identical query gives identical ranking | 8 of 8 |
| B: recall at 10 | 90 to 120 of 120 needle searches (point 105) |
| C: Adds stored and Searches answered 200 | 32 of 32 and 128 of 128 |
| C: Search median against B | 1.2 to 3 times B (point 1.6) |
| C: Searches over 60 s | 5 to 80 (point 30) |

**What would falsify "C9 handles the platform's concurrency":**

- a Search that returns another user's data
- any Add or Search that never reaches 200 within its retries
- a Search slower than the 120 s client timeout
- a graph fallback, or an inactive atomic rescue, under load
- a ranking that changes between identical queries in stage B, where no tenant is written

A Search slower than 60 s is a risk to report, not a failure. The platform's real Search timeout
is not known here, and the ceiling memo already records that its safe numbers assume the platform
waits about two minutes.

## Result

(appended after the run, below this line; nothing above is edited)
