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

## Result

(appended after the run, below this line; nothing above is edited)
