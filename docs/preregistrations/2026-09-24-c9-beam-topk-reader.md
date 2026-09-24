# Pre-registration: does answering from fewer of C9's items help the reader on BEAM?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

The C9 BEAM ability probe (`2026-09-24-c9-beam-ability-probe.md`, run 1) found the evidence in
the returned 100 items for summarization, multi-session reasoning and event ordering (coverage 0.91
to 0.95), while Qwen3-14B scored only 0.22 to 0.55. Re-ordering the same items by time hurt every
type except event ordering. That points to a reader that leans on the top of a long context.

This test re-answers the SAME stored retrieval (run 1's `retrieval.jsonl`, 400 questions, no new
Search) from only the first 10, 20 or 40 items, in returned order. Arms `top10`, `top20` and `top40`
run on all 10 BEAM types. The answer prompt, judge and model are unchanged. Coverage over the first
N items is also scored for the five in-scope types, to see how much evidence truncation drops.

The AML contract allows returning fewer than `top_k` items, so if a smaller N wins, it is a
serving change C9 can make.

## Baseline (run 1, top 100, returned order)

The mean over all 10 types is **0.456**. By type: summarization 0.304, multi_session_reasoning 0.550,
event_ordering 0.225, temporal_reasoning 0.290, information_extraction 0.831, abstention 0.500,
contradiction_resolution 0.063, instruction_following 0.494, knowledge_update 0.525,
preference_following 0.775.

## Predictions

| quantity | top10 | top20 | top40 |
| --- | --- | --- | --- |
| mean over 10 types | 0.40 to 0.50 (point 0.44) | 0.43 to 0.52 (point 0.47) | 0.44 to 0.51 (point 0.47) |
| summarization | 0.22 to 0.42 | 0.28 to 0.45 (point 0.35) | 0.28 to 0.42 |
| multi_session_reasoning | 0.40 to 0.58 | 0.45 to 0.60 (point 0.52) | 0.48 to 0.60 |
| event_ordering | 0.15 to 0.32 | 0.18 to 0.35 (point 0.26) | 0.18 to 0.33 |
| temporal_reasoning | 0.20 to 0.38 | 0.22 to 0.38 (point 0.30) | 0.24 to 0.36 |
| information_extraction | 0.70 to 0.88 | 0.75 to 0.88 (point 0.83) | 0.78 to 0.88 |

Predicted coverage over the first 10 items: summarization 0.60 to 0.90, multi-session 0.55 to
0.85, event ordering 0.55 to 0.85, temporal 0.30 to 0.50, information extraction 0.70 to 0.85.

## Decision rule, fixed now

A top-N arm is worth proposing as a C9 serving change only if both hold:

- its 10-type mean beats top 100 by 0.02 or more;
- no single type drops by more than 0.05 against top 100.

Otherwise C9 keeps returning 100. Either way, the result goes to the user as a recommendation, not a
deployment.

## What this cannot show

This is one public split, with a different reader and judge than the platform may use, and a
context format assumed rather than known. It reuses retrieval from C9 at `385c6074`, before #757
(bare anchor ids). That changes compiled records only for the Adds that fell back, and compiled
records are not what the top-N arms cut.
