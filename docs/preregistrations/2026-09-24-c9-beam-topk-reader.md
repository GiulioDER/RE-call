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

## Result

(appended after the run; nothing above is edited)

### Run 1, 2026-09-24 21:39 to 22:34 UTC, VPS2, offline from the probe's stored retrieval

The code is `benchmarks/beam/aml_c9_probe.py` at `74bfc681`. The top20 and top40 answers were
produced in parallel with the top10 judging, to save time. Every file holds 400 unique ids (200 for
coverage), so no question was answered twice. Model spend was $1.53, which brings the two probes
to $3.95. Raw outputs are on VPS2 under `/root/c9-beam-probe/out/`.

| type | top 100 | top10 | top20 | top40 |
| --- | --- | --- | --- | --- |
| abstention | 0.500 | 0.700 | 0.600 | 0.650 |
| contradiction_resolution | 0.062 | 0.069 | 0.062 | 0.044 |
| event_ordering | 0.224 | 0.241 | 0.298 | 0.321 |
| information_extraction | 0.831 | 0.785 | 0.783 | 0.756 |
| instruction_following | 0.494 | 0.431 | 0.450 | 0.475 |
| knowledge_update | 0.525 | 0.500 | 0.525 | 0.475 |
| multi_session_reasoning | 0.550 | 0.610 | 0.499 | 0.525 |
| preference_following | 0.775 | 0.700 | 0.769 | 0.846 |
| summarization | 0.304 | 0.353 | 0.311 | 0.257 |
| temporal_reasoning | 0.289 | 0.171 | 0.250 | 0.205 |
| **mean over 10 types** | **0.456** | **0.456** | **0.455** | **0.455** |

Paired mean difference against top 100 over about 398 scored questions, with a bootstrap 95% CI:

- top10: +0.001 [-0.038, +0.040]
- top20: -0.001 [-0.037, +0.033]
- top40: -0.000 [-0.039, +0.037]

Coverage (evidence present in the first N items):

| type | top10 | top20 | top40 | top 100 |
| --- | --- | --- | --- | --- |
| summarization | 0.755 | 0.862 | 0.945 | 0.950 |
| multi_session_reasoning | 0.780 | 0.799 | 0.902 | 0.923 |
| event_ordering | 0.805 | 0.880 | 0.921 | 0.911 |
| temporal_reasoning | 0.500 | 0.628 | 0.615 | 0.529 |
| information_extraction | 0.850 | 0.854 | 0.850 | 0.838 |

**Predictions:**

- **Held:** the three 10-type means (all inside their ranges, though under the points of 0.47);
  every coverage range; and all event_ordering ranges.
- **Falsified low:** summarization at top40 (0.257), temporal_reasoning at top10 (0.171) and top40
  (0.205), and information_extraction at top40 (0.756).
- **Falsified high:** multi_session_reasoning at top10 (0.610).

**Decision rule: not met by any arm.** No arm beats top 100 by 0.02. Every arm has a type that drops
by more than 0.05: temporal -0.118 at top10, multi-session -0.051 at top20, temporal -0.084 at
top40. **C9 keeps returning 100.**

**Reading.** Context length is not what limits this reader. Cutting from 100 to 10 items drops
summarization coverage from 0.95 to 0.76, yet its score barely moves (0.30 to 0.35). So the gap
between coverage and score in run 1 is not the reader drowning in a long context. Per-type
movements trade against each other and rest on 40 questions each. The largest, abstention at +0.20
with top10, is inside the noise one type can show and is not acted on.

Two readings remain open, and neither is tested here:

1. The coverage judge is lenient, so the "evidence is present" figures overstate what the context
   really gives the reader. That needs validation: a stronger judge, or hand labels on a sample.
2. The content is the limit, not the amount: undated, speakerless windows (the other session's dates
   work), or evidence that needs synthesis, which a retrieval order cannot supply.
