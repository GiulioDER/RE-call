# Pre-registration: stop C9's compiler citing prior record ids as evidence

**Date:** 2026-09-25   **Status:** predicted, not yet measured

## The question

`2026-09-25-c9-compile-citation-diagnosis.md` (R2) found that gpt-4o-mini cites the ids of the
prompt's `prior_records` as `evidence_anchor_ids`: 41 of 41 unknown ids in the fallback calls, 6 of
31 later chunks fell back, and later calls that did not fall back kept 55 of 107 proposals. Those
ids have one use, `supersedes`, which was set on 0 of 2,242 C9 and 0 of 261,420 C8 compiled
records, and on 0 of 1,005 replay proposals. Does sending prior records without ids (P1), or not
at all (P2), remove the loss without making the compiler repeat what is already stored?

The user decided on 2026-09-25 to change C9 for the first official Textual Full with the result,
so this measurement decides a served build, the same day.

## Arms and sets

`OpenAICompiler(prior_record_mode=...)` (this commit), default `with-ids`, byte-identical to v3
(`tests/test_aml_compiler_prior_records.py`). Served C9 is unchanged by this commit.

| arm | `prior_record_mode` | sets |
|---|---|---|
| K0 (R2, already measured, `out-r2`) | `with-ids` | diagnosis |
| K0b | `with-ids` | diagnosis (same-session replicate of K0) |
| K0h | `with-ids` | heldout |
| P1 | `without-ids` | diagnosis and heldout |
| P2 | `none` | diagnosis and heldout |

Sets (`SETS` in `scripts/aml_c9_compile_citation_diagnosis.py`): **diagnosis** is the 71 calls of
R1 and R2. **heldout** was never run: LoCoMo sessions 3 and 4 of the same 10 conversations, and the
first two chunks of the second batch of the same 20 BEAM conversations, cut the same way. All
six new runs run concurrently on VPS3, gpt-4o-mini through OpenRouter, no provider pin (as C9).
Spend cap USD 0.60 per run.

## Measures (`arm_measures`, tested)

Over **later calls** (after the first chunk of their session): later-call fallbacks; accepted
records per later call; the share of accepted records that near-duplicate an earlier accepted
record of the same session (token Jaccard at least 0.6 over kind, task_shape, problem, action,
outcome, validation); prompt tokens per later call; prior ids cited.

## Predictions

| quantity | band | point |
|---|---|---|
| K0b later fallbacks, diagnosis (K0 was 6 of 31) | 2 to 10 | 6 |
| K0h later fallbacks, heldout | 2 to 12 | 6 |
| P1 later fallbacks, each set | 0 to 2 | 0 |
| P2 later fallbacks, each set | 0 to 2 | 0 |
| P1 prior ids cited in later calls, each set | 0 to 3 | 0 |
| P1 minus K0-mean accepted per later call, pooled | +0.3 to +2.5 | +1.0 |
| P2 minus K0-mean accepted per later call, pooled | +0.3 to +3.0 | +1.2 |
| P1 minus K0-mean near-duplicate share, pooled | -0.05 to +0.05 | 0.00 |
| P2 minus K0-mean near-duplicate share, pooled | 0.00 to +0.15 | +0.05 |
| P1 prompt tokens per later call vs K0 | -2% to -15% | -6% |
| P2 prompt tokens per later call vs K0 | -15% to -60% | -30% |

K0-mean is the mean of the two with-ids runs on that set (K0 and K0b on diagnosis; K0h alone on
heldout, which has no replicate). Pooled means over both sets, weighted by later calls.

## Decision rule, fixed now

1. **Apparatus failure** (any check below fails): report, change nothing.
2. An arm **passes** if all hold: later fallbacks at most 1 on each set; accepted per later call,
   pooled, at least K0-mean's; near-duplicate share, pooled, at most K0-mean's plus 0.05.
3. **If P1 passes, recommend P1** (the smaller change: the same context without citable ids),
   unless P2 also passes AND exceeds P1 on accepted per later call by at least 0.5 with a
   near-duplicate share no more than 0.02 above P1's; then recommend P2.
4. If only P2 passes, recommend P2. If neither passes, recommend no change for the Full.
5. The recommendation goes to the user, who decides; then it is served as a C9 variant setting,
   with the control and an AML smoke before the Full.

## Apparatus checks

1. Every successful call of every run has sent anchor ids; P1 requests carry no prior `id`; P2
   requests carry no `prior_records` (read back from the stored request by `sent_prior_ids`).
2. Per-call `accepted_records` equals the compile diagnostics line in every run.
3. Each run completes at least 60 calls within its cap.
4. |K0b minus K0| later fallbacks on diagnosis at most 4 (noise floor; if larger, say so and read
   every difference below that size as noise).

## What this does not measure, stated now

Downstream quality. Compiled records feed the graph sidecar and the Context4 index; on the
Textual smoke 47 of 48 Searches took the Code4 route, which does not index them, so a Textual
score change is expected to be small either way. The Coding K-screen and the LoCoMo answer check
are not run before the Full, for time; the AML Textual smoke on the new build is the end-to-end
check, and the Coding check comes before the Coding Full.

## Result, measured 2026-09-25 09:04 to 09:25 UTC (appended; nothing above edited)

Six runs on VPS3 at `0789f5b9` (P1 relaunched at `d434f023` after the harness crashed at call 0
with `KeyError: 'id'` reading an id-less prior record; harness only, fixed with a red-proved
test, `recall_aml` identical), 71 calls each, USD 0.49 in total.
`/home/sentiment/c9-cite-diag/arms-compare-final.json`.

| run | later fallbacks (of 31) | accepted per later call | near-duplicate share | prompt tokens per later call | prior ids cited |
|---|---:|---:|---:|---:|---:|
| K0 (R2), diagnosis | 6 | (not recorded) | (not recorded) | (not recorded) | 114 |
| K0b, diagnosis | 7 | 3.548 | 0.027 | 7,725 | 97 |
| K0h, heldout | 7 | 4.581 | 0.014 | 7,258 | 79 |
| **P1, diagnosis** | **0** | **6.742** | 0.005 | 7,752 | **0** |
| **P1, heldout** | **0** | **6.645** | 0.010 | 7,127 | **0** |
| P2, diagnosis | 4 | 5.097 | 0.000 | 2,976 | 0 |
| P2, heldout | 3 | 5.710 | 0.000 | 2,967 | 0 |

**Deviation, stated:** K0 (R2) was run before `accepted` and `prompt_tokens` were recorded, so
K0-mean on diagnosis uses K0b alone for those measures; its fallbacks (6) do count.

Apparatus checks: 1 passes (every call had sent anchors; P1 and P2 sent 0 prior ids; the P1
payload shape is covered by `tests/test_aml_compiler_prior_records.py`); 2 passes (0 mismatches in
all six runs); 3 passes (71 of 71 each); 4 passes (|K0b minus K0| = 1).

| prediction | band | measured |
|---|---|---|
| K0b later fallbacks | 2 to 10 | 7, in band |
| K0h later fallbacks | 2 to 12 | 7, in band |
| P1 later fallbacks, each set | 0 to 2 | 0 and 0, in band |
| P2 later fallbacks, each set | 0 to 2 | **4 and 3, falsified** |
| P1 prior ids cited | 0 to 3 | 0 and 0, in band |
| P1 minus K0-mean accepted per later call, pooled | +0.3 to +2.5 | **+2.63**, just above the band |
| P2 minus K0-mean accepted, pooled | +0.3 to +3.0 | +1.34, in band |
| P1 minus K0-mean near-duplicate share | -0.05 to +0.05 | -0.013, in band |
| P2 minus K0-mean near-duplicate share | 0.00 to +0.15 | -0.021, **falsified** (below) |
| P1 prompt tokens vs K0 | -2% to -15% | **+0.3% and -1.8%, falsified**: the ids are a small part of the prompt |
| P2 prompt tokens vs K0 | -15% to -60% | -61.5% and -59.1%, one just outside |

P2's fallbacks are a different failure: in all 7 the model proposed no record at all (0
proposals), which never happened in P1. Without prior records it declines small follow-up chunks.

**Decision by the rule:** P1 passes (0 and 0 later fallbacks; accepted per later call 6.69 pooled
against K0-mean 4.06; near-duplicate share 0.008 against 0.021). P2 fails the fallback condition
(4 and 3). **Recommendation: P1, `anchor_prior_records = "without-ids"`.** The user decided on
2026-09-25 to take this result into the first official Textual Full; downstream gates as stated
above are not run first.
