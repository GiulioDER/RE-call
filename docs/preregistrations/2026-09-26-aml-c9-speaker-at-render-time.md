# Pre-registration: K-1, mark who is speaking inside a returned window, at Search time

**Date:** 2026-09-26   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Baseline: C9 at `3eb447c4` with the Add-time compile off (C9-raw), the
configuration X-1 Stage B stores retrieval for (`docs/preregistrations/2026-09-25-aml-c9-source-coverage-baseline.md`,
amendment 2). Local directional experiment over public data; not an AML hosted evaluation.

## Why

C9 stores content-only windows: 160 words cut from a session's messages joined end to end, with no
role and no boundary (`build_chunks`, `content_only_windows=True`). Where a source puts the speaker's
name in the text (the LoCoMo collect here writes "Caroline: …"), the reader can tell who said what.
Where it does not, a window that spans a user turn and an assistant turn reads as one voice. That
is the case for LongMemEval-S, PersonaMem-v2 and CLBench, whose messages carry only `user` and
`assistant` roles, and it is exactly what LongMemEval's single-session-assistant questions ("what
did you recommend…") need. Putting roles into stored text cost Coding 0.09 MRR (2026-09-24),
because it changed what was embedded; a render at Search time cannot change ranking.

## The mechanism, fixed now

`speaker_marked_items(items, boundaries)` in a new `recall_aml/speaker_render.py`, flag
`speaker_at_render` (default off), override `RECALL_AML_SPEAKER_AT_RENDER`. For each returned text
item whose window spans known message boundaries, insert `[user]` or `[assistant]` (the message's
role, lower-cased) at the start of each message's words inside the window, and at the window's start
for the message it opens in. Nothing else: same items, same order, same scores; image items and items
with no known boundaries are returned unchanged.

**Where the boundaries come from.** In serving: per-window `(role, word offset)` pairs written into
chunk metadata at Add (stored text, embeddings and BM25 unchanged). For measurement on retrieval
already stored (X-1 Stage B, the LoCoMo collect): rebuilt offline by locating the window's words as
a contiguous run in the session's content-only word sequence, rebuilt from the source data, and
reading the message word ranges `build_chunks` itself computes. A window found at no position, or
at more than one, is left unmarked and counted.

## Stage 0, census (free: no model, no service)

On every returned top-10 text item, per source: the share that spans at least two messages of
different roles with no speaker name at a boundary.

| Set | Predicted share |
|---|---|
| LoCoMo (the committed 720-question collect) | 0.00 to 0.05 (names are in the text) |
| LongMemEval-S (X-1 Stage B, 120 questions) | 0.40 to 0.80 |
| PersonaMem-v2 (X-1 Stage B, 200 questions) | 0.30 to 0.70 |
| CLBench (X-1 Stage B, 71 tasks) | 0.20 to 0.60 |
| Windows left unmarked because no unique position was found | at most 0.02 of items |

**Gate:** Stage 1 runs only if the LongMemEval-S share is at least 0.30 and unmarked windows are at
most 0.05 of items. If LoCoMo's share is above 0.05, LoCoMo joins Stage 1 as a non-inferiority set.

## Stage 1, answers (only if Stage 0 passes; about USD 1 on DeepSeek)

Stored X-1 Stage B retrieval for LongMemEval-S, answered with AML's LongMemEval-S prompt and binary
judge from the pinned checkout, `deepseek/deepseek-v4.1-flash` pinned to one provider, reasoning
off. Arms: **R** (C9-raw as stored), **R′** (R again, noise), **K1** (R with speakers marked).
Interleaved per question; paired bootstrap over questions, 10,000 resamples, seed 20260925.

| Contrast | Set | Predicted |
|---|---|---|
| K1 − R | LongMemEval-S, all 120 | +0.02, band −0.02 to +0.06 |
| K1 − R | single-session-assistant (20) | +0.08, band 0.00 to +0.20 |
| K1 − R | single-session-user (20) | 0.00, band −0.05 to +0.05 |
| R′ − R | all 120 | within ±0.04 |

Predictions sit low: my record is two to four times too high on effects, though twice this week too
low where a mechanism measurement overshot first; Stage 0's shares are that mechanism measurement,
and an amendment revises these bands before Stage 1 if Stage 0 lands far outside its own.

## What would falsify this

- Stage 0: LongMemEval-S share below 0.30 (few windows mix speakers, so there is little to mark).
- Stage 1: K1 − R on single-session-assistant at or below 0, or K1 − R on all 120 below −0.02.

## Decision rule

Recommend K-1 if K1 − R on all 120 is at least +0.02 with a CI lower bound above −0.03, K1 − R on
single-session-assistant is at least +0.05, and |R′ − R| is smaller than the overall effect. Before it
could serve: the Add-time metadata path built and red-proved, byte-identical output when the flag is
off, a Coding check that ranking is unchanged (it must be, since nothing ranked changes), and the
user's decision. Never deployed during an AML job.

## Apparatus checks, fixed now

1. `speaker_marked_items` unit tests: a one-role window is unchanged; a two-role window gets exactly
   one marker per boundary; items, order and scores are unchanged; image items pass through; each
   seen red against a deliberate mutation first.
2. The offline boundary rebuild agrees with `build_chunks`' own `message_word_ranges` on synthetic
   sessions (test), and on real sessions its unmarked share is reported.
3. Valid-answer rate at least 98% per arm.

## What I already know

- The LoCoMo collect's windows carry speaker names in the text (seen in the T-1 prompts, 2026-09-26).
- LongMemEval-S, PersonaMem-v2 and CLBench adapters send `user` and `assistant` roles with no names
  (`scripts/aml_x1_sources.py`).
- Nothing from X-1 Stage B has been read: it is still running when this is written.

## Confounds I can name now

- **The platform's Add format is not public.** If AML's own adapters put speaker names in content,
  K-1 does less there than here.
- **Offline boundaries are a reconstruction.** A repeated phrase can place a window twice; those are
  left unmarked and counted, not guessed.
- **One reader.** DeepSeek here; the gpt-4o-mini confirmation round applies as for every candidate.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

### Stage 0 census, 2026-09-26 (PARTIAL: LongMemEval-S 95 of 120, PersonaMem-v2 and CLBench not yet)

Run on VPS3 by `scripts/aml_k1_census.py` (no model, no service, no key), whose classifier is
red-proved by three mutations (`tests/test_aml_k1_census.py`). Top-10 text items per question.
Inputs: LoCoMo `collected-S.json.gz` (SHA-256 prefix `243f48ebcd2dfbe1`, the file the T-1 record
names); X-1 Stage B `out/longmemeval_s.jsonl` as it stood when Stage B was paused (95 tenants,
SHA-256 prefix `8b63ef837c732343`). Output: `results/aml-k1/census-stage0-*.json`.

| Set | Items | Mixed-speaker share with no name at a boundary | Predicted | Unmarked |
|---|---:|---:|---|---:|
| LoCoMo | 15,350 | **0.000** | 0.00 to 0.05, held | 0.057 of all items; 0.000 of windows |
| LongMemEval-S (95 of 120) | 950 | **0.768** | 0.40 to 0.80, held, near the top | 0.002 |

Two things the table needs said:
- **LoCoMo's 876 unmarked items are all compiled records** (repository fact, procedure and seven more
  kinds; collected-S was collected with the compile on), which are not windows and cannot be located
  by construction. Every one of its 14,474 windows was located, none is mixed without names, and
  94.3% span two or more named speakers: the names in the text already do what K-1 would.
- **Deviation from the record: LoCoMo here is all 1,535 questions of `collected-S.json.gz`**, not the
  720-question collect the Stage 0 table names. No 720-question retrieval file was found beside it;
  the share is 0.000 either way, since every LoCoMo message carries a speaker name.

Gate, provisionally: LongMemEval-S share 0.768 is at least 0.30, and unmarked windows are 0.002 at
most 0.05, so **Stage 0 passes on the partial set**; LoCoMo's share is not above 0.05, so it does
not join Stage 1. The gate is decided on all 120 LongMemEval-S questions when X-1 Stage B finishes,
and PersonaMem-v2 and CLBench are counted then. Stage 1 has not run.

### Stage 0 census, LongMemEval-S complete, 2026-09-26 ~19:55 UTC

X-1 Stage B finished all 120 LongMemEval-S tenants (resumed after the user stopped the Textual Full;
`out/longmemeval_s.jsonl` SHA-256 prefix `0ae88d4507a6976f`). Same script and settings as the partial
run above. Output `results/aml-k1/census-stage0-lme-120.json`.

| Set | Items | Mixed-speaker share with no name at a boundary | Predicted | Unmarked |
|---|---:|---:|---|---:|
| LongMemEval-S (all 120) | 1,200 | **0.749** | 0.40 to 0.80, held | 0.0017 |

**Gate: Stage 0 passes** on the full LongMemEval-S set (share at least 0.30, unmarked at most 0.05).
The partial 0.768 on 95 was within 0.02 of it. PersonaMem-v2 and CLBench are still being collected by
Stage B and are counted when it finishes; they inform the result but do not change this gate, which
is defined on LongMemEval-S. Stage 1 has not run.
