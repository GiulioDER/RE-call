# Pre-registration: does a shorter compile answer make C9's Add faster without losing Coding retrieval?

**Date:** 2026-09-26   **Status:** predicted, not yet measured

## The question

An Add on C9 waits for its gpt-4o-mini compile, and that wait is output bound. Over the official
Textual Full of 2026-09-25 (C9 journal, 13,084 Adds with one compile call), Add time fitted
7.65 s plus 11.9 s per 1,000 completion tokens, with completion p50 1,046 tokens. The same journal
shows most of that output is thrown away: 57,222 of 75,709 accepted records (76%) had every
generated text field removed as not verbatim and were backfilled from their first cited anchor,
so for three records in four the only surviving model output was the record kind and the cited
anchor ids. 357 records were also rejected only because the model copied the session id wrong.

Two shorter answer shapes, built on `claude/c9-compile-output` (stacked on #775), both off by
default and C9 unchanged (`anchor_compile_output="full"`):

- **`lean`**: the full shape without `source_session_id` and `supersedes`, which the compiler
  overwrites or drops anyway, and with empty fields omitted.
- **`select`**: only `kind` and `evidence_anchor_ids`; every record is backfilled from its first
  cited anchor exactly as the full path backfills a record whose fields were all removed.

Do they make the compile faster and cheaper, and do they keep Coding retrieval? The decision is
for the Coding Full, whose configuration is not registered yet.

## Design

1. **Compile replay.** `scripts/aml_c9_compile_output_replay.py run --mode {full,lean,select}` over
   the 200 BEAM Adds already built for the cost replay (`/home/sentiment/c9-cost-replay/adds.jsonl`
   on VPS3, the prompt-size mix of the Textual Full), each arm with #775's bounds
   (`without-ids`, no prior records, 150,000 character limit), 6 workers, spend cap USD 1.5 per
   arm, arms run one after another in the order full, select, lean. `compare` gives the numbers.
2. **Coding retrieval.** `scripts/aml_c9_coding_window_check.py collect --dated-search-content
   --compile-output {full,select,lean}` as arms K4 (full, the existing K4 record), K5 (select)
   and K6 (lean), each in a fresh database on VPS3, the same frozen AMB Coding screen (196
   sessions, 34 prompts), compared by `report` with K0 and K0b.

**Keys.** Neither measurement may use the official C9 keys while an official run is live
([[never-share-the-official-keys-during-a-full]]). Before each run the runner compares the SHA-256
of its Voyage and OpenRouter keys with `/etc/recall-aml/c9-official.env` and does not start on a
match. Runs wait for the Textual Full to finish unless separate experiment keys exist.

## Baseline already measured (not a prediction)

Cost replay arm B (#775, 2026-09-26, same 200 Adds): 157 compiled, 36 skipped as too large,
1 truncated, 1,076 accepted records (6.85 per compiled Add), completion p50 805, p90 1,029,
USD 0.315. The full arm here re-measures this.

## Predictions

My effect-size predictions have run two to four times too high (memo
`i-over-predict-effect-magnitudes`), so these bands are deliberately modest.

| quantity | band | point |
|---|---|---|
| select / full completion tokens (total) | 0.30 to 0.55 | 0.40 |
| lean / full completion tokens (total) | 0.70 to 0.95 | 0.85 |
| select / full paired median compile seconds | 0.45 to 0.80 | 0.62 |
| lean / full paired median compile seconds | 0.80 to 1.00 | 0.92 |
| select / full cost | 0.75 to 0.93 | 0.85 |
| full compiled Adds (of 200) | 150 to 164 | 157 |
| select compiled Adds minus full | -3 to +8 | +2 |
| lean compiled Adds minus full | -3 to +5 | +1 |
| full accepted records per compiled Add | 6.3 to 7.3 | 6.85 |
| select accepted per compiled Add minus full | -1.0 to +0.5 | -0.2 |
| full backfilled share of accepted records | 0.55 to 0.85 | 0.72 |
| K5 (select) Coding MRR | 0.845 to 0.870 | 0.858 |
| K6 (lean) Coding MRR | 0.852 to 0.870 | 0.862 |
| K5 and K6 recall@10 | 33 to 34 | 34 |

Mechanism I expect: the saving is completion tokens only; prompt tokens stay equal, and they are
most of the cost, which is why the cost ratio is predicted far above the token ratio.

## Decision rule, fixed now

N is the noise floor |K0b minus K0| on Coding MRR (0.000 measured).

1. Apparatus failure (below): report, decide nothing.
2. **Serve `select`** in the Coding Full build when all hold: K5 MRR at least K0's minus
   (0.02 + N), K5 recall@10 at least 33 and recall@100 34, no Add failures; select compiled Adds
   at least full's minus 5; select paired median compile seconds ratio at most 0.80; select
   accepted per compiled Add at least full's minus 1.0.
3. Otherwise **serve `lean`** when the same hold for K6 and lean, with the seconds ratio at most
   0.92.
4. Otherwise keep `full`.

The Textual configuration is not decided here: any change to it needs its own check on a Textual
proxy.

## Apparatus checks

1. `/version` of K5 and K6 reports `anchor_compile_output` as the arm claims; K4, K5, K6 each
   write 1,220 raw windows.
2. Every replay arm covers the same 200 Add ids; no arm stops at its spend cap.
3. The key hashes differ from the official C9's, recorded in the result.

## Confounds

- BEAM Adds stand in for AML Textual, and the AMB Coding screen for AML Coding.
- OpenRouter does not pin a provider, and provider speed moves by the hour; arms run back to back
  once, so a provider shift between arms is not controlled. The paired median over 200 Adds is
  the guard, not a cure.
- `select` records keep the first 700 characters of the first cited anchor; the model's ordering
  of anchors now decides the stored text, which `full` did only for backfilled records.
- One run of each.
