# Pre-registration: does C9's compile cost bound (#775) save what the journal estimate says?

**Date:** 2026-09-26   **Status:** predicted, not yet measured

## The question

C9's journal from the official Textual Full (2026-09-25 10:49 to 2026-09-26 02:19 UTC) shows the
anchored compiler cost USD 85.27 at list price, of which USD 67.18 went to 2,802 Adds whose answer
stopped at `max_tokens` and was resent twice. From logged token counts, #775 (no resend of a
cut-off answer, and no call over 150,000 encoded characters) would have cost about USD 23 and lost
about 330 of 9,914 compiled Adds (3.3%). That figure is arithmetic on a log, not a run. Does the
saving hold when the served compiler actually runs both ways on the same inputs?

## Design

Harness `scripts/aml_c9_compile_cost_replay.py`, test `tests/test_aml_c9_compile_cost_replay.py`.

- **Inputs:** 200 Adds built from public BEAM 100K (`beam100k.jsonl`, the C9 BEAM probe's input),
  consecutive messages of one conversation, sized to follow the Full's first-call prompt-token mix
  (9 strata, seed 20260926; 45% under 5k tokens, 13% over 40k). Sizes are aimed with 3.7 characters
  per token and read back from the provider's `prompt_tokens`.
- **Arm A, served:** `recall_aml` from `origin/master` (`3eb447c4` content for the compiler: a
  cut-off answer is resent, no size limit).
- **Arm B, #775:** `recall_aml` from `claude/c9-compile-cost` (`b8fb4989`), with
  `max_anchor_payload_chars=150_000`.
- Both: `prior_record_mode="without-ids"`, no prior records, gpt-4o-mini through OpenRouter as C9
  calls it, 6 workers per arm, both arms concurrently. Cost at list price from the returned token
  counts. Spend caps USD 2.00 (A) and USD 1.00 (B). The key is the official C9's; the Full was
  running when this was written.

## Predictions

| quantity | band | point |
|---|---|---|
| A cost, 200 Adds | USD 0.80 to 2.00 | 1.30 |
| **B cost divided by A cost** | **0.20 to 0.45** | **0.28** |
| A compiled Adds, share of 200 | 0.65 to 0.90 | 0.77 |
| A compiled share among Adds whose first prompt is 40k tokens or more | 0.05 to 0.35 | 0.15 |
| A Adds with a cut-off first answer that compile after a resend, share of those Adds | 0.00 to 0.10 | 0.03 |
| **B compiled Adds divided by A compiled Adds** | **0.90 to 1.00** | **0.97** |
| B calls per non-skipped Add | 1.00 to 1.15 | 1.05 |

## Decision rule, fixed now

1. Apparatus failure (any check below fails): report, decide nothing.
2. **Saving confirmed** if B/A cost is at most 0.50 AND B keeps at least 0.90 of A's compiled Adds.
3. **Saving confirmed, loss larger than estimated** if B/A cost is at most 0.50 but B keeps less
   than 0.90: report the loss by size bin; the user decides whether 150,000 characters is right.
4. **Not confirmed** if B/A cost is above 0.50: the journal arithmetic does not survive a run.

## Apparatus checks

1. All 200 Adds complete in both arms within the caps.
2. B's skips (`CompilerInputTooLarge`) are exactly the Adds whose encoded payload is over 150,000
   characters, and B makes no call for them.
3. B never sends a second call after a `finish_reason == "length"` answer.
4. The test in `tests/test_aml_c9_compile_cost_replay.py` passes and was proved red.

## Confounds, stated now

- BEAM is not AML's data. Truncation depends on how long the verbatim fields are that the model
  copies, so the size at which answers get cut off may differ.
- One run per arm; OpenRouter routes gpt-4o-mini to more than one provider (C9 does not pin one),
  so single-Add outcomes are noisy. The rule is on 200-Add totals.
- No prior records are sent, while the Full's later chunks carried them; prompts here are anchors
  only.
