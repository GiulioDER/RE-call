# Pre-registration: which unknown anchor ids does C9's compiler cite?

**Date:** 2026-09-25   **Status:** predicted, not yet measured

## The question

On the 2026-09-25 AML Textual smoke (C9 at `efb79146`), 25 of 134 Adds fell back
(`compiler_fallback: true`), 20 of 137 the day before. In the journal about 20 of the 25 were
compiles whose every proposed record cited only anchor ids that were never sent, so C9 dropped the
compile and the Add kept no compiled record. `resolved_bare_anchor_ids` was 0, so this is not the
bare `a162` form #757 fixed. The journal could not say which ids were cited. What are they?

This is round 2 step 1 of the compiler-fallback plan. It decides which fix to build, not whether
a fix works. The official C9 service is not touched.

## Design

Harness: `scripts/aml_c9_compile_citation_diagnosis.py` (this commit), tests
`tests/test_aml_c9_compile_citation_diagnosis.py` (12, both red-proved; see the docstring).

- **The served compiler, in process.** `OpenAICompiler.compile_anchored_v3` at this commit, the
  code C9 runs apart from the logging-only change in the same branch, through the same
  `build_openrouter_client`, gpt-4o-mini, temperature 0, no provider pin (as C9). A wrapper keeps
  every request and raw answer.
- **Inputs, public only.** LoCoMo (`locomo10.json`): the first 2 sessions of each of 10
  conversations. BEAM 100K (`beam100k.jsonl`, the C9 BEAM probe's input): the first 2 chunks of
  each of 20 conversations. Both cut as AML's Textual adapter cuts (`aml_chunks`: 20 messages or
  2,000 words, never across a session), with AML millisecond timestamps. Chunks of one session run
  in order, and each compile receives the records of the earlier chunks of its session as `prior`,
  as C9 does.
- **Sorting.** Each cited id is classified against the ids that call actually sent, read back
  from the request's `<stored_data>` (`classify`, first rule wins): `known`, `bare_index_known`
  (resolved by #757), `bare_index_out_of_range`, `truncated_id` (a prefix of a sent id, or right
  index with a prefix of its hash), `index_out_of_range`, `hash_of_sent_anchor` (unsent index,
  a sent hash), `right_index_wrong_hash`, `hash_of_other_anchor` (sent index, another sent
  anchor's hash), `v2_anchor_form`, `quoted_text`, `other`.
- **A fallback** is a compile that raised or returned no record, as C9 counts it.
- **Spend cap** USD 0.60 at gpt-4o-mini list price; about 80 calls are planned.

## Predictions

[[i-over-predict-effect-magnitudes]] applies to effect sizes; these are shares, so the bands are
wide rather than shrunk.

| quantity | band | point |
|---|---|---|
| fallback rate, all calls | 0.08 to 0.30 | 0.15 |
| fallback rate, BEAM minus LoCoMo | above 0 | |
| median sent anchors, fallback calls minus other calls | above 0 | |
| share of unknown ids that are `right_index_wrong_hash` | 0.40 to 0.75 | 0.55 |
| share `truncated_id` | 0.05 to 0.25 | 0.12 |
| share `index_out_of_range` plus `hash_of_sent_anchor` | 0.05 to 0.25 | 0.12 |
| share `hash_of_other_anchor` | 0.00 to 0.15 | 0.05 |
| share `v2_anchor_form` | 0.00 to 0.05 | 0.00 |
| share `quoted_text` plus `other` | 0.00 to 0.15 | 0.05 |

Reasoning: v3 ids carry a 16-hex hash the model has to copy exactly; hex strings are what a small
model miscopies most, and more anchors in the prompt mean more of them to copy. Indices are short
and in order, so they should survive better than hashes.

## What each outcome points to (round 2 step 2)

- `right_index_wrong_hash` plus `truncated_id` at least 0.60 of the unknown ids in fallback calls:
  build **short ids** first (send `a001`, map to the hashed id server side; compiler version 4),
  and compare it with resolving an in-range index whose record text matches that anchor's quote.
- `index_out_of_range` plus `hash_of_sent_anchor` at least 0.40: the model loses track of which
  anchors exist; test fewer anchors per call or renumbering before the hash question.
- `quoted_text` plus `other` at least 0.40: extend the exact-text recovery to near matches.
- Anything else: report and decide with the user.

## Apparatus checks

1. `sent_anchor_ids` finds `<stored_data>` in every successful call (no call has 0 sent ids).
2. The run's per-call `accepted_records` agrees with the `compiler_anchor_compile_complete`
   diagnostics line of that call.
3. At least 60 calls complete within the cap.

## Confounds

- Public LoCoMo and BEAM, not AML's data (which may not be analysed). BEAM's long assistant turns
  resemble AML's 20-message Adds; LoCoMo's short turns do not.
- Temperature 0 is not determinism through OpenRouter, and C9 does not pin a provider, so the
  provider mix of the run is part of the result.
- The smoke's journal pairing that motivated this is approximate (compile lines had no
  request_digest); this run does not depend on it.
