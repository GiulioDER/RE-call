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

## Result R1, measured 2026-09-25 08:25 UTC (appended; nothing above edited)

Run of `6a50c6cb` on VPS3, `/home/sentiment/c9-cite-diag/out`: 71 calls (31 LoCoMo, 40 BEAM),
USD 0.0995. Apparatus checks 1 to 3 pass: every successful call had sent ids, per-call
`accepted_records` equals the diagnostics line on all 71, and 71 calls completed.

| quantity | predicted | measured |
|---|---|---|
| fallback rate, all | 0.08 to 0.30 | **0.113** (8 of 71) |
| BEAM minus LoCoMo | above 0 | **-0.086** (0.075 against 0.161), falsified |
| median anchors, fallback minus other | above 0 | **-8** (6 against 14), falsified |
| every unknown-id share band | as listed | **all falsified: 89 of 89 unknown ids are `other`** |

**All 89 unknown ids are the ids of prior compiled records.** The compile prompt carries
`prior_records` (up to 24, each with its `id`), and gpt-4o-mini cited those ids as
`evidence_anchor_ids`. Every fallback but one (a `JSONDecodeError`) is a call whose proposals cite
only prior-record ids: 8 of 31 calls that had prior records fell back, 0 of 40 first chunks of a
session. The small LoCoMo chunks (1 to 6 anchors after 8 prior records) are the clearest case.
No miscopied hash, truncation, out-of-range index or quoted text occurred at all.

**Apparatus deviation found in R1:** the harness named prior records `rec<call>_<i>`, while C9
names them `mem_` plus 64 hex characters (`service.py`, `"mem_" + canonical_digest(payload)`;
2,242 of 2,242 compiled rows on the official table are 68 characters). A short `rec14_0` may be
easier to mistake for an evidence id than a long hash, so R1 establishes the mechanism but not its
rate under C9's real ids. The step 0 logging had the matching blind spot: its id filter would have
logged a `mem_` id as `<non-id:68 chars>`, hiding exactly this class. Both are fixed in the next
commit (`prior_record_id` in the harness, `mem_` in `_ID_SHAPED`), each with a red-proved test.

## Amendment for R2, written before R2 runs

R2 is R1 with prior records named in C9's own form (`prior_record_id`: `mem_` plus a 64-hex
digest) and a `prior_record_id` class (a cited id among the prior ids that call sent) plus
`prior_record_form_unsent`. Same 71 calls, same cap.

| quantity | band | point |
|---|---|---|
| fallback rate, all | 0.03 to 0.20 | 0.08 |
| fallbacks among first chunks of a session (no prior records) | 0 to 1 | 0 |
| share of unknown ids that are `prior_record_id` | 0.60 to 1.00 | 0.90 |

Decision mapping, replacing the one above for this cause: if `prior_record_id` is at least 0.60 of
the unknown ids in fallback calls, round 2 step 2 tests, in this order, (a) a prompt and payload
change that separates the two id spaces (prior records labelled as not citable, or carried without
ids the model can copy), and (b) server-side handling of a cited prior id: resolving it to that
record's own evidence spans, or dropping the citation instead of the record. Otherwise report and
decide with the user.

## Result R2, measured 2026-09-25 08:44 UTC (appended; nothing above edited)

Run of `f2c43c2b` on VPS3, `/home/sentiment/c9-cite-diag/out-r2`: the same 71 calls, prior records
named as C9 names them, USD 0.0946. Apparatus checks 1 to 3 pass (no call without sent ids,
diagnostics agree on all 71, 71 completed).

| quantity | band | measured |
|---|---|---|
| fallback rate, all | 0.03 to 0.20 | **0.085** (6 of 71), in band |
| fallbacks among first chunks of a session | 0 to 1 | **0 of 40**, in band |
| share of unknown ids that are `prior_record_id` | 0.60 to 1.00 | **0.966** (114 of 118), in band; **41 of 41** in fallback calls |

All 6 fallbacks are later chunks of a LoCoMo session (6 of 31 later chunks; BEAM 0 of 40 in R2,
3 of 40 in R1). The other 4 unknown ids: 3 `quoted_text`, 1 `hash_of_sent_anchor`.

**The loss is wider than the fallbacks.** 21 of the 31 calls that had prior records cited at
least one prior id. The 15 of those that did not fall back proposed 107 records and kept 55.

**Exploratory, not pre-registered:** records citing only prior ids overlap an earlier proposal
of the same session somewhat more than anchored records do (median token Jaccard 0.42 against
0.32; at least 0.6 for 42% against 32%). So some of what is rejected restates stored records,
and rejecting it is harmless deduplication, but most of it does not look like a restatement. A
crude lexical measure; the step 2 evaluation should judge novelty properly.

**Decision, by the rule above:** `prior_record_id` is 1.00 of the unknown ids in fallback calls
(at least 0.60), so round 2 step 2 tests, in order, (a) separating the two id spaces in the prompt
and payload, and (b) server-side handling of a cited prior id (resolve it to that record's
evidence spans, or drop the citation rather than the record).

**Status:** measured (R1 and R2, 2026-09-25). The cause it found was fixed by
`2026-09-25-c9-prior-record-ids.md`.
