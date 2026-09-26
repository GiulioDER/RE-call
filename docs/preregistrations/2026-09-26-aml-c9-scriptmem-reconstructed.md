# Pre-registration: X-1 ScriptMem, measured on locally reconstructed script texts

**Date:** 2026-09-26   **Status:** predicted, not yet measured (Stage 0, which is free and
decides which works enter, ran while the texts were built; its results are recorded below)

Branch `claude/scriptmem-rebuild` (off `claude/mm1-mm3`). A sub-study of X-1
(`docs/preregistrations/2026-09-25-aml-c9-source-coverage-baseline.md`), which reported ScriptMem
**unavailable** at Stage A because the public release carries questions, options and answers but
not the script text. This record does not amend X-1 and does not change its result: X-1's ScriptMem
line stays "unavailable". What is measured here is a **proxy**, called *reconstructed ScriptMem*
throughout, and it is never reported as ScriptMem.

Local directional experiment over public data and privately held texts, on VPS3; not an AML hosted
evaluation. The user approved private local use of the copyrighted texts on 2026-09-26, and chose
DeepSeek V4.1 Flash as the reader to save credit. No text is committed, published, or copied
anywhere but the session scratchpad on this workstation and the testbench.

## Why

ScriptMem is one of AML's seven Textual sources and the only one X-1 could not measure. It is also
the most multi-party of them (dozens of speakers, dialogue plus narration), so it is the source most
likely to expose something LoCoMo and LongMemEval do not. Another AML participant
(melandlabs/opencontext, commit `13dfe14`) rebuilt the texts from outside sources and renamed the
Friends leads to ScriptMem's aliases; they published no score from it.

## What the public questions fix, and the texts used

Read from ScriptMem `22ac7e7e` (457 questions; no evidence ids, no session ids):

| Work | Questions | Version the questions describe | Text used here |
|---|---:|---|---|
| An Enemy of the People | 94 | the R. Farquharson Sharp translation (Ejlif, Morten Kiil, Horster, Aslaksen, Billing, Evensen) | Project Gutenberg #2446, public domain; five acts; speakers from the play's own labels |
| 12 Angry Men | 99 | the 1957 film (Juror #4 and the actress's name, Juror #12's Rice Pops sketch, Juror #7 and Baltimore) | a public subtitle-style transcript of the film's dialogue, with speakers aligned from the published shooting script (a scanned PDF whose OCR is too noisy to use as text) |
| Friends | 174 | Season 1 (Celia and Kristin, the only doubtful names, appear in S1E15 and S1E14), the six leads renamed (Ross Bennett, Rachel Ariel, Monica Chloe, Chandler Dexter, Joey Ethan, Phoebe Fiona) | the EmoryNLP character-mining JSON, Season 1, 24 episodes |
| The Man from Earth | 90 | the 2007 film | a public subtitle-style transcript; **no speaker-labelled source exists**; dropped at Stage 0 (below) |

SHA-256 of what was fetched and of what is ingested (the texts themselves stay private):

| File | SHA-256 |
|---|---|
| Gutenberg #2446 plain text | `a38988e897a473218fec4eb96f7d551e03ad68fba12a24be9299b562a980fa92` |
| EmoryNLP `friends_season_01.json` | `a181b0fe8e13607c107402c4f30f78516d024d1872a4b45a838c581777fc4532` |
| 12 Angry Men dialogue page | `46695c3d75c01e833a2757bc3ffb242e62acaddea51bbda20b06c22609c9ad39` |
| 12 Angry Men shooting script PDF | `6c9328ab9c5fbb6f0b471e4f0cd3ec8910653ec1be2f326d2bbb13a0a88a9e2c` |
| 12 Angry Men shooting script OCR text | `2da5ba75c54cb98cff7f44e3fb2f48b0746b29d70e7ec61b33d89fe7e349f09b` |
| The Man from Earth dialogue page | `f8d1e26882f4ff59916436ec6a8838e8fee8493b830b09d40d4e5822d21afbcd` |
| normalised `enemy.jsonl` | `30a8d9ce468eb77c68b805956cc74b83e0a38ff2aee7b2496dfef13b16667ab0` |
| normalised `friends.jsonl` | `3946db84ad90b0d1ad1177f22f30025ee4bf4099ff38abef180af8917a1c682c` |
| normalised `angry.jsonl` | `c6edd72fda387c218d7350f373f70c8c4b79362a9ac10f5fc7d3ec06aae126e7` |
| ScriptMem `angry.json` | `f9234a9d4bbd48e5d74fa94f22031ad4867a90756d285afeb1cf3bd039a11b88` |
| ScriptMem `enemy.json` | `459f7d91a2af8f23cf01954f4bdd8afc2f5a9ab32c596571249f85e5b918294c` |
| ScriptMem `friends.json` | `58da6f3dc20644aa35ee17b427e9cab3b6596b97171c6bf28ffc5889c1d0cf04` |
| ScriptMem `man_earth.json` | `4d2f51bcc5a817ab38d2195845458fdb379b593928efa844032911580b2b070b` |

## How the texts are normalised (`scripts/aml_x1_scriptmem.py normalize`)

- **Enemy**: each paragraph that opens with a speaker label (`Dr. Stockmann.`, `Billing (as he
  eats).`) is a line for that speaker, the parenthesised direction kept at its head; every other
  paragraph inside an act is narration. 1,519 lines: 1,440 attributed, 79 narration, 5 acts.
- **Friends**: one line per utterance with its speaker, renamed (full names first, then first names
  and the nicknames "Rach" and "Pheebs", on word boundaries, in labels and text). 5,968 lines.
- **12 Angry Men**: subtitle fragments are rejoined into utterances. Speakers come from the shooting
  script's cue lines (`#3`, `FOREMAN`, OCR's `#ll` read as `#11`): each film line is matched to the
  script block sharing most of its content words (function words excluded), anchors are kept only
  along the longest run increasing in script order, lines between anchors match only the blocks
  between them, and any single-speaker run over 30 lines is unattributed as a swallowed scene. 1,643
  lines: **785 attributed (48%), 858 left as unattributed dialogue**. Attribution precision is not
  measured; spot checks agreed with the film, including the exchange a ScriptMem question names
  (Juror #9 calling Juror #10 an ignorant man).
- Unattributed dialogue is rendered bare, never as `Narration:`.

## Mapping to AML requests, fixed now

AML's own ScriptMem ingest mapping is not public, so this follows the convention the LoCoMo harness
already uses (`turn_content` in `scripts/aml_locomo_route_compare.py`):

- **One tenant per work**, `user_id` from `aml_x1_sources.user_id_for("x1sm", "scriptmem", work)`.
- **One Add per session**; every line is a message with role `user` and content
  `"<Speaker>: <text>"`, `"Narration: <text>"`, or the bare text for unattributed dialogue.
- **Sessions**: Friends one per episode, timestamped with its first US air date; Enemy one per act;
  12 Angry Men consecutive runs of 60 lines. Works without a calendar carry 2000-01-01 UTC plus one
  minute per session index, so order is kept and no false gap appears.
- **Search**: the question with its options, `top_k` 100, exactly as
  `aml_x1_sources.scriptmem_questions` builds it.
- **Service**: C9 as X-1 Stage B serves it (VPS3, the `x1-repo` checkout, `RECALL_AML_COMPILER=0`,
  X-1 amendment 2), under this record's own run id, so no X-1 tenant is touched.

## Stage 0, coverage gate: run, and final

For each question, the capitalised words and numbers in the **gold** option, excluding the option's
first word and anything already in the question stem, are looked up in the work's text; a work below
**0.80** is dropped. The same share over the wrong options is context, not a gate.

**History, stated because the gate was adjusted once.** The first read counted every capitalised
word, including the option's first word, and gave Enemy 0.806 and Friends 0.907 with every Enemy
miss a sentence-initial common word ("Earliest", "Institutional"). The first word was then excluded.
Every number below is under that rule, and no further change was made to the matcher. The two films
were first read after that change; 12 Angry Men then read 0.733 on bare subtitles, all its misses
juror numbers, which is why speakers were aligned (a change to the TEXT, which this record's table
had already called for, not to the gate).

| Work | Gold share | Wrong-option share | Result |
|---|---:|---:|---|
| An Enemy of the People | 1.000 (51/51) | 0.835 | kept |
| Friends, Season 1 | 0.969 (127/131) | 0.960 | kept |
| 12 Angry Men, speakers aligned | 0.933 (42/45) | 0.916 | kept |
| The Man from Earth | 0.797 (59/74) | 0.740 | **dropped** |

The Man from Earth's misses include content absent from the subtitles ("United States", "Bronze
Age", "1890") as well as possessives, and no speaker-labelled source exists; it is dropped rather
than rescued by a matcher change made after seeing its number. **Kept: 367 questions** (Enemy 94,
Friends 174, 12 Angry Men 99).

## Arms

All arms answered with `deepseek/deepseek-v4.1-flash`, one pinned provider (DeepInfra, as the T-1
runs), reasoning off, temperature 0, output capped at 300 tokens; scored by ScriptMem's exact scorer
as vendored by AML (`gold_letters`, `predicted_letters`, `score_item`; multi-select all and only,
ordering exact). No judge.

| Arm | Memories given to AML's ScriptMem answer prompt | Scope |
|---|---|---|
| **C9** | the 100 items C9 returns, in AML's timestamped block format | all kept works |
| **C9p** | the same items, answered a second time, interleaved per question | all kept works |
| **N0** | none: the memory slot reads `(no memories)` | all kept works |
| **NK** | none, and the prompt's first instruction replaced by "Answer from what you know about *<work>*." | all kept works |
| **F** | the whole normalised text of the work | Enemy and 12 Angry Men (Friends Season 1 is too large for the cap) |

Speaker slots, as in the LoCoMo harness: `speaker_1_name` "the characters of *<work>*",
`speaker_2_name` "(none)", `speaker_2_memories` "(all memories are listed above)". Arm order rotates
per question so reader drift falls on every arm alike.

N0 is the floor the benchmark actually offers (AML's prompt tells the reader to pick "Cannot infer"
only when no memory is relevant). NK measures what the reader already knows about these works, which
is the contamination this proxy cannot remove: two of the three kept works are unrenamed and famous.

## What I predict

My effect predictions have run two to four times too high (memory
`i-over-predict-effect-magnitudes`). X-1 predicted ScriptMem exact match 0.30 to 0.60 for C9 on the
real data; that band is carried over unchanged as the only level prediction for C9.

| Quantity | Predicted |
|---|---|
| C9, exact match, 367 kept questions | 0.30 to 0.60 (X-1's band) |
| \|C9p − C9\| | at most 0.03 |
| N0 | 0.10 to 0.30; at least half of its answers pick "Cannot infer" |
| NK | 0.25 to 0.45 |
| C9 − N0 | +0.15 to +0.35 |
| C9 − NK | +0.00 to +0.15 |
| F − C9, Enemy and 12 Angry Men | +0.05 to +0.20 |
| lowest question type under C9 | ordering |
| Friends against the two single works, under C9 | Friends lower |

## What would falsify this

- C9 outside 0.30 to 0.60 falsifies the carried band (for the proxy; it says nothing about the real
  ScriptMem by itself).
- C9 − NK at or below 0: the memory adds nothing over what the reader already knows, and the proxy
  cannot judge a retrieval change on these works.
- F − C9 at or below 0: retrieval loses nothing against the whole text.
- \|C9p − C9\| above 0.05: unusable at this size.

## How the result is used, fixed now

1. If C9 − NK exceeds the C9p noise floor, reconstructed ScriptMem joins X-1's held-out set as a
   **seventh, labelled proxy source**, for round-two candidates that touch multi-party dialogue.
2. If not, it is recorded as contaminated and is not used to accept or reject any change.
3. Per-type and per-work scores enter the round-two plan as targets only where they sit below C9's
   own mean by more than the noise floor.

## Spend and host

On VPS3 only: Voyage for ingest and query embeddings, DeepSeek through OpenRouter for answers.
**Cap USD 10** (two C9 arms at about 367 × 21k prompt tokens, about USD 2.3 each at list price; F
about USD 1.6; N0 and NK under USD 0.2). **No paid call runs while an official AML Full is live**,
because VPS3 shares the official C9's Voyage and OpenRouter keys (memo
`2026-09-22-vps2-hands-off-during-official-aml-run`). The check before each paid stage is the
official C9's journal showing no AML-client POST in the last hour, and the OpenRouter balance above
the USD 5 floor. At the time this record was written (17:10 UTC) the Textual Full was live, so
nothing paid has run.

## Apparatus checks, fixed now

1. Every Add returns 200; each tenant is deleted and verified empty afterwards.
2. A canary per work: a verbatim line from the middle of its text, as the query, must return in the
   top 10.
3. Valid-answer rate (a parseable option string) at least 98% per arm; unparseable answers are
   scored wrong and counted separately.
4. The answer prompt is AML's `CHOICE_ANSWER_TEMPLATE` from the pinned AML checkout (`1b8142b`),
   byte for byte except NK's one stated line.
5. The question files on VPS3 hash to the values above before anything is sent.

## Confounds I can name now

- **The proxy is not the benchmark.** ScriptMem's private text has its own sessions, narration and
  possibly edits, and a subtitle transcript differs in wording from a script. Levels here predict
  AML's ScriptMem score only loosely; differences between arms are the product.
- **Half of 12 Angry Men is unattributed**, and the attributed half has unmeasured precision.
  Questions about who said what are harder here than on ScriptMem's own text.
- **Contamination**: measured by NK, not removed.
- **Reader**: DeepSeek V4.1 Flash, for cost; AML's reader for ScriptMem is not published and the
  ScriptMem leaderboard uses gpt-4o-mini, so no level here compares with either.
- **Small strata**: ordering has 30 kept questions (8, 8 and 14); per-type numbers flag, they do not measure.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No paid measurement had run when this record was committed.
