# Pre-registration: does an Add-time aggregation view fix C9's incomplete-list answers?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

Written before any extraction, record or answer for this design exists. The compiled-records
counterfactual had finished running but had not been read when this was committed.

## The question

On LoCoMo category 1 (multi-hop, 282 questions), does inserting up to two Add-time aggregation
records into C9's served evidence raise answer accuracy over the served evidence alone? Accuracy
is AML's own LoCoMo-Refined judge verdict. The comparison is paired on the same retrieval run,
with the same reader and judge, and must not cost the other 1,253 questions.

## Why it is being asked, and why LoCoMo alone cannot settle it

The loss diagnosis (`2026-09-24-aml-c9-locomo-loss-diagnosis.md`, result `1b9b2c91`) found that
incomplete lists are the largest loss: 120 of 379 wrong answers, 89 of them in category 1. Every
item was returned and spread across sessions, and the reader listed only some. So the fix belongs
in what the reader is shown, not in recall.

**This idea was generated from these very questions,** so a LoCoMo gain is development evidence
only. The design below is frozen before anything runs, and no part of it may be tuned on LoCoMo
answers ([[an-in-sample-best-cut-is-not-a-measurement]]). A held-out gate on a different dataset
is fixed here too, and must pass before the view goes anywhere near C9.

## The design, frozen

All of it is in `scripts/aml_locomo_aggregation_view.py`, committed with this record.

- **Extract, one call per session, as an Add would.**
  - `openai/gpt-4o-mini` at temperature 0 reads the session's turns with their ids and date.
  - It lists `(person, category, item, turn_id)` under the fixed `EXTRACT_PROMPT`.
  - The category comes from the fixed 19-entry `CATEGORIES` list; anything else becomes `other`.
- **Grounding.** An entry is dropped unless its `turn_id` is a turn of that session, its person is
  one of the session's two speakers, and its item is non-empty.
- **Build.** One record per `(conversation, person, category)`.
  - Items are ordered oldest first and deduplicated case-insensitively.
  - Each line carries the session date, the item, and up to 160 characters of its source turn.
  - The record's `created_at` is its latest session date. It is built from all sessions, which
    equals the state after the last Add, since LoCoMo runs every Add before any Search.
- **Select, at Search, with no LLM.**
  - Okapi BM25 (k1 1.5, b 0.75) scores the question against the conversation's records.
  - If the question names a speaker, only that speaker's records are eligible.
  - Up to 2 records with a score above zero are chosen.
- **Place.** The chosen records go after the protected top five, at ranks 6 and 7, and the list is
  cut back to 100 items. The fused top five is never touched (user decision 2026-09-23, recorded in
  `recall_aml/variants.py`).

## Arms

| arm | evidence | answers |
| --- | --- | --- |
| A | C9's served list from collect `full1` | the loss diagnosis answers |
| A′ | the same list answered again | new, all 1,535, the noise floor |
| C | the same list with the aggregation records inserted | new, all 1,535 |

Reader and judge are unchanged from the loss diagnosis: AML's LoCoMo-Refined prompts at
`1b8142bf`, `openai/gpt-4o-mini` through OpenRouter, temperature 0.

## What I predict

| quantity | prediction |
| --- | --- |
| extraction: share of proposed entries that survive grounding | 85% to 98% |
| records per conversation | 40 to 150 |
| questions with at least one record inserted, all 1,535 | 50% to 85% |
| the same, category 1 | 60% to 90% |
| **C minus A, category 1 (n = 282)** | **+2 to +6 points** |
| C minus A, the other 1,253 | -1.0 to +0.5 points |
| C minus A, all 1,535 | +0.2 to +1.5 points |
| A′ minus A, category 1 | -2.0 to +2.0 points |
| of category 1 wrong-to-right flips (C against A), share with a record inserted | at least 70% |

Reasoning. If every one of the 89 category 1 list errors were fixed, that would be +31.6 points,
but that is not a realistic ceiling. The extraction will miss items, BM25 will sometimes pick the
wrong category, and the reader may still stop early. I guess about a third of those errors are
reachable, which is about +10 points. My record says I capture a quarter to a half of a ceiling
([[i-over-predict-effect-magnitudes]]), so +2 to +6. Harm elsewhere should be small, because the
top five are untouched and at most two items are displaced from the tail.

## Decision rule, fixed now

**LoCoMo passes** only if all three hold:
- C minus A on category 1 is at least +2.0 points, with a 95% interval above zero;
- it is larger than the absolute value of A′ minus A on category 1;
- C minus A on the other 1,253 is at least -1.0 point.

**Then the held-out gate runs**, and nothing is built into C9 before it passes.
- **Questions:** 40 LongMemEval-S multi-session questions, drawn by `random.Random(0).sample` from
  the 133 sorted multi-session ids of `longmemeval_s_cleaned.json` (SHA256 `d6f21ea9…`), frozen
  here: `1192316e 129d1232 1a8a66a6 2311e44b_abs 2318644b 3a704032 4adc0475 51c32626 5a7937c8
  60036106 60bf93ed 6456829e 85fa3a3f 87f22b4a 8979f9ec 8e91e7d9 92a0aa75 9ee3ecd6 a1cc6108
  a4996e51 a96c20ee_abs bc149d6b c2ac3c61 d23cf73b d682f1a2 d905b33f e25c3b8d e3038f8c e56a43b9
  e5ba910e_abs edced276 edced276_abs ef9cf60a f35224e0 gpt4_15e38248 gpt4_194be4b3
  gpt4_2ba83207 gpt4_372c3eed gpt4_59c863d7 gpt4_a56e767c`.
- **Setup:** C9 serves the same Add then Search protocol, 1,893 sessions in all, and the same
  extraction prompt and rules are applied.
- **Adaptation, the only one allowed:** speakers are `user` and `assistant`, turn ids are
  `S<session>:<turn>`, and a question in the first person (I, my, me, mine) names `user`.
- **Reader and judge:** AML's LongMemEval pipeline prompts at `1b8142bf`, with the same models.
- **The gate passes** if C minus A is at least +5 points and wrong-to-right flips outnumber
  right-to-wrong flips. With 40 questions this is a sign check, not a significance test.

**If LoCoMo fails, stop.** Do not tune the prompt, categories, selection, count or placement on
these questions. A redesign needs a new record and questions this work has not seen.

Nothing here changes the first official Full run.

## What would falsify this

- C minus A on category 1 below +2 or above +6 points.
- Harm on the other 1,253 beyond -1.0 point.
- Grounding survival or insertion coverage outside its band. Either would mean the mechanism did
  not run as designed, whatever the accuracy says.

## How it will be measured

```bash
python scripts/aml_locomo_aggregation_view.py extract --data locomo10.json --out extracted.jsonl
python scripts/aml_locomo_aggregation_view.py build --data locomo10.json --extracted extracted.jsonl --out records.json
python scripts/aml_locomo_aggregation_view.py answer --collected full1.json.gz --data locomo10.json \
    --records records.json --aml-repo aml-official --out agg-answers.jsonl
python scripts/aml_locomo_loss_diagnosis.py answer --collected full1.json.gz --data locomo10.json \
    --aml-repo aml-official --out replicate-answers.jsonl
# then aml_locomo_loss_diagnosis.py judge on both, and compare A against C and A against A'
```

- **Statistics:** `compare` gives paired accuracy with a 10,000-resample percentile interval (seed
  0) and discordant counts, overall and by category. The "other 1,253" figure is computed from the
  per-category rows.
- **Tests:** `tests/test_aml_locomo_aggregation_view.py` covers grounding, record building,
  selection and placement. Each test is proved red by mutation.
- **Cost:** about 10 USD. Extraction is about 0.2, each of the two arms about 4.5, and judging
  about 0.4. Every output file is capped at 15 USD.

## Confounds I can name now

- **The idea came from these questions**, hence the held-out gate.
- **Arm C adds text** as well as structure. A gain could come from repeated evidence rather than
  from aggregation. The mechanism metric (flips where a record was inserted) speaks to this but
  does not settle it.
- **The streaming contract is not exercised.** In a live Add, the records would be updated session
  by session. The offline build equals the final state only because LoCoMo searches after all Adds.
- **The reader and judge are stand-ins**, and the speaker split is not AML's, as in the loss
  diagnosis.
- **The C9 build that would carry this does not exist yet.** This tests the evidence the reader
  would see, not a served implementation, its latency, or its Add cost.

## Amendment before measurement (2026-09-24, before any extraction, record or answer)

The compiled-records counterfactual finished after this record was committed. Its replicate
answered the identical evidence twice, hours apart, and got 71.7% then 69.3%: a paired
difference of -2.38 points [-4.52, -0.24]. That is drift between runs, of the same size as the
effect predicted here. So comparing arm C, answered now, against arm A, answered earlier, would
measure the clock as much as the view. What changes, and what does not:

- **Pinned provider, recorded provenance.** Every call in this experiment runs with
  `AML_DIAG_PROVIDER=OpenAI`, which the script turns into an OpenRouter provider order with no
  fallback. Every answer and judge record also stores the `provider` and `system_fingerprint` the
  response names. The earlier runs stored neither, which is why their drift cannot be explained.
- **Concurrent arms.** C, A′ and a second replicate A″ run at the same time, as three processes
  started together, and are judged at the same time. A″ covers category 1 only (282 questions),
  since only that category's floor enters the decision.
- **The primary comparison becomes C against A′**, both answered concurrently and pinned. The
  LoCoMo pass criteria keep their thresholds and apply to that pair:
  - C minus A′ on category 1 is at least +2.0 points, with a 95% interval above zero;
  - it is larger than the absolute value of A″ minus A′ on category 1;
  - C minus A′ on the other 1,253 is at least -1.0 point.
- **C minus A is still reported**, as a secondary comparison, labelled as spanning runs.
- **The held-out gate** uses the same pinning and concurrency, with A and C answered together.
- Unchanged: the design, the prompt, every prediction and every threshold.
- Added cost: about 0.8 USD for A″.

## Amendment 2 before measurement (2026-09-24, before any record or answer)

The first launch stopped itself at the pre-registered grounding metric. After 53 of 272 sessions,
only 42.6% of proposed entries survived grounding, against a predicted 85% to 98%.

- **Cause, reproduced on one session:** a harness defect, not the model. The prompt shows turns as
  `[D1:3] speaker: text` and asks for the id "copied exactly", so the model returned `"[D1:3]"`
  with the brackets, and `grounded_items` looked up the bracketed string.
- **Fix:** strip surrounding brackets and whitespace from the id, and nothing else. Covered by
  `test_a_bracketed_turn_id_is_the_same_turn`, proved red by removing that strip.
- **Also:** each extraction record now stores its raw reply, so a grounding drop can be audited
  without re-running the model.
- **Handling of the bad run:** it was stopped during extraction. No record was built and no answer
  requested. Its output is kept as `extracted-v1-bracket-bug.jsonl`, not used, and extraction
  restarts from empty.
- **Unchanged:** the prompt, every rule, prediction and threshold, including the 85% to 98%
  survival band, which now measures grounding rather than bracket handling.
