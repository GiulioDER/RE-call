# Pre-registration: do C9's coding-kind compiled records cost Textual answers on the Context4 route?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

Written while the loss diagnosis (`2026-09-24-aml-c9-locomo-loss-diagnosis.md`) was still
answering and judging. No answer, judge label or accuracy from that run had been read when this
was committed.

## The question

On the 420 LoCoMo questions that C9's router sends to the Context4 route, does removing the
compiled records from the served evidence list raise answer accuracy? Accuracy is AML's own
LoCoMo-Refined judge verdict. The comparison is paired, with the same reader, the same judge and
the same retrieval run.

## Why it is being asked

Measured on the 2026-09-24 collect (`full1`, C9 at `337f2537`), before any answer was read:

- **Where they appear.** Compiled records reach the Answer model only on the Context4 route, because
  with the graph on `persist_specialist` receives compiled chunks and the primary Code4 store does
  not. There they are 37.5 of 100 items per question, 1.98 of the top 10, and 21.3% of the
  characters.
- **What they are.** They come from a compiler whose prompt is coding-only ("You compile stored
  coding conversations…") and allows only nine coding kinds. So conversation turns are filed as
  `procedure`, `symptom`, `successful repair` and so on.
- **They repeat raw text.** 13,997 of 15,759 carry an evidence line that also appears verbatim in a
  raw item of the same list.
- **They carry no timestamp.** Every one has `created_at` None, while every raw item has one.
- **The route is mostly date questions.** 214 of the 420 are temporal (category 2), since "when
  did" is one of the router's context keywords.
- **Dropping them loses no gold.** Removing them leaves gold evidence present in 419 of 420 lists,
  the same as with them, and keeps 62.5 raw items per question on average.

So the undated, duplicated, mislabelled records land mostly on the questions where a date matters.

## Arms

| arm | evidence given to the reader | answers |
| --- | --- | --- |
| A | the served list, 100 items including compiled records | the loss diagnosis answers, restricted to these 420 |
| A′ | the same served list, answered again | new |
| B | the served list with every non-raw item removed, raw items in served rank order | new |

A′ measures the reader's own run-to-run variation at temperature 0. That is the floor any B
effect must clear.

## What I predict

Accuracy is in points over the 420 questions, unless stated.

| quantity | prediction |
| --- | --- |
| B minus A | +0.5 to +3.0 points, 95% interval including zero |
| B minus A, category 2 only (n = 214) | +1.0 to +5.0 points |
| A′ minus A | -1.0 to +1.0 points |
| discordant questions, A′ against A | 5 to 25 |
| discordant questions, B against A | 20 to 50 |
| of B's wrong-to-right flips, share in category 2 | 50% to 80% |

Reasoning. The records add no evidence the raw list lacks, so any effect is attention and dating,
not recall. The mechanism I expect is a date question answered from an undated record, or from a
record whose coding label misframes the turn. That argues for a positive effect concentrated in
category 2. But a reader that already sees every raw turn with its timestamp should mostly
recover, and my own record says I over-predict effects by two to four times
([[i-over-predict-effect-magnitudes]]). My unadjusted guess was +3 to +6, which I have cut to +0.5
to +3.0. With about 30 discordant pairs, the paired standard error is roughly 1.3 points, so I
expect the interval to include zero.

## Decision rule, fixed now

1. If B minus A is at least +1.0 point with a 95% interval above zero, and larger than the
   absolute value of A′ minus A, recommend to the user a C9 change for Textual: keep compiled
   records out of what the Context4 route serves. Two ways would do it: index only raw windows in
   the Context4 store, or filter non-raw items at render. The recommendation needs its own smoke
   before any official run, and the choice is the user's.
2. If B minus A is at least +1.0 but the interval includes zero, it is inconclusive. The next step
   is a larger temporal sample, not a change.
3. If B minus A is below +1.0, there is no quality case. Whether to stop compiling Textual Adds at
   all is then an operational question about Add cost, decided separately.

Nothing here changes the first official Full run.

## What would falsify this

- B minus A below +0.5 or above +3.0 points.
- The category 2 effect outside +1.0 to +5.0, or wrong-to-right flips not concentrated in
  category 2.
- A′ minus A outside plus or minus 1.0, which would mean reader noise is larger than I assumed and
  the design cannot resolve the predicted effect.

## How it will be measured

- **Script:** `scripts/aml_locomo_loss_diagnosis.py`, committed with this record.
  - `answer --route context` selects the 420 questions with the served `route_query`, which is
    identical between `337f2537` and this branch.
  - `answer --drop-compiled` builds arm B.
  - `judge` scores each arm.
  - `compare` gives paired accuracy, a 10,000-resample percentile interval with seed 0, and
    discordant counts, overall and by category.
- **Collect:** `full1`, the same retrieval run as the loss diagnosis, not re-run.
- **Reader and judge:** unchanged from the loss diagnosis. AML's LoCoMo-Refined Answer and accuracy
  prompts at `1b8142bf`, `openai/gpt-4o-mini` through OpenRouter, temperature 0.
- **Comparisons:** B against A (primary), A′ against A (noise floor), and B against A′ (a
  sensitivity check, with both arms answered fresh).
- **Tests:** `tests/test_aml_locomo_loss_diagnosis.py`. The raw-only filter, the route filter and
  the paired statistic are each proved red by mutation.
- **Cost:** about 2 to 3 USD for two arms of 420 answers plus their judging, under the script's
  15 USD cap per output file.

## What I already know

- `2026-09-24-c9-context4-route-worse-on-locomo`: on retrieval, the Context4 route loses about 2.6
  points of turn hit@10 against Code4 on its own 420 questions. Displacement by compiled records
  was named as the likely mechanism there, but not measured.
- `2026-09-24-c9-llm-routing-headroom`, exploratory: Context4 losses carry fewer non-raw items in
  the top 10 than average (1.81 against 2.18), so displacement did not explain retrieval losses.
  This record asks about answers, which is a different consumer
  ([[measure-what-the-consumer-reads]]).

## Confounds I can name now

- **Arm B has fewer items and fewer characters than A.** It measures "without compiled records"
  and "with less text" together. Arm B is exactly what render-time filtering would serve. An
  index with raw windows only would instead fill all 100 slots with raw windows, which this does
  not test.
- **About 11% of compiled records carry text that no raw item in the list repeats.** That text is
  lost in B, though no gold evidence is.
- **The reader and judge are stand-ins**, and the speaker split is not AML's, both as in the loss
  diagnosis.
- **LoCoMo is not AML Textual.** 420 questions and one draw of each arm is small.

## Result (2026-09-24)

**Status:** measured. Read only after the aggregation view record (`1fa5b4af`) was committed.

All three arms: 420 answers each, 420 judge labels each, 0 unparsed. Artifacts are in
`docs/results/2026-09-24-aml-c9-compiled-records-counterfactual/`.

| quantity | predicted | measured | in band |
| --- | --- | --- | --- |
| B minus A | +0.5 to +3.0, interval including zero | **-0.95 [-3.81, +2.14]** (71.7% to 70.7%) | no, below |
| B minus A, category 2 (n = 214) | +1.0 to +5.0 | -0.93 [-6.07, +4.21] | no, below |
| A′ minus A | -1.0 to +1.0 | **-2.38 [-4.52, -0.24]** | no, below |
| discordant, A′ against A | 5 to 25 | 20 (5 wrong-to-right, 15 right-to-wrong) | yes |
| discordant, B against A | 20 to 50 | 40 (18 against 22) | yes |
| B's wrong-to-right flips in category 2 | 50% to 80% | 94% (17 of 18) | no, above; moot at a net loss |

B minus A′, with both arms answered fresh minutes apart: +1.43 [-1.67, +4.52], 24 against 18.

**Gap.**
- **The compiled records do not measurably cost answers.** Removing them moved accuracy by -0.95,
  against a predicted gain. That is within the run-to-run drift below, and far from the +1.0 bar.
  Undated, duplicated and mislabelled as they are, the reader works around them. The prediction
  assumed that what looks bad in the evidence costs answers. It did not, which is the same lesson
  as the retrieval result the day before: displacement looked like the mechanism and was not.
- **The larger finding is the apparatus.** Answering the identical evidence twice gave 71.7% and
  69.3%, a paired difference whose interval excludes zero. That is drift between runs, not
  symmetric noise.
  - Likely cause, not measured: OpenRouter routing gpt-4o-mini to different upstream providers at
    different times, since provider identity was not recorded.
  - Consequence: any arm answered hours apart from its control carries a bias of about 2 points in
    either direction. The loss diagnosis's buckets are unaffected, because they are one run.
    Every paired comparison against arm A is affected.

**Decision, by the rule fixed above:** B minus A is below +1.0, so rule 3 applies. There is no
quality case for removing compiled records from the Context4 route. Whether to stop compiling
Textual Adds at all is an operational question about Add cost, decided separately.
