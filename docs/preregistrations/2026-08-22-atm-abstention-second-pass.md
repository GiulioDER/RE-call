# Pre-registration: a second pass on ATM answers that carry no committed content

**Date:** 2026-08-22   **Status:** predicted, not yet measured

Arm A1 of the diagnosis recorded on 2026-08-22 against the submission package
`submission/atm-benchmark-20260821/`. That diagnosis is descriptive and spent nothing; this record
is the first arm of it that costs money.

## The question

On the 85 ATM questions of type `number` and `list_recall` whose final answer carries no committed
content, does a second generation pass with a decision-forcing framing, over byte-identical
evidence, produce a mean deterministic score above what the same prompt produces on a re-ask?

Answerable by two numbers: the mean official deterministic score of the treatment arm over those 85
questions, and the same for a control arm that re-asks with the unchanged prompt on the same day
against the same model alias.

## What I already know

Searched before predicting.

* `docs/preregistrations/2026-08-20-atm-evidence-allocation-and-selection.md`: **H17, one sentence
  of answer-first disposition added to all 1,013 questions, moved the refusal rate by 0.34 points
  and QS by -0.12.** A prompt sentence applied in advance does not reach this behaviour. This arm
  is deliberately not that: it observes an emitted state and spends a second decision on it.
* The same record: **H21 lost 4.22 QS** by asking the model to mark each of ten evidence items,
  which taught it to reject rather than to discriminate. Nothing in this arm asks the model to
  judge items one at a time.
* Memory `i-over-predict-effect-magnitudes`: eleven of twelve registered predictions in this study
  were falsified and every falsified magnitude was too high by two to four times. The bands below
  are already pulled down for that, and predict the mechanism metric beside the outcome.
* Memory `atm-over-abstention-is-the-largest-mechanism`, written today: on this run the model
  abstains on 156 of 1,013 questions and 139 of those are wrong, worth 13.72 QS. All 23
  gold-abstention questions are `open_end`, so on `number` and `list_recall` an abstention is never
  correct and always scores exactly 0.0000.
* Memory `atm-judge-runs-through-openrouter`: `--metrics atm` scores `number` and `list_recall`
  deterministically and calls the judge only for `open_end`. This arm touches neither `open_end`
  nor the judge, so it costs **zero judge calls**.

## The selection rule, which is part of the hypothesis

A question enters the arm when its answer in `answers.jsonl` contains **no evidence ID and no
digit**. That is a property of the answer against its own output contract, not a phrase list: a
`number` answer must carry a value and a `list_recall` answer must carry an identifier, and an
answer carrying neither has committed to nothing.

⛔ **The scorer's abstention vocabulary is deliberately not used to gate anything.** The runner's
`_REFUSAL_MARKERS` and the evaluator's `ABSTENTION_PHRASES` exist for measurement, and feeding a
scorer's vocabulary back into the serving path is tuning to the metric. Measured today, the
contract rule selects **85** questions and the official `is_abstention` selects **84**, the 85
being a strict superset: the extra one is a `list_recall` question answered with a prose menu and
no identifier. **All 85 score exactly 0.0000.**

Composition: 53 `number`, 32 `list_recall`.

## The two arms, stated exactly

Both arms re-use the ten saved hits from `retrieval.jsonl` and rebuild the evidence block with the
**greedy packer of commit `6c0ec26b`** at 8,192 characters, which is the packer that produced the
answer file. Model `deepseek/deepseek-v4-pro` through OpenRouter, `temperature` 0.0, reasoning
effort `medium`, `max_tokens` 2,048 with a single retry at 8,192 on a length stop.

**Control.** The official oracle baseline system prompt, verbatim and unchanged:

> You are a QA assistant. Use ONLY the provided evidence to answer. If the evidence is
> insufficient, answer 'Unknown'. Respond with only the answer. If the question asks to recall or
> list items (photos/emails/videos), respond with the corresponding evidence IDs only,
> comma-separated, with no extra text.

**Treatment.** The same text with one paragraph appended:

> This is a second attempt at a question that was declined on the first attempt. The evidence block
> is the entire memory available and no further retrieval is possible, so declining again cannot
> produce a better answer than deciding. Identify the item in the block that comes closest to what
> the question asks about, and answer the question from that item. Follow the answer rules above
> exactly.

The `'Unknown'` clause of the official prompt is **kept**, so the model retains the option. The arm
tests whether framing the call as a decision changes the choice, not whether removing the option
does.

## What I predict

| Quantity | n | Now | Predicted |
| --- | ---: | ---: | --- |
| Control, mean deterministic score | 85 | 0.0000 | **0.00 to 0.10** |
| Control, re-abstention rate | 85 | n/a | **above 70%** |
| **Treatment, mean deterministic score** | 85 | 0.0000 | **0.12 to 0.28** |
| **Treatment minus control** | 85 | n/a | **+0.10 to +0.25** |
| Treatment, re-abstention rate | 85 | n/a | **below 20%** |
| Treatment, `number` | 53 | 0.0000 | 0.10 to 0.25 |
| Treatment, `list_recall` | 32 | 0.0000 | 0.15 to 0.35 |
| Treatment, answers over 200 characters | 85 | n/a | below 5% |

**The ceiling, stated so the prediction is not mistaken for it.** Of the 84 deterministic wrong
abstentions, 45 have at least 80% of the gold answer's content tokens inside the packed evidence
they received. So at most about 0.53 of these questions are plausibly answerable at all from the
text, and reaching a quarter to a half of that ceiling is 0.13 to 0.27. That is where the band
comes from, not from optimism about the framing.

**What it is worth in QS, with the arithmetic shown.** A gain of `s` mean score on 85 of 1,013
questions is `85 s / 1013 x 100 = 8.39 s` QS points. The band 0.12 to 0.28 is therefore
**+1.01 to +2.35 QS**, applied to the full benchmark only under the router caveat below.

🔁 **Correction to a number I published this morning.** The research report accompanying this arm
stated a band of "+1.0 to +1.8 QS" from a predicted mean score of 0.25 to 0.45. That arithmetic was
wrong by roughly a factor of two in the opposite direction: 0.25 to 0.45 is +2.10 to +3.78 QS. The
band in the table above is the one that governs this record, and it is lower than both, because I
re-derived it from the 45-question answerability ceiling rather than from the type mean.

## What would falsify this

Each of these can fire, and one of them is the reason the control arm exists at all.

* **Treatment mean score below 0.10.** The second pass does not reach the behaviour, exactly as
  H17's sentence did not, and the mechanism is dead at the prompt level for the second time.
* **Treatment re-abstention rate above 20%.** The framing did not force a decision. This is the
  H21 failure mode wearing different clothes and it must be read before the score.
* **Control mean score within 0.05 of treatment.** The gain is a re-ask effect or model drift, not
  the framing, and the intervention is a wrapper around sampling noise.
* **Control re-abstention rate below 40%.** Then the original answers are not reproducible against
  today's model alias, the baseline file is not a stable reference, and **no comparison in this
  record is interpretable**, including the ones that look good.
* **More than 5% of treatment answers over 200 characters.** The framing broke the "respond with
  only the answer" contract, which would cost points wherever it were generalised even if it gains
  here.

A prediction that cannot fail is not a prediction, so it is worth naming what is **not** a
falsifier: no arm can score below baseline on these 85 questions, because the baseline is exactly
0.0000 on every one of them. The sign of the effect is not in question. Its size, its mechanism,
and its attribution are.

## How it will be measured

```
python scratchpad/a1_second_pass.py --arm control    --out results/a1/control
python scratchpad/a1_second_pass.py --arm treatment  --out results/a1/treatment
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth data/atm-bench/atm-bench.json \
  --predictions <arm>/answers.jsonl --metrics atm
```

n is 85 for every rate in this record, and the two per-type rates are over 53 and 32. The metric is
the official ATM deterministic score: exact multiset equality over extracted values for `number`,
Jaccard over extracted identifier sets for `list_recall`. **The judge is not called and is not
modified.** The re-abstention rate is over the 85 answers each arm produces, using the official
`is_abstention` for measurement only.

**Apparatus verification, before the outcome is read.** Three checks, each with an answer known in
advance: the untouched baseline file re-scored through the official evaluator must reproduce
0.6843 overall and 0.7278 / 0.5983 per type; the 85 selected questions must re-score to exactly
0.0000 each; and the rebuilt evidence block for a spot-checked question must contain the same ten
identifiers, in the same order, as `retrieval.jsonl` records. If any of the three fails, nothing
else in the run is read.

## Confounds I can name now

1. **The gold `qtype` is used to choose which questions to spend on.** The trigger itself reads
   only the answer, but the restriction to `number` and `list_recall` comes from the gold file, and
   it is there because those two types are free to score. This makes the arm a **diagnostic**: a
   promotable version needs a question-form router, and the QS figure quoted for the full benchmark
   assumes one exists and is accurate. The 85 answers themselves are not advantaged by this,
   because the second pass is identical regardless of type.
2. **The model alias moves.** `deepseek/deepseek-v4-pro` is not an immutable checkpoint and the
   baseline was generated on 2026-08-21. The control arm exists precisely to absorb this: it is
   generated today, against the same alias, with the unchanged prompt.
3. **The second pass sees no new information.** This is a property, not a defect: any gain is
   attributable to the framing rather than to evidence, which is what makes the arm interpretable.
   It also caps it, since 40 of the 85 have no gold tokens on screen to find.
4. **`open_end` is untouched and so is the 1.68 QS of correct abstentions.** All 23 gold
   abstentions are `open_end`. This arm therefore cannot measure the downside of the intervention,
   and a generalisation to all 156 abstentions must measure that separately before promotion.
5. **Two questions of the 85 may be affected by the `extract_times` defect** in the official
   scorer, which yields two tokens from `8:00 PM` and one from `8PM`. Incidence was 0 to 2 per set
   in the earlier study. It is not corrected for, and it is not tuned around.

## Result

Not yet measured at the time this record was committed.
