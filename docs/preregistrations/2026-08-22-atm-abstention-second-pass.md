# Pre-registration: a second pass on ATM answers that carry no committed content

**Date:** 2026-08-22   **Status:** measured

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

---

## Result (2026-08-22)

**Status:** measured. Every prediction above is unedited. 170 provider calls, zero judge calls,
zero dollars of judge credit. Both arms generated on the same day against the same model alias,
over byte-identical evidence rebuilt by the packer of `6c0ec26b`.

Apparatus verified before the outcome was read, all three checks passing: my scoring path
reproduces the saved official per-question accuracy on all **499** deterministic questions with
**zero mismatches**, per-type 0.7278 and 0.5983 as the record required; all **85** selected
questions re-score to exactly 0.0000; **zero** of 85 rebuilt evidence blocks disagree with the
identifiers in `retrieval.jsonl`.

| arm | n | score | `number` | `list_recall` | re-abstention | over 200 chars | exactly right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 85 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 0 |
| control | 85 | 0.1176 | 0.1132 | 0.1250 | 0.776 | 0.012 | 10 |
| **treatment** | 85 | **0.2853** | 0.2453 | 0.3516 | **0.447** | 0.024 | 24 |

**Treatment minus control: +0.1676, 95% paired bootstrap [+0.0853, +0.2559], which excludes zero.**
16 questions improved, 1 worsened, 68 tied.

### A registered falsifier fired, so the arm is not promotable as specified

**Treatment re-abstention is 44.7% against a registered ceiling of 20%.** By the criteria written
before the run, that invalidates the arm regardless of what the score did, and it is recorded here
before the number that looks good.

What the falsifier does **not** say, and what has to be stated beside it: re-abstention fell from
100% at baseline and 77.6% in the control to 44.7%, so the mechanism moved by 33 points against its
own control. The threshold I registered was an absolute level, and the informative version would
have been relative to the control arm, which did not exist when the threshold was written. **The
prediction stands as written and is not edited.** The lesson is about how the falsifier was framed,
not about whether it fired.

### Predicted against measured

| Prediction | Predicted | Measured | Verdict |
| --- | --- | ---: | --- |
| Control mean score | 0.00 to 0.10 | **0.1176** | **falsified, high** |
| Control re-abstention | above 70% | 77.6% | held |
| Treatment mean score | 0.12 to 0.28 | **0.2853** | **falsified, high** |
| Treatment minus control | +0.10 to +0.25 | **+0.1676** | **held** |
| Treatment re-abstention | below 20% | **44.7%** | **falsified, and it is the registered falsifier** |
| Treatment `number` | 0.10 to 0.25 | 0.2453 | held |
| Treatment `list_recall` | 0.15 to 0.35 | 0.3516 | held, one thousandth outside |
| Treatment answers over 200 chars | below 5% | 2.4% | held |

**Two magnitudes were under-predicted, which has not happened before in this study.** The standing
calibration note says every falsified magnitude here has been too high by two to four times, across
eleven predictions. This is the first record in which the measured effect exceeded the registered
band, and it did so on both arms at once. The band was derived from the 45-question answerability
ceiling and a quarter-to-half discount; the discount was too aggressive for an intervention that
changes a decision rather than a description.

### The control arm earned its 85 calls

It carries **0.1176 of the 0.2853 by itself, 41% of the raw gain**, without a single word of the
treatment framing. Had the arm been run alone, as the research report proposed this morning, the
whole +2.39 QS would have been attributed to the framing. The framing's own share is:

| quantity | QS on the full 1,013 split |
| --- | ---: |
| treatment arm, total | +2.39 |
| control alone, re-ask with the unchanged prompt | +0.99 |
| **framing only** | **+1.41, CI [+0.72, +2.15]** |

The control's own gain is not attributable here. Re-ask nondeterminism at temperature 0 with
reasoning enabled, and drift in the moving `deepseek/deepseek-v4-pro` alias between 2026-08-21 and
today, are both consistent with it and this design cannot separate them.

### The mechanism check, which the falsifier does not cover

Splitting the 85 by whether at least 80% of the gold answer's content tokens were inside the
evidence block the model received:

| | n | control | treatment | delta |
| --- | ---: | ---: | ---: | ---: |
| gold on screen | 46 | 0.1957 | 0.4565 | **+0.2609** |
| gold not on screen | 39 | 0.0256 | 0.0833 | +0.0577 |

**The gain is 4.5 times larger where the answer was actually available.** That is the difference
between a system that decides and a system that guesses, and it was not a registered prediction.

The second piece of the same evidence: the treatment committed to an answer on **47** of 85
questions against the control's **19**, and the share of committed answers that are exactly right is
**51.1% against 52.6%**. Volume rose by 2.5 times at unchanged precision. On the 32 `list_recall`
questions, answers emitting no usable identifier fell from 26 to 15, exact set matches rose from 4
to 11, and invented items fell from 36 to 27.

### Cost

| arm | calls | prompt tokens | completion tokens | truncations |
| --- | ---: | ---: | ---: | ---: |
| control | 85 | 199,055 | 45,689 | 0 |
| treatment | 85 | 204,785 | 68,099 | 0 |

### What this does and does not establish

1. **It does not establish a promotable intervention.** A registered falsifier fired. The next
   record must re-register the re-abstention criterion relative to a control, and must measure the
   downside this arm structurally cannot see: all 23 gold-abstention questions are `open_end`, so
   the 1.68 QS at risk was never exposed here.
2. **The `qtype` confound is unresolved and bounds the QS figure.** The 85 were chosen by an
   answer-side contract, but restricted to two types using the gold file. Every QS number above
   assumes a question-form router that does not yet exist.
3. **What it does establish** is that the largest measured mechanism on this benchmark responds to
   an intervention at all, after two prompt-level attempts, H17 and H21, that produced nothing and
   harm. The effect is present, its interval excludes zero, and it concentrates where the evidence
   is, which is the shape a real effect has.
