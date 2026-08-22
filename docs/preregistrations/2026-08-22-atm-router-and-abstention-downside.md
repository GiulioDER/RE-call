# Pre-registration: the question-form router, and the downside the A1 arm could not see

**Date:** 2026-08-22   **Status:** predicted, not yet measured

Follows `2026-08-22-atm-abstention-second-pass.md`, whose result section closed with two named gaps.
This record measures both, and nothing else.

* **Gap 1.** A1 selected its 85 questions by an answer-side contract but restricted them to
  `number` and `list_recall` using the **gold `qtype`**, so every QS figure in that record assumes
  a question-form router that did not exist.
* **Gap 2.** All 23 gold-abstention questions are `open_end`, so A1 structurally could not measure
  the 1.68 QS its own intervention puts at risk. An arm that cannot see its downside is not
  evidence about its net effect.

## The questions

1. On the held-out test partition, how accurately does a text-only router identify a request to
   retrieve memory items, measured as precision and recall against the gold `list_recall` label?
2. On the 17 questions where the gold answer is itself an abstention and the model abstained, how
   often does the second-pass framing keep abstaining?
3. On the 55 `open_end` questions where the model abstained and the gold answer is not an
   abstention, how much gold-answer content does the second pass recover?

## What I already know

* A1, measured today: the treatment scores 0.2853 on the 85 deterministic contentless answers
  against a same-day control of 0.1176, so the framing is worth +0.1676, CI [+0.0853, +0.2559]. It
  re-abstains 44.7% of the time, which fired a registered falsifier. The gain concentrates 4.5x on
  questions whose gold tokens were on screen.
* The diagnosis of the 2026-08-21 run: 156 abstentions, 139 wrong, 13.72 QS of raw loss; the 23
  gold abstentions currently score 0.7391 and the model abstains on 17 of them.
* On this run `open_end` score tracks the answer's gold-token coverage almost exactly: 0.2800 at
  coverage 0, 0.5185 at 0.50, 0.9825 at 1.00, n=442. That mapping is what makes question 3
  answerable without paying the judge, and it is a **proxy**, stated as one.
* Memory `i-over-predict-effect-magnitudes` and its one fresh counterexample from A1, where two
  magnitudes came in above the band.

## The router, frozen before measurement

`scratchpad/a2_router.py`, function `is_retrieval_request`. It reads the question and nothing else:
the **last** clause is taken as the request, a retrieval verb phrase or an imperative head makes it
a retrieval request, and a leading wh-word vetoes it. It was designed by reading the **development**
partition only, `sha256(id) mod 10 < 7`, 712 questions, and its dev numbers are **precision 0.8649,
recall 0.9897** on 97 `list_recall` questions with 15 false positives.

⚠️ **The router cannot reach 1.0 and the dev errors say why.** Thirteen of the fifteen dev false
positives are `open_end` questions whose surface form is a genuine retrieval request, of the shape
"I saw an object somewhere, help me recall the photo", where the gold answer is an attribute of the
object rather than the item. The label is a property of the gold answer, not of the question, so a
question-only router is measuring something the question does not fully determine. Any promoted
router therefore carries an irreducible error rate, and this record predicts it rather than hoping
it away.

The router deliberately does **not** attempt `number` against `open_end`. The dev sample carries
the same surface form on both sides of that line, so the distinction is not in the question.

## The two generation probes

All 72 `open_end` questions whose 2026-08-21 answer is an abstention, run in both arms, same
prompts, same packer, same model settings, same day as A1. 144 calls. The 17 gold-abstention
questions are the subset that answers question 2; the other 55 answer question 3.

**The judge is not called.** `open_end` is scored by the official judge and that spend is
deliberately deferred: this record measures re-abstention, which needs no judge, and gold-token
coverage, which is a proxy whose relationship to the judge is quoted above and is not re-derived
here.

## What I predict

| # | Quantity | n | Predicted |
| --- | --- | ---: | --- |
| 1 | Router precision, test partition | 301 | **0.78 to 0.90** |
| 1 | Router recall, test partition | 42 | **0.92 to 1.00** |
| 1 | Router fires on a gold-abstention question | 23 | **0 to 2** |
| 2 | Treatment re-abstention, gold-abstention questions | 17 | **0.50 to 0.80** |
| 2 | Control re-abstention, same | 17 | **above 0.80** |
| 2 | **QS destroyed by the treatment on those 17** | 17 | **0.34 to 0.84** |
| 3 | Treatment re-abstention, wrong `open_end` abstentions | 55 | 0.35 to 0.60 |
| 3 | Treatment mean gold-token coverage, same | 55 | **0.25 to 0.50** |
| 3 | Control mean gold-token coverage, same | 55 | 0.05 to 0.20 |

The damage figure in row 6 is `(1 - re-abstention) x 17 / 1013 x 100` and assumes every question
where the treatment commits loses its point outright, which is the pessimistic reading and is the
right one for a downside estimate.

**Net, if every prediction lands mid-band:** the framing is worth +1.41 QS on the deterministic
half, costs about 0.59 QS on the gold abstentions, and returns an unknown amount on the 55, which
is exactly why the judge spend is the next decision and not this one.

⚠️ **Calibration note.** A1 is the first record in this study where I under-predicted, on both arms
at once. I have not corrected for that here: the bands above are set by the same reasoning as
before, so that a second under-prediction is informative rather than absorbed.

## What would falsify this

* **Router precision below 0.75 on test**, or recall below 0.90. Then the dev numbers were fitted
  and the router is not usable for A2, whose whole safety argument rests on precision.
* **The router fires on more than 2 gold-abstention questions.** That is the one place a false
  positive is expensive rather than free, because emitting an identifier there destroys a point the
  system currently earns.
* **Treatment re-abstention on the 17 falls below 0.50.** Then the framing does reach the
  questions where refusing is right, the downside is worse than 0.84 QS, and the intervention must
  be gated by something other than the emitted abstention.
* **Treatment coverage on the 55 does not exceed control coverage by at least 0.10.** Then the
  framing produces commitment without content on `open_end`, and extending A1 past the
  deterministic types is not supported.
* **Control coverage above 0.20.** Then the re-ask effect explains `open_end` too and the framing
  is again not the active ingredient.

## How it will be measured

```
python scratchpad/a2_router_eval.py                 # question 1, zero calls
python scratchpad/a1_second_pass.py --arm control   --ids open_end_abstentions.json
python scratchpad/a1_second_pass.py --arm treatment --ids open_end_abstentions.json
python scratchpad/a3_score_open_end.py              # questions 2 and 3, zero judge calls
```

n is 301 for router precision and 42 for router recall; 17 for every rate in question 2 and 55 for
every rate in question 3. Re-abstention uses the official `is_abstention`, for measurement only.
Coverage is the share of the gold answer's non-stopword tokens appearing in the answer, the same
definition used throughout the diagnosis.

**Apparatus verification.** The test partition must contain 301 questions and 42 `list_recall`,
which is the complement of the dev counts already recorded; the 72 selected questions must all be
`open_end`, all currently abstentions, and 17 of them must have a gold answer that
`is_abstention` accepts. If any of those disagrees, nothing else is read.

## Confounds I can name now

1. **The router is evaluated against a label it cannot fully determine.** Its error rate is
   therefore a floor on ambiguity, not only on the router. Reported as such.
2. **Coverage is a proxy for the judge and the judge is the outcome.** A coverage gain that does
   not convert is precisely the failure mode arm E hit in the previous study, where `open_end` rose
   4.35 points while coverage did not move. This record predicts the mechanism and defers the
   outcome; it must not be read as a QS result.
3. **The 17 are few.** One question is 0.0987 QS, so the damage estimate has a granularity of a
   tenth of a point and no interval worth quoting.
4. **Same-day drift is controlled for by the control arm, not eliminated.** As in A1.

## Result

Not yet measured at the time this record was committed.
