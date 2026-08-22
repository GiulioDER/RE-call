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

---

## Result (2026-08-22)

**Status:** measured. Predictions above unedited. 144 provider calls, **zero judge calls**. The test
partition was read once, after this record was committed as `44a6979a`.

Apparatus verified first, all four counts as the record required: the test partition holds 301
questions and 42 `list_recall`; the selection holds 72 `open_end` abstentions of which 17 have a
gold answer that `is_abstention` accepts.

### Question 1, the router

| | dev, the fitting set | **test, read once** | predicted | verdict |
| --- | ---: | ---: | --- | --- |
| precision | 0.8649 | **0.9048** | 0.78 to 0.90 | **falsified, high by 0.0048** |
| recall | 0.9897 | **0.9048** | 0.92 to 1.00 | **falsified, low** |
| fires on a gold-abstention question | n/a | **0 of 23** | 0 to 2 | held |

Test counts: 38 true positives, 4 false positives, 4 false negatives on 42.

**The recall drop from 0.9897 to 0.9048 is the fitting gap, and it landed exactly where I put my
hands.** Two dev false negatives were fixed by adding an abbreviation rule and an imperative-head
rule; the test partition produced four fresh ones. Neither registered falsifier fired, precision
above 0.75 and recall above 0.90, the second by 0.0048.

The safety row is the one that matters for the A2 fallback: the router does not fire on a single
question where abstaining earns a point.

### Question 2, the downside A1 could not see

| arm | re-abstention on the 17 | commits on | pessimistic damage |
| --- | ---: | ---: | ---: |
| baseline | 1.0000 | 0 of 17 | 0.00 QS |
| control | 0.9412 | 1 of 17 | 0.10 QS |
| **treatment** | **0.9412** | **1 of 17** | **0.10 QS** |

| Prediction | Predicted | Measured | Verdict |
| --- | --- | ---: | --- |
| Treatment re-abstention, the 17 | 0.50 to 0.80 | **0.9412** | **falsified, high** |
| Control re-abstention, the 17 | above 0.80 | 0.9412 | held |
| QS destroyed on the 17 | 0.34 to 0.84 | **0.10** | **falsified, low** |

**The downside is a quarter of the bottom of my band, and the treatment does no more damage than
the control.** The single question where the treatment commits is one the control commits on too,
so it is a re-ask effect and not the framing. This is the strongest evidence in either record that
the framing discriminates rather than pushes: on questions where the evidence genuinely does not
support an answer it keeps refusing at 94%, while on the 85 deterministic questions where the
evidence often does, it dropped its refusal rate from 100% to 44.7%.

### Question 3, the upside on the 55 wrong `open_end` abstentions

| arm | re-abstention | gold-token coverage | commits on | coverage among committed | median chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.0000 | 0.0000 | 0 | n/a | n/a |
| control | 0.8000 | 0.1447 | 11 of 55 | 0.6137 | 81 |
| **treatment** | **0.4364** | **0.2864** | **31 of 55** | 0.4812 | 23 |

| Prediction | Predicted | Measured | Verdict |
| --- | --- | ---: | --- |
| Treatment re-abstention | 0.35 to 0.60 | 0.4364 | held |
| Treatment coverage | 0.25 to 0.50 | 0.2864 | held |
| Control coverage | 0.05 to 0.20 | 0.1447 | held |

**No registered falsifier fired in this record.** Coverage rises 0.1417 over control, above the
0.10 the record demanded.

⚠️ **A finding this record did not predict, and it is a caution rather than a win.** The
treatment's committed answers cover LESS of the gold than the control's, 0.4812 against 0.6137, and
are a third as long, 23 characters against 81. The extra twenty commitments are terse and
thinner. On this run `open_end` answers under 20 characters score 0.5977 while those between 50 and
100 score 0.8228, so the framing is buying commitment in exactly the shape the `open_end` judge
penalises. **A1 and a completeness contract are not independent on this type, and combining them
untested would be the mistake this study has already made twice.**

### The net, with the part that is still unbought marked as such

| component | n | measured | judge needed |
| --- | ---: | ---: | --- |
| framing, deterministic types | 85 | **+1.41 QS**, CI [+0.72, +2.15] | no |
| damage on the gold abstentions | 17 | **-0.10 QS** | no |
| upside on the wrong `open_end` abstentions | 55 | coverage +0.1417 over control | **yes, deferred** |
| **net measured without a judge** | | **+1.31 QS** | |

The `open_end` component is a mechanism, not a result. Arm E of the previous study raised
`open_end` by 4.35 points while the coverage it claimed as its mechanism did not move at all, and
this is the mirror image: the mechanism moved and the outcome is unbought. Both coverage figures
also fall in the same bucket of the measured coverage-to-score mapping, 0.25 and 0.29 against
bucket boundaries at 0.25 and 0.50, so the mapping does not resolve the gain either.

### What the two records together now support

1. **The intervention is real, small, and discriminating.** +1.31 QS measured without a judge, from
   a mechanism that keeps refusing where refusing is right.
2. **It is still not promotable**, because A1's registered re-abstention falsifier fired and has not
   been re-registered in the relative form the result section argued for.
3. **The router is usable for the A2 fallback** at precision 0.9048 with zero hits on the
   gold-abstention set, and is NOT usable as a general `qtype` classifier, which it never claimed.
4. **The next spend is a judge run on the 55**, and it is the only way to convert the largest
   remaining component from a mechanism into a number.
