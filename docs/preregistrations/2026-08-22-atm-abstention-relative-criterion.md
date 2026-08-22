# Pre-registration: the re-abstention criterion in relative form, tested on a fresh generation

**Date:** 2026-08-22   **Status:** predicted, not yet measured

Closing record of the A1 line. `2026-08-22-atm-abstention-second-pass.md` registered a re-abstention
ceiling of 20% in absolute form, measured 44.7%, and the falsifier fired. Its result section argued
the criterion should have been relative to a control that did not exist when the threshold was
written. This record writes the relative criterion.

## ⛔ What this record does NOT do

**It does not rehabilitate A1.** That record's falsifier fired and stays fired, permanently. A
criterion written after seeing the number it judges cannot be applied to that number: the whole
value of a pre-registration is that the threshold could not have been chosen to clear the bar, and
here it demonstrably could have been.

So the criterion below is tested on **data that does not exist yet**: a second, independent
generation of the treatment arm over the same 157 questions. Temperature is 0.0 and the arm is
still not deterministic, which the control already demonstrated by diverging from the 2026-08-21
baseline under the identical prompt, so a re-run is a genuinely fresh sample of the same
distribution. If the criterion fails there, the line closes as a measured effect that never passed
a pre-registration, and that is an acceptable ending.

## The criterion, argued from the mechanism rather than from the measured value

The falsifier exists to catch two failures, and only the second was ever expressed:

1. **The framing does not move the decision.** H17 added a disposition sentence to all 1,013
   questions and shifted refusals by 0.34 points. An arm that does that again is dead.
2. **The framing moves the decision indiscriminately.** H21 satisfied its instruction by rejecting
   everything, accepting 1.23 items of 10 and no items at all on 24% of questions. An arm that
   answers everywhere, including where the evidence does not support an answer, is worse than
   useless: it converts a system that knows what it does not know into one that does not.

An absolute ceiling expresses neither. It cannot separate "moved the decision" from "the questions
were easy", and it says nothing at all about discrimination. The relative form expresses both, as a
ratio to the same-day control on strata defined by whether an answer exists to be given.

**C1, the arm must move the decision.** Treatment re-abstention divided by control re-abstention
must be **at most 0.70** on the 85 deterministic questions and, separately, on the 55 wrong
`open_end` abstentions. A plain re-ask is the floor this must clear.

**C2, the arm must discriminate.** The same ratio must be **at least 0.95** on the 17 gold
abstentions, where refusing is the correct answer. This is the clause that separates this arm from
H21, and it is the reason the criterion is two-sided rather than a ceiling.

**C3, the effect must replicate.** The deterministic score on the 85 must be **at least 0.20**,
against a baseline of exactly 0.0000 and a first measurement of 0.2853.

⚠️ **Disclosure, because it is the weakest part of this record.** I chose 0.70, 0.95 and 0.20
knowing the first generation's values, which were 0.576, 1.000 and 0.2853. I argue each from the
mechanism above, and none is set at the measured value, but I cannot claim they are uninformed by
it. What makes the test real is the fresh generation, not the thresholds. A reader who discounts
this record entirely and reads only A1's fired falsifier is reading it correctly.

## What will run

The treatment arm again, same prompt, same evidence, same model settings, same day, over the same
157 questions: the 85 deterministic and the 72 `open_end` abstentions. 157 calls. **No judge**:
every quantity in the criterion is a re-abstention rate or a deterministic score. The control is
the one already generated and is not re-run, so the comparison is fresh treatment against the
existing control.

## What I predict

| Quantity | n | First generation | Predicted for the replication |
| --- | ---: | ---: | --- |
| Deterministic score | 85 | 0.2853 | **0.24 to 0.33** |
| Re-abstention, deterministic | 85 | 0.4471 | 0.38 to 0.52 |
| **C1 ratio, deterministic** | 85 | 0.576 | **0.49 to 0.67** |
| Re-abstention, wrong `open_end` | 55 | 0.4364 | 0.36 to 0.52 |
| **C1 ratio, `open_end`** | 55 | 0.545 | **0.45 to 0.65** |
| **C2 ratio, gold abstentions** | 17 | 1.000 | **0.95 to 1.06** |
| Answers identical to the first generation | 157 | n/a | 0.55 to 0.80 |

The last row is the one I am least sure of and it is there because it calibrates all the others: if
the arm reproduces itself almost exactly, the replication is a weaker test than this record claims,
and the reader should know that from a number rather than from my assurance.

## What would falsify this

* **C1 fails on either stratum**, ratio above 0.70. The framing does not reliably move the decision
  and the first generation was a favourable draw.
* **C2 fails**, ratio below 0.95. The framing erodes correct refusals on replication, which is the
  H21 failure and would end the line outright regardless of any score.
* **C3 fails**, deterministic score below 0.20. The effect does not replicate.
* **Identical-answer rate above 0.90.** Then the re-run is not an independent sample, the criterion
  was tested against a copy of the data that motivated it, and this record proves nothing. This is
  the falsifier that protects the design itself and it is read first.

## How it will be measured

```
python scratchpad/a1_second_pass.py --arm treatment --ids a1_ids.json --out a1out/rep_det
python scratchpad/a1_second_pass.py --arm treatment --ids open_end_abstentions.json --out a1out/rep_oe
python scratchpad/a6_replicate.py
```

n is 85, 55 and 17 for the three strata. Re-abstention uses the official `is_abstention`, for
measurement only. The deterministic score uses the official scorer's `number` and `list_recall`
paths, which is the same code verified against the saved official output on 499 questions with zero
mismatches.

## Confounds I can name now

1. **The control is not re-run**, so a drift between this hour and two hours ago moves the ratio
   without the framing changing. Both generations are same-day against the same alias, which is
   what this project can afford, not what would be ideal.
2. **The thresholds are informed by the first generation.** Stated above rather than buried.
3. **A replication over the same questions is not a new sample of questions.** It is a new sample of
   the model's decisions on fixed questions, which is what the criterion is about, and nothing here
   generalises to questions not in these 157.

## Result

Not yet measured at the time this record was committed.

---

## Result (2026-08-22)

**Status:** measured. Predictions above unedited. 157 provider calls, zero judge calls.

**Design falsifier read first, as the record requires.** 95 of 157 replicated answers are identical
to the first generation, a rate of **0.6051**, below the 0.90 that would have invalidated the test.
The replication is a fresh sample and the criterion was not tested against a copy of the data that
motivated it.

| stratum | n | replication | control | **ratio** | first generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic | 85 | 0.4588 | 0.7765 | **0.591** | 0.4471 |
| wrong `open_end` | 55 | 0.4000 | 0.8000 | **0.500** | 0.4364 |
| gold abstentions | 17 | 0.8824 | 0.9412 | **0.938** | 0.9412 |

Deterministic score on the 85: replication **0.2863**, first generation 0.2853, baseline 0.0000.

### The criterion

| clause | requirement | measured | |
| --- | --- | ---: | --- |
| C1 deterministic | ratio at most 0.70 | 0.591 | **pass** |
| C1 `open_end` | ratio at most 0.70 | 0.500 | **pass** |
| **C2 gold abstentions** | **ratio at least 0.95** | **0.938** | **FAIL** |
| C3 deterministic score | at least 0.20 | 0.2863 | **pass** |

**The criterion fails, on the discrimination clause.** The treatment re-abstains on 15 of 17
questions where refusing is correct, against the control's 16. One question, and the criterion is
written so that one question decides it.

### Predicted against measured

| Prediction | Predicted | Measured | Verdict |
| --- | --- | ---: | --- |
| Deterministic score | 0.24 to 0.33 | 0.2863 | held |
| Re-abstention, deterministic | 0.38 to 0.52 | 0.4588 | held |
| C1 ratio, deterministic | 0.49 to 0.67 | 0.591 | held |
| Re-abstention, wrong `open_end` | 0.36 to 0.52 | 0.4000 | held |
| C1 ratio, `open_end` | 0.45 to 0.65 | 0.500 | held |
| **C2 ratio, gold abstentions** | 0.95 to 1.06 | **0.938** | **falsified, low** |
| Identical-answer rate | 0.55 to 0.80 | 0.6051 | held |

Six of seven held, and the miss is the one the record was written to decide. That is the first
well-calibrated prediction set in this study, and it did not save the arm.

### What the failure actually says, and what it does not

**It does not say the effect is not real.** The deterministic score reproduced to within 0.001,
0.2863 against 0.2853, on an independent generation with 60% answer overlap. Both C1 clauses passed
with room. The +2.50 QS measured across the three earlier records stands.

**It says the arm erodes correct refusals, slightly, and that the erosion cannot be bounded here.**
Two of 17 in the replication against one of 17 in the first generation: measured damage moves from
0.10 to 0.197 QS, and the difference between those two numbers is one question.

⚠️ **The blocker is now the benchmark, not the arm.** ATM has **23 gold-abstention questions in
1,013**, and 17 of them are in scope. On a stratum that size the finest distinction expressible is
one question, worth 0.0588 of the ratio, so the C2 threshold of 0.95 was in effect a demand for
exact parity with the control. Any threshold looser than parity would have passed an arm that
commits on 2 of 17; any threshold at parity fails an arm that commits on 1. **There is no honest
place to put that line on 17 questions**, and choosing one after seeing which side the data fell on
is the failure this whole record exists to avoid.

### Verdict for the A1 line

**Measured, replicated, and not promoted.** Two registered falsifiers have now fired on it: A1's
absolute re-abstention ceiling, and this record's discrimination clause. The effect is +2.50 QS
attributable, the damage is between 0.10 and 0.20 QS, and both statements are supported by
committed pre-registrations with their falsifiers reported above their scores.

Promotion needs one of two things, and neither is a rerun of this:

1. **A larger population of questions where refusing is correct.** ATM cannot supply it. A
   different corpus, or a constructed unanswerable set held beside the benchmark, could.
2. **A gate that does not rely on the emitted abstention alone.** The sufficiency classifier of
   idea A13 is the candidate, and this line has now measured what it would have to beat: the
   framing alone moves the decision at ratio 0.50 to 0.59 while costing between one and two of
   seventeen correct refusals.

The line closes here.
