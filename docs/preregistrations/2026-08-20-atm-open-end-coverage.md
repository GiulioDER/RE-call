# Pre-registration: `open_end` answers are scored on coverage, not brevity

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Follows `2026-08-20-atm-evidence-allocation-and-selection.md`, where four arms were measured and
three produced nothing while the fourth produced harm. `open_end` fell in **every** arm, and that
regularity is what this record is about.

## The question

Does an answer contract that asks for complete coverage of what the question requests, applied only
to questions that are not a single value and not a list, raise `open_end` without costing `number`
or `list_recall`?

## What I already measured

On the 300 question subset, arm B, `open_end` n=118:

| gold coverage of the answer | n | mean score |
| --- | ---: | ---: |
| complete | 51 | **98.04** |
| partial | 67 | **43.28** |

Coverage is not correlated with the verdict, it very nearly IS the verdict. Of the 67 partial
answers, **17 are missing exactly one content token** of the gold.

Across arms, median `open_end` answer length and mean gold coverage move together with the score:

| arm | median chars | gold coverage | `open_end` |
| --- | ---: | ---: | ---: |
| old `0a0d6429` prompt | 53 | 0.701 | 69.83 |
| A official prompt | 17 | 0.629 | 68.97 |
| B official prompt + allocated | 17 | 0.640 | 66.38 |
| D + selection | 18 | 0.599 | 65.52 |

On the 10 questions the official prompt lost against the old one, gold coverage fell by 0.523 on
average and the answers went from 94 to 14 median characters. The shape of the loss, described
rather than quoted (see the redaction note at the end of this record): on a deadline question the
gold carried a date AND a time and the answer gave only the date; on an event question the gold
carried the event name, its venue, its city and its date, and the answer gave only the event name.

**"Respond with only the answer" is correct for `number`, where multiset equality punishes every
extra value, and wrong for `open_end`, where the official judge rubric marks accuracy true when the
ground truth is COVERED and explicitly permits additional information.** One prompt has been
applying the first rule to all three types.

## Mechanism

A third contract, selected from the question text alone with no gold `qtype`: a question that asks
for a description, an explanation, a reason, or several attributes of one thing is answered
completely, naming every element the question asks about, while single-value and list questions keep
the terse contracts that are already measured to work for them.

**Code point.** `benchmarks/atm_full_run.py`, a new `answer_policy` value `coverage` composed over
`BASELINE_SYSTEM` exactly as the other arms are.

**Cost.** Zero additional calls. `open_end` answers get longer; the other two types must not.

## What I predict

Arm E is `allocated` packer plus `coverage`, measured against arm B on the same 300 questions.

| Quantity | B now | Predicted for E |
| --- | ---: | --- |
| `open_end` gold coverage | 0.640 | **0.72 to 0.82** |
| `open_end` score | 66.38 | **+3 to +9 points** |
| `number` | 74.17 | **within 2 points, either direction** |
| `list_recall` | 64.41 | **within 2 points, either direction** |
| QS | 68.88 | **+1.5 to +4.5** |
| Refusal rate | 13.67% | **no higher than 15%** |

The ceiling arithmetic, stated so the prediction is not mistaken for it: if every partial answer
reached complete coverage and scored like the complete ones, `open_end` would rise 31.09 points and
QS 15.76. **That is not the prediction.** It assumes a rewrite succeeds every time, and no prompt
does that.

⚠️ **Calibration note, recorded because it is the most reliable thing known about my predictions
here.** Ten of my eleven registered predictions in this study were falsified, and every falsified
magnitude was too HIGH, by roughly two to four times. The range above is already pulled down for
that bias, and if the measured result lands below it, that is the eleventh instance of the same
error rather than a surprise.

## What would falsify this

* `open_end` does not rise by at least 3 points. Then coverage was a symptom of answers being right
  rather than a cause of them being scored right, and the whole diagnosis is post hoc.
* `number` or `list_recall` falls by more than 2 points. Then the contract has leaked across types
  and the router is the problem, not the instruction.
* Coverage rises but the score does not. That would be the cleanest refutation available: it would
  mean the judge is not doing what its own rubric says, and no amount of prompt work reaches it.
* The refusal rate rises above 15%. Longer answers must not be bought with more refusals.

## Confounds

1. **This optimises toward a published rubric.** The judge prompt ships in the benchmark repository
   and says accuracy is true when the ground truth is covered. Writing complete answers is a
   generic quality instruction and the instruction here names only the QUESTION, never the gold,
   never the rubric's wording, and never a phrase list. It still has to be disclosed, because the
   line between "answer well" and "answer the way this judge scores" is thinner here than anywhere
   else in this study.
2. **Coverage is measured with my own token overlap, not the judge's.** It is a proxy. The judge
   remains the outcome, and coverage is only the mechanism being claimed.
3. **`number` and `list_recall` are the control.** They are scored without a judge, so if they move
   at all under a prompt change aimed at `open_end`, the router leaked and the arm is invalid
   regardless of what QS does.

## Result

Not yet measured at the time this record was committed.

---

## Result (2026-08-20)

**Status:** measured. Prediction above unedited. Arm E, `allocated` packer plus `coverage`, judged
on the same OpenRouter route as every other set, scored on the 295 questions common to all six.

| set | QS | number | list_recall | open_end | `open_end` coverage | `open_end` chars | refusal % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old prompt | 67.83 | 70.83 | 53.64 | 69.57 | 0.700 | 53 | 11.86 |
| A official prompt | 69.65 | 72.50 | 62.56 | 69.57 | 0.634 | 17 | 15.25 |
| B + allocated | 68.73 | 74.17 | 64.41 | 66.09 | 0.645 | 17 | 13.90 |
| **E + coverage** | **69.81** | 70.83 | 64.83 | **70.43** | **0.632** | 18 | 16.27 |

| paired | delta QS | 95% CI |
| --- | ---: | ---: |
| E vs B | +1.08 | [-2.69, +4.93] |
| E vs A, the best previous arm | +0.16 | [-3.91, +4.30] |
| E vs old prompt | +1.98 | [-1.84, +5.92] |

### Two registered falsifiers fired, so the arm is not promotable

* **`number` fell 3.33 points**, against a registered band of 2 in either direction.
* **The refusal rate reached 16.27%**, against a registered ceiling of 15%.

By the criteria written before the run, arm E is invalid regardless of what QS did. Recording that
first, before the number that looks good, is the entire reason the criteria were written first.

### The prediction was right about the outcome and wrong about the mechanism

| Prediction | Predicted | Measured | Verdict |
| --- | --- | --- | --- |
| `open_end` gold coverage | 0.72 to 0.82 | **0.632**, from 0.645 | **falsified**, it did not move |
| `open_end` score | +3 to +9 | **+4.35** | held |
| `number` | within 2 | **-3.33** | falsified |
| `list_recall` | within 2 | +0.42 | held |
| QS | +1.5 to +4.5 | +1.08 | falsified, below, as the calibration note predicted |
| refusal rate | at most 15% | 16.27% | falsified |

**`open_end` rose by the predicted amount while the quantity it was supposed to rise THROUGH did
not move at all.** The registered falsifier anticipated the opposite failure, coverage up and score
flat, and named it the cleanest refutation available. This is the mirror image, and it is worse:
the outcome moved, so a reader who checked only the outcome would call this a success and ship a
mechanism that is not doing anything.

Scale check on that gain: 10 questions improved and 5 worsened, a net of 5 on 115. The judge's own
measured disagreement rate with itself across routes was 3 of 59, about 5%, which predicts roughly
6 flips on 115 from noise alone. The 15 questions that moved exceed noise; the NET of 5 does not.

### Why `number` fell, and it is not what the falsifier said

The falsifier blamed the contract leaking into the terse types. Measured, it did not leak:

| | B | E |
| --- | ---: | ---: |
| median `number` answer chars | 10 | 10 |
| numeric values emitted per answer | 1.67 | 1.72 |

Answers did not get longer. Of the 5 `number` questions E lost, two are new refusals ("Unknown"
where B answered a correct date), one is a wrong value, and one is the scorer defect below. So the
registered cause is wrong for the second time in this study, and the real cause is the same
refusal pressure that has followed every arm.

### A defect in the official scorer, found by a question E answered correctly and lost

`extract_times` applies its 24-hour regex and its am/pm regex to the same substring, so
`"8:00 PM"` yields **two** tokens, `08:00` and `20:00`, while the gold `"8PM"` yields one. Multiset
equality then fails a correct answer. Verified directly:

Demonstrated on a CONSTRUCTED pair rather than the benchmark row that exposed it, so this record
carries no benchmark content and the evidence still stands on its own. Against a gold of
`March 3rd, 2024 at 9PM.`:

| prediction | times extracted | verdict |
| --- | --- | --- |
| `March 3, 2024 at 9 PM` | `['21:00']` | pass |
| **`March 3, 2024 at 9:00 PM`** | **`['09:00', '21:00']`** | **fail** |

Incidence across the six sets: 0 to 2 questions each, worth 0.12 to 0.24 QS. Small, real, and not
ours to fix in a serving prompt: writing `8 PM` instead of `8:00 PM` to satisfy it would be tuning
to a scorer bug rather than answering better. It is recorded so that a future gain of a quarter of
a point on `number` is not mistaken for a reasoning improvement.

### Verdict

No arm in this study is established as better than **A**, the official prompt with the ORIGINAL
greedy packer. E is nominally the highest at 69.81 but its advantage over A is +0.16 with a CI of
[-3.91, +4.30], and it fails two of its own registered criteria. The one durable result remains the
deterministic half, where the official prompt plus the allocated packer beats the old prompt by
+5.41 with CI [+0.99, +10.13].

---

## Redaction note (2026-08-20)

⛔ **This record was EDITED after it was committed, which the standing rule forbids.** The edit was
authorised explicitly for this file and this reason, and it is declared here rather than made
silently, because a correction nobody can see is the failure the rule exists to prevent.

**What was removed.** Three verbatim ATM Bench gold answers, quoted as illustrations: a deadline
answer carrying a date and a time, an event answer carrying a name, venue, city and date, and a
reservation answer carrying a date and an hour. They are replaced by descriptions of their SHAPE,
which is the only property the argument ever used. The scorer-defect table, which quoted a gold
answer and three near-copies of it, is replaced by a constructed pair verified to reproduce the
same defect.

**Why.** This repository is public, and `docs/ENTERPRISE_RAG_SUBMISSION.md` states that it
deliberately carries no benchmark questions, gold answers or document text. That rule was applied
correctly two hours earlier, when the 300 question subset file was kept out of the tree and only
its ids committed, and then contradicted in the prose of this record without anyone noticing.

**What was NOT touched.** Every prediction, every measured value, every sample size and every
verdict. The redaction removed third-party benchmark content and nothing that could make a
prediction look better than it was. The ten falsified predictions in this study are all still here.

**Still outstanding.** The phase 0 record on branch `claude/recall-answer-selection-e17d7c` carries
the same class of content, three further gold answers. It is not part of this push and has not been
edited. It needs the same decision before that branch is ever published.
