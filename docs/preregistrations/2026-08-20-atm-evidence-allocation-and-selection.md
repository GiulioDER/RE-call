# Pre-registration: ATM evidence allocation, answer disposition, and item selection

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Continuation of `2026-08-20-atm-answer-selection-phase0.md`, which is committed on branch
`claude/recall-answer-selection-e17d7c` and carries the phase 0 diagnosis this record builds on.
Measured there, and treated as given here:

* 22.15 QS points are lost on questions where the evidence was already retrieved.
* At least 5.63 of those are a text modality ceiling and cannot be won by answer selection.
* 61 questions never see a retrieved gold item, 39 more see it truncated, 85 distinct questions.
  They score 0.4067 against 0.6892 overall.
* 35 of 58 refusals had the answer fully on screen. 14 of 58 refusals were correct.
* The 23 gold-abstention questions currently score 0.7391, so 1.68 QS is at risk from any change
  that makes the system answer more.
* 47 questions are the reader choosing a real but wrong item from among the ones on screen.

## Facts about the current implementation, checked before predicting

`benchmarks/atm_bench.py:113` `format_media` already emits ID, Type, Timestamp and Location as the
FIRST four fields, and `format_email` emits ID and Timestamp first. `_evidence_text` truncates with
`part[:remaining]`, which cuts from the tail, so a truncated item keeps its identifying header and
loses Tags, OCR and Caption. That is better than a blind cut and it narrows H7: **the damage is
concentrated in items dropped whole, not in items trimmed.**

`_compact_text` collapses the newlines between fields, so the labelled structure survives as inline
labels on one line. That is cosmetic and is not part of any hypothesis here.

## H7. Fair-share evidence allocation

**Mechanism.** Replace the greedy first-come packer with a quota pass over the same 8,192 character
budget. Every item gets `budget / n` as a baseline; items shorter than their quota release the
surplus; the surplus is redistributed to the longer items; every item is guaranteed a floor so that
no item is dropped entirely while another holds three thousand characters; the highest ranked item
keeps a larger share, because rank 1 is the most likely to carry the answer and starving it to feed
rank 10 is the obvious way this change could backfire. Truncation stays tail-first, at a field
boundary rather than mid-token.

**Code point.** `benchmarks/atm_full_run.py:82` `_evidence_text`. Nothing else changes.

**Cost.** Zero API calls. Input tokens unchanged by construction, since the budget is unchanged.

### What I predict

| Quantity | n | Now | Predicted after |
| --- | ---: | ---: | --- |
| Questions where a retrieved gold item is never shown | 1,013 | 61 | **0**, by construction |
| Items presented whole, of 10 | 1,013 | mean 6.13 | **9.0 to 10.0** |
| Questions with at least one truncated item | 1,013 | 74.8% | 75% to 95%, **allowed to RISE** |
| QS | 1,013 | 68.92 | **+1.0 to +2.5** |

The third row is deliberately allowed to get worse. Trading whole-item drops for more tail trims is
the point of the change, and a design that improved both numbers at this budget would be a sign I
had miscounted rather than a better design.

The QS range is bounded above by the arithmetic: bringing all 85 affected questions from 0.4067 to
the 0.7453 that comparable complete-evidence questions score is +2.84 QS, and those 85 carry other
failure modes too, so the realistic band sits below it. The 4.98 figure from phase 0 is what they
would be worth at a perfect score and is NOT the prediction.

### What would falsify it

* Gold-hidden questions do not reach 0. That is a bug, not a result, because the floor guarantees
  it arithmetically.
* QS does not move, or moves below +1.0. Then evidence presence is not what is limiting those 85
  questions, and the phase 0 correlation between hidden gold and a 0.4067 score was confounded by
  those questions simply being harder.
* **QS falls.** The named cause would be starving rank 1 to feed the tail, and the check is the
  subset of questions whose gold is the rank 1 item: if those regress while the rest improve, the
  rank weighting is wrong and the fix is the weighting, not the hypothesis.

## H17. Answer-first disposition

**Mechanism.** One added sentence in the system prompt telling the model that the evidence block is
the complete memory available and that it should answer when the evidence supports an answer, even
partially, reserving refusal for when no item bears on the question. No change to retrieval,
packing, or output format.

**Code point.** `benchmarks/atm_full_run.py:137` system prompt, which `8ae5bd53` already rewrote to
the official baseline text. H17 is an addition to that text and must be measured against it, not
against the `0a0d6429` prompt.

**Cost.** Zero additional calls.

### What I predict

| Quantity | n | Now | Predicted after |
| --- | ---: | ---: | --- |
| Refusals with the answer on screen | 58 | 35 | **10 to 20** |
| Correct refusals preserved | 58 | 14 | **at least 10** |
| Score on the 23 gold-abstention questions | 23 | 0.7391 | **at least 0.65** |
| QS | 1,013 | baseline | **+1.0 to +2.5** |

**Both directions are registered on purpose.** The upside is capped at 3.46 and the downside at
1.68, so an arm that gains 2 while losing 1.5 is a null dressed as a win, and the only way to see
that is to report the abstention score beside the total. **The refusal rate is a required column of
every result table from here on**, and a QS gain reported without it is not acceptable evidence.

### What would falsify it

* The gold-abstention score falls below 0.65, whatever QS does.
* Refusals with the answer on screen do not fall below 25.
* QS rises while the unsupported claim rate rises: that is a model bluffing rather than answering.

## H21. Item selection before answer, in one call

**The target is the largest mechanism found and the one with no implementation yet:** 47 questions,
roughly 4.6 QS, where the reader answered from a real but wrong item that was on screen beside the
right one. Two cases, described rather than quoted (see the redaction note at the end): a question
that named a city by name AND country, answered with the same-named city in the other country while
items for both were on screen; and 18 of 21 wrong-date answers giving a date that was itself on
screen, 8 of them with the gold date on screen too.

**Mechanism.** Evidence first, answer second, in a single structured call. The model returns
`{"qualifiers": [...], "items": [{"id", "matches": "yes|no|partial", "failing_qualifier"}],
"answer": ...}`. It must extract the question's qualifiers before looking at the items, mark each
of the ten against them, and compose the answer only from items marked `yes`, or from `partial`
ones when no item is `yes`. The runner submits the `answer` field. The qualifier list and the
per-item marks are diagnostics, not gates: **no deterministic filter drops an item**, because a
filter that drops the gold item converts a wrong answer into a refusal and phase 0 already showed
refusals cost more than wrong answers on this benchmark.

**Code point.** `benchmarks/atm_full_run.py:115` `generate_answer` payload and a new
`select_answer(payload)`. Fails closed to the raw text if the JSON does not parse.

**Cost.** Zero additional calls. Roughly +200 to +400 output tokens per question, which lands
inside the existing ceiling.

### What I predict

| Quantity | n | Now | Predicted after |
| --- | ---: | ---: | --- |
| Wrong-item answers, `open_end` D2 | 29 | 29 | **15 to 22** |
| Wrong-date answers whose date was on screen | 21 | 18 | **10 to 15** |
| JSON parse failure | 1,013 | n/a | **below 2%** |
| Refusal rate | 1,013 | baseline | **no higher than baseline + 1 point** |
| QS | 1,013 | baseline | **+1.0 to +3.0** |

I predict this recovers between a quarter and a half of its 4.6 QS target, not more. The mechanism
addresses selection, and an unknown share of those 47 questions are wrong for a reason the marking
step will reproduce rather than catch, because the same model does both.

### What would falsify it

* Wrong-item answers do not fall below 25 of 29.
* The refusal rate rises by more than 1 point: marking every item `no` is the cheap way for a model
  to satisfy this instruction, and it would show up here before it showed up in QS.
* Parse failure above 5%.
* QS rises while wrong-item answers do not fall. Then something else produced the gain and the
  mechanism is unproven, whatever the total says.

## How all three will be measured

Mechanism metrics for H7 are computed offline by replaying the packer over the saved
`retrieval.jsonl`, 1,013 rows, at zero cost, and that replay is the gate for spending anything.
H17 and H21 need generation, so they are phase 2: a 300 question stratified subset, 120 `number`,
120 `open_end`, 60 `list_recall`, with QS reweighted to the true 35.5 / 50.7 / 13.7 quotas and
never reported as the raw subset mean. That subset detects roughly 0.10 on `list_recall` and 0.08
on `number`; below those a null is not a rejection.

Every result table must carry, beside QS: the refusal rate, the gold-abstention score, and the
count of questions whose gold evidence was hidden by packing.

## Confounds I can name now

1. **The baseline moved.** `8ae5bd53` changed the prompt after the measured run, so H17 and H21
   must be measured against a rerun of `8ae5bd53` itself, not against the 68.92 file. Comparing
   either arm to 68.92 would credit it with `8ae5bd53`'s own effect.
2. **H1 and H6 are entangled in that same commit** and cannot be separated by rerunning it.
3. **H7 and H21 overlap.** Nine of the refusals and nine of the wrong answers in the packing-hit
   set are also in H17's and H21's target sets. Measured together they will not sum, and the
   attribution needs one arm at a time.
4. **The same model marks the items and writes the answer**, so H21's marking step inherits the
   error that produced the wrong selection. This is why the prediction is a quarter to a half of
   the target rather than most of it.
5. **The judge is an LLM for 514 of the questions.** A prompt change alters answer style, and style
   moves an LLM judge independently of correctness. The deterministic 499 are the control: a gain
   that appears only on `open_end` and not on `number` or `list_recall` is a style effect until
   something else says otherwise.

## Result

Not yet measured at the time this record was committed.

---

## Result, H7 mechanism (2026-08-20)

**Status:** mechanism measured, QS not measured. Zero API calls. Implemented in `94feb574`. The
prediction above is unedited.

Both packers were replayed over the same 1,013 saved retrievals, the old one copied verbatim into
the harness so the comparison is against the code that produced the 68.92 file rather than against
a description of it.

| Metric | Old | New | Predicted | Verdict |
| --- | ---: | ---: | --- | --- |
| Questions hiding a retrieved gold item | 61 | **0** | 0 | met |
| Items presented whole, of 10 | 6.13 | **5.93** | 9.0 to 10.0 | **falsified** |
| Questions with a truncated item | 74.53% | 74.83% | 75% to 95% | roughly met |
| Evidence characters used, mean | 7,956 | 7,930 | unchanged | met |
| Blocks with an unreadable identifier | 3 | **0** | not predicted | found and fixed |
| Questions OVER the stated budget | **757** | **0** | not predicted | found and fixed |

### Where the prediction was not just wrong but incoherent

"Items presented whole, 9.0 to 10.0" cannot happen at this budget. In roughly three quarters of
questions the ten rendered items exceed 8,192 characters in total, so someone MUST be trimmed. I
predicted a number the arithmetic forbids, by confusing "presented at all", which the change fixes
completely, with "presented whole", which no allocation can fix without a bigger budget. The
recorded number is the one that matters and it is the one I did not predict.

### Two defects the replay found that no hypothesis named

The old packer never counted the `\n\n` separators against its own budget, so **757 of 1,013
questions were sent more than the 8,192 characters the manifest and the pre-registration both
claim**. The overrun is small, at most 18 characters, but every record of this run states a budget
that was not the budget. It is now exact.

On 3 questions the old packer's final block was cut mid-identifier, producing a block the reader
could not attribute to any memory item.

### The design was falsified once, by its own registered falsifier

The first implementation shared the budget by weighted water filling. Its replay: whole items fell
from 6.13 to **4.09**, and the top ranked item lost text on **252 of 1,013** questions. That is
exactly the failure the record registered as H7's third falsifier, spreading a fixed budget evenly
buys visibility for rank 10 with text taken from rank 1. Redesigned as rank-greedy with a lookahead
reserve: rank 1 now loses text on **15** questions.

### The trade, stated as a trade rather than as a win

At the level of individual gold evidence items, floor 100:

| Gold item, old state to new state | count |
| --- | ---: |
| whole to whole | 1,061 |
| **hidden to trimmed** | **84** |
| **whole to trimmed** | **21** |
| trimmed to trimmed | 39 |

84 gold items go from invisible to visible, 21 go from complete to partial. That is a favourable
trade rather than a free win, and it is worth stating that the floor sweep offered a better looking
net at 60 characters (84 against 8) which I did NOT take, because a usable `Timestamp` survives in
0% of blocks at 60 and in 100% at 100. Choosing 60 would have been fitting the floor to the outcome
metric on the same data I would then evaluate on. The floor is chosen by what it renders.

### What is still unmeasured

The QS prediction, +1.0 to +2.5. The 61 questions whose hidden gold is now shown currently score
0.3959, and bringing them to the 0.7453 that comparable questions reach is **+2.10 QS**, which sits
inside the registered band. Nothing here demonstrates they will move: the mechanism is fixed, the
outcome is not measured, and it cannot be measured without generation.

---

## Implementation note (2026-08-20)

Not a result. H17 and H21 are implemented in `5cd43430` and the control restored in the commit
above. Nothing has been generated, so no prediction in this record is resolved.

Both arms are selected by flag, and so is the packer, giving four independent switches over one
runner:

```
--evidence-packer {greedy,allocated}      greedy is the code that produced 68.92, defects included
--answer-policy {baseline,disposition,selection,both}
```

`baseline` is the official oracle prompt verbatim, pinned by a test, so the control is the
benchmark's own text rather than a paraphrase of it. The arms are additive over it, so a difference
between two arms is attributable to the sentences that differ.

Every question now writes a `diagnostics.jsonl` row with the refusal flag, the answer length, the
items presented, and for the selection arm the qualifiers, the per-item marks and the parse
outcome. `answers.jsonl` is untouched and still carries exactly the `{"id", "answer"}` rows the
official evaluator expects. The manifest carries `refusal_rate_this_invocation`, so the number this
record requires beside every score is produced by the run rather than reconstructed afterwards.

### Three defects found while wiring the arms, each fixed and pinned by a test

1. **The output ceiling was retried up to four times.** The request is byte identical on every
   attempt, so the retries buy the same refusal at four times the price. It now raises
   `CompletionTruncated` immediately and names the remedy. This matters for `selection`
   specifically: its envelope adds output tokens on top of the reasoning tokens OpenRouter already
   counts inside the ceiling, and the default here is 1,024 while the last completed full run
   needed 8,192.
2. **A failed envelope surrendered the raw JSON blob as the answer.** On a `list_recall` question
   the official scorer harvests evidence ids out of free text, so submitting the blob would have
   predicted every id in the `items` array against a gold set that is a singleton in 100 of 139
   cases: a truncated closing brace would have turned a good answer into the worst possible one.
   The answer field is now rescued by pattern first, and `rescued_answer` is recorded.
3. **The refusal marker list was too broad.** A bare "does not contain" marks "the email does not
   contain a price, but the total is £50" as a refusal, corrupting the one number these arms must
   be judged against. Removed, and the narrower wording this runner's own earlier prompt used is
   kept.

### The operational consequence for whoever runs this

`--answer-policy selection` or `both` must be run with a larger `--max-output-tokens` than the
1,024 default. Run the arms with the same ceiling as each other, because the ceiling is a config
difference and a run that dies on question 16 is not an arm.

---

## Result: arms A and B, judged (2026-08-20)

**Status:** measured for two of four arms. C and D are still generating. Predictions above are
unedited. Every number here was judged through the OpenRouter route, INCLUDING the baseline, which
is why the 68.92 figure does not appear in any difference.

The baseline row is the old `0a0d6429` answer file restricted to the same 300 questions and
re-judged on the same route, so no comparison below carries a route term.

| set | QS reweighted | number | list_recall | open_end | refusal % | gold abstention | median chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `0a0d6429` prompt | 67.50 | 70.83 | 53.64 | 68.91 | 11.67 | 100.00 | 33 |
| A greedy + official prompt | 68.89 | 72.50 | 62.56 | 68.07 | 15.33 | 100.00 | 14 |
| B allocated + official prompt | 69.17 | 74.17 | 64.41 | 66.95 | 13.67 | 83.33 | 14 |

⚠️ **That table is the wrong one to draw conclusions from, and it is printed to show why.** Each
row is scored on its own set of successfully judged questions, and the judge failed on a different
one or two per arm. Restricted to the 296 questions scored in ALL THREE, the order of A and B
REVERSES: A 69.34, B 68.88. A comparison that lets each arm choose its own denominator is not a
comparison.

### Paired, on the 296 questions common to all three

| comparison | delta QS | 95% CI, paired bootstrap | questions that changed |
| --- | ---: | ---: | ---: |
| A vs old prompt | +1.38 | [-3.28, +5.81] | 45 / 296 |
| B vs old prompt | +0.91 | [-3.31, +5.18] | 47 / 296 |
| **B vs A, H7 alone** | **-0.47** | [-3.84, +2.95] | 24 / 296 |

**Not one of these excludes zero.** The 300 question subset cannot resolve effects of this size,
which the pre-registration said in advance for the per-type metrics and which turns out to bind for
the aggregate too.

### The half of the benchmark that has no judge in it

`number` and `list_recall` are scored by exact match and Jaccard, with no model in the loop and no
run-to-run noise. On the 180 such questions common to all three:

| comparison | delta | 95% CI | verdict |
| --- | ---: | ---: | --- |
| A vs old prompt | +3.69 | [-0.88, +8.37] | includes zero |
| **B vs old prompt** | **+5.41** | **[+0.99, +10.13]** | **excludes zero** |
| B vs A, H7 alone | +1.72 | [-0.76, +4.47] | includes zero |

Levels: old 66.04, A 69.73, B 71.45. **This is the only comparison in the whole exercise whose
interval excludes zero**, and it says the official prompt plus the fair-share packer together beat
the old prompt on the part of ATM that is measured exactly.

### Predicted against measured

| Prediction | Predicted | Measured | Verdict |
| --- | --- | --- | --- |
| H1 QS gain | +3.5 to +6.5 | +1.38 full, CI includes zero; +3.69 deterministic | **falsified on QS** |
| H1 `list_recall` Jaccard | +0.20 to +0.35 | +0.089 | **falsified** |
| H1 ID emission rate | at least 0.95 | 0.833 | **falsified** |
| H7 QS gain | +1.0 to +2.5 | **-0.47** full, +1.72 deterministic | **falsified on QS** |
| H7 named risk: rank 1 starved | the cause if QS falls | rank 1 trimmed on 15 of 1,013 | **not the cause** |

I have now over-predicted the value of answer format three times in a row, in the same direction
each time. The mechanisms are real and reproduce; my estimates of what they are worth are not.

### Where H7 actually goes, and it is not where the prediction said

B changes only 24 of 296 answers against A, and the per-type paired means split cleanly:

| type | n | paired mean, B minus A | better | worse |
| --- | ---: | ---: | ---: | ---: |
| `number` | 120 | +1.67 | 3 | 1 |
| `list_recall` | 60 | +1.85 | 5 | 2 |
| `open_end` | 116 | **-2.59** | 5 | 8 |

Showing all ten items helps where the answer is a value to be found and hurts where a judge reads
prose. The registered falsifier blamed rank 1 starvation, and that is measurably not it: the top
item loses text on 15 of 1,013 questions. The cost lands on `open_end`, and neither the record nor
I predicted that.

### The refusal column, which the record required for exactly this reason

Refusals rise in both arms, +3.67 points in A and +2.00 in B, and `open_end` falls in both, -0.84
and -1.96. The two move together, as the phase 0 diagnosis said they would when the refusal wording
became shorter and easier. **The gold-abstention column is NOT evidence either way**: the subset
holds 6 such questions, so B's 100.00 to 83.33 is a single question and must not be read as a
trend.

---

## Result: all four arms, judged (2026-08-20)

**Status:** measured, complete. Predictions above unedited. Every set, INCLUDING the baseline, was
judged through the OpenRouter route, so no comparison carries a route term. Scored on the 296
questions common to all five sets, so no arm chooses its own denominator.

| set | QS reweighted | number | list_recall | open_end | refusal % | median chars | truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `0a0d6429` prompt | 67.96 | 70.83 | 53.64 | 69.83 | 11.67 | 33 | 0 |
| **A** greedy + official prompt | **69.34** | 72.50 | 62.56 | 68.97 | 15.33 | 14 | 0 |
| **B** allocated + official prompt | 68.88 | **74.17** | **64.41** | 66.38 | 13.67 | 14 | 0 |
| **C** B + H17 disposition | 68.76 | 73.33 | 62.49 | 67.24 | 13.33 | 15 | 0 |
| **D** B + H21 selection | **64.66** | 69.17 | **49.79** | 65.52 | **19.33** | 13 | 0 |

| paired comparison | delta QS | 95% CI | questions changed |
| --- | ---: | ---: | ---: |
| A vs old prompt | +1.38 | [-3.28, +5.81] | 45 / 296 |
| B vs A, H7 packer | -0.47 | [-3.81, +2.89] | 24 / 296 |
| C vs B, H17 disposition | -0.12 | [-3.51, +3.38] | 27 / 296 |
| **D vs B, H21 selection** | **-4.22** | [-8.76, +0.17] | 42 / 296 |
| best arm vs old prompt | +0.79 | [-3.20, +4.84] | 44 / 296 |

**The best arm on full QS is A**, which is the official prompt with the OLD greedy packer. Every
addition after it made the total worse. No interval excludes zero, so the ranking is not
established either, and the honest summary is that three of the four interventions produced
nothing measurable and the fourth produced measurable harm.

### H21 failed exactly as the record said it would fail

The registered falsifier read:

> The refusal rate rises by more than 1 point: marking every item `no` is the cheap way for a model
> to satisfy this instruction, and it would show up here before it showed up in QS.

Measured over 300 questions, with **zero parse failures**, so this is the protocol working rather
than breaking:

| quantity | measured |
| --- | ---: |
| Items marked `yes`, mean of 10 | **1.23** |
| Items marked `no`, mean | 8.10 |
| Questions accepting ZERO items | **72 of 300 = 24.0%** |
| Refusal rate | 19.33%, against 13.67% for B |
| `list_recall` | 49.79, against 64.41 for B |

Asking a model to justify each item against the question's qualifiers taught it to reject, not to
discriminate. On `list_recall` the gold list averages 1.95 items and D marks 1.38 as matching, so
it under-selects systematically, and that is the collapse from 64.41 to 49.79.

The prediction had been 15 to 22 wrong-item answers out of 29 and a refusal rate no more than one
point above baseline. Refusals rose 5.66 points. **Falsified on its own named mechanism**, which is
the most useful way for a hypothesis to die.

### H17 did nothing

Refusals moved 13.67 to 13.33, a third of a point, against a prediction that refusals with the
answer on screen would fall from 35 to between 10 and 20. One sentence of disposition did not shift
the behaviour it targeted. The mechanism identified in phase 0 is real, 35 refusals with the answer
fully on screen, but a prompt sentence does not reach it.

### What survives

Only one thing, and it is not visible in the QS column: on the 180 questions ATM scores WITHOUT a
judge, B beats the old prompt by +5.41 with CI [+0.99, +10.13]. `number` and `list_recall` improve
in every arm that carries the official prompt. The gains are real and they are consistently erased
by `open_end`, which falls in every arm and is 50.7% of the weight.

### The cost of the arm that lost

D generated at roughly 2.4 answers per minute against A's 11, so the selection envelope cost about
four times the wall clock for a 4.22 point loss.

### Predicted against measured, all of it

| Prediction | Predicted | Measured | Verdict |
| --- | --- | --- | --- |
| H1 QS | +3.5 to +6.5 | +1.38, CI includes zero | falsified |
| H1 `list_recall` Jaccard | +0.20 to +0.35 | +0.089 | falsified |
| H1 ID emission | at least 0.95 | 0.833 | falsified |
| H7 QS | +1.0 to +2.5 | -0.47 | falsified |
| H7 named risk | rank 1 starved | rank 1 trimmed on 15 of 1,013 | not the cause |
| H17 refusals with answer on screen | 35 to 10-20 | refusal rate moved 0.34 points | falsified |
| H17 QS | +1.0 to +2.5 | -0.12 | falsified |
| H21 QS | +1.0 to +3.0 | **-4.22** | falsified, wrong sign |
| H21 refusal rate | no more than +1 point | **+5.66 points** | falsified, and it was the named failure mode |
| H21 parse failure | below 2% | **0.0%** | **held** |
| Ceiling: completions over 8,192 | fewer than 5 of 900 | 1 of 333 recorded | held |

One prediction of eleven held on the upside, and it was the one about whether the protocol would
work mechanically rather than whether it would help.

---

## Redaction note (2026-08-20)

⛔ **This record was EDITED after it was committed, which the standing rule forbids.** Authorised
explicitly, for the same reason as the note in `2026-08-20-atm-open-end-coverage.md`, and declared
here rather than made silently.

One fragment of an ATM Bench question, two words naming a city and its country, was replaced by a
description of its shape. This repository is public and
`docs/ENTERPRISE_RAG_SUBMISSION.md` states that it deliberately carries no benchmark questions,
gold answers or document text.

No prediction, measured value, sample size or verdict was touched. The same fragment was removed
from the `SELECTION_SYSTEM` docstring in `benchmarks/atm_full_run.py`, which is code and not
covered by the pre-registration rule.
