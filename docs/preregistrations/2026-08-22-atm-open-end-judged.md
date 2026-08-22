# Pre-registration: does the second-pass commitment on `open_end` survive the judge

**Date:** 2026-08-22   **Status:** predicted, not yet measured

Third and last record of the A1 line. `2026-08-22-atm-router-and-abstention-downside.md` measured a
coverage gain on the 55 `open_end` abstentions and deliberately deferred the judge, because
**coverage is a proxy and the judge is the outcome**. This record buys the outcome.

## The question

On the 55 `open_end` questions whose 2026-08-21 answer was an abstention and whose gold answer is
not one, what does the official judge score the second-pass arm, and does it beat the same-day
control?

## What I already know

* Treatment on those 55: re-abstention 0.4364 against the control's 0.8000, gold-token coverage
  **0.2864 against 0.1447**, commits on 31 against 11.
* ⚠️ **The committed answers cover LESS gold than the control's, 0.4812 against 0.6137, and are a
  third as long, 23 characters against 81.** On this run `open_end` under 20 characters scores
  0.5977 while 50 to 100 scores 0.8228. The arm buys commitment in the shape this judge punishes,
  and that is the reason the outcome cannot be inferred from the coverage number.
* **The baseline for these 55 needs no re-judging.** The submission package judged them through the
  same OpenRouter transport and they score **exactly 0.0000, 0 of 55 nonzero**. The 17
  gold-abstention questions score **exactly 1.0000, 17 of 17**.
* Arm E of the earlier study raised `open_end` by 4.35 points while the coverage it named as its
  mechanism did not move. This record is the mirror case and must not repeat the error of reading
  one for the other.

## What will run

`run_llm_judge` from the official evaluator, on all 72 questions in both arms, 144 calls. Provider
`openai`, model identity `openai/gpt-5-mini`, `max_tokens` 600, the official `LLM_JUDGE_PROMPT`
unmodified, the official parsing unmodified, four workers.

⚠️ **Disclosed deviation, transport only.** The OpenAI account holding the official judge key is out
of credit, so the client is pointed at an OpenAI-compatible endpoint by the two environment
variables of `scripts/atm_judge_openrouter.patch`. Verified by diff against the pristine file: the
patch adds `import os` and replaces the client construction, and touches **no scoring code, no
prompt, no temperature and no retry policy**. With both variables unset the line is the original.
The 2026-08-21 submission was judged the same way, so the baseline and both arms share the route
and no comparison here carries a route term.

## What I predict

| Quantity | n | Now | Predicted |
| --- | ---: | ---: | --- |
| Baseline judged score | 55 | 0.0000 | 0.0000, it is already measured |
| Control judged score | 55 | 0.0000 | **0.08 to 0.18** |
| **Treatment judged score** | 55 | 0.0000 | **0.15 to 0.30** |
| **Treatment minus control** | 55 | n/a | **+0.05 to +0.18** |
| Treatment, share of its 31 committed answers judged true | 31 | n/a | 0.30 to 0.50 |
| Both arms, the 16 questions that re-abstained on gold abstentions | 16 | 1.0000 | **1.0000** |

A gain of `s` on 55 of 1,013 questions is `55 s / 1013 x 100 = 5.43 s` QS points, so the treatment
band is **+0.81 to +1.63 QS** and the delta band is **+0.27 to +0.98 QS**.

The reasoning behind the treatment band, stated so it can be checked rather than admired: 31
committed answers at a gold-token coverage of 0.4812 sit in the 0.5185 bucket of the measured
coverage-to-score mapping, which would give 0.29 mean. These 55 are the questions the reader
already declined once, so they are harder than the population that mapping was fitted on, and the
band is set below that arithmetic rather than at it.

## What would falsify this

* **Treatment score below 0.10.** The commitment carries no content this judge accepts, and
  extending A1 past the deterministic types is not supported.
* **Treatment minus control at or below zero.** The framing adds nothing on this type and the whole
  `open_end` extension dies here.
* **Coverage rose and the score did not.** That is arm E's failure in mirror image and it kills the
  coverage proxy as a mechanism claim for the rest of this programme, whatever the totals say.
* **The 16 re-abstained gold-abstention questions do not all score 1.0000.** Then the judge is not
  reproducing its own 2026-08-21 verdicts on byte-similar answers, and nothing in this record is
  interpretable, including the parts that look good. This is the apparatus check, and it is read
  first.

## Confounds I can name now

1. **Judge run-to-run variance.** Measured earlier in this project at 3 disagreements in 59 on
   identical triples, about 5%. On 55 questions that is roughly 3 flips from noise alone, so a
   delta smaller than about 0.05 is not resolvable and the record says so before the number
   arrives.
2. **The baseline was judged on 2026-08-21 and the arms today.** Same route, different day. The
   baseline is exactly 0.0000 with no nonzero rows, which is the floor and the least sensitive
   place for that difference to matter, but it is not zero risk.
3. **This is a subset chosen because the model abstained on it.** Nothing here generalises to
   `open_end` as a whole, and the QS figures apply to these 55 questions only.

## Result

Not yet measured at the time this record was committed.

---

## Result (2026-08-22)

**Status:** measured. Predictions above unedited. 144 judge calls through the disclosed transport,
official prompt and parsing untouched.

**Apparatus read first, as the record requires.** The 16 gold-abstention questions that re-abstained
in both arms score **1.0000 in both**, 0 of 16 below. The judge reproduces its own 2026-08-21
verdicts on those rows, so the comparison is interpretable.

### The 55 `open_end` abstentions that were wrong

| arm | judged score | committed on | share of committed judged true | gold-token coverage |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0000 | 0 of 55 | n/a | 0.0000 |
| control | 0.0909 | 11 of 55 | 0.4545 | 0.1447 |
| **treatment** | **0.3091** | 31 of 55 | **0.5484** | 0.2864 |

**Treatment minus control: +0.2182, 95% paired bootstrap [+0.1091, +0.3273], excludes zero.**
In QS on the full split: treatment +1.68, control +0.49, **framing only +1.18 [+0.59, +1.78]**.

**No registered falsifier fired.**

### The 17 gold abstentions, now judged rather than assumed

| arm | score | points lost | damage |
| --- | ---: | ---: | ---: |
| baseline | 1.0000 | 0 of 17 | 0.00 QS |
| control | 0.9412 | 1 of 17 | 0.10 QS |
| treatment | 0.9412 | 1 of 17 | **0.10 QS** |

The previous record's pessimistic bound assumed the one committed answer would score zero. It does.
The bound was exact, and the damage is identical in both arms, so it is a re-ask effect rather than
the framing.

### Predicted against measured

| Prediction | Predicted | Measured | Verdict |
| --- | --- | ---: | --- |
| Control judged score | 0.08 to 0.18 | 0.0909 | held |
| Treatment judged score | 0.15 to 0.30 | **0.3091** | **falsified, high by 0.0091** |
| Treatment minus control | +0.05 to +0.18 | **+0.2182** | **falsified, high** |
| Share of the 31 committed judged true | 0.30 to 0.50 | **0.5484** | **falsified, high** |
| The 16 re-abstained gold abstentions | 1.0000 | 1.0000 | held |

### The caution from the previous record was wrong, and wrong in the useful direction

That record flagged that the treatment's committed answers cover **less** gold than the control's,
0.4812 against 0.6137, and are a third as long, and warned that the framing was buying commitment
in the shape this judge punishes. **It is not.** The treatment's committed answers are judged true
**54.84%** of the time against the control's **45.45%**: shorter, lower-coverage answers were judged
correct more often, not less.

So the gold-token coverage proxy **understated** this arm rather than overstating it, which is the
opposite of the arm E failure the record was watching for. Coverage remains a mechanism worth
reporting and is now measurably not a substitute for the judge in either direction.

### The net across all three records

Framing only, with the same-day control subtracted everywhere, which is the part attributable to
the intervention rather than to re-asking:

| component | n | QS | interval |
| --- | ---: | ---: | --- |
| deterministic types | 85 | **+1.41** | [+0.72, +2.15] |
| wrong `open_end` abstentions | 55 | **+1.18** | [+0.59, +1.78] |
| gold abstentions | 17 | **-0.10** | measured, not assumed |
| **total** | **157** | **+2.50** | |

Against the shipped baseline of **68.4264**, the attributable projection is **70.93**. The larger
figure obtained by comparing the treatment to the baseline directly, **+3.97 QS to 72.40**, includes
the re-ask and drift effect the control isolates, and **must not be quoted as the value of the
intervention**: a control that gains 1.48 QS by re-asking with the unchanged prompt would gain it
for the baseline too.

### What is now established, and the one thing that still blocks promotion

Established: the intervention is real on **both halves of the benchmark**, its interval excludes
zero on each, it costs 0.10 QS where abstaining is correct, and it concentrates where the evidence
is. That is three independent records, 458 provider calls and 144 judge calls, with the
deterministic half free to re-check.

Not established: A1's registered re-abstention falsifier fired at 44.7% against a ceiling of 20%
and **has never been re-registered in the relative form its own result section argued for**. Until a
record predicts that criterion relative to a control and it holds, this line is a measured effect
without a passed pre-registration, and shipping it would set the precedent that a falsifier can be
outrun by later evidence. The next record is that one, and it is cheap: the arms are already
generated.
