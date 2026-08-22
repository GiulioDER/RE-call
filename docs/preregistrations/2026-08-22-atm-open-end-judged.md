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
