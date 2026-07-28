# Blind judge arbitration — how to label

199 items in `judge_labelling.csv`. These are **every case where our two LLM judges disagreed**
(`gpt-4o-mini` vs `gpt-4o`) on the full 10-conversation run. On each one, exactly one judge is
wrong — your labels decide which.

## What is hidden from you, and why

The file deliberately does **not** show which memory system produced the answer, or what either
judge said. If you could see that, your labels would be contaminated by it and the whole exercise
would prove nothing. The mapping lives in `.judge_labelling_key.json`, which you should not open
until you are finished. The row order is shuffled with a fixed seed for the same reason.

## The rule to apply

Use the **same criterion the judges were given** — otherwise we would be scoring them against a
standard they never saw, which is not a fair test of the judges:

> Grade the FACTS, not the wording. A different phrasing, format or level of verbosity that
> conveys the same facts is **correct** (`YES` matches `Yes, she is supportive`; `14 July 2023`
> matches `the Friday before 15 July 2023`, because they denote the same day).
> When the gold answer lists several items, the prediction is correct **only if it covers every
> one of them**; additional items or extra detail beyond the gold answer do **not** make it
> incorrect.

## How to fill it in

Put `Y` or `N` in the `your_verdict_Y_or_N` column. Leave a row blank to skip it — skipped rows
are excluded from scoring rather than counted against either judge.

Open it in a spreadsheet; it is UTF-8 CSV. Save as CSV when done (not .xlsx).

## Two things worth knowing before you start

**These are the hard cases by construction.** Every easy call is already in the 90% the judges
agreed on. Expect to find several genuinely arguable — that is the finding, not a failure of the
task. Where you truly cannot decide, leave it blank rather than guessing; a blank is honest data
and a coin-flip is not.

**Your labels become the ground truth we publish.** Please do not look up the source conversation
to resolve an ambiguity — the judges did not have it either. Judge only on the question, the gold
answer and the prediction in front of you, which is exactly the information the judges had.

## When you are done

Tell me and I will score both judges against your labels: each judge's accuracy on the contested
set, whether the difference is significant, and — the part that actually matters — whether either
judge's errors fall **asymmetrically between the two memory systems**. An asymmetry there would
mean judge choice biases the head-to-head, which is the thing we need to rule in or out before
publishing anything.
