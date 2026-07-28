# Blind system comparison on BEAM — how to label

Each row is **one question and one predicted answer**. Your labels decide which memory system
answered better. Build a sheet with `build_beam_labelling.py`; score it with `score_beam_labels.py`.

## What is hidden from you, and why

The file does **not** show which system produced the answer, or what any LLM judge said about it.
Both systems appear, shuffled together with a fixed seed, so consecutive rows are usually not the
same question and never labelled as a pair.

This matters more here than in the judge-arbitration round. There you were refereeing two judges on
one system's output; **here you are grading your own project against a competitor.** Unblinded
labels would be worth nothing no matter how carefully you made them — a reader can say "the author
graded his own system" and stop reading. The mapping lives in the `_key.json`; do not open it until
you are finished.

## The rule to apply

The same criterion the systems were held to by BEAM's own judge:

> Grade the FACTS, not the wording. A different phrasing, format or level of verbosity that conveys
> the same facts is **correct**.
> The gold answer is a list of **rubric nuggets**, one per line. The prediction is correct **only if
> it covers every one of them**; extra detail beyond the gold does **not** make it incorrect.

**Abstention questions are the ones to read closely.** Their gold reads *"there is no information
related to X"*, so the correct answer is a refusal. An answer that supplies confident specifics is
**incorrect**, however plausible it sounds — that is the behaviour the category exists to catch, and
it is scored the same way whichever system produced it.

## How to fill it in

Put `Y` or `N` in `your_verdict_Y_or_N`. Free text is accepted and mapped the way the last round's
labels were: anything containing "missing" reads as **incorrect** (covering only part of the gold is
wrong), and `no gold answer` / `not sure golden is vague` are **excluded** rather than counted
against either system. Leave a row blank to skip it.

UTF-8 CSV; save as CSV, not .xlsx.

## Keeping it proportional

600 items for the full 300 questions is about five hours. Two switches cut that without weakening
the result:

- `--disagreements-only` — only questions where the two systems' recorded LLM scores differ (160
  questions, 320 items here). Questions both arms already got right, or both got wrong, cannot tell
  you which system is better; this is the same argument that reduced the judge-arbitration set to
  its 199 contested items.
- `--category abstention` — 30 questions, **60 items**, and the axis this project is actually about.
  The cheapest useful sheet in the repo.

## What comes out

`score_beam_labels.py` reports each system's accuracy on the labelled items and a **paired McNemar**
exact test over the discordant questions — the ones where exactly one system was right. Concordant
questions carry no information about which is better and are excluded from the test by construction,
not by choice.
