# Human arbitration of the two LLM judges

Reproduce with `python -m benchmarks.labelling.score_labels`.

## What was measured

Both judges (`openai/gpt-4o-mini`, `openai/gpt-4o`) graded the same full 10-conversation run of
both memory systems. They disagreed on **199** answerable questions. On each of those, exactly one
judge is wrong, so hand-labelling them arbitrates between the judges directly.

The 199 were presented **blind**: shuffled with a fixed seed, with the memory system and both
judges' verdicts withheld until labelling was complete. Annotator instructions are in
`README.md`, and they quote the judge prompt verbatim — the point being to grade the judges by the
standard they were actually given, not a stricter one invented afterwards.

## Result

**195 scored, 4 excluded as undecidable.**

| judge | correct on contested items |
|---|---|
| `openai/gpt-4o-mini` | 28 / 195 = **0.144** |
| `openai/gpt-4o` | 167 / 195 = **0.856** |

`gpt-4o-mini` is the wrong judge for this task: where a stronger judge disagrees with it, it is
wrong roughly six times out of seven.

Its failure mode is **under-crediting correct answers**, not accepting wrong ones — 180 of the 199
contested predictions were labelled correct, and mini had called them wrong. That is the opposite
direction from the [LoCoMo audit](https://github.com/dial481/locomo-audit), which measured the same
model *over*-accepting (62.81% of deliberately wrong answers) under the original judge prompt. Taken
together the two results say something stronger than either alone: the weak judge is unreliable in
whichever direction the prompt pushes it.

## The bias question — clean

The result that could have invalidated the head-to-head is whether judge error falls
asymmetrically between the two memory systems. It does not:

| items from | n | `gpt-4o-mini` correct | `gpt-4o` correct |
|---|---|---|---|
| RE-call | 108 | 0.130 | 0.870 |
| Mem0 | 87 | 0.161 | 0.839 |

14/108 versus 14/87 — a ~3-point gap on ~100 items per arm, well inside noise. The weak judge is
bad on both arms about equally, which is why the accuracy ranking held across both judges in every
configuration tested.

## Annotation rules

Labels were free text rather than Y/N, because the binary was too coarse for the recurring case: a
prediction covering most of a gold list but dropping one item. Mapping back to a binary is a
judgement call, so the rules are stated (and implemented in `score_labels.py`):

- **Extra detail is still correct.** `Yes +` / `Yes + day` / `Yes + data` = right, and said more
  than the gold. The judge prompt explicitly permits this.
- **Missing any gold item is incorrect** — every `partial missing …`, and any label containing
  "missing", including one written `yes missing advices from friend` (annotator confirmed).
- **Undecidable rows are excluded, not guessed.** `no gold answer` (3, the dataset's gold is
  unusable) and `not sure golden is vague` (1). Counting a dataset defect as a judge error would
  blame the judge for LOCOMO's problem; guessing would manufacture ground truth that does not exist.

## Consequence for the published numbers

The `gpt-4o`-judged results are the ones to report. The `gpt-4o-mini` figures are retained as the
incumbent-configuration baseline — it is the model published LOCOMO evaluations use — now carrying
the measured caveat that it is the inferior judge.

## Limits

Single annotator, no second-rater agreement measured, and the annotator is the project author —
which is why the blinding and the pre-committed instructions matter, and why the raw labels are
published here rather than summarised. Anyone can re-score them, or re-label the same blind set,
from the files in this directory.
