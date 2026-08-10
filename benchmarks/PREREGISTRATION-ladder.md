# Pre-registration — the Answerability Ladder

Written **before** the builder exists. The git history of this file is the evidence.
Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

## Corpus statistics (sizes only — inspected before fixing widths, as permitted)

```
conversations: 10
turns per conversation: min=369 median=646.0 max=689
total turns: 5882
```

## Fixed parameters

- **Ring widths** `d in {0, 4, 16, 64}` plus `d = max` (whole conversation). Powers of four so the
  ladder spans two orders of magnitude in five levels; the top rung is capped at the median
  conversation size, because a ring wider than its cluster is d=max under another name.
- **Saturation rule:** d=max excises every turn of the conversation the gold turn belongs to.
- **Lambda in {1, 3, 10}**, where lambda = cost(false answer) / cost(false abstention). lambda=1
  reproduces BEAM's implicit weighting. Choosing lambda after seeing results is forbidden by this
  file.
- **Tie-breaking:** equal BM25 scores rank by `doc_id` ascending.
- **Instances per question:** one per ring level, all paired to the same answerable original.
- **Question sample: 300, seed 0**, drawn from the 1 536 usable questions (non-category-5, with
  `evidence`) after sorting by `question_id`. Fixed here, before any curve is visible. *(Corrected
  to 1 533 below — see the 2026-07-29 addendum.)*
- **Ingest scope: one conversation.** A question is scored against its own conversation only, not
  all ten. This matches `recall/eval/locomo.py`, which indexes one conversation at a time, and
  LOCOMO's own protocol.

  **Why the sample exists, stated because it bounds every number below.** The full set is 1 536
  questions × 5 rungs ≈ **7 680 distinct corpus states**, and a state must be indexed before it can
  be queried. Against the whole 5 882-turn corpus that is ~45 million document-indexings. This is
  not merely expensive for us: it is what every third party adopting the benchmark would pay, and
  a benchmark nobody can afford to run is not adopted. 300 questions × 5 rungs ≈ 1 500 states,
  each scoped to one conversation (median 646 turns), is a run an adopter can finish. *(The 1 536
  figure is corrected to 1 533 below — see the 2026-07-29 addendum; the 300-question sample and
  every ratio here are unaffected.)*

  Power was checked **before** the number was chosen, not after: with 300 pairs the standard error
  of a paired mean over deltas in {−1, 0, +1} is at most 1/sqrt(300) ≈ 0.058, so the
  pre-registered 0.15 effect sits at roughly 2.6 SE. BEAM's own head-to-head used 300 questions.
  **The H1 verdict rests on 300 questions, not 1 536, and every write-up must say so.** *(Read
  1 536 here as 1 533 — see the 2026-07-29 addendum below.)*

## Predictions, committed now

- **P1 (H1).** Correct-abstain rate rises with d. Specifically, correct-abstain at d=max exceeds
  d=0 by **more than 0.15**, with a bootstrap 95 % CI on the paired difference excluding zero.
- **P2 (H2).** The d=0 rung reproduces the adversarial regime: RE-call's correct-abstain at d=0 is
  **below 0.25**, consistent with 0.00/446 on LOCOMO's own category-5 adversarials and 0.467 on
  BEAM's abstention category.
- **P3.** False-abstain on the *answerable* originals exceeds **0.30**, consistent with the 0.481
  measured on LongMemEval per-question.
- **P4 (H3).** Rebuilding rings with a random-within-cluster neighbour function preserves the P1
  difference: same sign, and within **0.10 absolute** of the BM25 arm's paired difference. If the
  sign flips or the gap exceeds 0.10, BM25 ordering is a confound — the curve is measuring the
  neighbour function rather than answerability, and it does not ship as an answerability result.

## What falsifies the benchmark

P1 failing is not a bad result — it is the **kill condition**. A flat curve means the axis is a
fiction, and no comparative arm (Mem0, BEIR) is run. Money is spent only after P1 passes.

## Known to cut against us

- RE-call false-abstains at 0.481 while retrieval hit@5 is 0.970 (LongMemEval, per-question).
- On BEAM's abstention category we score 0.467 against Mem0's 0.533; false-abstain 9.6 % vs 4.1 %.
- Six candidate abstention signals were already measured and all failed (dense cosine AUC 0.753
  ships and sits at its ROC ceiling). This benchmark does not reopen that question.
- LOCOMO's `evidence` labels share an annotation pass with its answer key, 6.4 % of which is wrong.
  Using evidence for excision is safer, not safe.

## What v1 does NOT measure

Whether an *answered* question was answered **correctly**. v1 has no judge, by design and by
budget. On the answerable arm, "answered" is scored as success, which makes every v1 accuracy
figure an **upper bound**. The write-up must say so wherever a number appears.

---

## Addendum, 2026-07-29 — the usable-question count is 1 533, not 1 536

Appended rather than edited in place. A pre-registration that is quietly corrected once
measurement has begun is worth nothing, so the original number stays above and this records what
changed, when, and why. **No prediction, parameter or threshold in this file is altered** — the
sample is still 300 at seed 0, sorted by `question_id`, and P1–P4 are untouched. Recorded before
any manifest was built and before any curve was visible.

**What happened.** Building the loader surfaced that 9 of LOCOMO's questions cite evidence turns
their own conversation does not contain. They fall into two kinds, and only one is a defect of
ours to fix:

- **6 were a parsing failure on our side.** LOCOMO packs several turn ids into a single evidence
  string in a handful of entries — `"D8:6; D9:17"`, `"D9:1 D4:4 D4:6"`, `"D22:1 D22:2 D9:10
  D9:11"`. The turns exist; the annotation was simply never split. These are now split on `;` and
  whitespace and are **recovered**.
- **3 are annotation typos in the corpus**: `D:11:26` (stray colon), `D30:05` (zero-padded),
  and ids naming turns that genuinely do not exist (`D10:19`, `D4:36`). These stay dropped and
  counted. **We do not repair them.** Guessing what an annotator meant would put our own invention
  into a corpus chosen precisely because we did not author it.

**Final composition of the ladder corpus**, arithmetic closed and asserted by test
(`raw == retained + sum(dropped)`):

| | count |
|---|---|
| raw QA entries | 1 986 |
| dropped, category-5 adversarial (unanswerable by annotation, not excision) | 446 |
| dropped, no usable evidence | 5 |
| dropped, evidence names a turn the conversation lacks | 2 |
| **retained** | **1 533** |
| documents (turns) across 10 conversations | 5 882 |

**A consequence for numbers this repo has already published, flagged not fixed.** `results/`
reports LOCOMO retrieval on n=1 536. By inspection of `recall/eval/locomo.py:353-367`, evidence is
matched by string equality against retrieved `dia_id`s — so an unsplit `"D8:6; D9:17"` matches
nothing and the question is a **permanent miss for a parsing reason rather than a retrieval one**.
Five questions are structurally unhittable that way, an understatement of at most 5/1 536 ≈
**0.33 pp** on hit@5. That is small, it moves our own number in our own favour once corrected,
and it is **out of scope for this benchmark** — recorded here so it is not rediscovered as new.
