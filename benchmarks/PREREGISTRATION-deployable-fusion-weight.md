# Preregistration: can down-weighting rescue the DEPLOYABLE query fusion?

Written **2026-08-06, before the arm was fused or scored.** Frozen.

## Prior work (searched 2026-08-06, `docs_search(source_type="memory")`, no gap warning)

- [[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]] — ⛔ **do not build an LLM
  rewriter**: gold rewriting is a ceiling, +0.0321 and INCONCLUSIVE, and it is the only lever that
  costs money. This preregistration honours that: the arm below uses **no gold and no LLM**.
- [[closed-hypothesis-recall-leg-disagreement-prf-trigger-2026-07-28]] — per-query *adaptive*
  weighting is FALSIFIED. The weight here is **fixed a priori and identical for every query**, so
  it is not that hypothesis returning.
- [[reference-mtrag-dev-query-variant-overlap-2026-08-06]] — the deciding cell.
- [[reference-validation-standards]] — bootstrap CI, permutation, Holm.
- The multi-query run of 2026-08-06 (`benchmarks/mtrag/multiquery.py`, `mq/`) — its arms, its
  five contrasts and its `mq_nested2_nogold` post-hoc result are the direct input here.

## What is already known, and why this is not a re-measurement

`mq_nested2_nogold` = `("last", "full")`, unweighted — **already run** as a post-hoc arm. It is
the only fully deployable fusion: both variants exist at inference, no gold, no LLM.

| arm | nDCG@5 | R@100 |
|---|---|---|
| `mq_last` (control) | 0.3573 | 0.7377 |
| `mq_nested2_nogold` = {last, full}, unweighted | **0.3126** | 0.8220 |

⇒ **+0.0842 R@100** (CI [+0.0644, +0.1050]) and **−0.0447 nDCG@5**, tripping the R@5, nDCG@5 and
nDCG@10 vetoes. Coverage bought with ranking. Its own author labelled it *"a hypothesis for a
future preregistration, not a decision"*. **This is that preregistration.**

## Hypothesis

Down-weighting the harmful variant in the RRF makes the deployable fusion **ranking-neutral while
retaining most of its coverage gain.**

Evidence it might: on the three-variant arm, `w_full = 0.5` recovered **+0.0208 nDCG@5** for only
**−0.0106 R@100** (`mq_nested3_vw` 0.3862/0.8507 vs `mq_nested3` 0.3654/0.8613).

## The arm (one, frozen)

    mq_nested2_nogold_vw   variants ("last", "full")   nested   weights (1.0, 0.5)

🔑 **`w_full = 0.5` is FIXED A PRIORI and is not tuned here.** It is the value already chosen a
priori and used by `mq_nested3_vw` in the previous run, so it carries no information from this
contrast. **No other weight will be run.** If 0.5 fails, that is the answer; sweeping weights
until one passes would convert this into a fitting exercise, which is precisely how the
falsified adaptive-weighting hypothesis went wrong.

Control: `mq_last`, unchanged. Only the fusion weight differs from `mq_nested2_nogold`.

## Decision rule, fixed before the numbers

**SHIP** iff BOTH hold:

1. R@100 delta >= **+0.020** (the same ship bar the previous run used), CI excluding zero, Holm-significant; AND
2. **no ranking veto trips** — nDCG@5, nDCG@10, R@5 and R@10 each either improve or have a CI
   containing zero.

Any other outcome is **NOT SHIP**. In particular, a large coverage gain with a significant nDCG@5
regression is a fail, not a trade to be argued about after seeing it.

## Predictions, written before running

| # | prediction | confidence |
|---|---|---|
| P1 | R@100 delta stays >= +0.020 (some coverage lost vs +0.0842, not all) | high |
| P2 | nDCG@5 regression SHRINKS from −0.0447 | high |
| P3 | **nDCG@5 veto no longer trips** | **genuinely uncertain — this is the experiment** |

🔑 **P3 is the deciding prediction and I do not know the answer.** Naive arithmetic says the
+0.0208 recovery seen on the three-variant arm leaves −0.024, which would still trip. But `full`
is 1 of 2 rankings here rather than 1 of 3, so halving its weight is proportionally a larger
intervention and the recovery could be bigger. **I expect it to be close to the line, and I am
recording that I cannot call it** rather than claiming a prediction I do not have.

## Population and statistics

Both populations, as established: the **deciding cell** (queries where `last` and `full` differ —
`full` is byte-identical to `last` on 102 of 777, so the cell is ~675) and **all 777** as the
deployment estimate. Paired bootstrap n=10000, sign-flip permutation n=5000, Holm across the
metric family, alpha=0.05. Dev split only; MTRAG-UN is sealed.

## Cost

**No retrieval, no GPU.** The `last` and `full` legs are already on disk in `mq/legs/`; this is a
re-fuse and a re-score, seconds of CPU. That is the entire reason it is worth doing now rather
than bundling into a later campaign.
