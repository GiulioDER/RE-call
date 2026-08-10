# The Answerability Ladder v1 — H1 verdict

**Date:** 2026-07-29 · **System:** RE-call at shipped defaults · **Corpus:** LOCOMO
**Archived manifest:** `results/ladder/manifest.jsonl`, digest
`6bfe2d2b094eefaf64409a3eddbb26d62b9e7709346540b2d068a4be300632b1`, 1 800 instances
**Archived responses:** `results/ladder/responses_recall.jsonl`, 1 800 scored, none missing
**Pre-registration:** [`benchmarks/archive/preregistrations/PREREGISTRATION-ladder.md`](../../benchmarks/archive/preregistrations/PREREGISTRATION-ladder.md) + its dated addendum

**Prior work searched** — `docs_search(source_type="memory")` on "BEAM benchmark evaluation results
answerability abstention" and on "distractor corpus far gap versus near miss abstention regime
unrelated documents present". Load-bearing hits, both cited in place below:
[[project-recall-abstention-bounded-domain-2026-07-24]] (the guard works on far gaps, 1.00 and 0.89,
and fails on near-miss; six candidate signals measured, all failed) and
[[project-recall-threshold-embedder-fragile-2026-07-28]] (the 0.50 floor is not comparable across
embedders). **This result corroborates the first and is confounded by the second** — neither was
rediscovered here, and this run does not propose a seventh signal.

## Headline: H1 PASS — and the PASS is an artefact

```
rung           n  corr-abst  false-ans  false-abst     L=1     L=3    L=10  surv-docs
original     300      0.000          0           0     0.0     0.0     0.0        629
d=0          300      0.000        300           0   300.0   900.0  3000.0        628
d=4          300      0.003        299           0   299.0   897.0  2990.0        624
d=16         300      0.003        299           0   299.0   897.0  2990.0        612
d=64         300      0.003        299           0   299.0   897.0  2990.0        564
d=max        300      1.000          0           0     0.0     0.0     0.0          0

H1 (pre-registered) paired Δ(correct-abstain), d=max − d=0: +1.000 [+1.000, +1.000]  n=300
H1: PASS
```

The pre-registered contrast passes at the maximum possible value. **It should not be cited as a
result**, and this document exists to say why in the same place the number appears.

`d=max` excises the question's whole conversation, and the ingested slice is one conversation — so
the last column is not incidental, it is the finding: **at `d=max` the corpus contains zero
documents.** A system abstaining there has nothing to retrieve. It has not recognised an
unanswerable question.

Every contrast whose corpus is non-empty is flat:

```
original vs d=0     Δ=+0.000 [+0.000, +0.000]  n=300  FAIL
d=0 vs d=4          Δ=+0.003 [+0.000, +0.010]  n=300  FAIL
d=4 vs d=16         Δ=+0.000 [+0.000, +0.000]  n=300  FAIL
d=16 vs d=64        Δ=+0.000 [+0.000, +0.000]  n=300  FAIL
d=64 vs d=max       Δ=+0.997 [+0.990, +1.000]  n=300  PASS   <- the cliff into the empty index
d=0 vs d=64         Δ=+0.003 [+0.000, +0.010]  n=300  FAIL
```

The harness's own derived qualification, printed from the computed numbers rather than asserted:

> the axis as built prices "is anything indexed at all", not answerability

## The result, stated plainly

**RE-call abstained on 3 of 1 200 unanswerable instances whose corpus was non-empty**, and on
300 of 300 whose corpus was empty. At `d=0` — the whole topic present, the one supporting turn
removed — it produced an answer **300 times out of 300**.

This corroborates prior work rather than overturning it.
[[project-recall-abstention-bounded-domain-2026-07-24]] already reported that the guard works on
far gaps (1.00, 0.89) and fails on near-miss. What is new here is the **magnitude and the shape**:
on this corpus the near-miss failure is total, and the transition is a cliff at the point of
corpus emptiness rather than a curve.

## Pre-registered predictions, including the ones that missed

| | prediction | measured | |
|---|---|---|---|
| **P1** | corr-abstain(d=max) − corr-abstain(d=0) > 0.15, CI excludes 0 | **+1.000 [1.000, 1.000]** | ✅ held — **but confounded**, see above |
| **P2** | corr-abstain at d=0 below 0.25 | **0.000** | ✅ held |
| **P3** | false-abstain on answerable originals **above 0.30** | **0.000** | ❌ **missed, and badly** |
| **P4** | random-ring arm preserves P1's sign and magnitude | **not run** — see below | ⚠️ uninformative by construction |

**P3 is the most interesting line in this table.** It was set from a LongMemEval measurement of
0.481 false-abstain, and it is wrong by the entire width of the scale: RE-call false-abstained on
**zero** of 300 answerable questions here. Taken with P2, the picture is not a badly-calibrated
guard — it is a guard that, on this corpus and this configuration, **essentially never fires while
anything is indexed.**

**P4 was not run, and running it would have been theatre.** The only separation in v1 comes from
`d=max`, whose excision set is *the whole cluster* — identical under BM25 ordering and under random
ordering. A random-ring arm is therefore structurally guaranteed to reproduce the same step, at a
cost of roughly an hour of compute, while discriminating nothing. Recorded as skipped-with-reason
rather than quietly omitted.

## Two defects in the ladder, both ours

1. **`d=max` was an empty corpus.** The pre-registered headline contrast could report PASS for a
   system whose only abstention trigger is an empty index — which is precisely what happened.
2. **The other rungs were near-duplicates.** Absolute widths 0/4/16/64 against a median cluster of
   629 turns remove at most ~10 % of the topic. The powers-of-four ladder was calibrated for
   clusters an order of magnitude smaller than LOCOMO's conversations.

Neither is a defect in the system under test. Both are defects in the instrument, and they were
found by looking at the instrument's own survivor counts rather than by reading its verdict.

## Disclosure

This arm ran with **no calibration for `bge-small`**, so abstention used the untuned **0.50 cosine
floor**. That is the correct configuration under the suite's shipped-defaults rule, and it is also
the exact constant already measured as not comparable across embedders
([[project-recall-threshold-embedder-fragile-2026-07-28]]) — on some models it never fires. The
result above is consistent with a floor that never fires on this embedder, and **v1 cannot separate
"the guard has no near-miss ability" from "this threshold never engages on `bge-small`."** A
calibrated arm would separate them. That is a named open question, not a hedge.

Also unmeasured, by design: whether any *answered* question was answered **correctly**. v1 has no
judge, so `answered_answerable` counts answering as success without checking content, and every
accuracy here is an upper bound.

## What v2 changes

[`benchmarks/archive/preregistrations/PREREGISTRATION-ladder-v2.md`](../../benchmarks/archive/preregistrations/PREREGISTRATION-ladder-v2.md), written
before any v2 arm ran: the ingested slice gains **2 distractor conversations** so the top rung is a
real far gap rather than an empty index, and rungs become **fractions of the question's own
cluster** so they cannot be mis-scaled by corpus size. Its P1 is explicitly demoted to a **positive
control**, because prior work has already measured both endpoints; the untested question is the
shape between them.

v2 is a new pre-registration, not an edit to this one. This verdict stands as measured.
