# The Answerability Ladder v2 — verdict

**Date:** 2026-07-29 · **System:** RE-call at shipped defaults · **Corpus:** LOCOMO + 2 distractor
conversations per question
**Manifest:** `results/ladder/manifest_v2.jsonl`, digest
`5534c61356acaa7b62ac5a79dbec7383674fc052984d10c1d0cc89e26a532bd5`, 1 200 instances (200 × 6)
**Responses:** `results/ladder/responses_v2.jsonl`, 1 200 scored, none missing
**Pre-registration:** [`benchmarks/PREREGISTRATION-ladder-v2.md`](../../benchmarks/PREREGISTRATION-ladder-v2.md) + its dated pre-run addendum
**Predecessor:** [`H1_VERDICT.md`](H1_VERDICT.md) (v1)

**Prior work searched** before the design was fixed — see the pre-registration's own header.
[[project-recall-abstention-bounded-domain-2026-07-24]] had already measured both endpoints of this
ladder, which is why P1 was demoted to a positive control rather than presented as a discovery.

---

## 1. Pre-registered verdict: FAIL — and the addendum said so before the run

```
rung           n  corr-abst  false-ans  false-abst  surv-docs
original     200      0.000          0           0       1774
r=0.00       200      0.000        200           0       1773
r=0.25       200      0.000        200           0       1669
r=0.50       200      0.000        200           0       1537
r=0.75       200      0.000        200           0       1367
r=1.00       200      0.005        199           0       1197

H1 (pre-registered) paired Δ(correct-abstain), r=1.00 − r=0.00: +0.005 [+0.000, +0.015]  n=200
H1: FAIL
```

**v1's confound is gone.** Every rung's ingested slice is non-empty — 1 774 documents at the
original down to 1 197 at `r=1.00`, where the question's whole conversation has been replaced by
distractor conversations. The far-gap rung is now a real far gap.

**The flatness is the threshold, not the axis.** The v2 pre-registration's addendum, committed
*before this arm ran*, predicted this FAIL and named its cause: a 3-question probe measured top-1
cosine at 0.7319 (`r=0.00`) against 0.5997 (`r=1.00`), while the shipped abstention floor sits at
**0.50** — below the entire distribution. A threshold that never fires produces a flat column
whatever the underlying signal does.

The harness prints a boilerplate line on FAIL — *"a flat curve means excision distance is not the
axis this benchmark claimed it was"*. **For v2 that line is wrong**, and it is wrong for a reason
recorded in advance rather than discovered afterwards. Section 2 is why.

## 2. The finding: the underlying score IS graded, and it is not close

Because the harness records top-1 cosine per response, the abstention decision is not the only
thing measurable. Paired change in top-1 cosine against each question's own `r=0.00`, bootstrap
95 % CI, n=200:

| rung | Δ top-1 cosine vs `r=0.00` | 95 % CI |
|---|---|---|
| `r=0.25` | **−0.0397** | [−0.0449, −0.0344] |
| `r=0.50` | **−0.0539** | [−0.0598, −0.0481] |
| `r=0.75` | **−0.0837** | [−0.0903, −0.0772] |
| `r=1.00` | **−0.1100** | [−0.1172, −0.1028] |

Monotone, every CI excluding zero, and non-increasing **per question** far more often than chance.
**Correction, 2026-07-29 (post-publication audit):** the original text of this section reported
these four per-step counts as "decreases" — they are **non-increases** (`<=`), which is not the
same claim: a flat top-1 cosine between two rungs counts toward "non-increase" but is not a
decrease. Both columns, recomputed from `results/ladder/manifest_v2.jsonl` +
`responses_v2.jsonl`, n=200:

```
step             strict decrease   ties   non-increase (<=)
r=0.00 -> 0.25         161          30          191 = 0.955
r=0.25 -> 0.50         114          78          192 = 0.960
r=0.50 -> 0.75         144          49          193 = 0.965
r=0.75 -> 1.00         109          87          196 = 0.980
monotone (non-increasing) across ALL FIVE rungs: 173/200 = 0.865
```

The tie rate is itself informative: at deeper rungs the top-1 retrieved document frequently does
not change at all — at the last step, 87 of the 196 non-increases (44 %) are flat ties, not actual
decreases, so the "0.980" figure overstates how often the score is still moving by that point.

**This is the benchmark's central claim, and it holds.** Answerability is not binary and not a
step: it is a graded function of how far the question sits from what the corpus contains. Existing
benchmarks sample one point of that function and report a scalar, which is why BEAM and LOCOMO
disagree about abstention.

**It also falsifies P2, which was my own prediction.** P2 said the curve would be a *step* — that
the guard engages only when the topic is essentially gone — and predicted it from v1's evidence.
The score says otherwise: the decline is smooth and begins at the very first rung, where only 25 %
of the topic has been removed.

## 3. The threshold sweep

Since abstention at threshold *t* is exactly `top_cosine < t`, one arm yields the whole family.
**This is a curve, not a recommendation.** No threshold below is proposed, and the shipped 0.50 is
marked; picking the row that maximises the effect after seeing this table would not be a result,
and adopting any of them needs its own pre-registration and its own held-out arm.

```
   thr   r=0.00   r=0.25   r=0.50   r=0.75   r=1.00  FALSE-ABST   Δ(1.00−0.00)
  0.50    0.000    0.000    0.000    0.000    0.005       0.000          0.005   <- shipped
  0.54    0.000    0.000    0.000    0.020    0.095       0.000          0.095
  0.58    0.000    0.015    0.030    0.150    0.265       0.000          0.265
  0.60    0.000    0.055    0.105    0.265    0.430       0.000          0.430
  0.62    0.015    0.090    0.170    0.395    0.630       0.005          0.615
  0.64    0.070    0.220    0.335    0.565    0.765       0.045          0.695
  0.66    0.150    0.385    0.490    0.735    0.875       0.085          0.725
  0.68    0.220    0.525    0.635    0.855    0.960       0.150          0.740
  0.70    0.355    0.700    0.780    0.930    0.980       0.260          0.625
```

Two things worth naming, neither of which is a threshold proposal:

- The **monotone gradient across rungs is visible at every threshold where anything fires at all**,
  which is the same claim as §2 seen through a different instrument.
- There is a region — up to about 0.60 — where correct-abstention on the far rung rises while
  **false-abstention on the answerable originals stays at 0.000**. That the two are separable at
  all is the interesting part; where exactly to sit is not something this arm can decide.

## 4. Pre-registered predictions

| | prediction | measured | |
|---|---|---|---|
| **P1** (positive control) | corr-abstain(r=1.00) − corr-abstain(r=0.00) > 0.15 | **+0.005** | ❌ failed — **predicted in advance**, cause named: threshold below the distribution |
| **P2** | the curve is a **step**, not a gradient | monotone gradient, 173/200 per-question | ❌ **falsified — my own prediction, wrong** |
| **P3** | false-abstain on originals **below 0.10** | **0.000** | ✅ held |
| **P4** | random-ring arm preserves P1 | **not run** | ⚠️ see below |

**P1's failure does not mean what the pre-registration originally said it would.** It was written
as "a FAIL means the harness is broken". The pre-run addendum corrected that reading *before* the
arm: the harness is fine, the axis is fine, and the shipped threshold is below the whole score
distribution on this embedder. That correction is dated and committed ahead of the result.

**P4 was not run.** With abstention flat at the shipped threshold, a random-ring arm would compare
two flat columns and discriminate nothing. The claim it guards — that BM25 ordering is not a
confound — is worth testing against the *cosine* curve rather than the abstention curve, and that
is a v3 question. Recorded as skipped-with-reason, as in v1.

## 5. Limits

- **One system, one embedder.** Everything here is RE-call on `bge-small`. The gradient is a
  property of that pairing until another system is run through the same manifest.
- **Uncalibrated by design.** No calibration exists for `bge-small`, so the arm used the untuned
  0.50 floor — correct under the suite's shipped-defaults rule, and the exact constant already
  measured as not comparable across embedders
  ([[project-recall-threshold-embedder-fragile-2026-07-28]]).
- **Distractors are same-domain.** They are other LOCOMO conversations: unrelated in topic, but
  identical in genre, register and speaker style. `r=1.00` is a far gap *within one domain*, not a
  far gap in general.
- **No judge.** Whether an answered question was answered *correctly* is unmeasured, so every
  accuracy here is an upper bound.
- **The excision ordering is BM25**, and its neutrality is untested against the cosine curve (P4).

## 6. What v1 and v2 together establish

v1 reported H1 PASS and meant nothing by it: its top rung had an empty index. v2 reports H1 FAIL
and the failure is entirely a threshold artefact. **Neither pre-registered verdict carries the
finding**, and in both cases the instrument's own diagnostics — survivor counts in v1, recorded
cosines in v2 — carried it instead.

The substantive result is that **excision distance moves retrieval confidence smoothly and
measurably (−0.11 cosine end to end, monotone per question in 173 of 200 cases), while the shipped
abstention decision is blind to all of it.** That gap between a graded signal and a binary decision
is the thing worth publishing, and it is not visible to any benchmark that reports only whether a
system abstained.
