# A query-time triage signal exists: the FLATNESS of the top-8 score curve

Measured 2026-08-16 from the frozen retrieval fixture `b6405b77…`, 500 questions, no model, no
judge. Exploratory, with the honesty protocol fixed before any number was read.

⛔ **This is hypothesis generation, not a pre-registered result.** The registered sweep in
[`PREREGISTRATION-retrieval-triage.md`](PREREGISTRATION-retrieval-triage.md) found seven scalars
inside the noise band and its T4 was FALSIFIED. That verdict stands. This is a wider search, and
searching harder over one dataset is exactly how a spurious winner is manufactured, so:

1. the 500 questions were split **train/test by a seeded hash of the question id**, ~235 each;
2. every feature was ranked on **train only**;
3. the winner is reported on **test**, which it never saw;
4. the train-minus-test gap is printed as **selection inflation** rather than hidden.

## The signal

**`ratio_8_over_1` = the retrieval score at rank 8 divided by the score at rank 1.**

| label | n positives (test) | TRAIN auc | **TEST auc** | inflation | 2 x SE | signal? |
|---|---:|---:|---:|---:|---:|---|
| `missed_any` (registered label) | 81 | 0.630 | **0.642** | **−0.012** | 0.111 | **yes**, 0.142 > 0.111 |
| `recoverable` (gold in pool, not top-8) | 48 | 0.581 | 0.632 | −0.051 | 0.144 | borderline, 0.132 < 0.144 |
| `absent` (gold not in pool) | 33 | 0.677 | 0.589 | +0.088 | 0.174 | no |

🔑 **A FLAT score curve predicts a retrieval miss.** When rank 8 scores nearly as well as rank 1,
nothing stood out and the gold document is likely outside the cut. The effect size is stable at
**0.63 to 0.64 across all three labels**; significance simply tracks the number of positives, which
is what a real effect looks like under varying power.

⚠️ **The inflation is NEGATIVE on both of the top two labels.** The feature did BETTER on data it
had never seen, which is the opposite of selection luck.

**It is a SHAPE, not a magnitude.** `top1`, `mean_top8` and every absolute score landed in the
noise band. That is why the registered sweep missed it: `score_decay` (a difference) is
scale-dependent, while a ratio is not, and the registered feature set had the difference and not
the ratio.

⚠️ **The train/test protocol earned its place here.** On the `recoverable` label the TRAIN winner
was `n_clauses` at 0.589, which collapsed to 0.538 on test (+0.051 inflation) while
`ratio_8_over_1`, which did not win on train, reached 0.632. A single-split search would have
reported the wrong feature.

## Is 0.64 useful? The budget curve, on held-out data

Ranking the 235 test questions by flatness and spending extra depth on the worst slice:

| depth budget | queries | recoverable misses caught | recall | lift vs random |
|---:|---:|---:|---:|---:|
| 10% | 23 | 7 | 14.6% | 1.46x |
| 20% | 47 | 15 | 31.2% | 1.56x |
| **25%** | 58 | 19 | **39.6%** | **1.58x** |
| 33% | 77 | 23 | 47.9% | 1.45x |
| 50% | 117 | 31 | 64.6% | 1.29x |

**Spending depth on the flattest quarter of queries catches about 40% of the recoverable misses,
1.58x better than choosing at random.** That is a real but modest lever: it is not confident
per-query routing, it is a ranking that makes a fixed budget go further.

## What this does NOT establish

- **Not that the caught rows become correct answers.** Retrieving the gold document is necessary,
  not sufficient: rows with gold fully retrieved still answer wrong 16.5% of the time.
- **Not a pre-registered result.** It needs its own registration and, ideally, a different corpus.
  A held-out split within one dataset controls selection, not dataset-specific quirks.
- **Not a replacement for `gap_warning`, which is chance (0.5015).** But it does say the shipped
  flag is measuring the wrong thing, and a ratio of ranked scores is nearly free to compute.
- **Nothing about the reranker.** This fixture has no cross-encoder, and reranking is precisely
  what would reshape the score curve this feature reads.

---

## Correction, 2026-08-16: the number stands, the MECHANISM story was wrong

Found by checking a number that should have been impossible: `ratio_8_over_1` reaches **2.726**,
meaning the item at rank 8 outscores the item at rank 1. That cannot happen in a list sorted by
score, so the list is not sorted by score.

**It is not.** All **500 of 500** ranked lists are non-monotonic in `score`. `recall/retriever.py:301`
sorts by the **RRF fused rank**, while `hit.score` carries a different quantity that the dense leg
supplied. A real example of the first ten: `0.7321, 0.7450, 0.7327, 0.6811, 0.6879, …`.

### What survives

**The empirical result is unaffected.** `ratio_8_over_1` is a real, cheap, query-time feature and it
predicts:

- whole-set AUC **0.6375**;
- across **10 independent split seeds**, test AUC mean **0.637**, stdev **0.025**, range **0.593 to
  0.681**. Every seed beats chance;
- miss rate by quintile of the feature: **19.1% → 36.2% → 46.8%**, roughly monotone through the
  middle, with an inversion at each end (Q1 22.3% above Q2, Q5 43.6% below Q4).

The budget curve is unaffected too, because it was computed from the same feature values.

### What does NOT survive

⛔ **The sentence "a FLAT score curve predicts a retrieval miss, because nothing stood out" is
withdrawn.** It described a ranking-score curve, and these values are not the ranking criterion.
The feature is the ratio of the dense score of the 8th RRF-ranked chunk to that of the 1st, which
is a different and less intuitive thing. **I do not currently have an explanation for why it
predicts.**

That distinction matters more than it might look. A signal with a mechanism generalises to other
corpora and rerankers by argument; a signal without one generalises only by measurement. Until the
mechanism is established this is an empirical regularity of this fixture, and the honest next step
is to find out what the feature is actually reading.

### Two further hazards this exposed

- **5 of 500 lists contain a NEGATIVE score.** A ratio whose denominator can be negative or near
  zero is not a well-behaved feature, and no guard exists.
- **`at()` returns 0.0 out of range.** Every list here is exactly 200 long so it never fired, but
  on a shorter list it would manufacture a ratio of 0.0.

Neither changes the measured numbers on this fixture. Both must be fixed before the feature is
computed anywhere else.
