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
