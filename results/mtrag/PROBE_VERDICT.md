# Verdict — MTRAGEval RB_alg probe

**Date:** 2026-08-05 · Pre-registration:
[`benchmarks/PREREGISTRATION-mtrag-rbalg.md`](../../benchmarks/PREREGISTRATION-mtrag-rbalg.md),
written before any measurement and **not edited**. This file records what happened.

**Headline: the thesis the probe was built to test is FALSIFIED, and the probe found a larger
lever than the one it was looking for.**

## 0. The invariant failed, and that was load-bearing

§1 of the pre-registration asserted that recomputing the per-model Task C harmonic mean from
IBM's published `RAG.json` must reproduce the published MTRAG table to within ±0.01. It does not:

| model | published | recomputed | delta |
|---|---|---|---|
| Target | 0.81 | 0.7947 | −0.0153 |
| GPT-4o | 0.53 | 0.5591 | +0.0291 |
| Llama 3.1 405B | 0.53 | 0.5691 | +0.0391 |
| Qwen 2.5 (72b) | 0.52 | 0.5625 | +0.0425 |
| Llama 3.1 70B | 0.52 | 0.5378 | +0.0178 |
| Mixtral 8x22B | 0.48 | 0.5230 | +0.0430 |
| Llama 3.1 8B | 0.45 | 0.4710 | +0.0210 |

**Diagnosis: the aggregation formula is right, the instance set is not the published one.**

The formula is the harmonic mean of the three per-metric means. It reproduces 5ting's
independently self-reported SemEval Task C triple to four decimals:

```
HM(RL_F 0.7692, RB_llm 0.6784, RB_alg 0.3867) = 0.5597   published 0.5597   d = -0.0000
HM(0.69, 0.92, 0.88) = 0.8169                            MTRAG-UN target, published 0.81
```

So `RAG.json` is internally valid but its absolute levels are not the paper's. **It licenses
paired, within-file comparison and nothing else.** This is
[[project-recall-token-f1-harness-offset-2026-07-29]] reproduced on a second benchmark: absolute
cross-harness levels are uncomparable, only anchored lift is. Every number below is a
within-file contrast, and none of them is a claim about parity with a published figure.

## 1. Predictions, as they landed

| | prediction | result | verdict |
|---|---|---|---|
| **P1** | median length ratio > 2.0 | **1.096** (mean 1.702) | **FALSIFIED** |
| **P2** | RougeL is the min component > 70% | **99.0%** of 7,578 | **CONFIRMED** |
| **P3** | RB_alg gap ≥ +0.10 by length band | **+0.1529** | **CONFIRMED** |
| **P4** | correct IDK scores exactly 1.0 | **120/120 = 100%** | **CONFIRMED** |
| **P5** | re-expression lifts RB_alg ≥ +0.10 | **not run, premise dead** | **WITHDRAWN** |

### P1 falsified: the models are not verbose

Median prediction 80 tokens against median target 76, ratio **1.096**. The falsification
threshold was ≤1.5 and it cleared it comfortably. **"Models write long, compress them" is wrong**,
and it was the premise the whole probe was built on.

The mean ratio of 1.702 against a median of 1.096 says the distribution has a heavy right tail,
not a shifted centre. That distinction is the whole difference between the thesis and the truth.

### P2 confirmed: RB_alg is RougeL

RougeL is the minimum of the three components in **99.0%** of instances. The other two are
BERTScore terms rescaled against a baseline and mapped through `(x+1)/2`, which parks them near
0.5 to 0.7; RougeL is a raw LCS F-measure and is not rescaled. RB_alg is a harmonic mean, so it is
governed by RougeL almost everywhere.

### P3 confirmed, but the shape refutes the reading

| length-ratio band | mean RB_alg |
|---|---|
| Q1, ratio 0.02–0.70 (too short) | 0.3617 |
| Q2, ratio 0.70–1.10 | **0.4591** |
| Q3, ratio 1.10–1.85 | 0.4374 |
| Q4, ratio 1.85–41.50 (too long) | 0.3068 |

It is an inverted U with its peak at ratio ≈ 1.0. **Being too short costs almost as much as being
too long** (0.3617 against 0.3068). So the lever is not compression, it is variance reduction
toward the reference length. And because the reference length is unknown at inference time, using
it means *predicting* the right length from the question and passages, which is a harder problem
than the one P5 proposed to solve.

**Ceiling:** if every response sat in the Q2 band, mean RB_alg would move from ~0.391 to 0.459,
about **+0.068**. Applied to MTRAG-UN's published gpt-oss-120b row that is a harmonic mean of
**0.545**, against the SemEval rank-1 of **0.586**. Length calibration alone does not win.

### P5 withdrawn, not redefined

P5 tested whether re-expressing a response at the reference length lifts RB_alg. Its premise was
P1, and P1 is dead. Running a modified P5 would be answering a different question under the old
question's name. A new lever needs a new pre-registration.

## 2. What P4 actually exposed

A correct IDK on an UNANSWERABLE task scores **exactly 1.0 on `rl_f`, `rb_llm` and `rb_agg`
simultaneously**, in 120 of 120 cells. Not "high". Exactly 1.0.

And the published baselines are very bad at it:

| model | correct abstention on UNANSWERABLE | overall HM |
|---|---|---|
| Llama 3.1 8B | **32.7%** | 0.4710 *(worst overall)* |
| Llama 3.1 70B | 29.1% | 0.5378 |
| GPT-4o-mini | 23.6% | 0.5437 |
| Command-R+ (104b) | 20.0% | 0.5502 |
| GPT-4o | 12.7% | 0.5591 |
| Llama 3.1 405B | **5.5%** | **0.5691** *(best overall)* |
| Qwen 2.5 (7b) | 5.5% | 0.5447 |
| Qwen 2.5 (72b) | 1.8% | 0.5625 |
| Mixtral 8x22B | **0.0%** | 0.5230 |

**The rank order is close to inverted.** The best model overall abstains least; the worst model
overall abstains most. Scale does not buy this capability, and on this evidence it costs it.

**Ceiling, measured on MTRAG:** setting every UNANSWERABLE cell to a correct abstention and
touching nothing else is worth **+0.0595** mean harmonic mean, on a set where UNANSWERABLE is only
**6.5%** of tasks.

On MTRAG-UN, UNANSWERABLE is **19.1%**, a 2.93× share, plus PARTIAL at 9.3% and UNDERSPECIFIED at
15.4%. A naive linear scale gives +0.174, **but linear scaling is an approximation and not a
measurement**, and MTRAG-UN is held out. The honest statement is directional: the lever is worth
several times more there than the +0.0595 measured here.

**What clears the bar,** on MTRAG-UN's published gpt-oss-120b RAG row (0.59 / 0.65 / 0.37,
HM 0.5054):

| change | HM | vs rank-1 0.586 |
|---|---|---|
| +0.05 to all three | 0.5584 | −0.028 |
| **+0.10 to all three** | **0.6110** | **+0.025** |
| +0.13 to all three | 0.6423 | +0.056 |
| RB_alg alone → 0.50 | 0.5732 | −0.013 |
| RB_alg alone → 0.53 | 0.5859 | −0.000 |

Because a correct abstention lifts all three metrics at once, it moves the harmonic mean far more
per unit of effort than anything that moves RB_alg alone.

## 3. Consequences

1. **The dominant lever in MTRAGEval is calibrated abstention, not output calibration.** It is
   also, by some distance, the thing RE-call already is.
2. **It is not a scale lever.** Llama-405B abstains 5.5%, Llama-8B abstains 32.7%. This bears
   directly on the decision to go unconstrained: the biggest lever does not appear to be bought
   with a bigger model.
3. **It is blocked on the calibration work already open in `MEMORY.md`** (`recall_calibrations`
   absent on the host, schema 0010 against migration 0011, strict trust refuses, every verdict
   `unverified`). That was named as the critical path before this probe ran, and the probe
   confirms it is the critical path.
4. **UNDERSPECIFIED (15.4% of MTRAG-UN) is scored by a separate clarification judge** and is
   excluded from these three metrics entirely. It is untouched by everything above and needs its
   own treatment.

## 4. Limits

- Everything here is measured on **MTRAG (842 human tasks)**. **MTRAG-UN was not touched** and
  remains held out.
- `RAG.json`'s absolute levels do not reproduce the published table (§0). No number here is a
  parity claim.
- The MTRAG-UN projections in §2 are arithmetic on published figures, not measurements.
- The abstention ceiling is an oracle bound. It assumes a perfect unanswerability detector and so
  is an upper limit, not a forecast. **How much of it a real detector recovers is unmeasured**,
  and that is the next pre-registration, not a conclusion of this one.
- `Answerability` was read from the task metadata for stratification and diagnosis only, per §5 of
  the pre-registration. It reached no inference path, because nothing here performs inference.
