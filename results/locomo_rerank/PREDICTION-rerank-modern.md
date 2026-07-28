# Prediction: `bge-reranker-base` vs the shipped `ms-marco-MiniLM-L-6-v2`

**Written 2026-07-27, with the arm at 2 760 / 5 882 rows indexed and `rerank_modern.json` not yet
in existence.** Committed before the result so it cannot be quietly revised afterwards.

## What is already measured (n = 1 536, artifacts row-count verified)

| k | baseline | rerank-shipped | delta | headroom left to hit@20 |
|---|---|---|---|---|
| 1 | 0.3978 | 0.5527 | +0.155 | — |
| 5 | 0.6706 | **0.7767** | **+0.106** | **0.078** |
| 20 | 0.8548 | 0.8704 | +0.016 | — |

## The prediction

**`bge-reranker-base` beats the shipped model, but modestly — the second gain will be far smaller
than the first.**

| k | shipped (known) | predicted `bge-reranker-base` | predicted delta |
|---|---|---|---|
| 1 | 0.5527 | **0.59** [0.57, 0.62] | +0.04 |
| 5 | 0.7767 | **0.80** [0.79, 0.81] | **+0.02 to +0.03** |
| 20 | 0.8704 | **0.878** [0.87, 0.89] | +0.008 |

**cat3 (multi-hop), k=5:** shipped reached 0.533. Predicted **0.55**, i.e. a *smaller* relative gain
than the other categories.

**Cost:** rerank overhead was 2 075 s for the shipped model (2 728 − 653). `bge-reranker-base` is
roughly 12× the parameters, so predicted total **7 000–11 000 s**, about **3–5×** the shipped arm.

## Reasoning

**1. The ceiling binds harder than the model gap.** Only **0.078** of headroom remains at k=5
before hit@20. The shipped model already took 57% of what was available. Whatever is left is what a
better model competes for, and the easy reorderings are the ones already made — what remains are
cases where the correct turn is genuinely hard to distinguish, which is where cross-encoders of any
size struggle.

**2. Both models perform the same operation.** This is not a mechanism change, it is a quality
upgrade to a stage that already exists. The first delta (+0.106) came from *introducing* reordering;
the second can only come from doing it *better*. Those are different orders of magnitude, which is
why I expect roughly a quarter of the first gain at most.

**3. Short documents compress the advantage of size.** LOCOMO documents are single dialogue turns —
often one sentence. A 278M-parameter model earns its capacity on long passages where there is
structure to exploit jointly with the query. Against a one-line turn there is much less for the
extra layers to read, so the parameter advantage does not convert fully.

**4. Multilingual training dilutes English performance.** `bge-reranker-base` is XLM-RoBERTa-based
and multilingual; `ms-marco-MiniLM-L-6-v2` is English-only and trained precisely on short-query /
short-passage ranking, which is structurally what this task is. Size should still win, but by less
than the parameter ratio suggests.

**5. cat3 should gain least, for an architectural reason.** A cross-encoder scores *one document*
against the query. Multi-hop questions need evidence combined across turns, which no pointwise
reranker can do at any size. §10c already located this boundary. The shipped model's +0.054 on cat3
is most likely better ranking of the single most relevant turn, not reasoning — and a bigger model
does not change what the architecture can represent.

## Where I could be wrong

**`bge-reranker-base` could be WORSE — I put this at roughly 20–25%.** Multilingual models
underperform English-only specialists on short English text more often than parameter counts
suggest, and MS MARCO training is unusually well matched to this exact shape of task. If that
happens, it is a genuine finding rather than a disappointment: it would say the shipped default is
already the right choice and that reranker selection here is about *task match*, not model size.

**A large gain (>+0.05 at k=5) would falsify my reasoning about the ceiling** and mean the residual
misses are more separable than I assumed.

## Decision rule, fixed in advance

| outcome at k=5 | recommendation |
|---|---|
| bge beats shipped by **≥0.02** with disjoint CIs | switch the default, and publish the latency cost beside it |
| gain **<0.02** or CIs overlap | **keep the shipped default** — smaller, faster, already revision-pinned |
| bge **worse** | keep shipped; report it as evidence that task match beats model size here |

Stating this now removes the temptation to justify whichever number arrives.

---

# Result — scored 2026-07-27

Artifacts row-count verified (5 882 each) before any number was read.

| k | shipped | **predicted bge** | **actual bge** | predicted delta | actual delta |
|---|---|---|---|---|---|
| 1 | 0.5527 | 0.59 [0.57, 0.62] | **0.5645** | +0.04 | **+0.012** |
| 5 | 0.7767 | 0.80 [0.79, 0.81] | **0.7734** | +0.02 to +0.03 | **−0.003** |
| 20 | 0.8704 | 0.878 [0.87, 0.89] | **0.8724** | +0.008 | +0.002 |

CIs overlap heavily at every depth. Cost **13 842 s vs 2 728 s = 5.1× the shipped arm.**

## Verdict: WRONG on the number, and wrong in an instructive direction

The actual k=5 value (**0.7734**) falls **outside** my committed interval [0.79, 0.81], and on the
wrong side: I predicted a modest *gain* and the measurement shows **no gain at all** — marginally
negative, comfortably inside noise.

**`bge-reranker-base` is not better than the shipped model on this task. It is 5.1× slower for
nothing.**

## Was the reasoning right for the right reason? Partly — and the failure is the interesting part

All four arguments were **directionally correct**, and every one of them argued for *compression* of
the advantage:

1. ceiling binds harder than the model gap ✓
2. quality upgrade to an existing stage, not a new mechanism ✓
3. single-turn documents give a 278M model little extra to read ✓
4. multilingual backbone dilutes English specialisation ✓

**And I still predicted +0.02 to +0.03.** Four arguments all pointing at "the advantage will not
materialise", and I converted them into a positive number anyway. An unstated prior — *a 12×
larger, more recent model must be somewhat better* — survived reasoning that had already refuted it.

That is the lesson, and it is more useful than the measurement: **the arguments were right and I
did not follow them to their conclusion.** Predicting ~0.00 would have been the honest reading of my
own case. Writing the reasoning down is what makes this visible instead of comfortable.

The stated falsifier did fire: I gave "bge is WORSE" a **20–25%** chance, and that is what happened.
Correct calibration, then discarded when I wrote the point estimate.

## The sub-prediction I got exactly backwards

I predicted **cat3 would gain LEAST**, arguing that a pointwise cross-encoder cannot combine
evidence across turns at any size, so extra capacity could not help multi-hop.

**cat3 gained MOST** — `+0.022`, the only category with a meaningful positive, while cat2 and cat4
went slightly *negative*.

So the §10c boundary is real but I over-applied it. Multi-hop questions still benefit from *better
single-document scoring*: getting the one most-relevant turn ranked correctly helps even when the
full answer needs several. "The architecture cannot represent multi-hop reasoning" does not imply
"nothing about a better reranker helps multi-hop questions" — I collapsed those two claims.

## Cost prediction: correct

Predicted 3–5× the shipped arm; actual **5.1×**, at the top of the range.

## Decision, by the rule fixed in advance

> gain <0.02 or CIs overlap → **keep the shipped default**

**Keep `ms-marco-MiniLM-L-6-v2`.** It is 5.1× faster, already revision-pinned, and statistically
indistinguishable from a model 12× its size. The rule was written before the data, so this
recommendation is not reverse-engineered from it.

Usefully, this also answers the question the second arm existed for: reranking's large gain is
**not** an artefact of one particular cross-encoder. Two independent models, 12× apart in size and
four years apart in vintage, land within noise of each other — so the +0.106 belongs to *reranking*,
not to a model choice.
