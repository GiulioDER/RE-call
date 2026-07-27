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
