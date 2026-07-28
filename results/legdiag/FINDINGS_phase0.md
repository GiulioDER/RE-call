# Phase 0 diagnostic — leg disagreement does not select for retrieval failures

**Date:** 2026-07-28 · **Verdict: the preregistered trigger is dead. Phase 2 (PRF) is not built.**
**Design:** [`2026-07-28-weighted-fusion-prf-phase0-design.md`](../../docs/superpowers/specs/2026-07-28-weighted-fusion-prf-phase0-design.md)
(predictions and decision rules committed in `ef68bb1`, before this run existed)
**Artifact:** [`locomo_phase0.json`](locomo_phase0.json) — per-question records behind every figure below.

## Reproduce

```bash
python -m recall.eval.legdiag --data locomo10.json --dsn "$RECALL_DSN" --out results/legdiag/locomo_phase0.json
```

LOCOMO, 10 conversations, `bge-small`, hybrid, pool 20 per leg, no rerank — the §9a configuration.

## The apparatus check passed

Asserted in code before any figure was computed:

| quantity | measured | expected | tolerance |
|---|---|---|---|
| answerable questions | **1536** | 1536 | exact |
| hit@5 | **0.6673** | 0.671 (§9a) | ±0.01 |
| hit@20 | **0.8574** | 0.855 (§9a) | ±0.01 |

The instrumented pipeline is the one §9a measured. A second, independent guard ran on every
question: `classify_gold`'s bucket was cross-checked against the harness's separately-computed
`hit`, and a disagreement would have aborted the run. It never fired. (That guard was itself
verified to bite — inverting its comparison aborts a run on the first question.)

## Q1 — does leg disagreement select for failures? **No. It selects for successes.**

| group | hit@5 | 95% Wilson | n |
|---|---|---|---|
| trigger fired | **0.7081** | [0.6767, 0.7376] | 853 |
| trigger did not fire | **0.6164** | [0.5794, 0.6521] | 683 |
| **delta** | **+0.0917** | intervals disjoint | 1536 |

**The gate:** *firing-group hit@5 ≥ non-firing group → stop, no PRF.* It fired, with the sign
reversed and the confidence intervals disjoint.

### The confound control clears it

The binding control from the design amendment — a gap that exists only pooled and vanishes inside
every depth bin would be sparse-leg depth talking, not the trigger:

| sparse-leg depth | n | firing | not firing | delta |
|---|---|---|---|---|
| n_sparse ≥ 20 | **1527** | 0.706 (n=847) | 0.615 (n=680) | **+0.0913** |
| n_sparse 10–19 | 4 | 1.000 (n=4) | — | n/a |
| n_sparse 5–9 | 1 | 1.000 (n=1) | — | n/a |
| n_sparse 0–4 | 4 | 1.000 (n=1) | 1.000 (n=3) | +0.0000 |

**+0.0913 within the dominant bin against +0.0917 pooled.** The effect is not the sample-size
residual. It is real.

Incidentally the control turned out to be nearly moot *on this corpus*: 1527 of 1536 questions
have a full 20-candidate sparse leg, so the depths barely varied. That is a property of LOCOMO —
its documents are single conversational turns and its questions share common vocabulary — and
should not be assumed on a corpus with rarer terms. The control stays.

## Q2 — firing rate: **0.5553** [0.5304, 0.5800], n=1536

Outside the predicted 15–35%, and outside the 5–50% usable band. The gate fires here too.

By category: cat1 0.440 (n=282) · cat2 0.567 (n=321) · cat3 0.467 (n=92) · cat4 0.599 (n=841).
Notably cat3 — multi-hop, the weakest category — is **not** where the trigger concentrates.

## Q3 — where the gold chunk actually was

| bucket | all scored | share | among the 249 firing misses |
|---|---|---|---|
| `hit` (inside top-5) | 1025 | 66.7% | — |
| `a_misranked` (in the pool, below k) | **349** | 22.7% | 180 (72.3%) |
| `b_unretrieved` (in neither leg's pool) | **162** | 10.5% | 69 (27.7%) |

`n_excluded_unlabelled` is 0, and per the module docstring that is **structurally guaranteed via
this CLI**, not a clean bill of health for the labels — the harness filters unlabelled questions
upstream before the diagnostic sees them.

## Prediction scorecard

All three preregistered predictions missed. They are quoted from `ef68bb1`.

| # | prediction | outcome | verdict |
|---|---|---|---|
| Q1 | "firing-group hit@5 lower by ≥0.10 absolute" | **+0.0917, opposite sign** | ❌ MISS |
| Q2 | "LOCOMO 15–35%" | **55.5%** | ❌ MISS |
| Q3 | "roughly 30–50% into bucket (a)" | **72.3%** of firing misses | ❌ MISS |

The Q1 miss is the one that matters, and the design doc named this exact outcome in advance:

> **Q1 falsification path, and it is live.** Sparse may be decisive precisely *because* it matched
> a rare term — a name, a number, a codename — in which case the firing group has *higher* hit@5,
> the trigger selects for successes, and the design is dead. This is the single load-bearing
> assumption of the whole proposal and the cheapest thing in it to check. It is stated here so
> that outcome cannot be reinterpreted as a partial win afterwards.

It is not being reinterpreted. The mechanism is the one predicted: a decisive lexical leg means
the question carried a sharp lexical signal, and those questions are *easy*. Firing on them spends
latency where retrieval already works, and stays silent on exactly the queries that need help.

## What this does and does not close

**Closed: PRF gated on this trigger.** Firing on the easy half is worse than useless — it inverts
the entire latency argument, which was that the second pass would cost nothing on confident
queries.

**Not closed: weighted fusion (Phase 1).** It never depended on the trigger. Its target is
measured here and is substantial: **349 questions (22.7%) have the gold chunk inside the fused
candidate pool but ranked below 5.** §9a's mechanism — symmetric RRF diluting a deeper pool —
remains diagnosed and unaddressed.

**Not tested: rank disagreement.** An honest limitation, stated without using it to rescue the
result. The design's *concept* was "sparse found something dense buried" — a statement about the
two legs ranking **different documents**. What shipped, `more_decisive`, compares how **peaked**
each leg's score distribution is. Those are different quantities, and the peakedness version is
now falsified. A set-overlap or rank-displacement measure is a **new hypothesis needing its own
preregistration**, not a second chance for this one — and Q3 bounds what it could ever buy at
`b_unretrieved` = **162/1536 (10.5%)**, since anything already in the pool is fusion's job, not a
second retrieval's. The retained records carry per-leg counts and confidences but not the leg id
lists, so this cannot be answered from this artifact; it would need another run.

## Cost

One CPU afternoon, `$0` in API spend, no production behaviour changed. The alternative was
building weighted fusion, then PRF, then discovering the trigger fires on the wrong half.
