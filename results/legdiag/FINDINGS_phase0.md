# Phase 0 diagnostic — leg disagreement does not select for retrieval failures

**Date:** 2026-07-28 · **Verdict: the preregistered trigger is dead. Phase 2 (PRF) is not built.**
**Design:** [`2026-07-28-weighted-fusion-prf-phase0-design.md`](../../docs/superpowers/specs/2026-07-28-weighted-fusion-prf-phase0-design.md)
(predictions and decision rules committed in `ef68bb1`, before this run existed)
**Artifact:** [`locomo_phase0.json`](locomo_phase0.json) — per-question records behind every figure below.

## Reproduce

```bash
python -m recall.eval.legdiag --data locomo10.json --dsn "$RECALL_DSN" --out results/legdiag/locomo_phase0.json
```

`locomo10.json` is gitignored and fetched on demand — see the `curl` command in
`recall/eval/locomo.py`'s module docstring for the exact source URL.

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

Two-proportion z-test across the two groups (pooled proportion, the test the design preregistered):
**z = 3.79, p = 1.5×10⁻⁴**.

**The gate:** *firing-group hit@5 ≥ non-firing group → stop, no PRF.* It fired, with the sign
reversed and the confidence intervals disjoint.

### The confound controls clear it

Two confounds could produce a spurious pooled delta without the trigger meaning anything: sparse-leg
depth (the residual bias the design amendment measured) and LOCOMO category (a much stronger
difficulty proxy than either). Both are checked directly against the artifact.

**Sparse-leg depth: not merely controlled for, structurally absent in the bin that carries the
finding.** `more_decisive`'s residual bias comes from truncating a *larger* pool to its top `m` —
order statistics from a truncated draw cluster more tightly near the maximum than a fresh
`m`-sized draw (design amendment). Every record has `n_dense == 20` (all 1536) and `n_sparse == 20`
(1527 of 1536), so in that dominant bin `m = min(20, 20) = 20` and **neither leg is truncated** —
both legs are scored over their full native depth, not a truncated one:

| sparse-leg depth | n | firing | not firing | delta |
|---|---|---|---|---|
| n_sparse ≥ 20 | **1527** | 0.706 (n=847) | 0.615 (n=680) | **+0.0913** |
| n_sparse 10–19 | 4 | 1.000 (n=4) | — | n/a |
| n_sparse 5–9 | 1 | 1.000 (n=1) | — | n/a |
| n_sparse 0–4 | 4 | 1.000 (n=1) | 1.000 (n=3) | +0.0000 |

_A reading note for anyone querying `locomo_phase0.json` directly rather than reading this table:
`_rate` returns `{"rate": 0.0, "n": 0}` for an empty group, so an empty not-firing cell (the "—"
above, e.g. `n_sparse 10–19` and `n_sparse 5–9`) shows up in the raw JSON as `"rate": 0.0` — that is
"no data", not a measured 0% hit rate. Always read `rate` together with `n`._

**+0.0913 within the dominant bin against +0.0917 pooled.** In the bin holding 99.4% of the data
the truncation mechanism that produces the residual bias cannot fire at all — the delta there is
not a controlled-for confound, it is measured with the confound structurally absent. That is a
stronger claim than "the control clears it", and it is the strongest defence available for this
result, not a weakness to apologize for.

**Category: a second, independent confound, also checked.** cat4 is 841/1536 (54.8%) of the corpus
and both the easiest-scoring category on average and the highest-firing one (0.599) — a setup where
a pooled effect could be entirely "the trigger fires more on the easy category" rather than leg
disagreement doing anything. Per-category deltas, computed from the artifact:

| category | n | firing hit@5 | not-firing hit@5 | delta |
|---|---|---|---|---|
| cat1 | 282 | 0.653 (n=124) | 0.608 (n=158) | +0.046 |
| cat2 | 321 | 0.747 (n=182) | 0.669 (n=139) | +0.078 |
| cat3 | 92 | 0.535 (n=43) | 0.449 (n=49) | +0.086 |
| cat4 | 841 | 0.722 (n=504) | 0.623 (n=337) | +0.099 |
| **category-adjusted (n-weighted)** | 1536 | — | — | **+0.0841** |
| pooled | 1536 | 0.708 (n=853) | 0.616 (n=683) | +0.0917 |

Holding category fixed, the delta is **+0.0841** against **+0.0917** pooled — most of the pooled
effect survives, so category is not driving it either. Per conversation (n=10) the delta is
positive in **8**, and the two exceptions are flat rather than reversed: conv-41 at **−0.004** and
conv-49 at **−0.002**. **The direction never inverts anywhere in the data** — not across sparse-depth
bins with enough data to read, not across categories, not in all but two of ten conversations, and
those two sit at zero within noise, not on the other side of it.

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

**Not run: the private-46 arm.** It is in the design's measurement set (`design.md`, "What Phase 0
measures") and does not appear above. That is correct sequencing, not an omission: the design
sequences it **after** the LOCOMO arm clears its gates, because LOCOMO is public, n=1,536, and
decides. The LOCOMO gate fired, so the private-46 arm was correctly not run.

**Not tested: rank disagreement.** An honest limitation, stated without using it to rescue the
result. The design's *concept* was "sparse found something dense buried" — a statement about the
two legs ranking **different documents**. What shipped, `more_decisive`, compares how **peaked**
each leg's score distribution is. Those are different quantities, and the peakedness version is
now falsified. A set-overlap or rank-displacement measure is a **new hypothesis needing its own
preregistration**, not a second chance for this one — and Q3 bounds what it could ever buy at
`b_unretrieved` = **162/1536 (10.5%)**, since anything already in the pool is fusion's job, not a
second retrieval's. The retained records carry per-leg counts and confidences but not the leg id
lists, so this cannot be answered from this artifact; it would need another run.

Part of that gap is not an inherent limit of what was measured. The design (`design.md:140`)
specified that bucket (a) be **sub-split by which leg held it** — dense-only, sparse-only, or both.
That sub-split was never implemented: `classify_gold` pools `_retrieved_dia_ids(probe.dense)` and
`_retrieved_dia_ids(probe.sparse)` into one undifferentiated `pool` set before checking membership,
so a misranked gold chunk's per-leg origin is discarded, not merely unrecorded. Stated plainly: the
inability to answer the rank-disagreement question from this artifact is **partly a dropped
deliverable**, not purely an inherent limit of the design.

## Cost

One CPU afternoon, `$0` in API spend, no production behaviour changed. The alternative was
building weighted fusion, then PRF, then discovering the trigger fires on the wrong half.
