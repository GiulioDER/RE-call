# Predictions for the regime sweep — written BEFORE the results

Committed while the sweep is running and before any condition has reported. The git timestamp
against `regime_sweep.json`'s mtime is the evidence. The point is not to be right; it is that a
wrong prediction here tells us the reasoning behind §9l was wrong, which is worth more than the
sweep itself.

## What is already measured (the basis for predicting the rest)

| condition | min | q05 | median | max |
|---|---|---|---|---|
| BEAM / bge-small (3 conv, n=60) | 0.629 | 0.728 | **0.825** | 0.939 |
| BEAM / text-embedding-3-small (15 conv, n=300) | 0.403 | 0.498 | **0.635** | 0.852 |

## Prediction 1 — bge-large clusters with bge-small, NOT with the cloud model

**Median on BEAM: 0.78–0.86.** Nowhere near 0.635.

Reasoning: bge-small and bge-large are the same family — same lab, same contrastive recipe,
same normalisation. Cosine calibration is a property of how a model is trained, not of its size.
A bigger model discriminates better, which widens the distribution *downward* for non-matches,
but top-1 is the BEST match and should stay high.

**This is the discriminating prediction of the whole sweep.** If the split were really "local vs
cloud" — a plumbing artefact rather than a model property — bge-large would land near the cloud
model. If it lands at ~0.65, §9l's reasoning is wrong and the embedder-family explanation dies.

## Prediction 2 — everything shifts UP on the memory corpus, order preserved

**bge-small ~0.88–0.93 · bge-large ~0.88–0.93 · text-embedding-3-small ~0.70–0.80.**

Reasoning: the memory queries are each memo's own `description:` line, so a query nearly
duplicates text inside its target. Near-duplicate matching saturates every model. The absolute
levels are therefore inflated and NOT comparable to BEAM — but the ordering between embedders
should survive, because the inflation applies to all three equally.

If the ordering flips on this corpus, then corpus dominates embedder and the framing of §9l
("the constant is embedder-fragile") is the wrong frame — it would be corpus-fragile too, which
is a different and larger problem.

## Prediction 3 — the headline spreads

**`absolute@0.50`: spread 0.05–0.09** (roughly 0 % starved in five conditions, ~6–7 % in
BEAM/cloud). The damage concentrates in one cell because it is the only one whose distribution
straddles 0.50; everywhere else the floor sits below the observed minimum and does nothing.

**`quantile@0.05`: spread 0.00–0.02.**

## ⚠ The weakness in my own design, stated before the numbers

Prediction 3's second half is **largely tautological**. In this script the quantile floor is
computed from the same scores it is then applied to, so it starves ~5 % by construction. A near-
zero spread there is arithmetic, not evidence.

The honest version fits the floor on a HELD-OUT sample of queries and applies it to a disjoint
one; only then does a low spread mean the rule *generalises* rather than *describes*. I did not
build that, and I am recording the flaw now rather than presenting the result as stronger than it
is. **Treat spread_quantile as unproven regardless of what it shows.**

The half that IS informative is `spread_absolute`: nothing tautological about it, since 0.50 is
fixed in advance and knows nothing about any of the six distributions.

## What would make me abandon the third intervention

- bge-large lands near the cloud model → not an embedder-family effect, §9l reasoning wrong.
- All six conditions starve at similar rates under 0.50 → the constant is not fragile, and the
  problem exists only in the one cell we happened to hit.
- The memory corpus reverses the ordering → corpus, not embedder, is the dominant variable.
