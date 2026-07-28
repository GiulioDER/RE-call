# What the regime sweep establishes, and what it only suggests

Six conditions: 2 corpora (BEAM dialogue, 787 curated memos) x 3 embedders, n=100 and n=775 per
cell, no LLM involved. Raw numbers in `regime_sweep.json`; predictions written beforehand in
`PREDICTIONS-regime-sweep.md`.

## Established

| | median (memory) | median (BEAM) | range | starve @0.50 |
|---|---|---|---|---|
| bge-small | 0.852 | 0.819 | 0.284 / 0.300 | 0 % / 0 % |
| bge-large | 0.827 | 0.782 | 0.316 / 0.344 | 0 % / 0 % |
| text-embedding-3-small | 0.710 | 0.608 | 0.422 / 0.376 | 0.3 % / **16 %** |

**E1. The shipped constant is inert in five conditions and destructive in one.** 0.50 sits at the
0th percentile of five distributions and at the 16th of the sixth. `absolute@0.40` starves nothing
anywhere — spread 0.0000 — so 0.50 is not "slightly high", it is the one value that happens to cut
into a single model's range while sitting below every other model's observed minimum.

**E2. Model effect dominates corpus effect, and the two INTERACT.** Model spread is 0.142
(memory) and 0.211 (BEAM). Corpus shift is +0.033 for bge-small, +0.044 for bge-large, **+0.102**
for text-embedding-3-small — three times larger for one model than another. So a per-model offset
cannot be stored once and applied everywhere; whatever calibrates must see the actual corpus.

**E3. Lower median goes with WIDER range, monotonically across all three models.** bge-small
saturates (median 0.85, range 0.28); text-embedding-3-small spreads (median 0.61, range 0.42).

## Hypotheses, with the reasoning

### H1 — cosine is ordinal across models, not cardinal

Within one model, higher means more similar and the ordering is meaningful. Across models, 0.70
in one is not 0.70 in another: the level is an artefact of training temperature, normalisation and
loss, not a property of the pair being compared. E1 and E2 are what that looks like from outside.

**Consequence:** any absolute cosine threshold commits a cardinal comparison between quantities
that are only ordinally comparable. It is not mis-tuned — it is the wrong TYPE of statistic.

### H2 — an absolute floor penalises the better model

E3 is the uncomfortable one. A model that compresses everything toward 0.85 has ~0.28 of dynamic
range to express "how relevant"; one spread over 0.40-0.78 has ~0.42. Wider range at a lower
centre is the signature of a model that refuses to call things similar unless they are — i.e. of
better discrimination.

An absolute floor is inert against a saturating model and bites a discriminative one. **The better
your embedder, the more the shipped default costs you** — which is exactly backwards, and it is
why the defect stayed invisible: the default embedder is the one it cannot hurt.

### H3 — there are TWO problems here and they have been conflated (including by me)

**Problem A — cross-model comparability.** Solvable by any monotone rescaling: a corpus quantile,
or the calibrated logistic `Calibration.scale` already carries. Rescaling to a probability makes
thresholds portable because probabilities mean the same thing everywhere.

**Problem B — the score does not separate answerable from unanswerable.** On BEAM the unanswerable
questions score HIGHER than the answerable ones (§9h: median 0.676 vs 0.641). **No monotone
transform can fix this**, because monotone transforms preserve order and the order is what is
wrong. Calibration, quantiles, gaps — all preserve order, so all fail identically.

That decomposition explains every negative result of the last two days: the count rule (§9i), the
newest-wins dedup (§9j), the percentile and gap rules — each was an attempt to solve B with an
instrument that can only solve A. Problem B needs evidence of a different KIND, which is what the
entailment guard is and why it is the only mechanism that moved the abstention number at all.

### H4 — the floor can be derived at index time with no labels

If A is real and B is out of reach, the useful fix is narrow: stop shipping a constant, derive the
floor from the corpus after indexing. Sample K chunks, use them as queries, measure the top-1
distribution excluding each chunk's self-match, take a low quantile.

**Predicted failure mode:** self-queries are drawn from the indexed text, so the distribution will
sit HIGH relative to real user queries, and the derived floor will be too strict. The self-match
exclusion is what makes it approximately honest; whether "approximately" is close enough is the
open question, not a detail.

## What would test each

- **H1**: a fourth model in a fourth regime, with an offset not predictable from the other three.
- **H2**: measure retrieval QUALITY (not just distribution) per model — if the wider-range models
  also rank better, E3 is discrimination rather than a scaling quirk. This is the one that could
  overturn H2, because a wider range could equally be noise.
- **H3**: no experiment needed for the impossibility half — it is arithmetic. The productive half
  is measuring how much of B the entailment guard recovers on a NON-adversarial corpus.
- **H4**: cheap and next. Derive the floor from self-queries in all six conditions and compare it
  against the floor derived from the real queries already measured. Agreement within a few points
  of starve rate would make it shippable; disagreement kills it.

## What this does NOT establish

That a rate-based floor is better. `spread_quantile = 0.0123` is low and **tautological** — the
floor is computed from the same scores it is applied to, as flagged in the pre-registration before
the numbers existed. It describes; it has not been shown to generalise. H4 is the experiment that
would change that.
