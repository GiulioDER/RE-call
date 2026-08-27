# Pre-registration: can any fusion of the two legs separate a hazard from a coincidence?

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_fusion_frontier.py`, committed with this record.

## The question

Draft-time search retrieves the right memo (lexical leg, 14 of 14 at top-5) and production loses
it (7 of 14), and the threshold frontier showed **no threshold can recover it**: negatives track
positives across the whole confidence range, so there is no cut. That leaves exactly one measured
lever, the one that changes the RANKING rather than relabelling it.

**Is there a weighting of the dense and lexical legs whose fused score both ranks the governing
memo into the top 5 AND separates a hazard-bearing draft from a hazard-free one?**

The second half is the one that matters. A fusion that improves recall while leaving positives and
negatives equally scored buys nothing: the threshold frontier stays flat and the direction stays
unbuildable. So this record measures recall and separability **together**, as one endpoint pair,
which is what the previous two records in this lane failed to do in different ways.

## Why the mechanism predicts this lever specifically

`recall/retriever.py` fuses with **unweighted Reciprocal Rank Fusion**, `1/(60 + rank + 1)` per
leg, no weights anywhere. RRF rewards **agreement between legs**. On a draft query the legs are not
equally informative: lexical reaches 14 of 14 and dense 7 of 14. So a document both legs mildly
like (dense 3, lexical 5 → 0.0154 + 0.0152 = 0.0306) outranks a document one leg is certain about
and the other never returns (lexical 1, dense absent → 0.0161). **Production is systematically
preferring consensus over conviction, on a query type where one leg is right and the other is
noise.** That is a precise, falsifiable account of why fused (11/14 raw, 7/14 served) sits below
lexical (14/14), and it is what a weight would correct.

## Design

**Collect once, sweep offline.** One retrieval pass captures the full top-200 ranked list and
scores from EACH leg separately for every query; every fusion variant is then computed from those
lists without touching the database. This is the discipline that closed the calibration question
in seconds (`[[sweep-the-threshold-before-refitting-a-calibration]]`): buy the data once, explore
the parameter space for free.

- **Corpus:** `probe2_control`, unchanged, via the shipped `HybridRetriever` with each leg isolated
  (`use_sparse=False` / `use_dense=False`), `candidate_k=200`.
- **Positives:** the 14 registered miss sessions and their recorded draft payloads.
- **Negatives:** the 18 `ctl-stage-by-pathspec` draft queries, whose hazard is not in this corpus.
- **Variants**, all computed from the same captured lists:
  - `unweighted_rrf` — the production baseline, `w = 1`
  - `weighted_rrf` — `w_lex/(60+rank+1) + w_dense/(60+rank+1)`, sweeping the lexical weight over
    1, 2, 3, 5, 10, 20
  - `lexical_only` — the degenerate limit, and the cheapest possible intervention: route a
    draft-shaped query to one leg
  - `score_fusion` — min-max normalised cosine and `ts_rank` combined linearly over the same grid,
    because RRF discards score magnitude and magnitude is what a threshold needs
- **Endpoints, per variant:**
  1. **recall@5** — of the 14 sessions, how many surface the governing memo in the top 5 of some
     draft query under that variant's fused ordering.
  2. **the (recall, false_trigger) frontier over that variant's fused SCORE**, swept as in the
     threshold record: `false_trigger(t)` is the share of the 18 negative queries returning any hit
     scoring >= t. Reported as: does a point exist with recall >= 9 and false_trigger <= 0.35.

  Endpoint 2 is deliberately computed on the raw fused score rather than a calibrated confidence,
  because no calibration exists for a fusion that has never been fitted, and because the question
  is precisely whether a separating score EXISTS — which is the precondition any calibration needs
  and the thing the current fusion was shown to lack.

## What I predict

Per `[[i-over-predict-effect-magnitudes]]`. Recall is a ranking change and this lane has
under-called those; separability is the thing three records have now failed to find, so it gets the
pessimistic end.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **`lexical_only` recall@5 = 13 or 14 of 14.** The screen measured 14 at `candidate_k=200`; production's 20 could cost one | 12 or fewer |
| 2 | **Best `weighted_rrf` recall@5 is 12 to 14, and rises monotonically in the lexical weight** up to the lexical-only limit | non-monotone, or best below 12 |
| 3 | ⚠️ **THE DECISIVE ONE: no variant has a point with recall >= 9 and false_trigger <= 0.35.** Lexical overlap is not relevance, and a hazard-free draft written in this repository's vocabulary matches this repository's memos however the legs are weighted | such a point exists in any variant |
| 4 | **At recall = 9, the best variant's false_trigger is 0.70 to 1.00** — an improvement on the current 1.00 is possible but small | below 0.70 |
| 5 | **`score_fusion` beats `weighted_rrf` on separability at equal recall**, because RRF throws away magnitude and a threshold needs it | it does not |
| 6 | **Cost: one retrieval pass under 25 minutes, 0.00 USD**, then seconds per variant | either bound exceeded |

## Decision rule, as an explicit cross product

Per `[[state-the-partition-over-the-cross-product]]`. Endpoint A = best recall@5 over all variants
(`<=8`, `9-11`, `>=12`). Endpoint B = lowest false_trigger achievable at recall >= 9 (`<=0.35`,
`0.36-0.70`, `>0.70`, and `n/a` when recall never reaches 9). Every cell assigned:

| A \ B | `<=0.35` | `0.36-0.70` | `>0.70` | `n/a` (recall < 9) |
|---|---|---|---|---|
| **`<=8`** | impossible | impossible | impossible | **KILL the retrieval-side lane.** Ranking was the last lever and it did not move recall |
| **`9-11`** | **BUILD** weighted fusion; preregister the live A/B | **GATE**: fusion helps, still needs a trigger discipline; preregister that, not a rollout | **KILL the fusion lever**; recall improved, separability did not, so the score cannot gate | impossible |
| **`>=12`** | **BUILD**, and route draft-shaped queries to the winning variant | **GATE**, with the winning variant named and its operating point recorded | **KILL the fusion lever**, and record that recall is solved while gating is not — the remaining question is a TRIGGER, not a retriever | impossible |

The three `>0.70` cells all kill the lever, deliberately: a retriever that finds the memo and
cannot tell a hazard from a coincidence is not deployable on a per-write trigger, whatever its
recall. That is the same standard the threshold record was held to.

## What I already know

- Per-leg recall on draft queries at top-5: lexical 14/14, fused 11/14, dense 7/14
  (`direction-screen.json`).
- The production served figure is 7/14, of which 5 are `low_confidence` on a retrieved memo and 2
  are never retrieved (`draft-precision.json`).
- The current fusion admits **no** viable threshold: 0 points with recall >= 9 and ft <= 0.35, and
  the frontier is flat (`threshold-frontier.json`).
- `recall/retriever.py` has no weight parameter of any kind, so every variant here is a proposal,
  not a configuration.

## Confounds I can name now

1. **18 negatives from 4 sessions.** Same small denominator as the threshold record. Counts are
   quoted beside every rate, and a decision resting on one or two queries is reported as "needs a
   bigger negative set" rather than as a verdict.
2. **Negatives are git-flavoured against a git-heavy corpus**, biasing false triggers upward,
   against the direction. Safe direction.
3. **A raw fused score is not a calibrated confidence.** Endpoint 2 asks whether a separating score
   exists at all, which is necessary but not sufficient for a deployable gate.
4. **Weights fitted on the same 14 sessions that define recall.** This screen deliberately reports
   the whole sweep rather than a best weight, and any BUILD outcome must fit the weight on held-out
   families before it means anything. Named here so the result cannot be read as a tuned number.
5. **`candidate_k=200` again exceeds production's 20**, so recall figures are a ceiling; a BUILD
   branch inherits the obligation to re-measure at 20.

<!-- frozen_above -->
