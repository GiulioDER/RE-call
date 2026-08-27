# Pre-registration: can ANY threshold make draft-time search usable?

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_threshold_frontier.py`, committed with this record.

## The question

The precision record found that 5 of 7 production failures were the trust layer marking a
correctly-retrieved memo `low_confidence`, and that 0 of 18 hazard-free draft queries abstained.
The licensed next step was "one calibration run". This record asks the question that must come
first and costs nothing:

**Is there ANY confidence threshold at which draft-time search has acceptable recall AND an
acceptable false-trigger rate at the same time?**

## Why this precedes a recalibration, and why it can settle it

A calibration in this system fits a score-to-confidence mapping and a threshold on it. **A monotone
remapping cannot change which hits outrank which** — it relabels the axis. So for a fixed retrieval
ranking, the achievable (recall, false-trigger) pairs are exactly those a threshold sweep traces
out, and **no recalibration can reach a point off that frontier.**

That makes this sweep decisive rather than indicative: if no point on the frontier is acceptable,
refitting the calibration cannot help, and the only remaining levers are ones that change the
RANKING (fusion weights, a reranker, a different query) rather than the label on it. If a good
point does exist, the recalibration becomes a well-posed engineering task with a target.

It also costs nothing: it re-analyses `benchmarks/artifacts/agent_ab/draft-precision.json`, which
already records a numeric confidence for all 2,270 positive and 90 negative hits. No retrieval, no
model, no build.

⚠️ **Disclosed before predicting:** I have seen the verdict labels (`ok`, `low_confidence`) and the
aggregate counts from the precision run — recall 7 of 14, false triggers 18 of 18 at the CURRENT
threshold. I have deliberately NOT looked at the confidence distributions, the per-hit values, or
any relationship between them. The predictions below are anchored on those two aggregates only.

## Design

- **Data:** the committed precision artifact, unchanged. 14 registered miss sessions (454 positive
  draft queries across 48 sessions, of which the 14 are scored), 18 negative draft queries.
- **Sweep:** threshold `t` over confidence, from 0.00 to 1.00 in steps of 0.01.
- **recall(t)** = of the **14 registered miss sessions**, how many have their governing memo in
  some draft query's top-5 with `confidence >= t`. Per SESSION: a rescue need happen only once.
- **false_trigger(t)** = of the **18 negative draft queries**, how many return at least one hit
  with `confidence >= t`. Per QUERY: the agent pays this cost on every write.

  **The asymmetry is deliberate and favours the direction**: recall gets credit for any one of a
  session's ~10 searches, while noise is charged per search. Both are also reported the other way
  round so the choice cannot hide a result.
- **Reported:** the full frontier, the current operating point, and the best achievable recall at
  each false-trigger ceiling.

## What I predict

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **No threshold achieves recall >= 9 with false_trigger <= 0.35.** At the current point false triggers are 18 of 18, so the negatives' top hits must already be scoring high; pushing recall up can only lower the bar further | such a point exists |
| 2 | **At the threshold where recall first reaches 9, false_trigger >= 0.80** | below 0.80 |
| 3 | **Driving false_trigger to <= 0.35 costs recall <= 4 of 14** | recall above 4 there |
| 4 | **Maximum achievable recall over the whole sweep is 11 to 14 of 14** — at t=0 every hit counts, so this is bounded by whether the memo is in the top-5 at all, which the screen measured at 11 of 14 fused | outside that band |
| 5 | **Cost: under 2 minutes, 0.00 USD, no retrieval** | either bound exceeded |

## Decision rule, as a CROSS PRODUCT this time

Written per `[[state-the-partition-over-the-cross-product]]`, because the previous record in this
lane wrote "full partition" over a rule with a hole and the result landed in it. Bands: recall at
the best point with false_trigger <= 0.35 (`<=4`, `5-8`, `>=9`) crossed with maximum achievable
recall at any threshold (`<=8`, `9-11`, `>=12`). Nine cells, every one assigned:

| max recall \ recall at ft<=0.35 | `<=4` | `5-8` | `>=9` |
|---|---|---|---|
| **`<=8`** | KILL the threshold lever AND the direction on this corpus | KILL the lever, direction stays unproven | impossible by construction |
| **`9-11`** | KILL the threshold lever; escalate to RANKING (weighted fusion, reranker) | GATE: build a trigger discipline, not a threshold | BUILD, preregister the live A/B |
| **`>=12`** | KILL the threshold lever; escalate to RANKING, with a strong prior it will pay | GATE, and recalibrate to the identified point | BUILD, preregister the live A/B |

"KILL the threshold lever" means precisely that a recalibration is not licensed, because the
frontier argument above shows it cannot reach a better point. It says nothing about the ranking
levers, which the cells route to explicitly.

## What I already know

- Current operating point: recall 7 of 14, false_trigger 18 of 18, judged precision 0.253 on
  misses and 0.056 actionable on negative slots.
- 5 of the 7 current failures are `low_confidence` on a retrieved memo; 2 are never retrieved at
  all, and those 2 are **outside** what any threshold can fix — they bound `max recall` below 14.
- The lexical leg reaches 14 of 14 and production fused reaches 11 of 14 raw
  (`direction-screen.json`), so the ranking lever has measured headroom the threshold lever does
  not.

## Confounds I can name now

1. **18 negative queries, from 4 sessions.** A rate over 18 is quoted as a count beside every
   percentage, and a frontier drawn through 18 points is coarse. If the decision turns on a
   difference of one or two queries, the honest outcome is "needs a bigger negative set".
2. **The negatives are git-flavoured** and this corpus is full of git memos, which biases false
   triggers upward, i.e. against the direction. Safe direction for a confound.
3. **The frontier is conditional on THIS ranking.** It bounds thresholds, not rerankers, not
   fusion weights, not a different query formulation. The decision rule routes to those explicitly
   rather than treating a threshold null as a verdict on the direction.
4. **Confidence is calibrated, not raw.** The monotonicity argument holds for any monotone
   recalibration; a non-monotone one would be a different system and is not contemplated here.

<!-- frozen_above -->
