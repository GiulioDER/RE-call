# Pre registration: successor expansion under a certified calibration

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Second measurement of the mechanism registered in
[2026-08-19-successor-directed-expansion.md](2026-08-19-successor-directed-expansion.md). That
record is closed: its prediction was falsified and its result stands unedited. This is a separate
record because it is a separate claim, and folding it into the first would let a resolved
prediction be reread in the light of a later number.

## What the first run established, and what it left open

Of 6 absent-successor queries, **6 fetched a successor and 0 failed to fetch. 2 were promoted.**
So retrieval works at 6 of 6 and every miss is the promotion rule refusing. Promotion requires the
**stale** hit to sit at or above the calibrated threshold, and that run reported its threshold
uncertified in the log: 0.7110 fitted from 10 answerable and 6 unanswerable samples against a
stated minimum of 20 each (`recall/calibration.py:44`).

Two defects in that calibration, not one, and the second was not noticed until the result was
written up:

1. **Too few samples.** 10 and 6 against a minimum of 20 each.
2. **It was fitted on the queries being measured.** The 10 answerable samples WERE the 10
   supersession queries. Those are worded from v1 and match v1 strongly, so the answerable
   distribution was built from the highest-scoring queries in the study and the threshold inherited
   that. Calibrating on the evaluation set is a leak regardless of sample count.

## The question

With a calibration fitted on at least 20 samples per class, drawn from queries disjoint from the 10
being measured, does successor recovery on the absent stratum rise above the 0.40 floor that the
first record fixed in advance?

## Treatment and baseline

Unchanged code. `benchmarks/successor_expansion_probe.py` and the authored corpus are byte
identical to the first run; the only edit is the labelled set the in-run calibration is fitted from.
That is deliberate, so any movement is attributable to the threshold and to nothing else.

## Prediction

| Metric | Denominator | Prediction |
|---|---|---|
| Calibrated threshold | n/a | **falls below 0.7110** |
| Calibration certified | n/a | **yes**, both sample counts >= 20 and the separability interval clears the floor |
| Successor recovery, treatment | stratum B | **0.50 to 0.85** (3 to 5 of 6) |
| Successor recovery, baseline | stratum B | **0.00**, structurally |
| Superseded trust rate | all trust queries | **0.00**, both arms, unchanged |
| Abstention accuracy, treatment | unanswerable controls | **falls, but not below 0.67** |
| Stratum sizes | all trust queries | **unchanged at 6 by hits and 3 by pool** |

**Why the threshold should fall.** Removing the 10 v1-worded queries removes the top of the
answerable distribution, and replacing them with queries drawn across the whole corpus lowers its
floor. `best_threshold` sits between the two distributions, so a lower answerable floor pulls it
down. A lower threshold means more stale hits clear it, which is exactly the gate promotion is
waiting on.

**Why abstention accuracy should fall and why that is not a defect on its own.** A lower threshold
admits more, so a control that scored just under the old floor can now clear the new one. That is
the honest cost of the same change that buys recovery, and the record fixes 0.67 in advance as the
point where the cost stops being acceptable.

**Stratum sizes are the invariant that makes this a clean comparison.** Ranking does not depend on
the threshold, so the A/B split must not move. If it does, something other than the calibration
changed and no comparison between the two runs is legitimate.

## What would falsify this

- Recovery on stratum B at or below **0.33**, the first run's value. The diagnosis that promotion is
  threshold-bound would then be wrong, and the four misses need a different explanation.
- The threshold rising rather than falling, which falsifies the stated mechanism even if recovery
  happens to improve.
- Any rise in `str_trust` above 0.00. Rejects the change outright, as in the first record.
- Abstention accuracy below 0.67, meaning recovery was bought by answering unanswerable queries.
- Stratum sizes moving, which invalidates the comparison rather than the treatment.
- The calibration still failing certification, which makes this an apparatus failure and not a
  result.

## How the labelled set is built, stated before it is written

- **Answerable:** at least 20, each answerable from the indexed corpus, and **none of them one of
  the 10 supersession queries**. Drawn from the repository prose that forms the distractor mass, so
  they describe the corpus rather than the study.
- **Unanswerable:** at least 20, genuinely off topic. Not an answerable query with a nonsense
  suffix, which is the defect `results/FINDINGS.md:370` records as making the two classes
  inseparable.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| Recovery 0.50 or above, `str_trust` 0.00, abstention at or above 0.67, calibration certified | The first record's null is attributed to its calibration. Open a default-promotion decision as a separate record |
| Recovery above 0.33 but below 0.50 | Partial. The threshold is part of the story and not all of it; keep opt in and look at the promotion rule itself |
| Recovery at or below 0.33 | The threshold diagnosis is wrong. Say so plainly and reopen the question of why promotion refuses |
| `str_trust` rises, or abstention falls below 0.67 | Reject regardless of recovery |
| Calibration still uncertified, or stratum sizes move | Apparatus failure. Do not interpret the quality result |

## Result

Not yet measured.
