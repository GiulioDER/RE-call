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

## Result (2026-08-20)

**Status: measured. The prediction is FALSIFIED, and so is the diagnosis it rested on. Recovery did
not rise from 0.33 to 0.50 or better. It FELL, to 0.17.**

Apparatus passed, including the invariant that makes the two runs comparable. Corpus identical at
1064 chunks from 71 files. Stratum sizes **unchanged at 6 by hits and 3 by pool**, so ranking did
not move and the comparison is legitimate. Baseline recovery on stratum B still 0.00. No
uncertified-calibration warning was emitted, unlike the first run, which logged one; both classes
are above the minimum at 24 and 22.

| Metric | Predicted | First run | This run |
|---|---|---|---|
| Calibrated threshold | falls below 0.7110 | 0.7110 | **0.7070** |
| Successor recovery, stratum B | **0.50 to 0.85** | 0.33 | **0.17 [0.03, 0.56] n=6** |
| Fetched / promoted, stratum B | n/a | 6 / 2 | **6 / 1** |
| Successor recovery, stratum A | unchanged | 0.75 | 0.75, unchanged |
| Superseded trust rate | 0.00 | 0.00 | 0.00 [0.00, 0.28] n=10 |
| Trust coverage, baseline | n/a | 0.80 | 0.90 |
| Abstention accuracy | **falls, not below 0.67** | 1.00 | **1.00, did not fall** |
| Stratum sizes | unchanged | 6 / 3 | 6 / 3 |
| p50 latency, triggering | no more than 2x | 0.90x, invalid | 1.66x, same ordering flaw |

**Two of six predictions held. The threshold did fall, by 0.004, which is directionally right and
materially nothing: removing ten high-scoring queries from the answerable side moved the operating
point by less than half a percent. Abstention accuracy did not fall at all.**

### Why this falsifies the diagnosis and not merely the number

The first record concluded that promotion is threshold-bound, because 6 of 6 fetched and only 2
were promoted. This run lowered the threshold and got **fewer** promotions, 1 instead of 2. A lower
threshold can only make the promotion test easier, so if promotion were threshold-bound, recovery
could not fall. It fell. The stated mechanism is wrong.

### The leading explanation, stated as a hypothesis and NOT as a result

Recovery is a **top-1** metric: it asks whether the first verdict-`ok` hit is the successor. The
expander appends fetched chunks to the END of the merged pool, and `evaluate` returns `ok + rest`
preserving pool order, so a fetched successor wins top-1 only when nothing ahead of it is `ok`.

Lowering the threshold admits more hits as `ok`. Baseline coverage rose 0.80 to 0.90 in exactly
this run, which is that effect visible on the baseline arm. So the likely story is that the
successor is still being fetched and still being promoted, and is now being **outranked by a
distractor that the lower threshold newly admitted**, because position in the pool decides the
order and the fetched chunk is last by construction.

That is consistent with every number here, and it is not measured. What would test it: report the
successor's RANK among `ok` hits rather than only whether it is first, and separately, order `ok`
hits by score instead of by pool position. Neither is done, and no claim above depends on the
hypothesis being right.

### What follows, per the decision rule fixed in advance

"Recovery at or below 0.33: the threshold diagnosis is wrong. Say so plainly and reopen the question
of why promotion refuses." That is the outcome. Both records now stand as nulls. The feature stays
opt in, off by default, and unpromoted.

Two things survive both runs and are worth keeping separate from the failed claims:

- **The fetch works.** 6 of 6 in both runs, 0 failures to fetch. Whatever is wrong is downstream of
  retrieval, which is the half this feature actually changed.
- **The invariants never moved.** `str_trust` 0.00 in all four arms, abstention accuracy 1.00 in all
  four, stratum A identical at 0.75 throughout. Nothing here was bought by serving stale memory or
  by answering unanswerable questions.

### Carried forward, still not fixed

The latency comparison remains invalid as ordered: treatment still runs second in the same process.
It reads 1.66x here against 0.90x in the first run, which is a spread wide enough to show the
measurement is describing cache state rather than work. Neither arm ran a reranker.
