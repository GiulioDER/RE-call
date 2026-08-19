# Pre registration: ordering promoted successors, and what it costs elsewhere

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Fourth record. The first three are closed and stay unedited:
[2026-08-19-successor-directed-expansion.md](2026-08-19-successor-directed-expansion.md),
[2026-08-20-successor-expansion-recalibrated.md](2026-08-20-successor-expansion-recalibrated.md),
[2026-08-20-successor-rank-hypothesis.md](2026-08-20-successor-rank-hypothesis.md).

The third established that fetched successors are promoted **6 of 6** and then outranked by
distractors at ranks 5, 5, 5, 5 and 2, and that placing them first recovers **1.00** on that
fixture. That number is not evidence that the ordering is safe. It was a counterfactual computed
from one result set, over 6 queries, on a fixture built to contain only cases where the successor
IS the right answer. A fixture that cannot produce a regression cannot report one.

## The question

Two, and the second is the one that decides whether anything ships.

1. Does an ordering change recover the successor on a larger set of supersession queries?
2. **What does it cost on queries where the successor is NOT the right answer?**

## Why "promoted first" is probably the wrong design, stated before measuring

Placing every promoted successor at rank 1 is unconditional. It fires whenever any retrieved
document carries a supersession edge, including when that document was retrieved incidentally and
some third document actually answers the query. The successor then displaces a correct answer.

So this record measures a third ordering that the previous one did not consider, and predicts it
wins:

| Arm | Ordering of verdict-`ok` hits |
|---|---|
| **A. pool order** | as shipped: `ok + rest`, pool position preserved |
| **B. promoted first** | every promoted successor ahead of other `ok` hits |
| **C. inherit position** | a promoted successor takes **the rank its demoted predecessor held**, rather than rank 1 |

C is the principled one. The supersession edge transfers the topical relevance the stale memory
proved, and the stale memory proved it **at its own rank**, not at rank 1. If the predecessor was
third-best for this query, its successor should be third-best, not first.

## Fixture, with the rules stated before it is authored

- **30 supersession pairs**, up from 10. The existing 10 are unchanged. Each new successor must
  **reframe** the decision rather than renumber it, and must be worded unlike its predecessor, as
  in the existing set.
- **A regression set**, new: queries whose gold answer is a THIRD document, worded so that a
  superseded document is also retrieved. Promoting its successor displaces the gold answer. Gold is
  declared per query before any arm runs.
- Distractor mass unchanged: the repository's own `docs/`.
- Calibration set unchanged, and still disjoint from every measured query.

⚠️ **A regression query only tests anything if it actually retrieves a superseded document.** That
is a property of the embedder, not of my intent, so it is checked empirically and the count is
reported. Queries that fail the check are excluded and the exclusion count is printed. An excluded
majority means the regression arm measured nothing, and is an apparatus failure rather than a clean
bill of health.

## Prediction

| Metric | Denominator | A: pool order | B: promoted first | C: inherit position |
|---|---|---|---|---|
| Successor recovery | stratum B, larger n | 0.10 to 0.30 | **0.70 to 0.95** | **0.65 to 0.90** |
| **Gold answer displaced** | regression set | **0.00** | **0.15 to 0.40** | **0.00 to 0.10** |
| Superseded trust rate | all supersession queries | 0.00 | 0.00 | 0.00 |
| Abstention accuracy | unanswerable controls | 1.00 | 1.00 | 1.00 |

**The claim this record exists to test is that B and C are close on recovery and far apart on
regression.** If that holds, C ships and B does not, and the third record's 1.00 turns out to have
been measuring a fixture that could not see the cost.

I predict B's recovery edge over C is **at most 0.10**, because the cases where rank 1 beats
inherited rank are cases where the predecessor was itself outranked, which the third record found
to be a minority: the stale hit was rank 1 among `ok` in most queries there.

**Recovery will fall for every arm relative to the third record's 1.00.** Twenty new pairs written
without knowing which will land in which stratum is a harder set than ten written earlier, and the
1.00 came from 6 queries.

## What would falsify this

- **B's displacement rate at or below 0.05.** The concern motivating C is then unfounded, and the
  simpler ordering is the right one.
- **C's displacement rate above 0.10.** Inheriting the predecessor's rank does not avoid the cost,
  and neither ordering is shippable as specified.
- **C's recovery more than 0.10 below B's.** The principled ordering is then paying too much for
  its safety and the trade needs stating rather than assuming.
- Any arm raising `str_trust` above 0.00, or dropping abstention accuracy below 0.83.
- The regression set failing its apparatus check, which measures nothing and is not a pass.
- Recovery under A moving outside 0.10 to 0.30, which would mean the larger fixture is not
  comparable to the earlier ones and the baseline has drifted.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| C recovers within 0.10 of B and displaces at or below 0.10, while B displaces above 0.15 | Ship C as the ordering, opt in, behind the existing policy flag. B is rejected and the reason recorded |
| B and C both displace at or below 0.05 | Ship B, the simpler one. C is unnecessary complexity |
| Either arm displaces above 0.10 while the other does not | The safer arm ships only if its recovery clears 0.50; otherwise neither ships and the feature stays as it is |
| Both arms displace above 0.10 | Neither ships. Report that ordering cannot be fixed without a signal this design does not have |
| Regression apparatus check fails | Do not interpret the displacement column at all |

## How it will be measured

The two orderings are implemented as real behaviour selected by policy, not as counterfactuals
computed after the fact, because a counterfactual cannot exercise the code path that would ship.
Same corpus, same calibration, same database, all three arms in one process over one index.

```bash
eval "$(scripts/session-db.sh up)"
python -m benchmarks.successor_expansion_probe
```

## Result

Not yet measured.
