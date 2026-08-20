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

## Result (2026-08-20)

**Status: measured, and PARTLY UNINTERPRETABLE. The regression apparatus check failed. Recovery is
readable; displacement is not, and the decision this record exists to make cannot be made.**

### Apparatus

Corpus 1114 chunks / 121 files. Threshold 0.7070 from 24 and 22, disjoint from every measured
query. 30 pairs, stratum B **16**, stratum A 14, baseline recovery on B **0.00** as it must be.
Those are the checks the recovery column needs and they passed.

**The regression set did not.** Of 10 authored queries, **only 4 are usable**. Six never retrieved
the superseded document they were written to drag in, so they cannot show a displacement:
`api_pagination`, `backup_encryption`, `deploy_notification`, `test_parallelism`, `review_sla`,
`queue_metrics`. None were excluded for the other reason; gold was top at baseline in all four
survivors.

The record fixed the rule in advance: *"Regression apparatus check fails: do not interpret the
displacement column at all."* The probe printed the failure itself and said the column is not
interpretable.

### Recovery, which IS interpretable

| Arm | Recovery, stratum B, n=16 | Predicted |
|---|---|---|
| baseline, no expansion | 0.00 [0.00, 0.19] | 0.00 |
| A: pool order, as shipped | **0.25 [0.10, 0.49]** | 0.10 to 0.30 ✓ |
| B: promoted first | **0.94 [0.72, 0.99]** | 0.70 to 0.95 ✓ |
| C: inherit position | **1.00 [0.81, 1.00]** | 0.65 to 0.90 ✗, above the band |

`str_trust` **0.00 [0.00, 0.11] n=30** in every arm. Abstention accuracy **1.00** in every arm.
Stratum A rose 0.93 to 1.00 under both orderings, which is a gain rather than the regression the
record was watching for, and it is not something the record predicted either way.

Two of the three recovery predictions landed inside their bands. **C beat its band**, and C beating
B was not predicted at all: I said B would lead by at most 0.10, and C leads by 0.06. The
prediction that the two would be close held; the direction did not.

### The uncomfortable part, stated rather than quietly used

The displacement column came out **exactly as predicted**: `promoted_first` displaces 0.75 of gold
answers over its four usable cases (`store_timeout`, `log_shipping`, `flag_naming`), `inherit`
displaces 0.00, against a prediction of 0.15 to 0.40 and at most 0.10. That is the result the record
was designed to produce and the argument for C in one line.

**It is still not interpretable, and I am not going to treat it as though it were.** The rule was
fixed before the fixture was authored, precisely so that a pleasing number could not be the thing
that decides whether the rule applies. Four cases is not the denominator this record promised, the
six exclusions are an authoring failure of mine rather than a property of the mechanism, and a
0.75 point estimate on n=4 carries a Wilson interval of [0.30, 0.95].

What can be said honestly: the four surviving cases are valid tests, not noise, and they point the
way the prediction did. What cannot be said is a rate.

### What follows, per the decision rule fixed in advance

Apparatus failure on the regression set means **no ordering ships from this run**. C is not adopted,
B is not rejected, and the feature stays opt in with `ordering="pool"` as its default.

The fix is mine and it is specific: the six failed regression queries did not share enough
vocabulary with their intended predecessor to pull it into the pool. Authoring them against the
measured retrieval, rather than against my expectation of it, is the repair. **That must not be done
by tuning the queries until the displacement number improves.** The apparatus criterion is whether
the stale document is retrieved, which is checkable without looking at the displacement column at
all, and the repair should be made and re-run under a new record with the outcome still unseen.

### Carried forward, still not fixed

No reranker in any arm. Latency is not reported at all this run, which is better than reporting the
invalid comparison the previous two runs printed.

## Decision (2026-08-20): `inherit` becomes the default ordering, opt in

**This is a decision, not a measurement, and it was taken on evidence this record calls partly
uninterpretable.** Recorded here so the basis is visible next to the result rather than only in a
commit message. The result above is unchanged.

The author was told, in these words, that four valid cases plus a mechanism argument was not
something I would claim a rate from, and chose to adopt the ordering anyway as an opt in default.
That is their call to make. What follows is what it does and does not rest on.

**Scope.** `successor_expansion` remains **off** by default: `trusted_search` still expands nothing
unless a caller passes a policy. What changes is the default *within* that policy, from
`ordering="pool"` to `ordering="inherit"`. A caller who enables expansion now gets `inherit`.

**What supports it.**

- Recovery is measured and interpretable: 0.25 to 1.00 on stratum B, n=16, against a baseline of
  0.00, with `str_trust` 0.00 and abstention accuracy 1.00 in every arm. That column passed its
  apparatus checks.
- A **structural** non-displacement property, which is the part that does not depend on the failed
  column. `inherit` sorts a promoted successor by the pool index its predecessor held, so any hit
  that outranked the predecessor still outranks the successor. It cannot displace a better-ranked
  answer for any arrangement, not merely for the four that were measurable. Asserted directly in
  `tests/test_successor_expansion.py` over every predecessor rank rather than sampled.

**What does NOT support it, and is not claimed.**

- Any displacement *rate*. The regression set failed its apparatus check at 4 of 10 usable, and the
  0.00 measured for `inherit` there remains uninterpretable under the rule this record fixed in
  advance. The structural argument above is a different kind of claim and is not a substitute for
  that measurement.
- Behaviour with a reranker, which no run has tested.
- Any corpus other than this one.

**What would reverse it.** A repaired regression set, authored against measured retrieval and
verified blind, showing `inherit` displacing above 0.10. The structural argument says that should be
impossible for hits that outranked the predecessor, so such a result would mean the property is not
what I think it is, and it would be more informative than the default is convenient.
