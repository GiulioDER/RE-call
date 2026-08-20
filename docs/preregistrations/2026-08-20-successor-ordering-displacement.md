# Pre registration: what does each ordering displace, on a regression set that works

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Fifth record. The fourth
([2026-08-20-successor-ordering-regression.md](2026-08-20-successor-ordering-regression.md))
measured recovery cleanly and failed its regression apparatus check at 4 of 10 usable, so its
displacement column was declared uninterpretable and no ordering shipped from it. The six failed
queries have been reauthored and now pass the check **10 of 10**, verified by
`benchmarks/successor_regression_check.py`, which computes the two apparatus criteria and nothing
else so that iterating on it cannot fit the fixture to an answer.

## ⚠️ This prediction is not blind, and that changes what it is worth

The fourth record's uninterpretable column showed `promoted_first` at 0.75 displacement and
`inherit` at 0.00 over its four usable cases. I have seen those numbers. A prediction made after
seeing a related result is weaker evidence than one made before, and pretending otherwise would be
the exact dishonesty this whole series is set up to prevent.

So the value here is **not** "did I guess the rate". It is the one claim that is not an
extrapolation from those four cases:

> **`inherit` cannot displace a better-ranked answer, for any arrangement, and will therefore
> measure exactly 0.00 rather than approximately 0.00.**

That is a claim of impossibility derived from the sort key, not a rate estimated from a sample. It
is falsified by a single displacement, and a single displacement would mean the default currently
shipped in `SuccessorExpansionPolicy` rests on a property it does not have.

## The question

On 10 usable regression cases, what fraction of gold answers does each ordering displace?

## Prediction

| Metric | Denominator | Prediction |
|---|---|---|
| Displacement, `inherit` | 10 regression | **exactly 0.00**, structural |
| Displacement, `promoted_first` | 10 regression | 0.50 to 0.90 |
| Displacement, `pool` | 10 regression | 0.00, by definition of the baseline |
| Recovery, `inherit` | stratum B | 0.85 to 1.00 |
| Recovery, `promoted_first` | stratum B | 0.80 to 1.00 |
| Recovery, `pool` | stratum B | 0.10 to 0.40 |
| `str_trust` | 30 supersession queries | 0.00, every arm |
| Abstention accuracy | 6 controls | 1.00, every arm |
| Stratum B size | 30 pairs | **12 to 20**, not necessarily 16 |

**Stratum B is predicted as a range, not as an invariant, and that is a change from the fourth
record.** There the corpus was identical and only reporting changed, so any drift meant a broken
comparison. Here six documents were rewritten, so the corpus genuinely differs and the pool for the
supersession queries may shift. Recovery figures are therefore comparable to the fourth record only
approximately, and the displacement column is the part this record is actually for.

## What would falsify this

- **`inherit` displacing anything at all.** The structural argument is then wrong, the default
  shipped on it is unjustified, and that is the most important outcome this run can produce.
- `promoted_first` displacing at or below 0.10, which would mean the concern that motivated
  `inherit` was unfounded and the simpler ordering was adequate all along.
- `pool` displacing anything, which would mean the baseline definition is not what I think it is and
  the whole column is mismeasured.
- Any arm raising `str_trust` above 0.00 or dropping abstention below 0.83.
- Fewer than 10 usable regression cases at run time, which would mean the reauthoring did not
  survive the corpus it changed, and the column is uninterpretable again.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| `inherit` 0.00 and recovery at or above 0.85, `promoted_first` above 0.10 | The shipped default is vindicated on a working apparatus. Record it and close the series |
| **`inherit` above 0.00** | **Revert the default to `pool` in the same change that records this**, and open a record on why the sort key does not have the property claimed for it |
| `promoted_first` at or below 0.10 | Record that the simpler ordering was adequate and that `inherit` is unnecessary complexity, whatever its own number |
| Usable below 10, or any invariant falsifier | Do not interpret the displacement column. Again |

## How it will be measured

Unchanged probe, unchanged library, unchanged calibration set. Only the six regression queries and
their gold documents differ from the fourth record's run.

```bash
eval "$(scripts/session-db.sh up)"
python -m benchmarks.successor_expansion_probe
```

## Result (2026-08-20)

**Status: measured. Seven of eight predictions held. The eighth is the one this record staked
itself on, and it is FALSIFIED. `inherit` displaced one gold answer.**

Apparatus clean for the first time in the series: **10 of 10 usable**, no exclusions on either
criterion. Stratum B 16, inside the predicted 12 to 20. Baseline recovery on B 0.00.

| Metric | Predicted | Measured |
|---|---|---|
| Displacement, `inherit` | **exactly 0.00** | **0.10 [0.02, 0.40]**, `queue_metrics` ✗ |
| Displacement, `promoted_first` | 0.50 to 0.90 | 0.80 [0.49, 0.94], 8 of 10 ✓ |
| Displacement, `pool` | 0.00 | 0.00 [0.00, 0.28] ✓ |
| Recovery, `inherit` | 0.85 to 1.00 | 1.00 [0.81, 1.00] ✓ |
| Recovery, `promoted_first` | 0.80 to 1.00 | 0.94 [0.72, 0.99] ✓ |
| Recovery, `pool` | 0.10 to 0.40 | 0.25 [0.10, 0.49] ✓ |
| `str_trust` | 0.00 every arm | 0.00 [0.00, 0.11] n=30 ✓ |
| Abstention accuracy | 1.00 every arm | 1.00 ✓ |

### The claim that died, and it is the one that mattered

> "`inherit` cannot displace a better-ranked answer, for any arrangement, and will therefore
> measure exactly 0.00 rather than approximately 0.00."

One displacement falsifies it, and there is one. **The shipped default rested on this and nothing
else**, because the displacement evidence behind it was uninterpretable when the default was
adopted. So the justification is gone, and per the decision rule fixed in advance the default is
reverted to `pool` in the same change that records this.

### What was actually wrong, stated as a hypothesis and NOT as a finding

Earlier in this series a mechanism asserted confidently in prose turned out to rest on a mislabelled
print statement and survived two records before being caught. So this is written as what it is.

The sort key almost certainly does have the property proved of it. `tests/test_successor_expansion.py`
asserts over every predecessor rank that any `ok` hit which outranked the predecessor still outranks
the successor, and it passes. **The invalid step was my inference from that property to "gold is
never displaced".** They are not the same statement. The guarantee is relative to the PREDECESSOR:
if the superseded document outranked gold in the pool, then giving its successor the predecessor's
rank places the successor above gold, and nothing in the property forbids that. Gold ranked below
the predecessor was never protected.

If that is right, the fix is not to the sort key but to the claim, and a genuinely safe ordering
would need to know that gold is a better answer, which is a signal this design does not have.

**Unverified.** What would test it: for `queue_metrics`, report whether `queue_backpressure_v1.md`
outranked `queue_metrics.md` in the baseline pool. That is one number and it decides between "the
property is narrower than I said" and "the sort key is broken", which are different repairs.

### The trade this leaves, stated plainly rather than resolved here

| Arm | Recovery, stratum B | Gold kept, regression |
|---|---|---|
| `pool` | 0.25 | 1.00 |
| `promoted_first` | 0.94 | 0.20 |
| `inherit` | 1.00 | 0.90 |

`inherit` is plainly the best of the three on this fixture, and reverting the default is not a claim
that it is worse. It is a claim that the reason it was made the default was wrong, and that a
default resting on a disproved impossibility claim should go back to the conservative option until
somebody chooses the trade-off knowingly. That choice is the author's and is not made here.

## Follow up (2026-08-20): the hypothesis was right, and the sort key is not at fault

The section above named one number as deciding between two repairs. Here it is, for all ten
regression queries rather than only the displaced one, because a single case cannot separate "this
is how the mechanism works" from "this case is odd".

| query | gold rank | predecessor rank | |
|---|---|---|---|
| store_timeout | 1 | 4 | gold above |
| log_shipping | 1 | 3 | gold above |
| flag_naming | 1 | 3 | gold above |
| api_pagination | 1 | 2 | gold above |
| backup_encryption | 1 | 2 | gold above |
| deploy_notification | 1 | 2 | gold above |
| slo_dashboard | 1 | 3 | gold above |
| test_parallelism | 1 | 2 | gold above |
| review_sla | 1 | 2 | gold above |
| **queue_metrics** | **2** | **1** | **PREDECESSOR ABOVE GOLD** |

**`inherit` displaces gold in exactly the cases where the predecessor outranked gold, and in no
others.** Nine of ten, gold was rank 1 and the predecessor rank 2 to 4, so the successor inherited
a rank below gold and gold survived. In the tenth the predecessor was rank 1, the successor
inherited rank 1, and gold at rank 2 lost it. One rule accounts for all ten rows with no residue.

So the repair is to the claim and not to the code. `order_promoted` has the property proved of it,
`test_inherit_cannot_displace_a_hit_that_outranked_the_predecessor` passes for the right reason, and
the invalid step was mine: inferring "gold is safe" from "anything that outranked the predecessor is
safe". Gold at rank 2 behind a rank-1 predecessor was never covered by that guarantee.

### What this changes about the trade

`inherit`'s cost is not a rate to be sampled. It is a **characterised condition**: the successor
takes gold's place precisely when retrieval ranked the superseded document above gold. Whether that
is even an error is a further question this fixture cannot settle. For `queue_metrics` the query is
"what is reported about the work queue during a burst", retrieval judged
`queue_backpressure_v1.md` the best match for it, and a reader might reasonably say the successor
of the best match is the right answer and my gold label is the thing that is wrong.

That is speculation and is not scored here. What is established: `inherit` fails only where the
stale document was already winning, and `promoted_first` fails wherever anything carries an edge,
which is why they measure 0.10 and 0.80 on the same ten queries.

**The default is unchanged by this follow up.** It stays `pool`. The decision rule fired on a
falsified prediction and the revert stands; what the follow up supplies is a correct basis on which
the trade could be chosen deliberately, which the original adoption did not have.
