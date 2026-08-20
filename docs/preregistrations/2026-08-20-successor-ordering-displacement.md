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

## Result

Not yet measured.
