# Pre registration: does the ordering matter to the consumer that actually exists?

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Sixth and, if the prediction holds, last record in this series. The first five are closed and
unedited.

## The gap this exists to close

**Every recovery number in the previous five records is top-1.** They ask whether the successor is
the FIRST verdict-`ok` hit. The default consumer is not top-1: `EvidencePolicy` ships
`max_items=5` with `selection_mode="prefix"` (`recall/evidence.py:37`), so `build_evidence_bundle`
hands a generator the first five `ok` hits.

In the third record the successors that "failed" to recover sat at ranks 5, 5, 5, 5 and 2 among
`ok` hits. Every one of those is inside a five-item prefix. If that carries to the 30-pair fixture,
then the 0.25 versus 1.00 gap between `pool` and `inherit` is **invisible to the shipped default
path**, and five records have been optimising a metric no caller uses.

That possibility should have been checked before the first ordering was written. It was not, and
this record is the correction rather than a discovery.

## The question

Under each ordering, does the successor reach the evidence bundle, and does the regression gold
answer stay in it?

## Treatment

No library change. The probe additionally calls the real `build_evidence_bundle` with the default
`EvidencePolicy()` and reports membership. The real function, not a rank comparison standing in for
it: a reimplementation of the selection rule would be exactly the counterfactual mistake the third
record made.

## Prediction

Denominators as before: stratum B for the successor, the 10 usable regression cases for gold.

| Metric | `pool` | `promoted_first` | `inherit` |
|---|---|---|---|
| Successor in bundle | **0.85 to 1.00** | 0.90 to 1.00 | 1.00 |
| Successor is top-1 (already measured) | 0.25 | 0.94 | 1.00 |
| Gold in bundle | 1.00 | **0.90 to 1.00** | **1.00** |
| Gold is top-1 (already measured) | 1.00 | 0.20 | 0.90 |

**The claim: both quantities are high for every arm, and the large top-1 differences mostly vanish
at the bundle.** `promoted_first` displaced gold from rank 1 in 8 of 10 cases, but displacing from
rank 1 to rank 2 leaves gold inside a five-item prefix, so its 0.20 should recover to near 1.00.
`inherit` displaced gold once, from rank 1 to rank 2, so its gold-in-bundle should be exactly 1.00.

If that is right, the ordering choice is close to irrelevant for the default consumer, matters only
for a caller reading `hits[0]`, and the conservative default is free.

## What would falsify this

- **Successor in bundle under `pool` below 0.60.** Top-1 was then a fair proxy after all, the gap
  between the orderings is real for every consumer, and `inherit` has a much stronger case than the
  deflationary story allows.
- Gold in bundle below 1.00 for `inherit`, meaning its displacement survives the prefix and is a
  real cost rather than a reordering within the delivered set.
- Gold in bundle below 0.90 for `promoted_first`, meaning it is genuinely harmful and not merely
  reordering.
- Any arm changing `str_trust` from 0.00, which would mean the bundle path admits something the
  hits path does not.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| Every arm at or above 0.90 on both quantities | The ordering is near-irrelevant to the shipped consumer. **Keep `pool`**, the simplest and the only one that cannot displace, and document that `ordering` matters only for callers reading `hits[0]` |
| `pool` successor-in-bundle below 0.60 | The top-1 gap is real for every consumer. `inherit` becomes the recommended default and a change should be proposed on that basis |
| `inherit` gold-in-bundle below 1.00 | `inherit` carries a real cost at the consumer level. Keep `pool` and say so plainly |
| Mixed, or an invariant moves | Report it and change nothing |

## How it will be measured

Same fixture, same calibration, same database, one added column per arm.

```bash
eval "$(scripts/session-db.sh up)"
python -m benchmarks.successor_expansion_probe
```

## Result

Not yet measured.
