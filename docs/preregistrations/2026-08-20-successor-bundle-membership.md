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

## Result (2026-08-20)

**Status: measured. All six predictions held exactly. The ordering is irrelevant to the consumer
that ships, and the first two records were nulls about the wrong metric.**

Apparatus unchanged and clean: 10 of 10 usable, stratum B 16, baseline recovery 0.00.

| Metric | `pool` | `promoted_first` | `inherit` |
|---|---|---|---|
| **Successor in bundle** | **1.00 [0.81, 1.00]** | **1.00** | **1.00** |
| Successor is top-1 | 0.25 | 0.94 | 1.00 |
| **Gold in bundle** | **1.00 [0.72, 1.00]** | **1.00** | **1.00** |
| Gold is top-1 | 1.00 | 0.20 | 0.90 |

Predicted 0.85 to 1.00, 0.90 to 1.00, 1.00, 1.00, 0.90 to 1.00 and 1.00. Every one landed.

### What this says about the previous five records

**The fetch was the whole value. The ordering was noise.**

Under every ordering, including the shipped `pool`, the fetched successor reaches the evidence
bundle **every time**. `promoted_first` displaces gold from rank 1 in 8 of 10 cases and gold still
reaches the bundle every time, because rank 2 is inside a five-item prefix. The dramatic top-1
spreads, 0.25 against 1.00 for recovery and 1.00 against 0.20 for gold, are entirely invisible to
`build_evidence_bundle` with its default policy.

The uncomfortable consequence, stated plainly: **the first record's 0.33 and the second record's
0.17 were nulls about a metric no default consumer reads.** Both measured whether the successor was
the FIRST `ok` hit. Neither asked whether it was delivered. The behaviour they scored as a failure
is `pool` ordering, which measures 1.00 bundle membership on today's fixture. I cannot retroactively
claim a number those runs did not compute, and the fixture has changed since, so this is an
implication rather than a re-measurement. But it is a strong one, and the direction is not in doubt.

Three further records then went looking for an ordering fix to a problem that only existed at
top-1. The mechanism work in them stands: the fetch, the promotion, the rank diagnosis and the
characterised displacement condition are all real findings. What does not stand is the framing that
made ordering look load-bearing.

### What follows, per the decision rule fixed in advance

"Every arm at or above 0.90 on both quantities: the ordering is near-irrelevant to the shipped
consumer. **Keep `pool`**, the simplest and the only one that cannot displace, and document that
`ordering` matters only for callers reading `hits[0]`."

That is the outcome. `pool` stays the default. It is the only ordering with a sound structural
guarantee, since appending to the end of the pool cannot reorder anything already there, and this
run says nothing is paid for that guarantee at the bundle.

`inherit` remains selectable and is the right choice for a caller that reads `hits[0]` and only
`hits[0]`. That caller exists in principle and this measurement does not cover them.

### The lesson worth keeping, since it cost five records

**Check what the consumer reads before choosing what to optimise.** `EvidencePolicy.max_items = 5`
has been in the codebase throughout. Nothing prevented reading it in the first record except that a
top-1 metric was the obvious one to write, and once written it was never questioned. Four
subsequent records inherited it without asking, including two that reopened the diagnosis from
scratch.
