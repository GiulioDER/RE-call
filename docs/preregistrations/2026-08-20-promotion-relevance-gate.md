# Pre registration: gate promotion on the successor's own score

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Eighth record. It exists because turning the feature on by default surfaced a failure that six
records of quality measurement never looked for.

## The failure

`tests/test_mcp_service_search.py::test_search_memory_abstains_when_only_superseded` builds a
corpus where `v1.md` says "cats", `v2.md` says "dogs entirely" and supersedes it, and the two
vectors are **orthogonal**. Queried for "cats" with the feature on:

```
abstained: False
  verdict=ok           text='dogs entirely'
  verdict=superseded   text='cats'
```

The service answers "cats" with a document about dogs, marked `ok`. Before, it abstained.

**Every quality number in this series measures whether the successor was DELIVERED. None measures
whether it is RELEVANT.** The third record flagged exactly this and did not close it: *"A recovered
answer is not a correct answer."* `str_trust` does not cover it either, since that counts serving
the **stale** memory and the stale memory was correctly demoted. This is a different failure and
nothing measured it.

## Why promotion does this

`recall/trust.py` promotes a successor from `low_confidence` to `ok` when the **stale** hit cleared
the calibrated threshold. The successor's own score is never consulted. That is deliberate, and the
reason is good: promotion exists precisely to rescue a successor whose wording scores low, because
the declared edge transfers the relevance the stale memory proved. What the rule lacks is any floor,
so relevance transfers to a document with no relationship to the query at all.

## The design, and why it is derived rather than chosen

`recall/calibration.py` computes the threshold as the **midpoint between the answerable floor (q05)
and the unanswerable ceiling (q95)**. So the calibration has already measured where "unrelated"
sits on this corpus. It simply does not keep the number.

- `Calibration` gains `unanswerable_ceiling`, populated by `from_samples` from the quantile it
  already computes.
- Promotion additionally requires the successor's own cosine to be **at or above that ceiling**: not
  confident, which is what promotion is for, but not in the unrelated distribution either.
- **When the ceiling is unknown, nothing is promoted.** Fail closed.

That last clause is the load-bearing one and it needs its reason stated. Promotion is a claim that a
below-threshold document is trustworthy. Making that claim requires evidence about where unrelated
sits, and an uncalibrated corpus has none. This library already treats an uncalibrated corpus as
degraded; a confident promotion out of one is exactly the thing that posture exists to prevent. It
is also why the MCP case above stops: that search passes no calibration.

⚠️ A fraction of the threshold would have been easier and is what I nearly wrote.
`recall/calibration.py` carries a standing warning against shipping a constant that merely looks
principled, and `0.5 * threshold` is that constant.

## Prediction

| Metric | Denominator | Prediction |
|---|---|---|
| MCP orthogonal case | 1 | **abstains again**, `ok` hits empty |
| Successor in bundle | stratum B, n=16 | **1.00**, unchanged |
| Successor top-1 recovery | stratum B, n=16 | **0.25**, unchanged |
| `str_trust` | 30 | 0.00, unchanged |
| Abstention accuracy | 6 | 1.00, unchanged |
| Measured unanswerable ceiling | n/a | **0.45 to 0.62** |

**The whole prediction turns on one unmeasured number.** The fixture's successors score around
**0.64** against a threshold of **0.707**. If the unanswerable ceiling comes in below 0.64, they
clear the gate and recovery is untouched. If it comes in above 0.64, the gate eats the feature and
this design is wrong. I predict 0.45 to 0.62 because the threshold is the midpoint: with an
answerable floor somewhere near 0.85, a 0.707 midpoint implies a ceiling near 0.56.

## What would falsify this

- **Successor-in-bundle below 0.90.** The gate is then too aggressive and costs more than the case
  it fixes.
- The MCP case still answering, meaning the fail-closed path does not cover an uncalibrated search.
- The unanswerable ceiling landing above 0.64, which makes the derived floor incompatible with
  promotion on this corpus and sends the design back rather than the constant.
- `str_trust` moving off 0.00, or abstention accuracy falling, either of which would mean the gate
  changed something it should not touch.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| MCP case abstains, bundle delivery at or above 0.90, invariants hold | Ship the gate. The default stays ON, and the failure that prompted this is closed |
| Bundle delivery below 0.90 | The gate is too aggressive. Revert it AND revert the default to off, since the on-by-default decision assumed a relevance property that does not hold |
| Ceiling above 0.64 | The derived floor is wrong for this corpus. Do not substitute a tuned constant; report and stop |
| Any invariant moves | Do not ship, and say which |

## How it will be measured

```bash
eval "$(scripts/session-db.sh up)"
python -m pytest tests/test_mcp_service_search.py -q
python -m benchmarks.successor_expansion_probe
```

## Result (2026-08-20)

**Status: measured. BOTH falsifiers fired. The gate is reverted, and with it the on-by-default
decision, exactly as the rule required.**

| Metric | Predicted | Measured |
|---|---|---|
| MCP orthogonal case | abstains again | **abstains** ✓, 4 of 4 in that module |
| Successor in bundle | 1.00, unchanged | **0.38** [0.18, 0.61] n=16 ✗ |
| Successor top-1, `pool` | 0.25, unchanged | **0.00** ✗ |
| Gold in bundle | 1.00 | 1.00 ✓ |
| `str_trust` | 0.00 | 0.00 ✓ |
| Abstention accuracy | 1.00 | 1.00 ✓ |
| Unanswerable ceiling | **0.45 to 0.62** | **0.675** ✗ |

The gate works. It stops the orthogonal case precisely as designed, `str_trust` and abstention never
move, and gold is never lost. It also **blocks roughly two thirds of legitimate successors**, which
is a far worse trade than the failure it fixes.

### Why the design is wrong, not the constant

The ceiling came in at **0.675** against a threshold of **0.707**. Since the threshold IS the
midpoint of the answerable floor and the unanswerable ceiling, the floor is
`2 × 0.707 − 0.675 = 0.739`. So on this corpus:

```
unanswerable q95  0.675 ────┬──── 0.707 threshold ────┬──── 0.739 answerable q05
                            │                          │
                     promotable band                 already `ok`
                       0.032 wide
```

**Promotion needs a successor in a sliver 0.032 wide, and that sliver is half the gap between the
two distributions by construction.** A q95 of the unanswerable class is a deliberately conservative
upper bound chosen for setting a threshold; it sits just under the threshold whenever the classes
overlap at all, which is whenever calibration is doing real work. Using it as a promotion floor is
therefore self-defeating in general, not merely unlucky here: promotion exists to rescue successors
BELOW the threshold, and this floor removes almost all of the space below the threshold.

The fixture's successors score about 0.64, comfortably under 0.675, so they are treated as
unrelated. They are not.

Per the decision rule: *"Ceiling above 0.64: the derived floor is wrong for this corpus. Do not
substitute a tuned constant; report and stop."* I have not substituted one. A lower quantile of the
same distribution, q50 or q75, is the obvious next idea and is exactly the tuning that rule forbids
without its own record.

### What was reverted

Both, per *"revert it AND revert the default to off, since the on-by-default decision assumed a
relevance property that does not hold"*:

- the gate, and the `Calibration.unanswerable_ceiling` field it needed;
- the default, back to `successor_expansion=None`, off.

**The on-by-default change never entered history.** It was written, the suite surfaced the
orthogonal-successor failure before it was committed, and it is now withdrawn. So there is nothing
to revert upstream and nothing was published in that state. The feature remains opt in.

### What this leaves standing, and what it leaves open

Standing: the fetch, the promotion, the ordering findings, the bundle-delivery result, and the
latency measurement. None of them depended on the gate.

Open, and now known rather than suspected: **promotion can transfer relevance to a document
unrelated to the query, and nothing in the shipped code prevents it.** The MCP test documents the
case. That is a real limitation of the feature as it ships, it is the reason the feature stays opt
in, and it is not fixed by anything in this record.
