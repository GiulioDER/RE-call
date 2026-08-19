# Pre registration: is the fetched successor promoted but outranked?

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Third record on the mechanism. The first two are closed nulls and stay unedited:
[2026-08-19-successor-directed-expansion.md](2026-08-19-successor-directed-expansion.md) and
[2026-08-20-successor-expansion-recalibrated.md](2026-08-20-successor-expansion-recalibrated.md).
This one tests the hypothesis the second record wrote down and explicitly declined to claim.

## What is established

Across both runs, **6 of 6 absent successors were fetched, 0 failed to fetch**. Recovery was 2 of 6
at threshold 0.7110 and 1 of 6 at 0.7070. Lowering the threshold made the promotion test easier and
recovery went **down**, which rules out promotion being threshold-bound and leaves the reason
unknown.

Recovery is a **top-1** metric: it asks whether the first verdict-`ok` hit is the successor.
`evaluate` returns `ok + rest` preserving pool order, and `expand_retrieval_by_successor` appends
fetched chunks to the END of the merged pool. So a fetched successor wins top-1 only if nothing
ahead of it is `ok`.

## The question

For the stratum B queries that did not recover, is the successor present with verdict `ok` and
merely outranked, or is it not `ok` at all?

Those are different failures with different fixes and the current instrumentation cannot tell them
apart. That is the defect this record exists to close.

## Treatment and baseline

**No library change.** The probe is extended to report, per stratum B query: whether the successor
chunk is present in the treatment result, its verdict, and its 1-based rank among `ok` hits. Two
counterfactual orderings are computed from the same `TrustedResult`, changing nothing that ships:

- **score order:** `ok` hits sorted by cosine, descending.
- **promoted first:** any `ok` hit whose file is named by another hit's `superseded_by`, placed
  ahead of the rest, otherwise pool order.

## Prediction

Denominator throughout is the stratum B queries that did NOT recover under the recalibrated
threshold, which was 5 of 6.

| Claim | Prediction |
|---|---|
| Successor present in the treatment result | **5 of 5.** It was fetched; nothing removes it |
| Successor carries verdict `ok` | **at least 3 of 5** |
| Where `ok`, its rank among `ok` hits | **2 or worse**, never 1 |
| What outranks it | a repository distractor, **never the stale v1**, since `str_trust` is 0.00 |
| Recovery under **score order** | **no better than pool order, and plausibly worse** |
| Recovery under **promoted first** | **0.67 to 1.00** (4 to 6 of 6) |

**The score-order prediction is the counterintuitive one and it is the point of including it.**
Promotion exists precisely for a successor whose own wording scores LOW: the declared edge transfers
relevance the stale memory proved, which is why a hit at 0.64 can be promoted past a 0.70 threshold.
Sorting `ok` hits by cosine therefore pushes exactly those successors DOWN. If the obvious fix is
the wrong one, that is worth knowing before anybody writes it.

## What would falsify this

- Fewer than 3 of 5 successors carrying `ok`. The failure is then promotion refusing for a reason
  neither record has identified, and the ranking hypothesis is dead.
- Any successor at rank 1 among `ok` hits while not counting as recovered. That would mean the
  recovery metric and the rank metric disagree, which is an apparatus fault, not a result.
- The stale v1 outranking the successor among `ok` hits. That contradicts `str_trust` 0.00 measured
  twice, and would mean one of the three measurements is wrong.
- Score order improving recovery. The stated reason for promotion existing would then be wrong.
- Promoted-first recovery below 0.67, which would leave the ranking hypothesis insufficient even if
  the diagnosis is right.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| At least 3 of 5 `ok` at rank 2 or worse, and promoted-first recovery at or above 0.67 | The ranking hypothesis is supported. Open a SEPARATE record proposing an ordering change; do not ship one off this measurement |
| At least 3 of 5 `ok`, but promoted-first recovery below 0.67 | Partly right. Ordering is one cause and not the only one |
| Fewer than 3 of 5 `ok` | Hypothesis rejected. Report that promotion refuses for an unidentified reason and stop guessing at it in prose |
| Any apparatus falsifier above | Do not interpret. Fix the probe and rerun under this record |

## How it will be measured

Same corpus, same pairs, same disjoint calibration, same database. Only the probe's reporting
changes, so the retrieval and trust numbers must reproduce exactly: **stratum sizes 6 and 3,
recovery 0.17, `str_trust` 0.00**. Any drift there means the run is not comparable and the new
columns must not be read.

```bash
eval "$(scripts/session-db.sh up)"
python -m benchmarks.successor_expansion_probe
```

## Result

Not yet measured.
