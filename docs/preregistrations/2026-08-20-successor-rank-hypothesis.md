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

## Result (2026-08-20)

**Status: measured. The ranking hypothesis is CONFIRMED, on every column, and the measurement
retracts the central finding of BOTH earlier records.**

| Claim | Predicted | Measured |
|---|---|---|
| Successor present | 5 of 5 | **6 of 6** |
| Successor verdict `ok` | at least 3 of 5 | **6 of 6** |
| Rank among `ok` where not recovered | 2 or worse, never 1 | **5, 5, 5, 5, 2** |
| What outranks it | a repository distractor, never the stale v1 | **repository distractors, every time** |
| Recovery, score order | no better than pool order | **0.17, exactly equal** |
| Recovery, promoted first | 0.67 to 1.00 | **1.00 [0.61, 1.00] n=6** |

Reproduction held on everything the record named: recovery 0.17, `str_trust` 0.00, stratum A 0.75,
coverage 0.90 and 1.00, six stratum B queries in the per-query listing. ⚠️ The by-pool stratum size
of 3 was truncated out of the captured output by a `tail`, so it was **not** re-verified; every other
reproduction value matched exactly.

### The retraction

**Every fetched successor was promoted. 6 of 6, in the run whose headline recovery is 1 of 6.**

The first record concluded that promotion was the bottleneck, from a line reading "6 fetched a
successor, 2 of those were then promoted". **That line was mislabelled in my own probe.** It printed
`treat_recovered`, which is top-1 recovery, under the word "promoted". Promotion was never measured
by it at all.

So both earlier conclusions are withdrawn:

- The first record's "every miss is the promotion rule refusing" was wrong. Promotion refused
  nothing.
- The second record's inference that a lower threshold produced fewer promotions was wrong for the
  same reason. Promotions did not move between runs. What moved was how many distractors the lower
  threshold newly admitted as `ok`, each of which sits ahead of the appended successor.

Neither record is edited. The predictions and results in both stand as written, including the
mistaken diagnoses, because the mistake is the informative part: a wrong label on one print
statement survived two full measurement cycles and produced two confidently wrong causal stories.
The numbers in those records were all correct. Only the word "promoted" was false, and it was
enough.

### The counterintuitive prediction held, and it matters for the fix

Ordering `ok` hits by cosine scores **0.17, identical to pool order, not better**. Promotion exists
for a successor whose own wording scores low, so the obvious fix moves exactly those hits back down.
Anyone reaching for "sort by score" would have shipped a change with no effect and a plausible
rationale.

**Placing promoted successors first scores 1.00 on stratum B**, and it is the only one of the three
orderings that does anything.

### What follows, per the decision rule fixed in advance

"At least 3 of 5 `ok` at rank 2 or worse, and promoted-first recovery at or above 0.67: the ranking
hypothesis is supported. Open a SEPARATE record proposing an ordering change; do not ship one off
this measurement."

That is the outcome, and the separation matters more than usual here. A 1.00 on six queries from one
authored fixture is not evidence that reordering `ok` hits is safe in general: it changes what every
caller reads first, on every query with a supersession edge, and this fixture cannot see the
regression that would cause elsewhere. The ordering change gets its own record, its own prediction,
and its own falsifiers.

### Carried forward, still not fixed

The latency comparison is still invalid as ordered, now reading 1.81x and 1.27x against 0.90x in the
first run for the same code. Neither arm ran a reranker.
