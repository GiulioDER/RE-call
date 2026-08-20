# Pre registration: what does the fetch actually cost per query?

**Date:** 2026-08-20   **Status:** predicted, not yet measured

Seventh record. The six before it measured quality and are closed. This one measures the cost, which
is the number a default-on decision turns on and the only one this series has never obtained
honestly.

## Why the three existing latency numbers are worthless

Every previous run timed both arms sequentially in one process with the treatment second, so page
cache, connection warmth and model state all favoured it. The same code measured **0.90x, 1.66x and
1.81x** across three runs. A spread that wide on identical work is describing cache state, not work,
and each record said so rather than quoting it. A fourth run of the same design would produce a
fourth number and no more information.

## The question

On a query that fires a fetch, what is the median paired wall-clock ratio of treatment to baseline?
And on a query that fires nothing, is it 1.00?

## Design, and what each part is for

- **Warm up first.** Every query runs through both arms once, discarded. Cold model load and cold
  page cache then fall outside the measurement instead of landing on whichever arm ran first.
- **Alternate the order.** Even-indexed queries run baseline then treatment, odd-indexed the
  reverse. Any residual advantage to running second is then split across the sample rather than
  handed to one arm, which is the exact defect in the previous three runs.
- **Pair by query, then take the median of RATIOS.** Query-to-query variance dwarfs the effect: a
  pooled p50 of all baseline times against all treatment times compares different queries. The
  paired ratio removes that.
- **Repeat and take per-query medians** before ratioing, so one scheduling hiccup cannot decide a
  query's number.
- **Report the two populations separately.** Triggering queries (stratum B, the fetch fires) and
  non-triggering (stratum A, it does not). Averaging them would hide the only cost there is behind
  the queries that pay nothing.

## Prediction

| Metric | Denominator | Prediction |
|---|---|---|
| Median paired ratio, triggering | stratum B, n=16 | **1.7x to 2.3x** |
| Median paired ratio, non triggering | stratum A, n=14 | **1.00 to 1.05x** |
| Absolute added milliseconds, triggering | stratum B | reported, not predicted |

**Why close to 2x rather than a small increment.** The scoped fetch calls `retriever.search`, which
re-embeds the query at `recall/retriever.py:548` rather than reusing the vector the first search
already computed. So a triggering query pays a second embed plus a second dense and lexical leg:
close to a whole extra retrieval, not a cheap lookup.

**Why non-triggering must be 1.00.** With no supersession edge among the hits, the expander returns
the result object unchanged and issues no query. `tests/test_successor_expansion.py` already pins
that it makes zero scoped searches on an edge-free corpus. Anything above 1.05 here means the
feature costs something while doing nothing, which is a defect rather than a trade.

## What would falsify this

- **Non-triggering above 1.05x.** A defect, and it blocks any default-on decision regardless of the
  triggering number.
- Triggering above **3.0x**, meaning more than one extra retrieval is happening per query and the
  `max_sources` bound is not doing what it should.
- Triggering below **1.3x**, which would mean the fetch is much cheaper than a retrieval and my
  reading of the path is wrong. A pleasant surprise is still a falsified prediction.
- The triggering and non-triggering split not matching the quality runs at 16 and 14, which would
  mean this is not measuring the same populations and the numbers are not comparable to them.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| Non-triggering at or below 1.05 and triggering at or below 2.5 | The cost is bounded and proportional to the work. A default-on decision becomes defensible on the existing quality evidence, and gets its own record |
| Non-triggering above 1.05 | Defect. Fix it before any default discussion |
| Triggering above 2.5 | The cost is real enough that default-on needs a caching change first, most obviously reusing the query vector instead of re-embedding |
| The population split moves | Apparatus failure, do not compare to the quality runs |

## How it will be measured

Same fixture, same corpus, same calibration, same database. A separate script, so the quality probe
keeps producing exactly the numbers the earlier records were measured with.

```bash
eval "$(scripts/session-db.sh up)"
python -m benchmarks.successor_latency
```

## Result (2026-08-20)

**Status: measured. The non-triggering prediction held exactly. The triggering prediction missed
its band, narrowly and on the low side.**

Apparatus passed: 16 triggering and 14 non-triggering, matching the quality runs, so these numbers
are comparable to them.

| Metric | Predicted | Measured |
|---|---|---|
| Median paired ratio, triggering | 1.7x to 2.3x | **1.63x** [min 1.17, max 1.84] n=16 ✗ |
| Median paired ratio, non triggering | 1.00 to 1.05x | **1.00x** n=14 ✓ |
| Added milliseconds, triggering | reported | **+45.4 ms**, 70.0 to 115.4 |
| Added milliseconds, non triggering | reported | −2.8 ms, which is noise |

**The band was missed by 0.07 and no stated falsifier fired.** The record's falsifier was 1.3x or
below, meaning the fetch is far cheaper than a retrieval and my reading of the path is wrong. 1.63x
does not reach that. The mechanism reasoning was right in kind and slightly over-stated in degree:
a triggering query does pay a second embed and a second pair of legs, but the scoped search is
cheaper than the unfiltered one it follows, because filtering to a single source leaves far less to
rank and fuse. I predicted the cost of a whole extra retrieval and got roughly two thirds of one.

**The non-triggering row is the important one for a default, and it is clean.** Median 1.00x with a
−2.8 ms delta, which is noise and not a speedup. The per-query range is wide, 0.75 to 2.02, exactly
as an unchanged code path measured against itself should look: the spread is scheduling, and the
median is the statistic that survives it. Enabling the feature on a corpus with no supersession
edges costs nothing, which the zero-extra-query test already required and this now confirms in wall
clock.

### What follows, per the decision rule fixed in advance

"Non-triggering at or below 1.05 and triggering at or below 2.5: the cost is bounded and
proportional to the work. **A default-on decision becomes defensible on the existing quality
evidence, and gets its own record.**"

Both conditions are met. That does **not** turn the feature on. It removes the one blocker this
series had against considering it, and the consideration is a separate decision with its own
record, as every default change in this series has been.

The concrete trade an operator now has, and did not have an hour ago: **+45 ms on a query that
retrieves a superseded memory whose successor is absent, and nothing at all on every other query.**

### Still not measured

Behaviour with a reranker. Any corpus but this one. Any embedder but `bge-small` through fastembed,
where the embed is a large share of the 70 ms baseline and a hosted embedder would change the
ratio in both directions at once.
