# Pre-registration: calibrating the code corpus, and whether one threshold transfers

**Date:** 2026-08-17   **Status:** predicted, not yet measured

Written and committed **before** the code query set is labelled or any threshold is fitted. The
gap between these predictions and the measurement is the output; the pass rate is not.

Companion to `2026-08-15-bge-large-voyage-splade-memory-corpus.md`, which is now fully scored.
That record ended with a prediction it could not test — whether thresholds differ between corpora
— because its design collapsed to a single tenant. This one is built to answer it.

## The question

Two things, and the second matters more operationally than the first.

1. **What abstention threshold does the code corpus calibrate to** under `voyage:voyage-code-3`?
2. **Does one code threshold transfer between code tenants**, or does each of the 21 need its own?

**Corpus under test:** `re-call-code` — 6,812 chunks from 615 files, `voyage:voyage-code-3`
(1024-dim), in the `chunks` table of `recall_repos` on VPS2. Chosen over the larger
`sentiment-agent-code` (23,245 chunks) because I can author accurate relevance labels for
recall's own source and cannot for the trading strategy code. That is a labelling-quality
decision, and it costs representativeness: 615 files of one Python project is not 21 repositories.

**Transfer target:** `mem-bench-code` — 1,351 chunks from 140 files, same embedder, different
project.

## Why the code corpus cannot simply be merged

All 21 code tenants share one embedder, so a single corpus was the obvious design. It is not
available cheaply: **379 filenames collide across the code tenants** (`__init__.py`,
`conftest.py`, `setup.py` and similar), and `relevant_ids` are keyed `<file>:<ord>`. Merging would
make those 379 ambiguous. The memory corpus had 3 such collisions and merged safely; code has two
orders of magnitude more.

Separately, **every one of the 37,160 code chunks has `project` NULL** — they were indexed before
the `--project` flag existed. Provenance survives only in the absolute `source` path
(`src/RE-call`, `local/recall-peps`). A merge would need that backfilled first.

## What I predict

| Prediction | Value | Confidence |
|---|---|---|
| **Calibrated threshold for `re-call-code`** | **0.35 to 0.50** | medium |
| Threshold difference between `re-call-code` and `mem-bench-code` | **< 0.10** | low |
| Separability (AUC) on code | **0.90 to 0.99** | low |
| Unanswerable top-1 cosine ceiling | **below 0.45** | medium |

**The threshold prediction is far below the memory corpus's 0.7100, and that is the point.** It
is not a claim that code retrieval is worse. Voyage and bge place cosines on different scales:
measured this session, `voyage-4` on `re-call-docs` returned top hits at **0.269 to 0.413**, while
`bge-large` on the memory corpus returned **0.6 to 0.8** for comparable quality. A threshold is
bound to its embedder precisely because these numbers are not comparable across models. If the
code threshold landed near 0.71 I would take that as evidence I have misunderstood the scaling,
not as evidence that code retrieval is unusually confident.

**Why transfer might hold:** same embedder, same chunker, same content type (Python source), and
the score distribution of a dense embedder is driven more by the model than by which repository
the code came from.

**Why it might not:** `re-call-code` is one coherent project with a consistent idiom;
`mem-bench-code` is a benchmark harness. Boilerplate density differs, and boilerplate is what
raises the unanswerable ceiling.

## What would falsify each

- **Threshold outside 0.35 to 0.50.** Plausible in the low direction: code boilerplate could
  compress the whole distribution downward, putting the floor under 0.35.
- **Thresholds differing by 0.10 or more between the two tenants.** That would mean a single code
  threshold cannot serve 21 tenants, and per-corpus calibration is mandatory rather than tidy.
  This is the prediction I hold most loosely; I rate it low confidence precisely because the
  memory record never got to test the equivalent.
- **Separability below 0.90**, which would say the query set does not discriminate rather than
  that the corpus is bad — the same reading the memory record applied to its own labels.
- **A negative min-answerable-minus-max-unanswerable separation**, as the memory corpus showed
  (−0.048). If code overlaps too, that is evidence the overlap is a property of dense retrieval
  on technical corpora generally, not of prose memos.

## Confounds I can name now

- **I author the labels and I predict the outcome.** Same bias the companion record names, not
  retired. Mitigation offered: the labels are committed before the fit and readable.
- **`re-call-code` is recall's own source, which I have been reading all session.** My queries
  will be drawn from what I happen to know about it, which is not a random sample of what a user
  would ask.
- **n will be small.** Around 20 answerable queries, the same size that made the memory embedder
  effect indistinguishable from zero. Any threshold difference under ~0.05 between tenants should
  be treated as unresolved rather than measured, and the transfer test will carry a bootstrap
  interval for that reason.
- **Voyage query embeddings are not deterministic** (measured previously: 42.5% of repeat calls
  differ, ~0.4% of labels flip). Two runs of this calibration will not produce byte-identical
  thresholds. The reported number is one draw.

## Result

**Status:** not yet measured. Append below; do not edit anything above.

## Measured, 2026-08-17: `re-call-code`

Artefacts: `results/calibration-code-recall-2026-08-17.json`, query set at
`docs/preregistrations/2026-08-17-code-queries.json`.

| | value |
|---|---|
| **Calibrated threshold** | **0.6620** (scale 0.0273) |
| Answerable (n=19) | min 0.661, p25 0.709, median 0.728, max 0.834 |
| Unanswerable (n=26) | min 0.480, median 0.574, p75 0.605, max 0.668 |
| Separation (min ans − max unans) | **−0.007** |
| Separability | 0.996, CI [0.975, 1.000] |
| Leave-one-out false-confident | 0.077 |
| Leave-one-out false-abstain | 0.053 |
| Certified? | **NO** — 19 answerable samples, ≥20 of each required |

### ⛔ A retrieval bug found while measuring, which invalidated the obvious method

`recall.eval.calibrate.measure_top_cosines` calls `query_dense(k=1)` and scores an empty result
as `0.0`. On this tenant that is catastrophic. Measured:

```
k=  1 ->   0 hits
k=  5 ->   0 hits
k= 20 ->   1 hit,  top=0.5277   <- WRONG, and silently so
k= 50 ->  50 hits, top=0.5743
k=100 -> 100 hits, top=0.5743   <- the true nearest neighbour
```

**10 of the 26 unanswerable queries returned NOTHING at k=1.** Each would have entered the fit as
0.0, dragging the unanswerable distribution down and the threshold with it.

Cause: the HNSW index is built over the WHOLE table and `tenant_id` is applied as a POST-filter,
so for a tenant that is a fraction of the table the graph walk returns candidates belonging to
other tenants which are then filtered away. `_query_dense` widens `ef_search` only when
`k * multiplier` exceeds pgvector's default, which never fires at k=1. Note the failure is not
merely "fewer rows": at k=20 the reported top cosine is **0.047 below** the true nearest
neighbour, so any small-k calibration understates the distribution.

This calibration therefore retrieves at **k=200** and takes `hits[0]` — the same quantity
`measure_top_cosines` intends. 0 queries came back empty at that depth.

**The memory corpus was checked and is UNAFFECTED**: all 50 queries return hits at k=1 and none
differs from k=200, so the 0.7100 threshold in the companion record stands unaltered.

### Scoring the predictions

**"Calibrated threshold 0.35 to 0.50."** **FALSIFIED**, at **0.6620** — far above, and much
closer to the memory corpus's bge-large 0.7100 than to anything predicted.

The reasoning was wrong in a specific and instructive way. I extrapolated from `voyage-4` on
`re-call-docs`, which returned top hits at 0.269–0.413, and assumed Voyage models share a cosine
scale. They do not: `voyage-code-3` returns 0.480–0.834 on the same kind of task. **A threshold is
bound to its embedder, and "Voyage" is not an embedder — `voyage-4` and `voyage-code-3` are as
different from each other in scale as either is from bge-large.**

**"Unanswerable top-1 cosine ceiling below 0.45."** **FALSIFIED**, at **0.668**. Same root cause.

**"Separability 0.90 to 0.99."** **CONFIRMED** at 0.996, at the top of the band.

**"Threshold difference between tenants < 0.10."** **NOT YET MEASURED** — needs a labelled query
set for `mem-bench-code`, which does not exist.

### What held across both corpora

**The overlap is not a property of prose.** The memory corpus separated by −0.048 and this one by
−0.007: in both, the worst answerable query scores below the best unanswerable one, so no single
cosine threshold cleanly separates them. Two different embedders, two different content types,
same qualitative result. That is now the second observation of it and it looks like a property of
dense retrieval on technical corpora rather than of any one corpus.

⚠️ **The fit is not certified and should not be published as-is.** 19 answerable samples against a
required 20 — caused by a broken label: `recall/wizard/identity.py` was named in the draft but is
absent from the index, because the code corpus was indexed BEFORE #349 added that file. The
corpus is stale relative to master, which is itself worth knowing.
