# Pre-registration: the per-tenant BM25 index makes C9's canonical BM25 leg cheap per Search

**Date:** 2026-09-26   **Status:** predicted, not yet measured

## The question

On a synthetic 13,097 chunk Code4 tenant in a local session database, what is the per-Search wall
time of C9's canonical BM25 leg with `TenantSearchCache.rank_bm25` (warm) against the code it
replaces, `rank_bm25_chunks(list(store.iter_chunks()), ...)`, and do the two return identical
rankings?

## What I predict

The memo `i-over-predict-effect-magnitudes` says my magnitudes run two to four times high, so
these are deliberately moderated.

- **old**, sequential: median 1.5 s to 5 s per Search.
- **cold** (first Search: fingerprint, full read, build, rank): 1.2x to 2.5x the old median.
- **warm**, sequential: median 15 ms to 120 ms, so a speedup of roughly 20x to 100x.
- **fingerprint query** alone: median 3 ms to 30 ms.
- **supersession**: uncached scan 5 ms to 60 ms; a cache hit costs about one fingerprint query.
- **concurrent**: old with 4 threads at once is at least 2x its sequential median (GIL bound);
  warm with 8 threads stays under 400 ms median.
- **memory**: snapshot plus its rows 40 MB to 120 MB of Python heap.
- **parity**: 0 mismatches over every checked query.

## What would falsify this

- Any parity mismatch (that alone blocks shipping, whatever the timings).
- Warm speedup under 10x against old.
- Snapshot heap over 150 MB for this tenant, which would make `MAX_BM25_TENANTS = 8` unsafe.
- Fingerprint query over 100 ms, which would make the per-Search guard the new bottleneck.

## How it will be measured

```bash
eval "$(scripts/session-db.sh up)"
python scripts/bench_aml_search_cache.py --dsn "$RECALL_TEST_DSN"
```

13,097 raw windows built by `build_chunks` at 160/120 words, 1024 dim noise vectors, a Zipf-like
30,000 term vocabulary. old: 8 queries; warm: 60 queries; parity checked on the 8 old queries
plus the cold one; fingerprint 20 calls; supersession 10 each; concurrency 4 old threads and 8
warm threads. Wall milliseconds from `time.perf_counter`, medians with min and max.

## What I already know

The audit behind this change reported each production Search on a 13,097 memory tenant taking
seconds of GIL-bound Python, and 8 to 30 s under concurrency. Not re-measured here: production is
serving an official run and must not be touched.

## Confounds I can name now

- This workstation is not the production host (Windows, 12 GB, other sessions' load), so absolute
  numbers transfer poorly; the ratio is the claim.
- Synthetic text: real windows repeat far more (code, logs), which changes postings lengths.
  A query of very common terms scores more candidates, so warm time depends on the vocabulary.
- The local database is on the same machine, so network transfer in `iter_chunks` is cheaper
  than production's; that flatters old, which biases against the prediction.

## Result (2026-09-26)
**Status:** measured

Four runs of `scripts/bench_aml_search_cache.py` against this checkout's session database, 13,097
windows, 1024 dim. Runs 1 and 2 are the committed script; runs 3 and 4 add the graph tenant
section (1,682 compiled records from `build_chunks` and `attach_grounded_relations`), which also
changes the random stream, so the raw corpus differs slightly. Run 4 was taken on a visibly loaded
box (its old concurrent median doubled), and is kept as the top of the range.

| Measure | Predicted | Measured (runs 1 to 4) |
|---|---|---|
| old, sequential median | 1.5 s to 5 s | 2,354 / 2,442 / 3,225 / 2,603 ms |
| cold (first cached Search) | 1.2x to 2.5x old | 3,110 / 3,105 / 4,134 / 4,170 ms (1.3x to 1.6x) |
| warm median (n = 60) | 15 ms to 120 ms | 28.7 / 29.9 / 31.4 / 47.9 ms |
| warm speedup over old median | 20x to 100x | 82x / 82x / 103x / 54x |
| fingerprint query median | 3 ms to 30 ms | 11.9 / 11.5 / 12.5 / 14.5 ms |
| raw supersession, uncached | 5 ms to 60 ms | 10.5 / 8.8 / 11.9 / 14.4 ms |
| raw supersession, cached hit | about one fingerprint | 13.7 / 12.1 / 14.1 / 45.5 ms |
| graph supersession, uncached | (not predicted) | 58.6 / 71.6 ms |
| graph supersession, cached hit | (not predicted) | 3.5 / 4.7 ms |
| old, 4 concurrent, median | at least 2x sequential | 6,961 / 6,574 / 7,056 / 15,220 ms (2.7x to 5.8x) |
| warm, 8 concurrent, median | under 400 ms | 166 / 194 / 67 / 123 ms |
| heap per tenant | 40 MB to 120 MB | 48.3 MB rows plus 20.5 MB index, 68.8 MB |
| parity mismatches | 0 | 0 of 9 per run, 36 total; graph supersession equal in both runs |

**Gap.** The BM25 prediction held: every warm median sits inside the predicted band and the
speedup is 54x to 103x, near the top of my moderated range, which is the first time the
over-prediction correction has itself been slightly too cautious. Parity held on every check.

**The supersession prediction was wrong in a way that changed what ships.** I predicted a cache
hit would cost "about one fingerprint query" and implicitly that this would be cheaper than the
scan. For the RAW tenant it is not: raw windows carry no `supersedes`, so the unindexed scan is a
heap pass with a cheap filter, and the fingerprint is a heap pass too (it needs `xmin`, which no
index holds). The cached path measured 12 to 45 ms against 9 to 14 ms uncached. So the raw
tenant's supersession read stays uncached; only the graph sidecar's scan, over large compiled
record metadata, goes through the cache, where it is 15x cheaper. A cheaper fingerprint would need
an index covering `(tenant_id, indexed_at)` and would lose `xmin`, which the backdated update test
shows is the component that catches a same-count, same-latest-time update.

Memory: at 68.8 MB per 13,097 window tenant, the planned bound of eight tenants per retriever
(about 1.1 GB for the process's two retrievers) was too generous for a host that also runs live
trading services; `MAX_BM25_TENANTS` is four.

The absolute numbers are this workstation's with a local database; the ratio is the claim, and
production was not touched to confirm it, because an official run is live there.

## Re-measurement (2026-09-26, evening), on the combined branch

**Why.** The result above was written by a session that was interrupted before it could hand it
back; I checked it but had not run it myself. The user asked for it to be redone. Nothing above
this heading is edited.

**What ran.** `PYTHONPATH=. python scripts/bench_aml_search_cache.py --dsn "$RECALL_TEST_DSN"`,
twice back to back, on `claude/c9-speed` at `868491d0` (the Search, Add and compile output branches
merged), against this checkout's own session database. A first attempt without `PYTHONPATH`
imported `recall_aml` from another checkout's editable install and failed with
`ModuleNotFoundError: recall_aml.search_cache` before measuring anything; the script now puts its
own checkout first on `sys.path`, so that cannot recur.

| Measure | Predicted | Rerun 1 (loaded box) | Rerun 2 |
|---|---|---|---|
| old, sequential median | 1.5 s to 5 s | 21,979 ms | 2,827 ms |
| cold (first cached Search) | 1.2x to 2.5x old | 24,076 ms (1.1x) | 6,129 ms (2.2x) |
| warm median (n = 60) | 15 ms to 120 ms | 110.9 ms | 52.8 ms |
| warm speedup over old median | 20x to 100x | 198x | 54x |
| fingerprint query median | 3 ms to 30 ms | 47.7 ms | 16.3 ms |
| raw supersession, uncached | 5 ms to 60 ms | 32.3 ms | 12.8 ms |
| raw supersession, cached hit | about one fingerprint | 46.4 ms | 21.2 ms |
| graph supersession, uncached / cached | (not predicted) | 278.0 / 18.2 ms | 82.8 / 4.8 ms |
| old, 4 concurrent, median | at least 2x sequential | 26,974 ms | 21,216 ms (7.5x) |
| warm, 8 concurrent, median | under 400 ms | 746.5 ms | 372.6 ms |
| heap per tenant | 40 MB to 120 MB | 48.3 MB rows plus 20.5 MB index | same |
| parity mismatches | 0 | 0 of 9, graph equal | 0 of 9, graph equal |

**Rerun 1 is a loaded-box run and I count it as such, not as a clean sample.** My own mutation
test runs overlapped it (six pytest processes started while it ran), and the workstation had
about 3.5 GB of 12 GB free with other sessions active. Its old median is 7.8 times rerun 2's. It
misses two bands (fingerprint over 30 ms, warm concurrent over 400 ms) and passes the falsifiers;
its parity result stands, since parity does not depend on load.

**Rerun 2 agrees with the result above on every predicted band:** warm 52.8 ms, 54x, cold 2.2x,
fingerprint 16.3 ms, warm concurrent 373 ms, parity 0 of 9. Across all six runs now recorded the
warm speedup is 54x to 198x and parity has never mismatched (54 checks). It also confirms the
decision above to leave the raw supersession read uncached: cached 21.2 ms against 12.8 ms direct,
while the graph sidecar's read goes from 82.8 ms to 4.8 ms through the cache.

**No falsifier fired** in either rerun: no parity mismatch, warm speedup at least 54x, heap
68.8 MB, fingerprint at most 47.7 ms.
