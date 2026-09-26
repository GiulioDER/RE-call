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
