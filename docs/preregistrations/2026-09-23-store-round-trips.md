# Pre-registration: fewer database round trips per trusted search on the generation store

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

On a `GenerationStore` serving an active generation, how many statements does one
`trusted_search` send to PostgreSQL before and after P3 of the core-module optimization, and by
how much does its median wall time change?

P3 makes three changes on this path: the filtered HNSW tuning is one `set_config` statement
instead of two `SET LOCAL`s; `generation_binding` is cached inside `snapshot()`; and
`max(indexed_at)`, a full scan of the generation, is cached per `(generation, corpus
fingerprint)`.

## What I predict

1. **Statements per search, steady state** (after one warm-up search, so caches are filled):
   exactly **3 fewer** statements per `trusted_search` after P3 than before, counted by
   PostgreSQL's own statement log. One from the merged tuning, one from the binding, one from
   `max(indexed_at)`.
2. **Statements on the first search after the store opens** (caches cold): exactly **1 fewer**,
   the merged tuning only.
3. **Median wall time per search**, 2,000-chunk generation, this workstation, local Docker
   PostgreSQL over loopback, `fast` profile (no reranker), 300 searches per arm, arms
   alternating in blocks: down by **3% to 20%**. Loopback round trips are cheap here, so most of
   the saving is the `max(indexed_at)` scan, which grows with the generation.

## What would falsify this

* Prediction 1 or 2: any other difference, in either direction. A difference other than 3
  would mean a statement I did not account for moved, or one of the caches does not hit.
* Prediction 3: a median change below 1%, or above 30%, or an increase.

## How it will be measured

One tenant of 2,000 chunks is built once, by this branch:
`python scripts/measure_store_round_trips.py build --dsn <session container> --chunks 2000`.
Both arms then search that same tenant, so they read identical rows:
`python scripts/measure_store_round_trips.py search --dsn <...> --tenant <t> --searches 100`,
run from this branch (`after`) and from a worktree at the P2 head `f4031c0c` (`before`), in the
order before, after, after, before, before, after (three blocks of 100 per arm, n = 300 each).
Each arm runs with `PYTHONPATH` set to its own checkout, because a bare `python` here resolves
`recall` to the editable install in the main checkout; the script prints `recall.__file__`, and a
block whose path is not its arm's checkout is discarded.

* **Statements** come from the server, not from RE-call's counter: the store's own connection
  is opened with `-c log_statement=all` (that session only), and the script counts the container
  log lines carrying that connection's backend pid, split per search by a boundary statement it
  sends between searches and excludes from the count. Before any counting it runs a known-answer
  check (five `SELECT 1` must log exactly five statements), and it refuses to report if that
  check fails. Search 1 is reported separately as the cold case; searches 2 to n are the steady
  state.
* **Wall time** is `time.perf_counter()` around each `trusted_search`, median and interquartile
  range per arm, n = 300 each.

Not measured: shared-pool mode (P3 does not change its savepoint), VPS2, and the MCP server end
to end.

## What I already know

* The P3 analyst's reading of the code: a shared-mode dense query was 8 round trips and a
  direct-mode one 5 (BEGIN, SET, SET, SELECT, COMMIT). Unmeasured until now.
* `benchmarks/store_latency_share.py` names `newest_indexed_at()` as an uncached round trip
  outside every `stage_ms` bracket.
* The in-process statement counter undercounted transaction statements until this branch, so no
  earlier `db_statement_count` figure is comparable.

## Confounds I can name now

* Both arms read one tenant, so they see the same rows and the same indexes; the ABBA-style
  block order limits the effect of a warming buffer cache on wall time.
* Other sessions load this workstation; wall time is noisy, which is why the prediction is a
  wide band and the statement count, which noise cannot move, is the primary result.
* A cold first `max(indexed_at)` may be served from shared buffers after warm-up in the `before`
  arm, making its steady-state cost small; that would push the latency result toward the bottom
  of the band, not change the statement count.

## Result (2026-09-23)
**Status:** measured

Measured on `060f2570` (after) against `f4031c0c` (before), one tenant `measure-10258185bb` of
2,000 chunks built once by the after arm, this workstation's session container, six blocks of 100
searches in the registered order. Every block's `recall.__file__` was its own arm's checkout, and
every block passed the known-answer check (five `SELECT 1`, five logged statements).

### Statements per `trusted_search` (PostgreSQL's own log)

| | before | after | change | predicted |
|---|---:|---:|---:|---:|
| first search after open | 17 | 16 | -1 | -1 |
| steady state (searches 2 to 100, all six blocks) | 9 | 6 | -3 | -3 |

Every steady-state search in every block had the same count; there was no variance to report.
The steady-state statements, in order:

* before: active-generation read, generation-binding read, `BEGIN`, `SET LOCAL hnsw.ef_search`,
  `SET LOCAL hnsw.iterative_scan`, dense `SELECT`, `COMMIT`, sparse `SELECT`, `max(indexed_at)`.
* after: active-generation read, `BEGIN`, one `set_config` for both GUCs, dense `SELECT`,
  `COMMIT`, sparse `SELECT`.

(The dense and sparse `SELECT`s log with empty text because their SQL begins with a newline; each
is still one logged statement.)

**Gap:** none. Both predictions held exactly, and the three removed statements are the three
named in the prediction.

### Median wall time per search (n = 300 per arm)

| arm | block medians (ms) | pooled median | pooled IQR |
|---|---|---:|---|
| before | 28.53, 25.55, 26.94 | 26.66 ms | 25.36 to 28.90 |
| after | 20.91, 20.71, 21.89 | 21.04 ms | 20.12 to 22.54 |

Measured: down 5.61 ms, **21.1%**. Predicted: down 3% to 20%; falsified only below 1%, above
30%, or as an increase. **Gap:** just above the predicted band, and inside the not-falsified
range. Every after block was faster than every before block.

The reason is worth keeping, because it limits what this number means elsewhere. Three statements
cost 5.6 ms here, about 1.9 ms each, which is the round-trip cost of Docker Desktop's virtualised
loopback on Windows, not of PostgreSQL. On a host where the server and the database share a
native loopback (VPS2) a round trip is far cheaper, so the absolute saving there will be much
smaller than 5.6 ms; the statement count is the portable result, and the percentage is not.
Not measured on VPS2, as stated above.
