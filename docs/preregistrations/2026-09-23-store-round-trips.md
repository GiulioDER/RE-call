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
