# Pre-registration: where a `recall_search` call spends its time, Python side and Postgres side

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

For one `recall_search` call through the real MCP server on the production generation-store path,
how much of the time is Postgres, which statements dominate it, and would any extra index help?

Answerable by: `pg_stat_statements` totals per statement, client-side latency per call, py-spy
sample shares on the server process, and HypoPG planned-cost deltas.

## Setup

VPS3 only. Session-owned database `pyspy_profile` (tenant `pyspy`): the repository's `docs/**/*.md`
at `origin/master` `cbd54efa` (381 files, 5,117 chunks), embedded with
`voyage-context:voyage-context-4` (the live memory tenant's model), generation
`gen_03a17146e8ce4668a2b2d8ff683a1ca0` promoted with `--unsafe-development-promotion`, so the server
reads it through the generation store as production does. The server runs with
`RECALL_TRUST_MODE=development` because this corpus has no certified calibration. PostgreSQL 17.11,
pgvector 0.8.6, `pg_stat_statements` preloaded, HypoPG 1.4.3. Code: `origin/master` `cbd54efa`, in
the lean venv (no `torch`).

## What I predict

The workload: the 80 queries of `docs-queries-40-40.vps2-20260824.json` (40 answerable, 40 not),
each sent once as `recall_search(query=...)` over stdio, after one warm-up call that is excluded.
`pg_stat_statements` is reset after the warm-up and read filtered to this database.

| id | claim | predicted |
|---|---|---|
| G1 | Postgres execution time per search: `sum(total_exec_time)` / 80 | 5 to 40 ms |
| G2 | the statement with the largest `total_exec_time` | the dense vector (HNSW) query over the chunk table, at 40% or more of all execution time |
| G3 | client-side latency per call, median | 250 to 800 ms |
| G4 | Postgres share of the client-side total, G1 × 80 / sum of latencies | under 20% |
| G5 | SQL statement executions per search, `sum(calls)` / 80 | 5 to 30 |
| G6 | py-spy on the server, share of all samples (idle included) that are idle waits | 50% or more: the server is I/O bound |
| G7 | HypoPG: for each of the 3 statements with the largest `total_exec_time`, the best single hypothetical B-tree index on columns from its WHERE clause lowers `EXPLAIN (GENERIC_PLAN)` total cost by 20% or more | in none of the 3 |

G7 is a null prediction on purpose: the chunk table already carries its HNSW index and the
generation filters, and I expect no cheap index to be missing. A hit would be the actionable result.

## What would falsify this

- G1 above 40 ms, or G2 naming a non-vector statement: the database is doing work I do not expect.
- G4 at 20% or more: Postgres is a real share of search latency and worth optimizing.
- G7 finding an index that cuts a top statement's cost by 20% or more.

## How it will be measured

- A harness on VPS3 starts `python -m recall_mcp.server` over stdio with the `mcp` client, finds the
  server's pid from `/proc` (the child of the harness running `recall_mcp.server`), and attaches
  `py-spy record --pid <pid> --idle --rate 100 --format raw` as root for the length of the run.
- Per call: `time.perf_counter()` around `call_tool`, and whether the result is an error.
- After the run: `pg_stat_statements` rows for this database, ordered by `total_exec_time`
  (`calls`, `total_exec_time`, `mean_exec_time`, `rows`, `shared_blks_hit`, `shared_blks_read`).
- HypoPG afterwards, per top statement: `EXPLAIN (GENERIC_PLAN)` total cost with no hypothetical
  index, then with each candidate B-tree index, one at a time, reset between candidates.
- n: 80 search calls, one run. About 81 Voyage calls.

## What I already know

- `2026-09-23-voyage-cold-start-profile.md`: a Voyage context-4 `embed_query` takes about 0.22 s
  from VPS3, so network time should dominate a single search.
- The live VPS2 servers do not rerank (`RECALL_RERANK` unset), and neither will this one.

## Confounds I can name now

- VPS3 has other sessions' MCP servers and services. Their statements are excluded by filtering
  `pg_stat_statements` on this database's `dbid`, but they share CPU and buffers.
- 5,117 chunks is small beside VPS2's memory tenant (about 9,800), so absolute database times
  will be low; the shares and the statement ranking are what transfer.
- Warm cache: the corpus was just built, so most pages are in shared buffers. A cold server would
  read more from disk.
- Development trust mode may skip gates that a certified corpus would run, so some trust-layer
  statements may be missing from this workload.

## Apparatus failure, run 1 (2026-09-23, 13:44 UTC), disclosed before the rerun

The first run measured the wrong path and is **void**. The server started with
`retrieval profile legacy`: `RECALL_ENV` was unset, so `recall/runtime_route.py` defaulted
`RECALL_INDEX_MODE` to `legacy` (production gets `generation` from `RECALL_ENV=production`). It
therefore queried the legacy `chunks` table, which is empty for this tenant: the top statement
returned 0 rows in all 160 of its calls, and every search came back empty without an error. The
80 calls, the 0.227 s median latency and the `pg_stat_statements` rows from that run describe an
empty legacy path, and none of G1 to G7 is scored from them.

What I missed is the check this record should have had from the start: a search that returns
nothing is not a failed call, so "80 calls, 0 errors" looked like success. The rerun adds
`RECALL_INDEX_MODE=generation` (development trust is kept, since this corpus has no certified
calibration), records the hit count of every call, and is void unless the answerable queries
return hits. The predictions above are unchanged.

## Result, run 2 (2026-09-23, 13:47 UTC)

**Status:** measured

Apparatus checks passed: the server bound generation `gen_03a17146e8ce4668a2b2d8ff683a1ca0`, all 80
calls returned without error, and every call returned hits (40 of 40 answerable, 40 of 40
unanswerable; the unanswerable ones return hits because this corpus has no calibrated threshold to
abstain on). py-spy exited 0 with 14,693 samples.

| id | predicted | measured | held |
|---|---|---|---|
| G1 | 5 to 40 ms of Postgres execution per search | 14.07 ms (1,125.6 ms over 80) | yes |
| G2 | the dense vector query is the top statement, at 40% or more | **no: the top statement is `SELECT max(indexed_at)` at 36.1%; the vector query is third at 29.9%** | **no** |
| G3 | median latency 250 to 800 ms | **243 ms** (p90 264 ms, max 584 ms) | **no, 7 ms under the band** |
| G4 | Postgres under 20% of client time | 5.62% | yes |
| G5 | 5 to 30 statement executions per search | 7.15 | yes |
| G6 | 50% or more idle samples | 85.4% | yes (see caveat) |
| G7 | no WHERE-column B-tree cuts a top-3 cost by 20% or more | 0.0% change on all three | yes, but vacuously (see below) |

The three statements that make up 99.5% of Postgres time, per search:

| rank | statement | share | mean | buffers per call |
|---|---|---:|---:|---:|
| 1 | `SELECT max(indexed_at) FROM recall_chunks_v1 WHERE tenant_id = $1 AND generation_id = $2` | 36.1% | 5.08 ms | 1,070 |
| 2 | the sparse (`tsv @@ tsquery`, `ts_rank`) leg | 33.5% | 4.72 ms | 1,227 |
| 3 | the dense leg, `ORDER BY embedding <=> $1 LIMIT $4` | 29.9% | 4.21 ms | 1,977 |

**G2 and the finding it exposed.** Statement 1 is `GenerationStore._newest_indexed_at`
(`recall/generation_store.py`), which `HybridRetriever.search` calls once per query for its
staleness report. `EXPLAIN (ANALYZE, BUFFERS)` with the real tenant and generation shows a
sequential scan over all 5,117 rows of the generation (10.7 ms, 1,070 buffers) to return one
timestamp. A promoted generation is immutable, so that value cannot change for a given
`generation_id`: the query recomputes a constant on every search.

**G7 was vacuous, and I should have seen that before registering it.** The only B-tree on the
WHERE columns of all three statements, `(tenant_id, generation_id)`, already exists
(`recall_chunks_v1_generation_idx`), so the registered test could only return 0%. The extended
candidate, outside the registered prediction and labelled as such, is the one that matters: a
hypothetical `(tenant_id, generation_id, indexed_at)` turns statement 1 into a backward
index-only scan and takes its generic-plan cost from 1159.6 to 0.1. Either that index or caching
the value per `generation_id` removes about a third of all Postgres time per search.

**A second finding, not predicted.** The dense leg (statement 3) is planned as a sequential scan
plus a top-N heapsort, an exact nearest-neighbour search over the whole generation, not an HNSW
index scan (confirmed with `EXPLAIN (ANALYZE, BUFFERS)` using a stored embedding). The
`SET LOCAL hnsw.*` statements ran in only 5 of the 80 searches, so a different path takes HNSW. At
5,117 rows the exact scan is cheap (4.21 ms); its cost grows with generation size, and whether
VPS2's memory tenant plans it the same way was not measured here.

**G3.** The miss is small and one-directional: a search is almost exactly one Voyage round trip
(the context-4 query embed measured 0.22 s in `2026-09-23-voyage-cold-start-profile.md`), plus
about 14 ms of Postgres and a few ms of Python. I padded the band for server overhead that is not
there.

**G6 caveat.** "Idle" is classified from the leaf frame's name (`wait`, `select`, `read` and
similar), so it is a heuristic. The largest leaf, `threading.wait` at 57.1%, is the event loop's
side of the worker thread that runs the synchronous store call.

**Gap.** Four of seven held, two missed, and one held only because it was built so that it could
not fail. The useful surprise is the reverse of what I expected: the most expensive statement per
search is not retrieval at all but a freshness check, and it is also the cheapest to remove.
