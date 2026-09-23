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
