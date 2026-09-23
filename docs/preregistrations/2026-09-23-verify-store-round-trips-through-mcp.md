# Pre-registration: what PR #711 changes for a `recall_search` call through the MCP server

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

PR #711 (squash commit `2997b00c`, "fewer round trips per trusted search on the generation store")
measured 9 → 6 statements and 26.66 → 21.04 ms per search with a direct store benchmark on this
workstation's Docker Desktop, and its record says the absolute saving would be smaller on a native
loopback. Through the real MCP server, on a native host, how many statements and how much
Postgres execution time does it remove per search, and is the end-to-end latency change visible?

Answerable by: `pg_stat_statements` per run (executions and `total_exec_time`, per statement) and
client-side latency per call.

## Setup

Identical to `2026-09-23-mcp-search-postgres-profile.md` (run 2), which is the baseline: VPS3,
database `pyspy_profile`, tenant `pyspy`, generation `gen_03a17146e8ce4668a2b2d8ff683a1ca0`
(5,117 chunks, `voyage-context:voyage-context-4`), `RECALL_INDEX_MODE=generation`,
`RECALL_TRUST_MODE=development`, the lean venv, the same 80 queries and one excluded warm-up call
per server process. Two arms, each a git worktree selected by `PYTHONPATH` and checked by
`recall.__file__`:

- **base**: `0365d30d`, master immediately before #711;
- **pr**: `2997b00c`, #711's squash commit on master.

#711 adds no migration and changes no dependency, so both arms share the database unchanged.

n = 3 runs per arm, alternating base, pr, base, pr, base, pr. Each run is a fresh server process;
`pg_stat_statements` is reset after its warm-up call and read after its 80 calls. py-spy is not
attached (it would add sampling overhead to a latency comparison).

## What I predict

| id | claim | predicted |
|---|---|---|
| H1 | statement executions per search, base arm (excluding the reset) | 6.8 to 7.5 (the baseline measured 7.15) |
| H2 | the same, pr arm | 4.6 to 5.6: about two fewer, the generation-binding read and `max(indexed_at)` |
| H3 | `max(indexed_at)` executions inside the 80-call window: base / pr | 80 / 0 (pr caches it during the warm-up call) |
| H4 | Postgres execution time per search: base | 12 to 17 ms |
| H5 | the same, pr, as a reduction from base | 25% to 45% lower |
| H6 | the dense and sparse statements' mean execution times, pr against base | each within ±25% |
| H7 | pooled median client latency, pr minus base | between -12 ms and +5 ms, with the two arms' interquartile ranges overlapping |

H7 predicts that the end-to-end change is too small to see next to the Voyage round trip, not that
it is zero.

## What would falsify this

- H2 not below H1 by at least 1: #711's cut does not reach the MCP path.
- H3 pr above 0: the cache does not survive from the warm-up call into the measured calls.
- H5 below 15%: the removed statements were not the expensive ones on this path.
- H7 showing pr slower than base by more than 5 ms.

## What I already know

- `2026-09-23-mcp-search-postgres-profile.md`: base-equivalent code (`cbd54efa`) spent 14.07 ms of
  Postgres execution per search; `max(indexed_at)` was 36.1% of it (5.08 ms mean, a sequential scan
  of the generation); 7.15 executions per search; `SET LOCAL hnsw.*` ran in 5 of 80 searches only;
  median latency 243 ms.
- #711's record: statements removed were the generation-binding read, `max(indexed_at)`, and one of
  the two `SET LOCAL`s (merged into one `set_config`).

## Confounds I can name now

- Voyage latency varies from minute to minute; alternating the arms spreads that across both.
- Other sessions' processes on VPS3 share CPU and buffers; statements are filtered to this database.
- The warm-up call is where pr fills its cache; if the cache is keyed on something the warm-up does
  not touch, H3 fails for a reason worth knowing.
- `SET LOCAL` ran in only 5 of 80 baseline searches, so #711's `set_config` merge affects few calls
  here, unlike in its own benchmark where it ran every search.
