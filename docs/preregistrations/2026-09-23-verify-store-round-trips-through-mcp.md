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

## Result (2026-09-23)

**Status:** measured

Six runs, alternating base (`0365d30d`) and pr (`2997b00c`), each a fresh server whose cwd was its
own arm's worktree. Every run: 80 calls, 0 errors, 40 of 40 answerable and 40 of 40 unanswerable
queries with hits. At the start of the sequence one other session's `pytest` or C8 process was
running on VPS3 (the pre-run check counted 1); it had finished by the time I looked again. It
shares CPU and buffers with both arms, and the alternation spreads it across them.

| id | predicted | measured | held |
|---|---|---|---|
| H1 | base: 6.8 to 7.5 executions per search | 7.14 counted (all three runs); about **9.0 actually executed**, see below | only as counted |
| H2 | pr: 4.6 to 5.6 | **6.01** (all three runs) | **no** |
| H3 | `max(indexed_at)` in 80 calls: base / pr | 80 / 0 in every run | yes |
| H4 | base Postgres time per search: 12 to 17 ms | 12.69 ms (12.18, 11.69, 14.21) | yes |
| H5 | pr: 25% to 45% lower | 8.41 ms (7.71, 7.95, 9.57), **-33.8%** | yes |
| H6 | dense and sparse means within ±25% | dense -1.5%, sparse +4.6% | yes |
| H7 | pooled median latency pr minus base: -12 to +5 ms, IQRs overlapping | **-2.7 ms** (241.4 against 238.7; IQR 237.1 to 246.5 against 233.1 to 244.8) | yes |

Per search, the statements that changed: the generation-binding read (80 → 0) and
`max(indexed_at)` (80 → 0) are gone, as #711's record says.

**Why H2 missed, and what it exposed.** The HNSW settings statements were counted 10 times in each
base run (5 searches × 2) and 80 times in each pr run. #711 did not start running them more often:
it merged two `SET LOCAL`s into one `SELECT set_config(...)` at the same code site. The difference
is in the measuring instrument. psycopg prepares a statement after 5 executions
(`prepare_threshold=5`), and `pg_stat_statements` stops counting a utility statement such as
`SET LOCAL` once it runs prepared; `set_config` is a `SELECT` and is counted every time. Checked
directly on the same server with a known answer: 12 executions of `SET LOCAL hnsw.ef_search` through
psycopg's defaults were counted 6 times, and 12 times with `prepare_threshold=None`.

So the base arm actually executed about 9.0 statements per search (7.14 counted, plus the 150 of 160
`SET LOCAL`s that went uncounted), and #711 cuts that to 6.01: **three fewer per search, matching
#711's own record exactly** (9 → 6 in its steady state). My H1 and H2 were built on the undercounted
baseline, which is why H1 "held" only in the counted sense and H2 missed. The uncounted statements
run in about 0.01 ms each, so the execution-time results (H4, H5) are unaffected.

**This also corrects my baseline record** (`2026-09-23-mcp-search-postgres-profile.md`): its 7.15
executions per search are an undercount of the same kind, and its inference that
"`SET LOCAL hnsw.*` ran in only 5 of the 80 searches, so a different path takes HNSW" is wrong.
The settings ran in every search, and the dense leg is planned as an exact sequential scan anyway.
The correction is appended to that record itself as well.

**Gap.** Five of seven held, and the miss taught more than the hits: a statement counter that
silently stops counting after five executions made a three-statement cut look like a
one-statement cut. On a native loopback, #711 removes a third of Postgres execution time per
search (4.3 ms of 12.7 ms), which is about 2.7 ms of a 240 ms search, too small to see end to end
next to the Voyage round trip.
