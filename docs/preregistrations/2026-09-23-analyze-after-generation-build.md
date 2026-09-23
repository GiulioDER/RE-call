# Pre-registration: what refreshing planner statistics after a generation build changes

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

`GenerationManager.build` now ends with a best-effort `ANALYZE recall_chunks_v1 (tenant_id,
generation_id)` (branch `claude/analyze-generation`). In a shared chunk table shaped like VPS2's,
does that make the planner's estimate for a freshly built generation correct, does it change the
plan or the execution time of the generation-scoped search statements, and what does it add to a
build?

Answerable by: `EXPLAIN` row estimates against the true row count, `EXPLAIN (ANALYZE, BUFFERS)` plans
and times, and the ANALYZE statement's own duration.

## Background

Measured read-only on VPS2 on 2026-09-23 (`new-generation-invisible-to-planner-stats` in memory):
the active memory generation held 11,825 rows of about 410,000 in `recall_chunks_v1` (57
generations), `n_mod_since_analyze` was exactly 11,825, and the planner estimated 41 rows for it.
The dense leg was planned as an index scan on `(tenant_id, generation_id)` plus a sort (exact
nearest-neighbour search), with or without the HNSW settings. The unit test on this branch shows
the same failure in miniature: 1 row estimated for a 20-row generation on the pre-fix code.

## Setup

VPS3, a database of this session's own (`analyze_bench`), PostgreSQL 17.11 with pgvector 0.8.6,
schema applied from this branch with the offline `hashing` embedder (64 dimensions; no Voyage
calls). The corpus is the repository's `docs/**/*.md` at `origin/master`, chunked by the default
pipeline (about 5,100 chunks per generation).

1. **Background:** 3 tenants × 10 generations of that corpus, about 150,000 rows, then one explicit
   `ANALYZE recall_chunks_v1`, so the statistics describe the background and nothing else.
2. **Arm base:** one new generation for a fourth tenant, built with `recall/generations.py` from
   `origin/master` `d9e661d7` (no refresh). Measured immediately.
3. **Arm fix:** one new generation for a fifth tenant, built with this branch. Measured immediately.

The base arm runs first on purpose: the fix arm's ANALYZE would otherwise also refresh the base
arm's generation. Each new generation is about 3% of the table, under autovacuum's threshold of
50 rows plus 10% (about 15,000), so no background ANALYZE can intervene; that is checked with
`pg_stat_user_tables.last_autoanalyze` before each measurement.

## What I predict

| id | claim | predicted |
|---|---|---|
| A1 | base: planner estimate for `generation_id = <new>` against its true row count | under 5% of the true count |
| A2 | fix: the same | within ±30% of the true count |
| A3 | fix: the same estimate for `tenant_id = <t> AND generation_id = <new>` | within a factor of 3 of the true count (the planner multiplies the two selectivities as if independent) |
| A4 | the dense statement's plan node that reads the table, base against fix | the same node type in both (an index scan on the generation index, then a sort) |
| A5 | the dense statement's execution time, fix against base, median of 11 runs each | within ±20% |
| A6 | the sparse statement's execution time, fix against base, median of 11 runs each | within ±20% |
| A7 | duration of the column-limited ANALYZE issued by the fix arm's build | 0.05 to 2 s |
| A8 | duration of a full `ANALYZE recall_chunks_v1` on the same table, for comparison, median of 3 | 0.5 to 10 s, and at least 3 times A7 |

A4 to A6 predict that, at this scale, the fix corrects the planner's picture without changing what
it does: the fix is statistical hygiene, and its value would show in larger or more varied plans,
not in these two statements. I am predicting no measurable speed-up on purpose.

## What would falsify this

- A2 outside ±30%: the column-limited ANALYZE does not give the planner a usable estimate.
- A4 showing a different plan node, or A5/A6 moving by more than 20% in either direction: the
  estimate matters more for these statements than I believe.
- A7 above 2 s: the refresh is too expensive to run unconditionally on every build.

## How it will be measured

- Row estimates: `EXPLAIN (FORMAT JSON)` of `SELECT 1 FROM recall_chunks_v1 WHERE ...`, read from
  `Plan Rows`; true counts from `count(*)`.
- Statements: the dense and sparse SQL exactly as `GenerationStore` sends them on `origin/master`,
  with a stored embedding of the new generation as the query vector and a fixed word as the sparse
  query, run with `EXPLAIN (ANALYZE, BUFFERS)` 11 times per arm after one discarded warm-up;
  the median `Execution Time` is reported with its range.
- A7: the build's own ANALYZE timed from `pg_stat_statements` (`total_exec_time` for that
  statement), reset just before the fix arm's build.
- A8: `\timing`-equivalent wall time around `ANALYZE recall_chunks_v1`, 3 runs, after both arms.

## Confounds I can name now

- Other sessions' processes share VPS3's CPU and buffers.
- 64-dimensional vectors make every row far smaller than VPS2's 1024-dimensional ones, so absolute
  times and buffer counts do not transfer; estimates and plan shapes do.
- The same corpus in every generation gives identical `tsv` and text distributions across
  generations, which is the most favourable case for leaving content-column statistics stale.
- The base and fix arms' new generations belong to different tenants but have identical content.
