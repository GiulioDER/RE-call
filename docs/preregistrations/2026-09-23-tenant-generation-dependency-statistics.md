# Pre-registration: what a tenant-generation dependency statistic changes

**Date:** 2026-09-23   **Status:** predicted, not yet measured

## The question

Migration 0026 (branch `claude/extended-stats`) creates `CREATE STATISTICS
recall_chunks_v1_tenant_generation_deps (dependencies) ON tenant_id, generation_id FROM
recall_chunks_v1` and analyzes those two columns. Does it make the planner's estimate for
`tenant_id = t AND generation_id = g`, the predicate every generation-scoped query carries,
correct, and does that change the plan or the time of the dense and sparse search statements?

Answerable by: `EXPLAIN` row estimates against the true count, `EXPLAIN (ANALYZE, BUFFERS)` plans and
times, `pg_statistic_ext_data`, and the migration's own duration.

## Background

`2026-09-23-analyze-after-generation-build.md` (result, A3): after the post-build ANALYZE (#728),
the single-column estimate for a new 5,176-row generation was 5,026, but the pair estimate was
**152**, a factor of 34 low, because the planner multiplies the two columns' selectivities as if
they were independent (3.1% × 3.1% of 165,632 rows). The unit test on this branch shows the same
in miniature: 6 rows estimated for a 20-row generation without 0026, and it passes with 0026.

## Setup

VPS3, the existing database `analyze_bench` from that record: 30 background generations of 3
tenants plus the base and fix arms' generations, 165,632 rows, all 64-dimensional (`hashing`
embedder), at migration 0025. The generation measured is the fix arm's `new-fix` generation
(5,176 rows). Code: this branch (`21fac7d2` or later, same migration bytes).

1. **Without the statistic:** run `ANALYZE recall_chunks_v1 (tenant_id, generation_id)` so both
   phases start from freshly analyzed columns, then measure.
2. **With the statistic:** apply 0026 through `recall.cli schema apply` with this branch's code
   (it creates the statistic and analyzes the two columns), then measure the same generation.

Same data, same session, back to back; only the statistics object differs.

## What I predict

| id | claim | predicted |
|---|---|---|
| E1 | pair estimate without the statistic | 100 to 250 (the independence product, about 160) |
| E2 | pair estimate with the statistic | within ±30% of the true 5,176 |
| E3 | single-column `generation_id` estimate, both phases | within ±30% of the true count, and within ±10% of each other |
| E4 | `pg_statistic_ext_data` for the new statistic after the migration's column-limited ANALYZE | populated (non-null `stxddependencies`), with a `generation_id → tenant_id` degree of at least 0.95 |
| E5 | the dense statement: plan node reading the table, with against without | the same node (index scan on the generation index, then a sort) |
| E6 | the dense statement's median time, 11 runs each | within ±20% |
| E7 | the sparse statement: plan node reading the table | the same node in both phases (index scan on the generation index, filtered by `tsv @@`) |
| E8 | the sparse statement's median time, 11 runs each | within ±20% |
| E9 | `schema apply` wall time for 0026 on this 165,632-row table | under 3 s |

E5 to E8 predict no plan change, the same stance as the previous record: this corrects the
planner's picture, and at this scale the cheapest plans do not depend on it. E7 is my least
certain prediction. With an estimate of 5,176 instead of 152 rows, a GIN bitmap on `tsv` becomes
more attractive for the sparse leg, and a switch would be the most informative result here.

## What would falsify this

- E2 outside ±30%: the dependency statistic does not fix the estimate every serving query uses.
- E4 empty: the column-limited ANALYZE issued after every build does not maintain the statistic,
  so the fix would decay as new generations arrive.
- E6 or E8 slower by more than 20%: a better estimate produced a worse plan.
- E9 at 3 s or more: the migration is too heavy to apply casually on VPS2's larger table.

## How it will be measured

- Estimates: `EXPLAIN (FORMAT JSON) SELECT 1 FROM recall_chunks_v1 WHERE ...`, `Plan Rows`; true
  count from `count(*)`.
- Statements: the dense and sparse SQL as `GenerationStore` sends them, with a stored embedding
  of the generation as the query vector and the fixed text `generation promotion calibration
  threshold`, 11 `EXPLAIN (ANALYZE, BUFFERS)` runs after one discarded warm-up; the median
  `Execution Time` is reported with its range, plus the plan nodes that read the table.
- E4: `SELECT stxddependencies FROM pg_statistic_ext s JOIN pg_statistic_ext_data d ON
  d.stxoid = s.oid WHERE s.stxname = 'recall_chunks_v1_tenant_generation_deps'`.
- E9: wall time around the `schema apply` command.

## Confounds I can name now

- Other sessions share VPS3's CPU and buffers; the two phases run minutes apart.
- 64-dimensional vectors, so absolute times do not transfer to VPS2; estimates and plans do.
- Every generation holds the same corpus, so content-column statistics are as favourable as they
  can be.
