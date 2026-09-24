# Pre-registration: what migration 0026 costs in recall, now that it moves dense search onto HNSW

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

`2026-09-23-tenant-generation-dependency-statistics.md` found that migration 0026 fixes the
planner's estimate for `tenant_id AND generation_id` and, as a direct consequence, moves the dense
leg from an **exact** scan of the generation (B-tree plus sort) to the **approximate** HNSW index.
In a table shaped like VPS2's, with real 1024-dimensional vectors, how much of the exact top-k does
the HNSW path return, and does what a caller of `recall_search` receives change?

Answerable by: recall@k of the HNSW result against the exact result for the same stored query
vector, and the overlap of the top 5 results `recall_search` returns before and after 0026.

## Setup

VPS3, the session-owned database `pyspy_profile`, at migration 0025. Its one generation
`gen_03a17146e8ce4668a2b2d8ff683a1ca0` (tenant `pyspy`, 5,117 chunks of the repository's docs,
`voyage-context:voyage-context-4`, 1024 dimensions) is the target.

To give the table VPS2's shape, the target generation's rows are copied into 29 further
generations with new generation ids: 9 more in tenant `pyspy` (state `retired`) and 10 each in
tenants `pyspy-b` and `pyspy-c`. That makes 30 generations and about 153,500 rows, with the target at
about 3.3% of the table (VPS2's active memory generation was 11,825 of about 410,000, 2.9%). The
copies carry **identical vectors**, which is what repeated memory builds with the embedding cache
produce, and it is the hardest case for a filtered HNSW index, because every point's nearest
neighbours are its own duplicates in other generations. The HNSW index is dropped before the copy
and rebuilt afterwards with the migration's own definition, then `ANALYZE recall_chunks_v1` runs.

Code: `origin/master` for the arm without 0026 (it only knows up to 0025); this branch
(`eccbdcdb` or later) for the arm with it. Their runtime code is identical: the branch adds only
the migration, its checksum, tests and records.

## Measurements

**M1, dense leg, same query vector.** The 80 queries of `docs-queries-40-40.vps2-20260824.json`
are embedded once with `voyage-context-4` (`input_type=query`) and stored. After 0026 is applied,
for each stored vector and for k = 20 (the MCP server's candidate count) and k = 100 (the hosted
quality profile's):
- **exact:** `ORDER BY embedding <=> q LIMIT k` over the generation's rows, isolated with an
  `OFFSET 0` subquery so the index cannot be used;
- **HNSW:** the SQL `GenerationStore._query_dense` sends, inside one transaction with
  `set_config('hnsw.ef_search', max(200, min(4k, 1000)))` and `hnsw.iterative_scan =
  relaxed_order`, exactly as `_hnsw_filtered_tuning` computes them. The plan is checked per query:
  a query whose plan does not read `recall_chunks_v1_embedding_idx` is reported, not counted.

**M2, what the caller receives.** `recall_search` over stdio through the real MCP server (the
harness from `2026-09-23-mcp-search-postgres-profile.md`, `RECALL_INDEX_MODE=generation`,
development trust), the 80 queries, the top 5 chunk ids per call recorded. Three runs: **A1** and
**A2** without 0026 (master), then **B** with 0026 (this branch). A1 against A2 is the noise floor,
which exists because the provider returns one of two vectors for some queries
(`2026-09-23-verify-voyage-http-client-in-vps2-mirror.md`); A against B is the effect.

## What I predict

| id | claim | predicted |
|---|---|---|
| P1 | apparatus: target generation's share of the table | 3.0% to 3.6% |
| P2 | apparatus: without 0026 the dense plan reads the generation index (exact); with it, the HNSW index | both, in every query checked |
| R1 | M1 mean recall@20, HNSW against exact | 0.80 to 0.98 |
| R2 | M1 queries with recall@20 = 1.0 | at least 50% of 80 |
| R3 | M1 mean recall@100 (ef_search 400) | 0.75 to 0.98 |
| R4 | M1 queries where HNSW returns fewer than k rows | 0 at k = 20 |
| R5 | M2 noise floor: mean top-5 overlap A1 against A2 | at least 0.97 |
| R6 | M2 effect: mean top-5 overlap A (both runs) against B | 0.85 to 0.99, and below the noise floor |
| R7 | M1 median dense execution time, exact against HNSW (EXPLAIN ANALYZE, k = 20) | HNSW faster, by 2 to 10 times |

R1 to R3 are deliberately wide and centred below 1.0: duplicates across generations are exactly
where a filtered graph search struggles, and I do not know how far `iterative_scan` recovers it.

## Decision rule, fixed now

- Mean recall@20 **at least 0.95 and** R6 overlap at least 0.95: 0026 ships as it is.
- Mean recall@20 **below 0.95 or** R6 overlap below 0.95: 0026 does not ship as it is. The next step
  is a separately pre-registered choice between raising `ef_search` for filtered queries and
  keeping the exact path deliberately (for example a generation-scoped exact scan below a size
  threshold), measured on this same table.

## Confounds I can name now

- Identical vectors across generations are a worst case; if VPS2's generations differ more, its
  recall is higher than measured here. That VPS2's copies are identical is inferred from the
  embedding cache, not measured.
- 29 copies of one corpus in 3 tenants, not VPS2's mix of memory, code and bench tenants with
  different content.
- The HNSW index is built once over the whole table after the copy, not incrementally as
  generations arrive, and graph quality can differ between the two.
- M2 carries provider variance in the query vector; A1 against A2 measures it, but with three
  runs the floor itself is an estimate from 80 pairs.
- Other sessions share VPS3.
