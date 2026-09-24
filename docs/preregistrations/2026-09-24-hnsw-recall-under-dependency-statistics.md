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

## Result (2026-09-24)

**Status:** measured

**Void run, disclosed.** The first phase B attempt ran `schema apply` without `VOYAGE_API_KEY` in
the environment; the embedder could not be built, 0026 was **not** applied, and the three steps
after it measured the pre-0026 state (plans: exact, 80 of 80) or failed (`recall_search` refused
to start, since the branch code expects 0026). Nothing in the database changed. The rerun loaded
the key, applied 0026 in 1.39 s, and checked the statistic existed before measuring.

Also running on VPS3 during phase A: one other session's `pytest` or C8 process.

| id | predicted | measured | held |
|---|---|---|---|
| P1 | target share 3.0% to 3.6% | 3.33% (5,117 of 153,510; 30 generations, 3 tenants) | yes |
| P2 | exact plan without 0026, HNSW with it | generation index 80/80 without; `recall_chunks_v1_embedding_idx` 80/80 with | yes |
| R1 | mean recall@20: 0.80 to 0.98 | **0.335** (median 0.25; 13 queries at 0) | **no** |
| R2 | at least 50% of queries perfect at k = 20 | **2 of 80** | **no** |
| R3 | mean recall@100: 0.75 to 0.98 | **0.327** (0 perfect) | **no** |
| R4 | no short pages at k = 20 | **2** short at k = 20 (10 at k = 100) | **no** |
| R5 | noise floor A1 against A2 at least 0.97 | 0.995 (78 of 80 identical top-5 sets) | yes |
| R6 | A against B: 0.85 to 0.99 | **0.588** (12 of 80 identical; answerable 0.655, unanswerable 0.520) | **no** |
| R7 | HNSW 2 to 10 times faster | 52.5 → 3.6 ms median, 14.5 times | no (faster than predicted) |

**Decision, by the rule fixed before the run: 0026 does not ship as it is.** Mean recall@20 is
0.335, far below 0.95, and what a caller of `recall_search` receives changes (top-5 overlap 0.588
against a 0.995 noise floor).

**Checks after the result, labelled as outside the registration.**
- Known answer: a stored chunk's own vector as the query, production settings, k = 20: the chunk
  itself came back in 18 of 20. The harness can see a correct answer; HNSW misses some even of
  those.
- `ef_search` 1000 (pgvector's maximum) raises mean recall@20 only to 0.513 (7 of 80 perfect);
  `strict_order` gives the same numbers as `relaxed_order` at both 200 and 1000. Tuning does not
  rescue it.
- Flip point, EXPLAIN only, 0026 dropped again and copies moved into the target tenant step by step:
  the plan is exact for all 80 queries at a pair estimate of 1,691 and HNSW for all 80 from 2,602
  upward (3,366, 4,047, 4,268, 5,144 all HNSW).
- VPS2, read-only: the table holds 419,761 rows; the memory tenant has 13 generations (151,907
  rows), and each chunk text of its active generation appears a **median of 13 times** (mean 12.6,
  maximum 30) across the table. Identical text is not proven to mean identical vectors there
  (generations from before the Context 4 cutover carry different ones); that was not measured.

**The consequence for #728, which is merged but not yet on VPS2's serving checkout.** #728 makes
every build refresh `tenant_id` and `generation_id` statistics. On VPS2 the memory tenant is 36%
of the table and its active generation 2.8%, so the independence product after that refresh is
about 419,761 × 0.36 × 0.028 ≈ **4,300** rows, well past the flip point measured here. Once #728
reaches VPS2 and the next memory build runs, the memory tenant's dense search is expected to move
from exact to HNSW. The same would happen, less predictably, whenever autovacuum's own ANALYZE runs
after a new generation exists, so the exact path production relies on today holds only because its
statistics are stale.

**Confounds that changed during the run.** The HNSW index is 177 MB for 153,510 vectors of 4 KB
each, which fits pgvector storing an identical vector once with several row pointers. So identical
copies are not simply "the hardest case" as the setup assumed; how pgvector's graph behaves with 30
identical copies against VPS2's median of 13 (possibly not all identical) is not established here.

**Gap.** Three of nine held, and every miss points the same way: I expected an approximate index
to cost a few points of recall, and in this shape it loses two thirds of the exact top 20. The
predictions were anchored on HNSW's usual behaviour on distinct vectors; the defining feature of
this table, the same content in many generations, is exactly what a filtered graph search handles
worst.

## 🔁 Correction the same morning: VPS2's live HNSW path is far better than this table (appended)

Read-only on VPS2 at about 04:30 UTC, after Postgres's own autoanalyze (04:23:54) and #728's
refresh (04:24:08) had given the planner correct statistics, the dense plan for the active memory
and code generations **is** the HNSW index (pair estimates 4,739 and 3,313). A diagnostic, not
pre-registered: 40 stored chunk vectors per tenant as queries, production tuning, HNSW top 20
against exact top 20:

| tenant | mean recall@20 | perfect | own chunk found |
|---|---:|---:|---:|
| memory | **0.995** | 38/40 | 39/40 |
| re-call-code-gen | **0.955** | 38/40 | 38/40 |

Against this table's 0.335 and its known answer of 18/20, the synthetic setup here (30 identical
copies of one corpus) was far harsher than VPS2's live table, and the "#728 consequence" paragraph
above overstated the harm. Two differences are plausible and neither is measured: VPS2's copies
are not all identical vectors (Context 4 document vectors depend on their group, and older
generations predate the cutover), and its index was built incrementally. Stored-vector queries are
also easier than real queries, so VPS2's figure for real queries is not established either.

What stands: HNSW is not exact on VPS2 either (0.955 on the code tenant), and the planner reaches it
through statistics, not by design, whether or not #728 is live.
