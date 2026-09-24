# Pre-registration: live HNSW recall on VPS2 with real queries

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

Since 04:23 UTC today, VPS2's memory and code search plan their dense leg on the HNSW index
(autoanalyze gave the planner correct statistics; see the correction in
`2026-09-24-hnsw-recall-under-dependency-statistics.md`). A read-only diagnostic with stored chunk
vectors as queries gave mean recall@20 against exact of 0.995 (memory) and 0.955 (code). Stored
vectors are easy queries. With real queries, how much of the exact dense result does the live HNSW
path return?

## Setup

VPS2, read-only (a `READ ONLY` transaction, 30 s statement timeout), the active generations:
memory `gen_38c69d3f0904…` (r208, 11,879 rows, `voyage-context:voyage-context-4`) and
`re-call-code-gen` `gen_ccbb95f4a78b…` (12,275 rows, `voyage:voyage-code-3`).

Queries: each tenant's most recent stored calibration query set, `recall_calibration_query_sets`,
memory `05db40780b…` (40 queries) and code `4b1b280620…` (48 queries), answerable and unanswerable
alike. Each is embedded once as a query with the tenant's own embedder (88 Voyage calls).

Per query and per k in {5, 20, 100}:
- **HNSW:** the production dense SQL with `hnsw.ef_search = max(200, min(4k, 1000))` and
  `hnsw.iterative_scan = relaxed_order`; the plan is checked, and a query whose plan does not read
  `recall_chunks_v1_embedding_idx` is reported, not counted;
- **exact:** the same ordering over the generation's rows behind an `OFFSET 0` subquery.

## What I predict

| id | claim | predicted |
|---|---|---|
| L1 | memory, mean recall@20 | 0.90 to 0.99 |
| L2 | code, mean recall@20 | 0.85 to 0.97 |
| L3 | memory, mean recall@5 | at least 0.90 |
| L4 | code, mean recall@5 | at least 0.85 |
| L5 | both tenants, mean recall@100 | within 0.05 of that tenant's recall@20 |
| L6 | queries planned on HNSW | all of them, both tenants |

Real queries sit farther from stored points than a chunk's own vector, so I expect them to lose a
few points against the stored-vector diagnostic, more on code, which was already the weaker one.

## Decision rule, fixed now

- Mean recall@20 **at least 0.95 on both tenants**: HNSW is acceptable as served; no exact-by-design
  change. Migration 0026 is then reconsidered on its own merits in a separate record.
- **Below 0.95 on either tenant**: build the exact-by-design change for generation-scoped dense
  search, pre-registered separately, and measure its latency before shipping.

## Confounds I can name now

- Two query sets of 40 and 48, so each tenant's mean has a wide interval; per-query values are kept.
- The query sets were written for calibration, not to be representative of live traffic.
- The table keeps changing (builds, gc); the measurement is a snapshot.
- Voyage query vectors can vary between calls; each query is embedded once and both paths use the
  same vector.
