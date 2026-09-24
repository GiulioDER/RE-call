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

## Result (2026-09-24)

**Status:** measured

VPS2, read-only, 88 real queries each embedded once. The first attempt crashed before reporting
anything (no query planned on HNSW at k = 100 left the "not counted" rule with nothing to average);
the rerun reports that case instead. Memory measured on r208 `gen_38c69d3f0904…`; the code tenant's
active generation had moved to `gen_2b6237957e3e…` by the time of the run (a scheduled refresh
promoted it), so code was measured on that one.

| id | predicted | measured | held |
|---|---|---|---|
| L1 | memory recall@20: 0.90 to 0.99 | **1.000** (40/40 perfect) | no (above the band) |
| L2 | code recall@20: 0.85 to 0.97 | **1.000** (48/48 perfect) | no (above the band) |
| L3 | memory recall@5 at least 0.90 | **0.790** (30/40 perfect, minimum 0.0) | **no** |
| L4 | code recall@5 at least 0.85 | 1.000 (48/48) | yes |
| L5 | recall@100 within 0.05 of recall@20 | not measurable: at k = 100 the planner chose the **exact** path for all 88 queries | n/a |
| L6 | every query planned on HNSW | at k = 5 and 20, yes; at k = 100, none | no |

**Decision, by the rule fixed before the run: HNSW is acceptable as served.** Mean recall@20 is 1.0 on
both tenants, and 20 is the dense candidate count the MCP server asks for ("candidates 20/leg"). No
exact-by-design change is built.

**What I did not predict.**
- **Recall@5 is worse than recall@20 on memory, with the same `ef_search` (200).** 10 of 40 queries
  lost part of their top 5 and one lost all of it, while the top 20 was always complete. The
  likely mechanism is `iterative_scan = relaxed_order`: it returns the first matching rows the scan
  reaches rather than re-sorting a fixed candidate list, so a small LIMIT takes an early, unsorted
  slice. That was not tested here. It matters for any caller that asks the dense leg for 5.
- **At k = 100 the planner prefers the exact path** (index scan on the generation, then a sort),
  so the hosted-quality profile's 100 candidates are exact even with correct statistics.
- Both tenants did better than the stored-vector diagnostic this morning (0.995 and 0.955), the
  opposite of what I expected; real queries were not harder here.

**Gap.** Two of six held in the strict sense, and every miss but one was in the favourable
direction. The one unfavourable miss (L3) points at a specific setting, not at HNSW as a whole.
