-- recall:transactional
-- Tell the planner that a generation belongs to exactly one tenant.
--
-- Every generation-scoped query filters on `tenant_id AND generation_id`. Without this the planner
-- multiplies the two selectivities as if they were independent, so on a table shared by many
-- tenants and generations it underestimates a generation by about the tenant's share of the
-- table: measured on the VPS3 testbench on 2026-09-23, 152 rows estimated for a 5,176-row
-- generation (docs/preregistrations/2026-09-23-analyze-after-generation-build.md, A3). A
-- functional-dependency statistic lets it use `generation_id`'s selectivity alone.
--
-- The statistic is populated by ANALYZE. `GenerationManager.build` issues
-- `ANALYZE recall_chunks_v1 (tenant_id, generation_id)` after every build, and the statement
-- below populates it once now, so an existing corpus does not wait for its next build.
CREATE STATISTICS IF NOT EXISTS recall_chunks_v1_tenant_generation_deps (dependencies)
    ON tenant_id, generation_id FROM recall_chunks_v1;

ANALYZE recall_chunks_v1 (tenant_id, generation_id);
