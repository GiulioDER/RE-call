-- recall:transactional
-- Learned sparse (SPLADE) vectors, as a sidecar to whichever chunk table produced them.
--
-- GLOBAL, not per-target, and that is forced rather than chosen: schema.py routes every migration
-- numbered >= GLOBAL_MIGRATION_START ("0008") to the __global__ ledger, applied ONCE per database.
-- A per-target templated sidecar at this number would be created for whichever table migrated first
-- and silently skipped for every other one -- and the evaluation harness provisions one table per
-- corpus, so that is the normal path here, not an edge case. `chunk_table` in the key is what lets
-- one global table serve every target.
--
-- No FOREIGN KEY: the parent table is a COLUMN VALUE, not a schema reference, so referential
-- integrity cannot be delegated to Postgres here. That is exactly why the learned sparse leg
-- refuses to initialise under RECALL_ENV=production -- the existing erasure paths
-- (generations.forget, control_plane.erase_sources_from_pending) cannot know about this table, and
-- SPLADE weights over the vocabulary are partially reconstructable content, not an opaque hash.
-- Wiring erasure is the precondition for lifting that refusal.
--
-- Dimension is fixed at 30522 (bert-base-uncased vocabulary), which covers both supported
-- checkpoints. A model with a different vocabulary needs its own table and its own migration; that
-- is deliberate, because a `sparsevec` carries its dimension in the type and vectors of different
-- widths must never share a column.
CREATE TABLE IF NOT EXISTS recall_sparse_v1 (
    tenant_id   TEXT NOT NULL,
    chunk_table TEXT NOT NULL,
    profile_id  TEXT NOT NULL,
    id          TEXT NOT NULL,
    vec         sparsevec(30522) NOT NULL,
    -- Stored, not derived. "How much did pruning remove?" is then answerable from the table
    -- without re-encoding the corpus, and a load that quietly pruned everything to one term is
    -- visible as a number rather than as an unexplained drop in retrieval quality.
    nnz         INTEGER NOT NULL CHECK (nnz > 0 AND nnz <= 1000),
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, chunk_table, profile_id, id)
);

-- Answers "is this corpus encoded under this profile?" as a cheap row lookup rather than a scan.
CREATE INDEX IF NOT EXISTS recall_sparse_v1_corpus_idx
    ON recall_sparse_v1 (tenant_id, chunk_table, profile_id);

ALTER TABLE recall_sparse_v1 ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_sparse_v1 FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_sparse_v1_tenant_isolation ON recall_sparse_v1;
CREATE POLICY recall_sparse_v1_tenant_isolation ON recall_sparse_v1
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));
