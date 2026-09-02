-- recall:transactional
-- Immutable, generation bound authored source dependencies and their diagnostics.
ALTER TABLE recall_chunks_v1
    ADD COLUMN IF NOT EXISTS first_indexed_at TIMESTAMPTZ;

ALTER TABLE recall_chunks_v1
    ALTER COLUMN first_indexed_at SET DEFAULT clock_timestamp();

CREATE TABLE IF NOT EXISTS recall_dependency_edges_v1 (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    edge_id TEXT NOT NULL,
    dependent_source TEXT NOT NULL,
    prerequisite_source TEXT NOT NULL,
    asserting_chunk_id TEXT NOT NULL,
    authority TEXT NOT NULL CHECK (
        authority IN ('policy', 'user_confirmed_decision', 'tool_observation', 'model_inference', 'unknown')
    ),
    asserted_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, generation_id, edge_id),
    FOREIGN KEY (tenant_id, generation_id)
        REFERENCES recall_generations (tenant_id, generation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS recall_dependency_edges_dependent_idx
    ON recall_dependency_edges_v1 (tenant_id, generation_id, dependent_source);
CREATE INDEX IF NOT EXISTS recall_dependency_edges_prerequisite_idx
    ON recall_dependency_edges_v1 (tenant_id, generation_id, prerequisite_source);

CREATE TABLE IF NOT EXISTS recall_dependency_diagnostics_v1 (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    diagnostic_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (
        kind IN (
            'malformed_metadata', 'inconsistent_authority', 'duplicate_dependency',
            'self_dependency', 'unresolved_dependency', 'dependency_cycle'
        )
    ),
    source TEXT NOT NULL,
    dependency TEXT,
    message TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, generation_id, diagnostic_id),
    FOREIGN KEY (tenant_id, generation_id)
        REFERENCES recall_generations (tenant_id, generation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS recall_dependency_diagnostics_source_idx
    ON recall_dependency_diagnostics_v1 (tenant_id, generation_id, source);

ALTER TABLE recall_dependency_edges_v1 ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_dependency_edges_v1 FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_dependency_edges_v1_tenant_isolation ON recall_dependency_edges_v1;
CREATE POLICY recall_dependency_edges_v1_tenant_isolation ON recall_dependency_edges_v1
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_dependency_diagnostics_v1 ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_dependency_diagnostics_v1 FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_dependency_diagnostics_v1_tenant_isolation ON recall_dependency_diagnostics_v1;
CREATE POLICY recall_dependency_diagnostics_v1_tenant_isolation ON recall_dependency_diagnostics_v1
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));
