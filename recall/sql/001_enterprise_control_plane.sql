CREATE TABLE IF NOT EXISTS recall_schema_versions (
    version integer PRIMARY KEY,
    checksum text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recall_index_generations (
    generation_id text PRIMARY KEY,
    physical_table text NOT NULL UNIQUE,
    embedding_profile text NOT NULL,
    dimension integer NOT NULL CHECK (dimension > 0),
    state text NOT NULL CHECK (state IN ('building', 'ready', 'active', 'retired', 'failed')),
    chunk_count bigint NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    source_count bigint NOT NULL DEFAULT 0 CHECK (source_count >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    ready_at timestamptz,
    retired_at timestamptz
);

CREATE TABLE IF NOT EXISTS recall_tenant_routes (
    tenant_id text PRIMARY KEY,
    active_generation text NOT NULL REFERENCES recall_index_generations(generation_id),
    shadow_generation text REFERENCES recall_index_generations(generation_id),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recall_migration_events (
    sequence_id bigserial PRIMARY KEY,
    tenant_id text NOT NULL,
    operation_id text NOT NULL,
    operation_kind text NOT NULL CHECK (operation_kind IN ('index', 'forget')),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'complete', 'failed')),
    payload jsonb,
    active_count bigint NOT NULL DEFAULT 0,
    shadow_count bigint NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    UNIQUE (tenant_id, operation_id)
);

ALTER TABLE recall_tenant_routes ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_tenant_routes FORCE ROW LEVEL SECURITY;
ALTER TABLE recall_migration_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_migration_events FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS recall_tenant_routes_isolation ON recall_tenant_routes;
CREATE POLICY recall_tenant_routes_isolation ON recall_tenant_routes
    USING (tenant_id = current_setting('recall.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('recall.tenant_id', true));

DROP POLICY IF EXISTS recall_migration_events_isolation ON recall_migration_events;
CREATE POLICY recall_migration_events_isolation ON recall_migration_events
    USING (tenant_id = current_setting('recall.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('recall.tenant_id', true));

CREATE INDEX IF NOT EXISTS recall_migration_events_pending_idx
    ON recall_migration_events (tenant_id, sequence_id) WHERE status = 'pending';
