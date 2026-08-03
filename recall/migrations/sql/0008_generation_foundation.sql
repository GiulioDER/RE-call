-- recall:transactional
-- Global v1 lineage, generation, ingestion, audit, and erasure state.
CREATE TABLE IF NOT EXISTS recall_generations (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('building', 'validating', 'ready', 'active', 'retired', 'failed',
                  'legacy_unverified')
    ),
    pipeline_identity JSONB NOT NULL,
    pipeline_fingerprint CHAR(64) NOT NULL,
    corpus_fingerprint CHAR(64) NOT NULL,
    manifest JSONB NOT NULL,
    manifest_digest CHAR(64) NOT NULL,
    corpus_version TEXT NOT NULL,
    parent_generation_id TEXT,
    failure_reason TEXT,
    validation_summary JSONB,
    created_by TEXT NOT NULL DEFAULT current_user,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    validating_at TIMESTAMPTZ,
    ready_at TIMESTAMPTZ,
    activated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    PRIMARY KEY (tenant_id, generation_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS recall_generations_one_active_idx
    ON recall_generations (tenant_id) WHERE state = 'active';
CREATE INDEX IF NOT EXISTS recall_generations_state_idx
    ON recall_generations (tenant_id, state, created_at DESC);

CREATE TABLE IF NOT EXISTS recall_tenant_state (
    tenant_id TEXT PRIMARY KEY,
    active_generation_id TEXT,
    previous_generation_id TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (tenant_id, active_generation_id)
        REFERENCES recall_generations (tenant_id, generation_id)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (tenant_id, previous_generation_id)
        REFERENCES recall_generations (tenant_id, generation_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS recall_chunks_v1 (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    object_version_id TEXT NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    chunk_ordinal INTEGER NOT NULL CHECK (chunk_ordinal >= 0),
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(__RECALL_DIM__) NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    tsv tsvector NOT NULL,
    PRIMARY KEY (tenant_id, generation_id, chunk_id),
    FOREIGN KEY (tenant_id, generation_id)
        REFERENCES recall_generations (tenant_id, generation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS recall_chunks_v1_source_idx
    ON recall_chunks_v1 (tenant_id, generation_id, source_uri);
CREATE INDEX IF NOT EXISTS recall_chunks_v1_generation_idx
    ON recall_chunks_v1 (tenant_id, generation_id);

CREATE TABLE IF NOT EXISTS recall_ingest_jobs (
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    generation_id TEXT,
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'succeeded', 'failed')),
    manifest_uri TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, job_id)
);

CREATE INDEX IF NOT EXISTS recall_ingest_jobs_claim_idx
    ON recall_ingest_jobs (state, lease_expires_at, created_at);

CREATE TABLE IF NOT EXISTS recall_audit_events (
    tenant_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    generation_id TEXT,
    source_uri TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, event_id)
);

CREATE INDEX IF NOT EXISTS recall_audit_events_lookup_idx
    ON recall_audit_events (tenant_id, event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS recall_audit_events_source_idx
    ON recall_audit_events (tenant_id, source_uri, created_at DESC);

CREATE TABLE IF NOT EXISTS recall_source_tombstones (
    tenant_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    event_id TEXT NOT NULL,
    erased_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, source_uri),
    FOREIGN KEY (tenant_id, event_id)
        REFERENCES recall_audit_events (tenant_id, event_id)
);

ALTER TABLE recall_generations ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_generations FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_generations_tenant_isolation ON recall_generations;
CREATE POLICY recall_generations_tenant_isolation ON recall_generations
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_tenant_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_tenant_state FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_tenant_state_tenant_isolation ON recall_tenant_state;
CREATE POLICY recall_tenant_state_tenant_isolation ON recall_tenant_state
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_chunks_v1 ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_chunks_v1 FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_chunks_v1_tenant_isolation ON recall_chunks_v1;
CREATE POLICY recall_chunks_v1_tenant_isolation ON recall_chunks_v1
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_ingest_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_ingest_jobs FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_ingest_jobs_tenant_isolation ON recall_ingest_jobs;
CREATE POLICY recall_ingest_jobs_tenant_isolation ON recall_ingest_jobs
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_audit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_audit_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_audit_events_tenant_isolation ON recall_audit_events;
CREATE POLICY recall_audit_events_tenant_isolation ON recall_audit_events
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_source_tombstones ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_source_tombstones FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_source_tombstones_tenant_isolation ON recall_source_tombstones;
CREATE POLICY recall_source_tombstones_tenant_isolation ON recall_source_tombstones
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

-- Register legacy tenants as evidence only. No legacy row becomes active or enters v1 chunks.
-- The migration role owns these tables but FORCE RLS applies even to an owner. Temporarily
-- relaxing FORCE inside this transaction lets the owner enumerate tenants and insert their
-- evidence records without granting BYPASSRLS. Other sessions see either side of the transaction,
-- never this intermediate state; ordinary RLS remains enabled throughout.
ALTER TABLE __RECALL_TABLE__ NO FORCE ROW LEVEL SECURITY;
ALTER TABLE recall_generations NO FORCE ROW LEVEL SECURITY;
ALTER TABLE recall_tenant_state NO FORCE ROW LEVEL SECURITY;

INSERT INTO recall_generations (
    tenant_id, generation_id, state, pipeline_identity, pipeline_fingerprint,
    corpus_fingerprint, manifest, manifest_digest, corpus_version
)
SELECT DISTINCT tenant_id,
       'legacy-v08',
       'legacy_unverified',
       '{"status":"legacy_unverified","schema":"v0.8"}'::jsonb,
       repeat('0', 64),
       repeat('0', 64),
       jsonb_build_object('legacy_table', '__RECALL_TABLE__'),
       repeat('0', 64),
       'legacy-v0.8'
  FROM __RECALL_TABLE__
ON CONFLICT (tenant_id, generation_id) DO NOTHING;

INSERT INTO recall_tenant_state (tenant_id)
SELECT DISTINCT tenant_id FROM __RECALL_TABLE__
ON CONFLICT (tenant_id) DO NOTHING;

ALTER TABLE __RECALL_TABLE__ FORCE ROW LEVEL SECURITY;
ALTER TABLE recall_generations FORCE ROW LEVEL SECURITY;
ALTER TABLE recall_tenant_state FORCE ROW LEVEL SECURITY;
