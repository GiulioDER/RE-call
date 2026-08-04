-- recall:transactional
CREATE TABLE IF NOT EXISTS recall_calibration_query_sets (
    tenant_id TEXT NOT NULL,
    query_set_digest CHAR(64) NOT NULL,
    queries JSONB NOT NULL,
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    created_by TEXT NOT NULL DEFAULT current_user,
    PRIMARY KEY (tenant_id, query_set_digest)
);

CREATE TABLE IF NOT EXISTS recall_calibrations (
    tenant_id TEXT NOT NULL,
    calibration_id TEXT NOT NULL,
    generation_id TEXT,
    embedder_identity JSONB NOT NULL,
    pipeline_fingerprint CHAR(64),
    corpus_fingerprint CHAR(64),
    query_set_digest CHAR(64),
    threshold DOUBLE PRECISION NOT NULL CHECK (threshold BETWEEN -1.0 AND 1.0),
    scale DOUBLE PRECISION NOT NULL CHECK (scale > 0.0),
    separability DOUBLE PRECISION CHECK (separability BETWEEN 0.0 AND 1.0),
    ci_low DOUBLE PRECISION CHECK (ci_low BETWEEN 0.0 AND 1.0),
    ci_high DOUBLE PRECISION CHECK (ci_high BETWEEN 0.0 AND 1.0),
    n_answerable INTEGER CHECK (n_answerable >= 0),
    n_unanswerable INTEGER CHECK (n_unanswerable >= 0),
    certified BOOLEAN NOT NULL,
    certification_reason TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL CHECK (
        lifecycle_state IN ('draft', 'published', 'superseded', 'rejected', 'legacy_unbound')
    ),
    scores JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    created_by TEXT NOT NULL DEFAULT current_user,
    published_at TIMESTAMPTZ,
    superseded_at TIMESTAMPTZ,
    artifact_checksum CHAR(64) NOT NULL,
    PRIMARY KEY (tenant_id, calibration_id),
    FOREIGN KEY (tenant_id, generation_id)
        REFERENCES recall_generations (tenant_id, generation_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, query_set_digest)
        REFERENCES recall_calibration_query_sets (tenant_id, query_set_digest),
    CHECK (lifecycle_state != 'published' OR certified),
    CHECK (
        (lifecycle_state = 'legacy_unbound' AND generation_id IS NULL
         AND pipeline_fingerprint IS NULL AND corpus_fingerprint IS NULL
         AND query_set_digest IS NULL)
        OR
        (lifecycle_state != 'legacy_unbound' AND generation_id IS NOT NULL
         AND pipeline_fingerprint IS NOT NULL AND corpus_fingerprint IS NOT NULL
         AND query_set_digest IS NOT NULL AND separability IS NOT NULL
         AND ci_low IS NOT NULL AND ci_high IS NOT NULL
         AND n_answerable IS NOT NULL AND n_unanswerable IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS recall_calibrations_one_published_idx
    ON recall_calibrations (tenant_id, generation_id)
    WHERE lifecycle_state = 'published';
CREATE INDEX IF NOT EXISTS recall_calibrations_history_idx
    ON recall_calibrations (tenant_id, generation_id, created_at DESC);

CREATE OR REPLACE FUNCTION recall_reject_query_set_update()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'calibration query sets are immutable';
END;
$$;
DROP TRIGGER IF EXISTS recall_calibration_query_sets_immutable
    ON recall_calibration_query_sets;
CREATE TRIGGER recall_calibration_query_sets_immutable
    BEFORE UPDATE ON recall_calibration_query_sets
    FOR EACH ROW EXECUTE FUNCTION recall_reject_query_set_update();

CREATE OR REPLACE FUNCTION recall_reject_calibration_payload_update()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (
        to_jsonb(NEW) - ARRAY['lifecycle_state', 'published_at', 'superseded_at']
    ) IS DISTINCT FROM (
        to_jsonb(OLD) - ARRAY['lifecycle_state', 'published_at', 'superseded_at']
    ) THEN
        RAISE EXCEPTION 'calibration artifact payloads are immutable';
    END IF;
    RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS recall_calibrations_payload_immutable ON recall_calibrations;
CREATE TRIGGER recall_calibrations_payload_immutable
    BEFORE UPDATE ON recall_calibrations
    FOR EACH ROW EXECUTE FUNCTION recall_reject_calibration_payload_update();

ALTER TABLE recall_calibration_query_sets ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_calibration_query_sets FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_calibration_query_sets_tenant_isolation
    ON recall_calibration_query_sets;
CREATE POLICY recall_calibration_query_sets_tenant_isolation ON recall_calibration_query_sets
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_calibrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_calibrations FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_calibrations_tenant_isolation ON recall_calibrations;
CREATE POLICY recall_calibrations_tenant_isolation ON recall_calibrations
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));
