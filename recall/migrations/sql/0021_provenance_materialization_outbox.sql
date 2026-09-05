-- recall:transactional
-- Durable delivery state for downstream materialization of already appended fact events.
-- The event snapshot is immutable. Only delivery state and retry diagnostics may change.
CREATE TABLE IF NOT EXISTS recall_fact_materialization_outbox (
    event_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    event JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'failed', 'applied')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    lease_until TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS recall_fact_materialization_pending_idx
    ON recall_fact_materialization_outbox (tenant_id, status, updated_at);

ALTER TABLE recall_fact_materialization_outbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_fact_materialization_outbox FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_fact_materialization_outbox_tenant_isolation
    ON recall_fact_materialization_outbox;
CREATE POLICY recall_fact_materialization_outbox_tenant_isolation
    ON recall_fact_materialization_outbox
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

CREATE OR REPLACE FUNCTION recall_fact_materialization_immutable_event() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.event_id <> OLD.event_id OR NEW.tenant_id <> OLD.tenant_id OR NEW.event <> OLD.event THEN
        RAISE EXCEPTION 'recall materialization event snapshot is immutable';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS recall_fact_materialization_immutable_event
    ON recall_fact_materialization_outbox;
CREATE TRIGGER recall_fact_materialization_immutable_event
    BEFORE UPDATE ON recall_fact_materialization_outbox
    FOR EACH ROW EXECUTE FUNCTION recall_fact_materialization_immutable_event();

REVOKE DELETE ON recall_fact_materialization_outbox FROM PUBLIC;
