-- recall:transactional
-- Append-only, tenant-isolated ledger for deterministic structured fact applications.
CREATE TABLE IF NOT EXISTS recall_fact_ledger_events (
    event_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('asserted', 'superseded', 'rejected', 'abstained')),
    fact_id TEXT,
    fact JSONB,
    evidence_cards JSONB NOT NULL DEFAULT '[]'::jsonb,
    supersedes_fact_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    request_id TEXT NOT NULL,
    writer TEXT NOT NULL,
    decision_code TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    controller_version INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT recall_fact_ledger_fact_shape CHECK (
        (event_type IN ('asserted', 'rejected') AND fact IS NOT NULL)
        OR event_type IN ('superseded', 'abstained')
    ),
    PRIMARY KEY (tenant_id, event_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS recall_fact_ledger_request_idx
    ON recall_fact_ledger_events (tenant_id, request_id);
CREATE INDEX IF NOT EXISTS recall_fact_ledger_current_idx
    ON recall_fact_ledger_events (tenant_id, fact_id, created_at);

ALTER TABLE recall_fact_ledger_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_fact_ledger_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_fact_ledger_tenant_isolation ON recall_fact_ledger_events;
CREATE POLICY recall_fact_ledger_tenant_isolation ON recall_fact_ledger_events
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

CREATE OR REPLACE FUNCTION recall_fact_ledger_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'recall fact ledger is append-only';
END;
$$;

DROP TRIGGER IF EXISTS recall_fact_ledger_no_update ON recall_fact_ledger_events;
CREATE TRIGGER recall_fact_ledger_no_update
    BEFORE UPDATE OR DELETE ON recall_fact_ledger_events
    FOR EACH ROW EXECUTE FUNCTION recall_fact_ledger_append_only();
