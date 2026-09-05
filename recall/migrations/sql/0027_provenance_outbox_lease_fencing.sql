-- recall:transactional
-- A lease token fences completion from a worker whose lease has expired and been reclaimed.
ALTER TABLE recall_fact_materialization_outbox
    ADD COLUMN IF NOT EXISTS lease_token TEXT;

CREATE INDEX IF NOT EXISTS recall_fact_materialization_lease_idx
    ON recall_fact_materialization_outbox (tenant_id, event_id, lease_token);
