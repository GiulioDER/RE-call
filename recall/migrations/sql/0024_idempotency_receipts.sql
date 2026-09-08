-- recall:transactional
-- Replay receipts are operational state, not append-only audit history. They have an explicit
-- expiry so durable recovery cannot make recall_audit_events grow with every keyed mutation.

CREATE TABLE IF NOT EXISTS recall_idempotency_receipts (
    tenant_id TEXT NOT NULL,
    idempotency_key_hash TEXT NOT NULL,
    operation TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, idempotency_key_hash)
);

CREATE INDEX IF NOT EXISTS recall_idempotency_receipts_expiry_idx
    ON recall_idempotency_receipts (tenant_id, expires_at);

ALTER TABLE recall_idempotency_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_idempotency_receipts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_idempotency_receipts_tenant_isolation
    ON recall_idempotency_receipts;
CREATE POLICY recall_idempotency_receipts_tenant_isolation
    ON recall_idempotency_receipts
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));
