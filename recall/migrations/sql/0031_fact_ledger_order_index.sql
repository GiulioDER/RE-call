-- recall:concurrent-index recall_fact_ledger_order_idx
-- The ledger replay, `recall provenance rebuild` and the reconcile read one tenant's events by
-- `ORDER BY created_at, event_id`, which `(tenant_id, fact_id, created_at)` cannot serve.
-- Built CONCURRENTLY, so the append-only ledger keeps taking writes while it builds: 0030 had this
-- index as a plain CREATE INDEX inside its transaction, and the migration lint refused it.
CREATE INDEX CONCURRENTLY IF NOT EXISTS recall_fact_ledger_order_idx
    ON recall_fact_ledger_events (tenant_id, created_at, event_id);
