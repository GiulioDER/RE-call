-- recall:transactional
-- Align the ledger policy name with the schema validator's table-scoped convention.
DROP POLICY IF EXISTS recall_fact_ledger_tenant_isolation ON recall_fact_ledger_events;
DROP POLICY IF EXISTS recall_fact_ledger_events_tenant_isolation ON recall_fact_ledger_events;
CREATE POLICY recall_fact_ledger_events_tenant_isolation ON recall_fact_ledger_events
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));
