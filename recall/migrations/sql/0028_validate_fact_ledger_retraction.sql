-- recall:transactional
-- 0026 installed these constraints without scanning the ledger under its exclusive table lock.
-- This separate transaction validates the existing rows while allowing concurrent writes.
ALTER TABLE recall_fact_ledger_events
    VALIDATE CONSTRAINT recall_fact_ledger_events_event_type_check;

ALTER TABLE recall_fact_ledger_events
    VALIDATE CONSTRAINT recall_fact_ledger_fact_shape;
