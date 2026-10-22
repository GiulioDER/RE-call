-- recall:transactional
-- The fact ledger's current facts, materialized (`recall.fact_current`): derived state, rebuilt
-- from the immutable ledger at any time with `recall provenance rebuild`. None of these tables
-- holds a fact value, only ids and hashes, and a read uses them only when they reflect every
-- ledger event of its tenant; otherwise it replays the ledger.
--
-- Numbered 0030 because PR 950 holds 0029 (fact value redaction). The two touch disjoint objects,
-- so a database that reaches 0030 first applies 0029 after it without conflict.

-- Every ledger event the tables reflect, filed under the conflict key it bears on.
CREATE TABLE IF NOT EXISTS recall_fact_current_events (
    tenant_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    conflict_key TEXT,
    fact_id TEXT,
    event_type TEXT NOT NULL,
    live BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (tenant_id, event_id)
);
CREATE INDEX IF NOT EXISTS recall_fact_current_events_key_idx
    ON recall_fact_current_events (tenant_id, conflict_key);
CREATE INDEX IF NOT EXISTS recall_fact_current_events_fact_idx
    ON recall_fact_current_events (tenant_id, fact_id) WHERE live;

-- One row per effective interval of an applied, unretracted assertion: its window less the
-- windows of the successors that hide it, half-open. Open ends are 0001-01-01 and 9999-12-31,
-- the bounds the projection itself uses.
CREATE TABLE IF NOT EXISTS recall_fact_current (
    tenant_id TEXT NOT NULL,
    conflict_key TEXT NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ NOT NULL,
    event_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    PRIMARY KEY (tenant_id, event_id, valid_from),
    CONSTRAINT recall_fact_current_interval CHECK (valid_from < valid_until)
);
CREATE INDEX IF NOT EXISTS recall_fact_current_window_idx
    ON recall_fact_current (tenant_id, conflict_key, valid_from, valid_until);

-- Per tenant, whether the tables may answer at all.
CREATE TABLE IF NOT EXISTS recall_fact_current_state (
    tenant_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK (mode IN ('table', 'replay')),
    reason TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- No index is added to the ledger itself: a plain CREATE INDEX there would block its writes while
-- it builds, and the table reads join it by primary key. The replay's `ORDER BY created_at,
-- event_id` sorts in memory as it always has; an index for it belongs in a concurrent-index
-- migration of its own, if a measurement asks for one.

ALTER TABLE recall_fact_current_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_fact_current_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_fact_current_events_tenant_isolation ON recall_fact_current_events;
CREATE POLICY recall_fact_current_events_tenant_isolation ON recall_fact_current_events
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_fact_current ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_fact_current FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_fact_current_tenant_isolation ON recall_fact_current;
CREATE POLICY recall_fact_current_tenant_isolation ON recall_fact_current
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

ALTER TABLE recall_fact_current_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_fact_current_state FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_fact_current_state_tenant_isolation ON recall_fact_current_state;
CREATE POLICY recall_fact_current_state_tenant_isolation ON recall_fact_current_state
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));
