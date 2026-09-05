-- recall:transactional
-- Durable immutable projection of trusted evidence cards. It intentionally has no foreign key to
-- chunks or generations so a historical card survives corpus garbage collection.
CREATE TABLE IF NOT EXISTS recall_evidence_cards (
    card_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    card JSONB NOT NULL,
    indexed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS recall_evidence_cards_lookup_idx
    ON recall_evidence_cards (tenant_id, generation_id, chunk_id);

ALTER TABLE recall_evidence_cards ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_evidence_cards FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_evidence_cards_tenant_isolation ON recall_evidence_cards;
CREATE POLICY recall_evidence_cards_tenant_isolation ON recall_evidence_cards
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

CREATE OR REPLACE FUNCTION recall_evidence_cards_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'recall evidence cards are immutable';
END;
$$;

DROP TRIGGER IF EXISTS recall_evidence_cards_no_update ON recall_evidence_cards;
CREATE TRIGGER recall_evidence_cards_no_update
    BEFORE UPDATE OR DELETE ON recall_evidence_cards
    FOR EACH ROW EXECUTE FUNCTION recall_evidence_cards_append_only();
