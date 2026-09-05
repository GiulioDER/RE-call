-- recall:transactional
-- Harden the provenance append boundary and fence recovered outbox workers.

ALTER TABLE recall_fact_materialization_outbox
    ADD COLUMN IF NOT EXISTS lease_token TEXT;

CREATE INDEX IF NOT EXISTS recall_fact_materialization_lease_idx
    ON recall_fact_materialization_outbox (tenant_id, event_id, lease_token);

CREATE OR REPLACE FUNCTION recall_append_fact_ledger_event(
    p_event_id TEXT,
    p_tenant_id TEXT,
    p_generation_id TEXT,
    p_event_type TEXT,
    p_fact_id TEXT,
    p_fact JSONB,
    p_evidence_cards JSONB,
    p_supersedes_fact_ids JSONB,
    p_request_id TEXT,
    p_writer TEXT,
    p_decision_code TEXT,
    p_policy_version TEXT,
    p_controller_version INTEGER,
    p_created_at TIMESTAMPTZ
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
BEGIN
    IF p_event_id IS NULL OR p_event_id !~ '^evt_[0-9a-f]{32}(_sup)?$'
       OR p_tenant_id IS NULL OR length(p_tenant_id) = 0
       OR p_generation_id IS NULL OR length(p_generation_id) = 0
       OR p_request_id IS NULL OR length(p_request_id) = 0
       OR p_writer IS NULL OR length(p_writer) = 0
       OR p_created_at IS NULL THEN
        RAISE EXCEPTION 'invalid fact ledger append envelope';
    END IF;
    IF p_event_type NOT IN ('asserted', 'superseded', 'rejected', 'abstained')
       OR p_decision_code NOT IN (
           'APPLIED', 'DUPLICATE', 'CARD_NOT_FOUND', 'CARD_TAMPERED', 'SOURCE_CHANGED',
           'VALIDITY_EXPIRED', 'VALIDITY_NOT_STARTED', 'GENERATION_MISMATCH',
           'LINEAGE_MISMATCH', 'TRUST_UNAVAILABLE', 'UNSUPPORTED_CLAIM',
           'CONTRADICTION_WITHOUT_SUPERSESSION', 'FRESH_SEARCH_UNAVAILABLE',
           'FRESH_SEARCH_INSUFFICIENT', 'LEDGER_UNAVAILABLE', 'MATERIALIZATION_UNAVAILABLE'
       ) THEN
        RAISE EXCEPTION 'invalid fact ledger decision envelope';
    END IF;
    IF jsonb_typeof(p_evidence_cards) <> 'array'
       OR jsonb_typeof(p_supersedes_fact_ids) <> 'array' THEN
        RAISE EXCEPTION 'fact ledger arrays must be JSON arrays';
    END IF;
    IF EXISTS (
        SELECT 1 FROM jsonb_array_elements(p_evidence_cards) AS card
        WHERE jsonb_typeof(card) <> 'object'
           OR card->>'card_id' IS NULL
           OR card->>'tenant_id' IS DISTINCT FROM p_tenant_id
           OR card->>'generation_id' IS DISTINCT FROM p_generation_id
    ) THEN
        RAISE EXCEPTION 'fact ledger evidence card envelope mismatch';
    END IF;
    IF current_setting('recall.tenant_id', true) IS DISTINCT FROM p_tenant_id THEN
        RAISE EXCEPTION 'recall fact ledger tenant context mismatch';
    END IF;

    INSERT INTO public.recall_fact_ledger_events (
        event_id, tenant_id, generation_id, event_type, fact_id, fact, evidence_cards,
        supersedes_fact_ids, request_id, writer, decision_code, policy_version,
        controller_version, created_at
    ) VALUES (
        p_event_id, p_tenant_id, p_generation_id, p_event_type, p_fact_id, p_fact,
        p_evidence_cards, p_supersedes_fact_ids, p_request_id, p_writer, p_decision_code,
        p_policy_version, p_controller_version, p_created_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION recall_append_fact_materialization(
    p_event_id TEXT,
    p_tenant_id TEXT,
    p_event JSONB
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    existing_event JSONB;
BEGIN
    IF current_setting('recall.tenant_id', true) IS DISTINCT FROM p_tenant_id THEN
        RAISE EXCEPTION 'recall materialization tenant context mismatch';
    END IF;

    SELECT event INTO existing_event
      FROM public.recall_fact_materialization_outbox
     WHERE event_id = p_event_id AND tenant_id = p_tenant_id
     FOR UPDATE;
    IF existing_event IS NOT NULL AND existing_event <> p_event THEN
        RAISE EXCEPTION 'materialization event collision for %', p_event_id;
    END IF;

    IF p_event_id IS NULL OR p_event_id !~ '^evt_[0-9a-f]{32}(_sup)?$'
       OR p_tenant_id IS NULL OR length(p_tenant_id) = 0
       OR jsonb_typeof(p_event) <> 'object'
       OR p_event->>'event_id' IS DISTINCT FROM p_event_id
       OR p_event->>'tenant_id' IS DISTINCT FROM p_tenant_id THEN
        RAISE EXCEPTION 'invalid materialization append envelope';
    END IF;

    INSERT INTO public.recall_fact_materialization_outbox
        (event_id, tenant_id, event, status)
    VALUES
        (p_event_id, p_tenant_id, p_event, 'pending')
    ON CONFLICT (event_id) DO NOTHING;
END;
$$;

REVOKE ALL ON FUNCTION recall_append_fact_ledger_event(
    TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, JSONB, JSONB, TEXT, TEXT, TEXT, TEXT, INTEGER, TIMESTAMPTZ
) FROM PUBLIC;
REVOKE ALL ON FUNCTION recall_append_fact_materialization(TEXT, TEXT, JSONB) FROM PUBLIC;
