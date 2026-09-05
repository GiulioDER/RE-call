-- recall:transactional
-- Keep fact appends behind owner-controlled SECURITY DEFINER functions. The isolated
-- controller role receives EXECUTE on these functions, not raw table INSERT privilege.

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
