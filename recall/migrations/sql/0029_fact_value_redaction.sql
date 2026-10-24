-- recall:transactional
-- recall_forget erases the fact values a forgotten memo supplied (recall.fact_redaction). The ledger,
-- the evidence cards and the materialization outbox stay append-only: their triggers still refuse
-- every DELETE and every UPDATE, except an UPDATE made while the transaction-local setting
-- `recall.redaction` names a redaction that changes only what a redaction erases, into the marker
-- shape. Only recall_redact_fact_values sets it, and only the controller role may call it; it
-- checks that the rewrite it is handed is a redaction and nothing else. The
-- triggers honour it only when current_user owns the table, which inside that SECURITY DEFINER
-- function it does: a role granted UPDATE (the outbox's delivery columns) that sets the setting
-- itself is still refused.

CREATE TABLE IF NOT EXISTS recall_fact_redactions (
    redaction_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    reason TEXT NOT NULL CHECK (reason IN ('forget', 'tenant_erase')),
    writer TEXT NOT NULL,
    facts_redacted INTEGER NOT NULL CHECK (facts_redacted >= 0),
    events_rewritten INTEGER NOT NULL CHECK (events_rewritten >= 0),
    cards_redacted INTEGER NOT NULL CHECK (cards_redacted >= 0),
    receipts_deleted INTEGER NOT NULL CHECK (receipts_deleted >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

ALTER TABLE recall_fact_redactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_fact_redactions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS recall_fact_redactions_tenant_isolation ON recall_fact_redactions;
CREATE POLICY recall_fact_redactions_tenant_isolation ON recall_fact_redactions
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

CREATE OR REPLACE FUNCTION recall_fact_redactions_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'recall fact redactions are append-only';
END;
$$;

DROP TRIGGER IF EXISTS recall_fact_redactions_no_update ON recall_fact_redactions;
CREATE TRIGGER recall_fact_redactions_no_update
    BEFORE UPDATE OR DELETE ON recall_fact_redactions
    FOR EACH ROW EXECUTE FUNCTION recall_fact_redactions_append_only();

-- The sorted keys of a JSON object, for the marker shape checks below.
CREATE OR REPLACE FUNCTION recall_jsonb_sorted_keys(p_value JSONB) RETURNS TEXT[]
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN jsonb_typeof(p_value) = 'object'
        THEN ARRAY(SELECT key FROM jsonb_object_keys(p_value) AS keys(key) ORDER BY key)
    END
$$;

-- Whether `p_new` is `p_old` with some cards replaced by redacted cards of `p_redaction_id`, each
-- keeping its tenant, generation and memo name (recall.types.RedactedEvidenceCard).
CREATE OR REPLACE FUNCTION recall_redacted_cards_ok(p_old JSONB, p_new JSONB, p_redaction_id TEXT)
RETURNS BOOLEAN
LANGUAGE sql IMMUTABLE AS $$
    SELECT jsonb_typeof(p_new) = 'array'
       AND jsonb_array_length(p_new) = jsonb_array_length(p_old)
       AND NOT EXISTS (
           SELECT 1
             FROM jsonb_array_elements(p_old) WITH ORDINALITY AS o(card, position)
             JOIN jsonb_array_elements(p_new) WITH ORDINALITY AS n(card, position) USING (position)
            WHERE n.card IS DISTINCT FROM o.card
              AND NOT (
                  public.recall_jsonb_sorted_keys(n.card)
                      = ARRAY['card_id', 'generation_id', 'redacted', 'redaction_id', 'source', 'tenant_id']
                  AND n.card->'redacted' = 'true'::jsonb
                  AND n.card->>'redaction_id' = p_redaction_id
                  AND n.card->>'card_id' ~ '^card_[0-9a-f]{32}$'
                  AND n.card->'tenant_id' = o.card->'tenant_id'
                  AND n.card->'generation_id' = o.card->'generation_id'
                  AND n.card->'source' = o.card->'source'
              )
       )
$$;

CREATE OR REPLACE FUNCTION recall_fact_ledger_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    redaction TEXT := CASE
        WHEN current_user = pg_catalog.pg_get_userbyid(
            (SELECT relowner FROM pg_catalog.pg_class WHERE oid = TG_RELID))
        THEN coalesce(current_setting('recall.redaction', true), '')
        ELSE ''
    END;
BEGIN
    IF TG_OP = 'UPDATE' AND redaction <> ''
       AND NEW.tenant_id = OLD.tenant_id
       AND NEW.generation_id = OLD.generation_id
       AND NEW.event_type = OLD.event_type
       AND NEW.writer = OLD.writer
       AND NEW.decision_code = OLD.decision_code
       AND NEW.policy_version = OLD.policy_version
       AND NEW.controller_version = OLD.controller_version
       AND NEW.created_at = OLD.created_at
       AND NEW.event_id ~ '^evt_[0-9a-f]{32}(_sup)?$'
       AND length(NEW.request_id) > 0
       AND (
           NEW.fact IS NOT DISTINCT FROM OLD.fact
           OR (
               OLD.fact IS NOT NULL
               AND public.recall_jsonb_sorted_keys(NEW.fact) = ARRAY['fact_id', 'redacted', 'redaction_id']
               AND NEW.fact->'redacted' = 'true'::jsonb
               AND NEW.fact->>'redaction_id' = redaction
               AND NEW.fact->>'fact_id' IS NOT DISTINCT FROM NEW.fact_id
           )
       )
       AND (NEW.fact_id IS NOT DISTINCT FROM OLD.fact_id OR NEW.fact IS DISTINCT FROM OLD.fact)
       AND public.recall_redacted_cards_ok(OLD.evidence_cards, NEW.evidence_cards, redaction)
       AND jsonb_typeof(NEW.supersedes_fact_ids) = 'array'
       AND jsonb_array_length(NEW.supersedes_fact_ids) = jsonb_array_length(OLD.supersedes_fact_ids)
       AND NOT EXISTS (
           SELECT 1
             FROM jsonb_array_elements_text(OLD.supersedes_fact_ids) WITH ORDINALITY AS o(id, position)
             JOIN jsonb_array_elements_text(NEW.supersedes_fact_ids) WITH ORDINALITY AS n(id, position)
                  USING (position)
            WHERE n.id IS DISTINCT FROM o.id AND n.id !~ '^fact_[0-9a-f]{32}$'
       )
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'recall fact ledger is append-only';
END;
$$;

CREATE OR REPLACE FUNCTION recall_evidence_cards_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    redaction TEXT := CASE
        WHEN current_user = pg_catalog.pg_get_userbyid(
            (SELECT relowner FROM pg_catalog.pg_class WHERE oid = TG_RELID))
        THEN coalesce(current_setting('recall.redaction', true), '')
        ELSE ''
    END;
BEGIN
    IF TG_OP = 'UPDATE' AND redaction <> ''
       AND NEW.tenant_id = OLD.tenant_id
       AND NEW.generation_id = OLD.generation_id
       AND NEW.indexed_at IS NOT DISTINCT FROM OLD.indexed_at
       AND NEW.created_at = OLD.created_at
       AND NEW.chunk_id = 'redacted'
       AND NEW.source_digest = 'redacted'
       AND NEW.card_id ~ '^card_[0-9a-f]{32}$'
       AND NEW.card->>'card_id' = NEW.card_id
       AND public.recall_jsonb_sorted_keys(NEW.card)
           = ARRAY['card_id', 'generation_id', 'redacted', 'redaction_id', 'source', 'tenant_id']
       AND NEW.card->'redacted' = 'true'::jsonb
       AND NEW.card->>'redaction_id' = redaction
       AND NEW.card->'tenant_id' = OLD.card->'tenant_id'
       AND NEW.card->'generation_id' = OLD.card->'generation_id'
       AND NEW.card->'source' = OLD.card->'source'
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'recall evidence cards are immutable';
END;
$$;

CREATE OR REPLACE FUNCTION recall_fact_materialization_immutable_event() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    redaction TEXT := CASE
        WHEN current_user = pg_catalog.pg_get_userbyid(
            (SELECT relowner FROM pg_catalog.pg_class WHERE oid = TG_RELID))
        THEN coalesce(current_setting('recall.redaction', true), '')
        ELSE ''
    END;
    rewritable TEXT[] := ARRAY['event_id', 'request_id', 'fact', 'evidence_cards', 'supersedes_fact_ids'];
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'recall materialization event snapshot is immutable';
    END IF;
    IF NEW.event_id <> OLD.event_id OR NEW.tenant_id <> OLD.tenant_id OR NEW.event <> OLD.event THEN
        -- Under a redaction, a snapshot changes only as its ledger row does: the same ids, the
        -- fact unchanged or the marker, cards unchanged or redacted cards, everything else equal.
        IF redaction = ''
           OR NEW.tenant_id <> OLD.tenant_id
           OR NEW.event_id !~ '^evt_[0-9a-f]{32}(_sup)?$'
           OR NEW.event->>'event_id' IS DISTINCT FROM NEW.event_id
           OR (NEW.event - rewritable) IS DISTINCT FROM (OLD.event - rewritable)
           OR NOT (
               NEW.event->'fact' IS NOT DISTINCT FROM OLD.event->'fact'
               OR (
                   jsonb_typeof(OLD.event->'fact') = 'object'
                   AND public.recall_jsonb_sorted_keys(NEW.event->'fact') = ARRAY['fact_id', 'redacted', 'redaction_id']
                   AND NEW.event->'fact'->'redacted' = 'true'::jsonb
                   AND NEW.event->'fact'->>'redaction_id' = redaction
               )
           )
           OR NOT public.recall_redacted_cards_ok(
               OLD.event->'evidence_cards', NEW.event->'evidence_cards', redaction
           )
           OR jsonb_array_length(NEW.event->'supersedes_fact_ids')
              IS DISTINCT FROM jsonb_array_length(OLD.event->'supersedes_fact_ids') THEN
            RAISE EXCEPTION 'recall materialization event snapshot is immutable';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS recall_fact_materialization_no_delete ON recall_fact_materialization_outbox;
CREATE TRIGGER recall_fact_materialization_no_delete
    BEFORE DELETE ON recall_fact_materialization_outbox
    FOR EACH ROW EXECUTE FUNCTION recall_fact_materialization_immutable_event();

-- `p_map` with `p_key` mapped to `p_value`; one old id never maps to two new ones.
CREATE OR REPLACE FUNCTION recall_redaction_map_put(p_map JSONB, p_key TEXT, p_value TEXT)
RETURNS JSONB
LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    IF p_key IS NULL OR p_value IS NULL THEN
        RAISE EXCEPTION 'fact redaction maps a missing id';
    END IF;
    IF p_map ? p_key AND p_map->>p_key IS DISTINCT FROM p_value THEN
        RAISE EXCEPTION 'fact redaction maps % to two ids', p_key;
    END IF;
    RETURN p_map || jsonb_build_object(p_key, p_value);
END;
$$;

-- `recall.fact_ledger.decision_event_id`: the id of a refusal or retraction event.
CREATE OR REPLACE FUNCTION recall_decision_event_id(p_tenant_id TEXT, p_request_id TEXT, p_event_type TEXT)
RETURNS TEXT
LANGUAGE sql IMMUTABLE AS $$
    SELECT 'evt_' || left(encode(sha256(
        convert_to(p_tenant_id, 'UTF8') || '\x00'::bytea || convert_to(p_request_id, 'UTF8')
        || '\x00'::bytea || convert_to(p_event_type, 'UTF8')
    ), 'hex'), 32)
$$;

-- Rewrite one redaction's rows. recall.fact_redaction plans the new values; this function does not
-- take them on trust. It learns from the plan which fact and card ids are replaced (each old id by
-- one new id), then holds every planned row to exactly what those replacements imply: the fact
-- unchanged or the marker of its new id, every card unchanged or the redacted card of its new id,
-- every superseded fact id mapped, and event and request ids changed only for an event whose
-- value is erased now, or derived from one (its retraction, its supersession) as the ledger
-- derives them. It then applies the rewrites, rebuilds each outbox snapshot from its rewritten
-- ledger row, deletes the replay receipts that echo what was erased, and records the redaction.
-- The triggers are the backstop for the shape of every row.
CREATE OR REPLACE FUNCTION recall_redact_fact_values(
    p_tenant_id TEXT,
    p_redaction_id TEXT,
    p_reason TEXT,
    p_writer TEXT,
    p_facts_redacted INTEGER,
    p_events JSONB,
    p_cards JSONB,
    p_receipt_needles JSONB,
    p_all_receipts BOOLEAN
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    item JSONB;
    old_row public.recall_fact_ledger_events%ROWTYPE;
    old_card JSONB;
    new_card JSONB;
    new_fact JSONB;
    expected JSONB;
    fact_map JSONB := '{}'::jsonb;
    card_map JSONB := '{}'::jsonb;
    event_map JSONB := '{}'::jsonb;
    erased_now BOOLEAN;
    holder TEXT;
    want_event TEXT;
    want_request TEXT;
    want_fact_id TEXT;
    slot INTEGER;
    touched INTEGER;
    v_events INTEGER := 0;
    v_cards INTEGER := 0;
    v_outbox INTEGER := 0;
    v_receipts INTEGER := 0;
BEGIN
    IF p_tenant_id IS NULL OR length(p_tenant_id) = 0
       OR p_redaction_id IS NULL OR p_redaction_id !~ '^red_[0-9a-f]{32}$'
       OR p_reason IS NULL OR p_reason NOT IN ('forget', 'tenant_erase')
       OR p_writer IS NULL OR length(p_writer) = 0
       OR p_facts_redacted IS NULL OR p_facts_redacted < 0
       OR p_all_receipts IS NULL
       OR jsonb_typeof(p_events) IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_cards) IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_receipt_needles) IS DISTINCT FROM 'array' THEN
        RAISE EXCEPTION 'invalid fact redaction envelope';
    END IF;
    IF current_setting('recall.tenant_id', true) IS DISTINCT FROM p_tenant_id THEN
        RAISE EXCEPTION 'recall fact redaction tenant context mismatch';
    END IF;

    -- 1. The replacements the plan implies.
    FOR item IN SELECT value FROM jsonb_array_elements(p_events) LOOP
        SELECT * INTO old_row FROM public.recall_fact_ledger_events
         WHERE tenant_id = p_tenant_id AND event_id = item->>'event_id' FOR UPDATE;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'fact redaction target event % is not in this tenant', item->>'event_id';
        END IF;
        IF event_map ? old_row.event_id THEN
            RAISE EXCEPTION 'fact redaction lists event % twice', old_row.event_id;
        END IF;
        event_map := event_map || jsonb_build_object(
            old_row.event_id,
            jsonb_build_object('event_id', item->>'new_event_id', 'request_id', item->>'request_id')
        );
        new_fact := CASE WHEN jsonb_typeof(item->'fact') = 'object' THEN item->'fact' END;
        IF old_row.fact IS NOT NULL AND NOT (old_row.fact ? 'redacted') AND new_fact ? 'redacted' THEN
            fact_map := public.recall_redaction_map_put(fact_map, old_row.fact_id, new_fact->>'fact_id');
        END IF;
        IF jsonb_typeof(item->'evidence_cards') IS DISTINCT FROM 'array'
           OR jsonb_array_length(item->'evidence_cards') <> jsonb_array_length(old_row.evidence_cards) THEN
            RAISE EXCEPTION 'fact redaction changes the cards of event %', old_row.event_id;
        END IF;
        FOR slot IN 0 .. jsonb_array_length(old_row.evidence_cards) - 1 LOOP
            old_card := old_row.evidence_cards->slot;
            new_card := item->'evidence_cards'->slot;
            IF new_card IS DISTINCT FROM old_card AND new_card ? 'redacted' AND NOT (old_card ? 'redacted') THEN
                card_map := public.recall_redaction_map_put(card_map, old_card->>'card_id', new_card->>'card_id');
            END IF;
        END LOOP;
    END LOOP;
    FOR item IN SELECT value FROM jsonb_array_elements(p_cards) LOOP
        card_map := public.recall_redaction_map_put(card_map, item->>'card_id', item->>'new_card_id');
    END LOOP;

    -- 2. Every planned row is exactly what those replacements make of it.
    FOR item IN SELECT value FROM jsonb_array_elements(p_events) LOOP
        SELECT * INTO old_row FROM public.recall_fact_ledger_events
         WHERE tenant_id = p_tenant_id AND event_id = item->>'event_id';
        new_fact := CASE WHEN jsonb_typeof(item->'fact') = 'object' THEN item->'fact' END;
        erased_now := old_row.fact IS NOT NULL AND NOT (old_row.fact ? 'redacted')
                      AND new_fact IS DISTINCT FROM old_row.fact;
        IF erased_now AND new_fact IS DISTINCT FROM jsonb_build_object(
               'fact_id', fact_map->>old_row.fact_id, 'redacted', true, 'redaction_id', p_redaction_id
           ) THEN
            RAISE EXCEPTION 'fact redaction rewrites the fact of event % into something else', old_row.event_id;
        END IF;
        IF NOT erased_now AND new_fact IS DISTINCT FROM old_row.fact THEN
            RAISE EXCEPTION 'fact redaction changes the fact of event %', old_row.event_id;
        END IF;
        -- (Computed first: plpgsql's IF reads its condition only up to the first THEN.)
        want_fact_id := CASE
            WHEN old_row.fact_id IS NOT NULL AND fact_map ? old_row.fact_id
            THEN fact_map->>old_row.fact_id ELSE old_row.fact_id END;
        IF item->>'fact_id' IS DISTINCT FROM want_fact_id THEN
            RAISE EXCEPTION 'fact redaction gives event % the wrong fact id', old_row.event_id;
        END IF;
        SELECT coalesce(jsonb_agg(to_jsonb(coalesce(fact_map->>value, value)) ORDER BY position), '[]'::jsonb)
          INTO expected
          FROM jsonb_array_elements_text(old_row.supersedes_fact_ids) WITH ORDINALITY AS ids(value, position);
        IF item->'supersedes_fact_ids' IS DISTINCT FROM expected THEN
            RAISE EXCEPTION 'fact redaction changes what event % supersedes', old_row.event_id;
        END IF;
        SELECT coalesce(jsonb_agg(
                   CASE WHEN NOT (card ? 'redacted') AND card_map ? (card->>'card_id')
                   THEN jsonb_build_object(
                       'card_id', card_map->>(card->>'card_id'), 'generation_id', card->'generation_id',
                       'redacted', true, 'redaction_id', p_redaction_id, 'source', card->'source',
                       'tenant_id', card->'tenant_id')
                   ELSE card END ORDER BY position), '[]'::jsonb)
          INTO expected
          FROM jsonb_array_elements(old_row.evidence_cards) WITH ORDINALITY AS cards(card, position);
        IF item->'evidence_cards' IS DISTINCT FROM expected THEN
            RAISE EXCEPTION 'fact redaction rewrites the cards of event % into something else', old_row.event_id;
        END IF;
        want_event := old_row.event_id;
        want_request := old_row.request_id;
        IF erased_now THEN
            want_event := NULL;  -- a fresh id: checked by shape below
        ELSIF old_row.event_type = 'retracted' AND old_row.request_id LIKE 'retraction:%' THEN
            holder := substr(old_row.request_id, length('retraction:') + 1);
            IF event_map ? holder THEN
                want_request := 'retraction:' || (event_map->holder->>'event_id');
                want_event := public.recall_decision_event_id(p_tenant_id, want_request, 'retracted');
            END IF;
        ELSIF old_row.event_type = 'superseded' AND old_row.event_id LIKE '%\_sup' THEN
            holder := left(old_row.event_id, length(old_row.event_id) - 4);
            IF event_map ? holder THEN
                want_event := (event_map->holder->>'event_id') || '_sup';
                want_request := (event_map->holder->>'request_id') || ':supersession';
            END IF;
        END IF;
        IF erased_now THEN
            IF item->>'new_event_id' !~ '^evt_[0-9a-f]{32}$'
               OR item->>'request_id' !~ '^redacted:[0-9a-f]{32}$' THEN
                RAISE EXCEPTION 'fact redaction gives event % ids of the wrong shape', old_row.event_id;
            END IF;
        ELSIF item->>'new_event_id' IS DISTINCT FROM want_event
              OR item->>'request_id' IS DISTINCT FROM want_request THEN
            RAISE EXCEPTION 'fact redaction renames event %, whose value it does not erase', old_row.event_id;
        END IF;
    END LOOP;

    -- 3. Apply them.
    PERFORM set_config('recall.redaction', p_redaction_id, true);

    FOR item IN SELECT value FROM jsonb_array_elements(p_events) LOOP
        UPDATE public.recall_fact_ledger_events
           SET event_id = item->>'new_event_id',
               request_id = item->>'request_id',
               fact_id = item->>'fact_id',
               fact = CASE WHEN jsonb_typeof(item->'fact') = 'object' THEN item->'fact' END,
               evidence_cards = item->'evidence_cards',
               supersedes_fact_ids = item->'supersedes_fact_ids'
         WHERE tenant_id = p_tenant_id AND event_id = item->>'event_id';
        v_events := v_events + 1;
        -- The snapshot of the event, if one was queued, follows its rewritten row.
        UPDATE public.recall_fact_materialization_outbox AS outbox
           SET event_id = ledger.event_id,
               event = outbox.event || jsonb_build_object(
                   'event_id', ledger.event_id, 'request_id', ledger.request_id,
                   'fact', ledger.fact, 'evidence_cards', ledger.evidence_cards,
                   'supersedes_fact_ids', ledger.supersedes_fact_ids)
          FROM public.recall_fact_ledger_events AS ledger
         WHERE outbox.tenant_id = p_tenant_id AND outbox.event_id = item->>'event_id'
           AND ledger.tenant_id = p_tenant_id AND ledger.event_id = item->>'new_event_id';
        GET DIAGNOSTICS touched = ROW_COUNT;
        v_outbox := v_outbox + touched;
    END LOOP;

    FOR item IN SELECT value FROM jsonb_array_elements(p_cards) LOOP
        SELECT card INTO old_card FROM public.recall_evidence_cards
         WHERE tenant_id = p_tenant_id AND card_id = item->>'card_id' FOR UPDATE;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'fact redaction target card % is not in this tenant', item->>'card_id';
        END IF;
        IF item->'card' IS DISTINCT FROM jsonb_build_object(
               'card_id', item->>'new_card_id', 'generation_id', old_card->'generation_id',
               'redacted', true, 'redaction_id', p_redaction_id, 'source', old_card->'source',
               'tenant_id', old_card->'tenant_id') THEN
            RAISE EXCEPTION 'fact redaction rewrites card % into something else', item->>'card_id';
        END IF;
        UPDATE public.recall_evidence_cards
           SET card_id = item->>'new_card_id',
               chunk_id = 'redacted',
               source_digest = 'redacted',
               card = item->'card'
         WHERE tenant_id = p_tenant_id AND card_id = item->>'card_id';
        v_cards := v_cards + 1;
    END LOOP;

    DELETE FROM public.recall_idempotency_receipts AS receipt
     WHERE receipt.tenant_id = p_tenant_id
       AND (
           p_all_receipts
           OR EXISTS (
               SELECT 1 FROM jsonb_array_elements_text(p_receipt_needles) AS needles(needle)
                WHERE length(needles.needle) > 0 AND strpos(receipt.result, needles.needle) > 0
           )
       );
    GET DIAGNOSTICS v_receipts = ROW_COUNT;

    INSERT INTO public.recall_fact_redactions (
        redaction_id, tenant_id, reason, writer, facts_redacted, events_rewritten, cards_redacted,
        receipts_deleted
    ) VALUES (
        p_redaction_id, p_tenant_id, p_reason, p_writer, p_facts_redacted, v_events, v_cards,
        v_receipts
    );

    PERFORM set_config('recall.redaction', '', true);
    RETURN jsonb_build_object(
        'events_rewritten', v_events,
        'cards_redacted', v_cards,
        'outbox_rewritten', v_outbox,
        'receipts_deleted', v_receipts
    );
END;
$$;

REVOKE ALL ON FUNCTION recall_redact_fact_values(
    TEXT, TEXT, TEXT, TEXT, INTEGER, JSONB, JSONB, JSONB, BOOLEAN
) FROM PUBLIC;
REVOKE ALL ON FUNCTION recall_redaction_map_put(JSONB, TEXT, TEXT) FROM PUBLIC;
