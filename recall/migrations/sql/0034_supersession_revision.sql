-- recall:transactional
-- Cache freshness must not count a tenant's lifetime removal history on every search.
-- The trigger shares the audit insert's transaction and serializes each tenant's increments.
CREATE TABLE IF NOT EXISTS recall_supersession_revisions (
    tenant_id TEXT PRIMARY KEY,
    revision BIGINT NOT NULL CHECK (revision > 0)
);
ALTER TABLE recall_supersession_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE recall_supersession_revisions FORCE ROW LEVEL SECURITY;
CREATE POLICY recall_supersession_revisions_tenant_isolation ON recall_supersession_revisions
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));

CREATE OR REPLACE FUNCTION recall_bump_supersession_revision() RETURNS trigger
LANGUAGE plpgsql SECURITY INVOKER SET search_path = pg_catalog, public AS $$
BEGIN
    INSERT INTO public.recall_supersession_revisions (tenant_id, revision)
    VALUES (NEW.tenant_id, 1)
    ON CONFLICT (tenant_id) DO UPDATE
        SET revision = public.recall_supersession_revisions.revision + 1;
    RETURN NEW;
END;
$$;
REVOKE ALL ON FUNCTION recall_bump_supersession_revision() FROM PUBLIC;
CREATE TRIGGER recall_supersession_revision_insert
    AFTER INSERT ON recall_audit_events
    FOR EACH ROW WHEN (NEW.event_type IN ('supersession_forgotten', 'supersession_withdrawn'))
    EXECUTE FUNCTION recall_bump_supersession_revision();
