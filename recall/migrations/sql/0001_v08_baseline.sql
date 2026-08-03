-- recall:transactional
-- Baseline the v0.8 schema and adopt older single-tenant tables in place.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS __RECALL_TABLE__ (
    tenant_id TEXT NOT NULL DEFAULT '__RECALL_DEFAULT_TENANT__',
    id TEXT NOT NULL,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(__RECALL_DIM__),
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_indexed_at TIMESTAMPTZ DEFAULT now(),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (tenant_id, id)
);

ALTER TABLE __RECALL_TABLE__
    ADD COLUMN IF NOT EXISTS first_indexed_at TIMESTAMPTZ;
ALTER TABLE __RECALL_TABLE__
    ALTER COLUMN first_indexed_at SET DEFAULT now();
ALTER TABLE __RECALL_TABLE__
    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT '__RECALL_DEFAULT_TENANT__';

DO $recall$
DECLARE
    current_pk name;
    key_count integer;
    tenant_in_key boolean;
    id_in_key boolean;
BEGIN
    SELECT c.conname,
           array_length(c.conkey, 1),
           EXISTS (
               SELECT 1 FROM unnest(c.conkey) AS key(attnum)
               JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = key.attnum
               WHERE a.attname = 'tenant_id'
           ),
           EXISTS (
               SELECT 1 FROM unnest(c.conkey) AS key(attnum)
               JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = key.attnum
               WHERE a.attname = 'id'
           )
      INTO current_pk, key_count, tenant_in_key, id_in_key
      FROM pg_constraint c
     WHERE c.conrelid = '__RECALL_TABLE__'::regclass AND c.contype = 'p';

    IF current_pk IS NULL OR key_count <> 2 OR NOT tenant_in_key OR NOT id_in_key THEN
        IF current_pk IS NOT NULL THEN
            EXECUTE format('ALTER TABLE __RECALL_TABLE__ DROP CONSTRAINT %I', current_pk);
        END IF;
        ALTER TABLE __RECALL_TABLE__ ADD PRIMARY KEY (tenant_id, id);
    END IF;
END
$recall$;

ALTER TABLE __RECALL_TABLE__ ENABLE ROW LEVEL SECURITY;
ALTER TABLE __RECALL_TABLE__ FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS __RECALL_TABLE___tenant_isolation ON __RECALL_TABLE__;
CREATE POLICY __RECALL_TABLE___tenant_isolation ON __RECALL_TABLE__
    USING (tenant_id = current_setting('__RECALL_TENANT_GUC__', true))
    WITH CHECK (tenant_id = current_setting('__RECALL_TENANT_GUC__', true));
