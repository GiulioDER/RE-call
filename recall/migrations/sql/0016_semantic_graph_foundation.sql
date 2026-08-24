-- recall:transactional
-- Deterministic, generation bound semantic graph.  Graph rows are derived evidence only.
CREATE TABLE IF NOT EXISTS recall_graph_entities_v1 (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    entity_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    entity_kind TEXT NOT NULL CHECK (
        entity_kind IN ('person', 'project', 'service', 'file', 'decision', 'event', 'concept', 'unknown')
    ),
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    extraction_method TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, generation_id, entity_id),
    FOREIGN KEY (tenant_id, generation_id)
        REFERENCES recall_generations (tenant_id, generation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS recall_graph_entities_lookup_idx
    ON recall_graph_entities_v1 (tenant_id, generation_id, normalized_name);

CREATE TABLE IF NOT EXISTS recall_graph_mentions_v1 (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    mention_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    mention_text TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, generation_id, mention_id),
    FOREIGN KEY (tenant_id, generation_id, entity_id)
        REFERENCES recall_graph_entities_v1 (tenant_id, generation_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, generation_id, chunk_id)
        REFERENCES recall_chunks_v1 (tenant_id, generation_id, chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS recall_graph_mentions_chunk_idx
    ON recall_graph_mentions_v1 (tenant_id, generation_id, chunk_id);
CREATE INDEX IF NOT EXISTS recall_graph_mentions_entity_idx
    ON recall_graph_mentions_v1 (tenant_id, generation_id, entity_id);

CREATE TABLE IF NOT EXISTS recall_graph_relations_v1 (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    relation_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (
        relation IN ('supports', 'contradicts', 'references', 'depends_on', 'caused', 'same_entity')
    ),
    extraction_method TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    status TEXT NOT NULL CHECK (status IN ('authored', 'candidate')),
    uncertainty JSONB NOT NULL DEFAULT '[]'::jsonb,
    pipeline_fingerprint CHAR(64),
    corpus_fingerprint CHAR(64),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, generation_id, relation_id),
    FOREIGN KEY (tenant_id, generation_id, subject_id)
        REFERENCES recall_graph_entities_v1 (tenant_id, generation_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, generation_id, object_id)
        REFERENCES recall_graph_entities_v1 (tenant_id, generation_id, entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS recall_graph_relations_subject_idx
    ON recall_graph_relations_v1 (tenant_id, generation_id, subject_id, relation);
CREATE INDEX IF NOT EXISTS recall_graph_relations_object_idx
    ON recall_graph_relations_v1 (tenant_id, generation_id, object_id, relation);

CREATE TABLE IF NOT EXISTS recall_graph_relation_evidence_v1 (
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    relation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (tenant_id, generation_id, relation_id, chunk_id),
    FOREIGN KEY (tenant_id, generation_id, relation_id)
        REFERENCES recall_graph_relations_v1 (tenant_id, generation_id, relation_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, generation_id, chunk_id)
        REFERENCES recall_chunks_v1 (tenant_id, generation_id, chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS recall_graph_relation_evidence_chunk_idx
    ON recall_graph_relation_evidence_v1 (tenant_id, generation_id, chunk_id);

DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'recall_graph_entities_v1',
        'recall_graph_mentions_v1',
        'recall_graph_relations_v1',
        'recall_graph_relation_evidence_v1'
    ] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS %I ON %I', table_name || '_tenant_isolation', table_name);
        EXECUTE format(
            'CREATE POLICY %I ON %I USING (tenant_id = current_setting(''%s'', true)) WITH CHECK (tenant_id = current_setting(''%s'', true))',
            table_name || '_tenant_isolation', table_name, '__RECALL_TENANT_GUC__', '__RECALL_TENANT_GUC__'
        );
    END LOOP;
END $$;
