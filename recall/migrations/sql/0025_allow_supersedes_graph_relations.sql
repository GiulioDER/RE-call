-- recall:transactional
-- Allow authored supersession relations in the semantic graph.
ALTER TABLE recall_graph_relations_v1
    DROP CONSTRAINT IF EXISTS recall_graph_relations_v1_relation_check;

ALTER TABLE recall_graph_relations_v1
    ADD CONSTRAINT recall_graph_relations_v1_relation_check
    CHECK (
        relation IN (
            'supports',
            'contradicts',
            'references',
            'depends_on',
            'caused',
            'same_entity',
            'supersedes'
        )
    );
