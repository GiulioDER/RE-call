-- recall:concurrent-index recall_chunks_v1_tsv_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS recall_chunks_v1_tsv_idx
    ON recall_chunks_v1 USING GIN (tsv);
