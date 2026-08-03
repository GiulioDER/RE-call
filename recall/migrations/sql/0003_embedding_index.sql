-- recall:concurrent-index __RECALL_TABLE___emb_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS __RECALL_TABLE___emb_idx
    ON __RECALL_TABLE__ USING hnsw (embedding vector_cosine_ops);
