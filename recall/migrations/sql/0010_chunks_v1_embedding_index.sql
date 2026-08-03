-- recall:concurrent-index recall_chunks_v1_embedding_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS recall_chunks_v1_embedding_idx
    ON recall_chunks_v1 USING hnsw (embedding vector_cosine_ops);
