-- recall:concurrent-index recall_sparse_v1_vec_idx
-- HNSW over the sparse vectors, scored by INNER PRODUCT: SPLADE ranks by dot product, so
-- `sparsevec_ip_ops` is the operator class and `<#>` (negative inner product) is the operator.
--
-- ⚠️ This index is what imposes the 1000-non-zero ceiling enforced by 0012's CHECK and by
-- SpladeEncoder. The `sparsevec` TYPE accepts 16000; the INDEX accepts 1000 and raises on INSERT
-- past it. Measured on pgvector 0.8.4, not read from documentation.
--
-- Separate file from 0012 because load_migrations() requires each migration to declare exactly one
-- execution mode, and a concurrent index cannot run inside the transactional block that creates
-- the table.
CREATE INDEX CONCURRENTLY IF NOT EXISTS recall_sparse_v1_vec_idx
    ON recall_sparse_v1 USING hnsw (vec sparsevec_ip_ops);
