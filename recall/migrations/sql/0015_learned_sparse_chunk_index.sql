-- recall:concurrent-index recall_sparse_v1_chunk_idx
-- Lets the erasure scrub seek to a chunk id. The sidecar's PK and corpus index both order
-- (tenant_id, chunk_table, profile_id, id), with profile_id BETWEEN chunk_table and id — but the
-- scrub `DELETE ... WHERE tenant_id=? AND chunk_table=? AND id = ANY(?)` (store.py, generations.py)
-- deliberately omits profile_id, so a dead chunk's rows die under every profile. Without a
-- profile-free index, that DELETE can only use the (tenant_id, chunk_table) prefix and then filters
-- id across every profile's rows for the whole corpus: forgetting one source from an N-chunk,
-- P-profile corpus scans O(N*P) index tuples instead of O(deleted*P). This index makes the scrub
-- cost proportional to the rows deleted. gc()'s anti-join scrub (id no longer in recall_chunks_v1)
-- benefits identically.
--
-- Separate file from 0012, like 0013: a concurrent index cannot run inside the transactional block
-- that creates the table, and load_migrations() requires one execution mode per migration.
CREATE INDEX CONCURRENTLY IF NOT EXISTS recall_sparse_v1_chunk_idx
    ON recall_sparse_v1 (tenant_id, chunk_table, id);
