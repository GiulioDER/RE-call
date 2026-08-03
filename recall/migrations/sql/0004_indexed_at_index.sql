-- recall:concurrent-index __RECALL_TABLE___indexed_at_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS __RECALL_TABLE___indexed_at_idx
    ON __RECALL_TABLE__ (indexed_at DESC);
