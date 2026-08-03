-- recall:concurrent-index __RECALL_TABLE___source_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS __RECALL_TABLE___source_idx
    ON __RECALL_TABLE__ (source);
