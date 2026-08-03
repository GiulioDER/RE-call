-- recall:concurrent-index __RECALL_TABLE___tsv_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS __RECALL_TABLE___tsv_idx
    ON __RECALL_TABLE__ USING GIN (tsv);
