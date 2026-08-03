-- recall:concurrent-index __RECALL_TABLE___file_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS __RECALL_TABLE___file_idx
    ON __RECALL_TABLE__ ((metadata->>'file'));
