-- recall:concurrent-index __RECALL_TABLE___tenant_idx
CREATE INDEX CONCURRENTLY IF NOT EXISTS __RECALL_TABLE___tenant_idx
    ON __RECALL_TABLE__ (tenant_id);
