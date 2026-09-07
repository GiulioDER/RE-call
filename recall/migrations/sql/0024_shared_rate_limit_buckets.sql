-- recall:transactional
CREATE TABLE IF NOT EXISTS recall_rate_limit_buckets (
    tenant_id TEXT NOT NULL,
    budget_key TEXT NOT NULL,
    tokens DOUBLE PRECISION NOT NULL CHECK (tokens >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, budget_key)
);

CREATE INDEX IF NOT EXISTS recall_rate_limit_buckets_updated_idx
    ON recall_rate_limit_buckets (updated_at);
