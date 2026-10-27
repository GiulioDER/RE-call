-- recall:concurrent-index recall_supersession_latest_idx
-- Latest-state reads rank identities before retrieving only their winning payloads.
CREATE INDEX CONCURRENTLY IF NOT EXISTS recall_supersession_latest_idx
    ON recall_audit_events (tenant_id, (payload->>'scope'), source_uri, created_at DESC, event_id DESC)
    WHERE event_type IN ('supersession_forgotten', 'supersession_withdrawn')
      AND source_uri <> '' AND jsonb_typeof(payload) = 'object'
      AND jsonb_typeof(payload->'scope') = 'string'
      AND jsonb_typeof(payload->'file') = 'string';
