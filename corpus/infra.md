# Infrastructure notes

The indexer runs nightly and records an indexed_at timestamp per chunk so we can
detect stale corpora. Freshness is checked before the agent trusts a retrieval.
