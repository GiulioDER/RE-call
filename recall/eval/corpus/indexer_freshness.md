# Indexer freshness

The indexer runs nightly at 02:00 UTC and records an indexed_at timestamp per chunk so we can
detect stale corpora. Anything older than two days is flagged stale before the agent trusts a
retrieval.
