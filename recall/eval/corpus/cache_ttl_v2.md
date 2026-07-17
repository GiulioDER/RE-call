---
supersedes: cache_ttl_v1.md
---
# Snapshot caching revision

After the stale-quote incident, snapshot entries now expire after 60 seconds and are refreshed
proactively by a background worker. This replaces the original lazy expiry choice.
