---
supersedes: rate_limits_v1.md
---
# API rate limits (revised)

Rate limiting was tightened after the June capacity incident: each client key is now limited
to 20 requests per second, enforced at the gateway. Burst credits allow short spikes up to
40 rps. Status: adopted, replaces the original limits.
