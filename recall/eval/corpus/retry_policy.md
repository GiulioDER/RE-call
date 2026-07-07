# Retry policy

Outbound HTTP calls use exponential backoff with jitter and a hard cap of three attempts. We do
not retry on 4xx client errors because they are not transient. Idempotency keys are required on all
POST endpoints so a retried write is safe. Status: adopted.
