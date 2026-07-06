# Decisions

## 2026-05-02 — Caching layer
We decided to add a read-through cache in front of the pricing service. The
cache is keyed by symbol and invalidated on every write. This cut median
latency from 180ms to 22ms. Decision status: adopted.

## 2026-05-10 — Retry policy
Outbound calls use exponential backoff with a 3-attempt cap. We deliberately do
NOT retry on 4xx responses. Decision status: adopted.
